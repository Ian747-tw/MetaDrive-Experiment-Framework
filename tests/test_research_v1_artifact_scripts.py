from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest

from scripts import download_research_v1_artifacts as download
from scripts import package_research_v1_artifacts as package


CHECKPOINT = Path("runs/research_v1/base_pretrain_s42/checkpoints/final.zip")
EVAL_CSV = Path("runs/research_v1/eval_base_pretrain/eval/heldout_random.csv")
BUFFER = Path("runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl")


def write_fake_artifacts(root: Path) -> None:
    checkpoint = root / CHECKPOINT
    eval_csv = root / EVAL_CSV
    buffer = root / BUFFER
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    buffer.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake checkpoint")
    eval_csv.write_text(
        "n_episodes,success_rate,route_completion_mean,timeout_rate\n100,0.44,0.699798,0.56\n",
        encoding="utf-8",
    )
    buffer.write_text(
        "\n".join(
            [
                json.dumps({"seed": 1, "failure_mode": "collision", "risk_score": 0.8, "route_completion": 0.5}),
                json.dumps({"seed": 2, "failure_mode": "offroad", "risk_score": 0.7, "route_completion": 0.6}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def fake_manifest(root: Path) -> dict:
    def entry(path: Path) -> dict:
        source = root / path
        return {"path": path.as_posix(), "sha256": package.sha256_file(source), "size_bytes": source.stat().st_size}

    return {
        "release_name": "research-v1-foundation-v1",
        "created_at": "2026-05-21T00:00:00+00:00",
        "repo": package.REPO,
        "artifacts": {
            "checkpoint": entry(CHECKPOINT),
            "eval_csv": entry(EVAL_CSV),
            "failure_buffer": {**entry(BUFFER), "record_count": 2},
        },
        "base_checkpoint_metrics": {
            "success_rate": 0.44,
            "route_completion_mean": 0.699798,
            "timeout_rate": 0.56,
        },
        "failure_buffer_quality": {
            "record_count": 2,
            "distinct_seed_count": 2,
            "unknown_fraction": 0.0,
            "failure_mode_counts": {"collision": 1, "offroad": 1},
        },
    }


def test_package_archive_includes_expected_paths_and_hashes(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_fake_artifacts(tmp_path)
    manifest = fake_manifest(tmp_path)
    manifest_path = Path("dist/research_v1_artifact_manifest.json")
    archive_path = Path("dist/research_v1_foundation_artifacts.tar.gz")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    package.package_archive(archive_path, manifest_path, manifest)

    with tarfile.open(archive_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert CHECKPOINT.as_posix() in names
    assert EVAL_CSV.as_posix() in names
    assert BUFFER.as_posix() in names
    assert "research_v1_artifact_manifest.json" in names
    assert manifest["artifacts"]["checkpoint"]["sha256"] == package.sha256_file(tmp_path / CHECKPOINT)


def test_download_skip_download_extracts_local_archive(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    write_fake_artifacts(source_root)
    manifest = fake_manifest(source_root)
    manifest_path = source_root / "research_v1_artifact_manifest.json"
    archive_path = tmp_path / download.DEFAULT_ARCHIVE
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(source_root)
    package.package_archive(archive_path, manifest_path, manifest)

    target_root = tmp_path / "target"
    target_root.mkdir()
    monkeypatch.chdir(target_root)
    monkeypatch.setattr(download, "run_readiness", lambda: None)
    monkeypatch.setattr(sys, "argv", ["download", "--archive", str(archive_path), "--skip-download"])

    assert download.main() == 0
    assert (target_root / CHECKPOINT).exists()
    assert (target_root / EVAL_CSV).exists()
    assert (target_root / BUFFER).exists()
    assert (target_root / download.DEFAULT_MANIFEST).exists()


def test_download_rejects_archive_members_outside_destination(tmp_path) -> None:
    archive = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("bad", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../outside.txt")

    with pytest.raises(ValueError, match="escapes destination"):
        download.extract_archive(archive, tmp_path / "extract")


def test_download_rejects_unsupported_archive_member_types(tmp_path) -> None:
    archive = tmp_path / "link.tar.gz"
    link = tarfile.TarInfo("link")
    link.type = tarfile.SYMTYPE
    link.linkname = "runs/research_v1/base_pretrain_s42/checkpoints/final.zip"
    with tarfile.open(archive, "w:gz") as tar:
        tar.addfile(link)

    with pytest.raises(ValueError, match="unsupported archive member type"):
        download.extract_archive(archive, tmp_path / "extract")


def test_download_release_pins_canonical_repo(monkeypatch) -> None:
    calls = []

    def fake_run(command, check):
        calls.append(command)

    monkeypatch.setattr(download.subprocess, "run", fake_run)

    download.download_release(download.DEFAULT_RELEASE, Path(download.DEFAULT_ARCHIVE))

    assert "--repo" in calls[0]
    assert download.REPO in calls[0]


def test_download_release_honors_archive_directory(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run(command, check):
        calls.append(command)

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    archive = tmp_path / "dist" / download.DEFAULT_ARCHIVE

    download.download_release(download.DEFAULT_RELEASE, archive)

    assert "--dir" in calls[0]
    assert archive.parent.as_posix() in calls[0]
    assert archive.parent.exists()


def test_download_rejects_wrong_manifest_repo(tmp_path) -> None:
    write_fake_artifacts(tmp_path)
    manifest = fake_manifest(tmp_path)
    manifest["repo"] = "someone/fork"

    with pytest.raises(RuntimeError, match="does not match expected repo"):
        download.validate_manifest_repo(manifest)


def test_download_skip_download_uses_existing_manifest_without_same_file_error(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    write_fake_artifacts(source_root)
    manifest = fake_manifest(source_root)
    archive_path = tmp_path / download.DEFAULT_ARCHIVE
    manifest_path = source_root / download.DEFAULT_MANIFEST
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(source_root)
    with tarfile.open(archive_path, "w:gz") as tar:
        for entry in manifest["artifacts"].values():
            tar.add(entry["path"], arcname=entry["path"])

    target_root = tmp_path / "target"
    target_root.mkdir()
    (target_root / download.DEFAULT_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(target_root)
    monkeypatch.setattr(download, "run_readiness", lambda: None)
    monkeypatch.setattr(sys, "argv", ["download", "--archive", str(archive_path), "--skip-download"])

    assert download.main() == 0


def test_force_does_not_delete_existing_artifacts_when_archive_is_incomplete(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    write_fake_artifacts(source_root)
    manifest = fake_manifest(source_root)
    archive_path = tmp_path / download.DEFAULT_ARCHIVE
    manifest_path = source_root / download.DEFAULT_MANIFEST
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(source_root)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(download.DEFAULT_MANIFEST, arcname=download.DEFAULT_MANIFEST)
        tar.add(EVAL_CSV, arcname=EVAL_CSV.as_posix())
        tar.add(BUFFER, arcname=BUFFER.as_posix())

    target_root = tmp_path / "target"
    target_root.mkdir()
    existing_checkpoint = target_root / CHECKPOINT
    existing_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing_checkpoint.write_bytes(b"existing checkpoint")
    monkeypatch.chdir(target_root)
    monkeypatch.setattr(download, "run_readiness", lambda: None)
    monkeypatch.setattr(sys, "argv", ["download", "--archive", str(archive_path), "--skip-download", "--force"])

    assert download.main() == 1
    assert existing_checkpoint.read_bytes() == b"existing checkpoint"


def test_existing_artifact_protection_fails_without_force(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_fake_artifacts(tmp_path)
    manifest = fake_manifest(tmp_path)
    (tmp_path / CHECKPOINT).write_bytes(b"different checkpoint")

    with pytest.raises(RuntimeError, match="existing artifact differs"):
        download.prepare_existing_targets(manifest, force=False)


def test_existing_artifact_protection_allows_force(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_fake_artifacts(tmp_path)
    manifest = fake_manifest(tmp_path)
    (tmp_path / CHECKPOINT).write_bytes(b"different checkpoint")

    download.prepare_existing_targets(manifest, force=True)

    assert not (tmp_path / CHECKPOINT).exists()
