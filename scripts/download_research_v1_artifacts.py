from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_RELEASE = "research-v1-foundation-v1"
DEFAULT_ARCHIVE = "research_v1_foundation_artifacts.tar.gz"
DEFAULT_MANIFEST = "research_v1_artifact_manifest.json"
REPO = "Ian747-tw/MetaDrive-Experiment-Framework"
EXPECTED_PATHS = [
    Path("runs/research_v1/base_pretrain_s42/checkpoints/final.zip"),
    Path("runs/research_v1/eval_base_pretrain/eval/heldout_random.csv"),
    Path("runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_artifact_paths(manifest: dict[str, Any]) -> dict[Path, dict[str, Any]]:
    return {Path(entry["path"]): entry for entry in manifest["artifacts"].values()}


def download_release(release: str, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "gh",
        "release",
        "download",
        release,
        "--repo",
        REPO,
        "--pattern",
        archive.name,
        "--pattern",
        DEFAULT_MANIFEST,
    ]
    if archive.parent != Path("."):
        command.extend(["--dir", archive.parent.as_posix()])
    subprocess.run(
        command,
        check=True,
    )


def extract_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        destination_resolved = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination_resolved):
                raise ValueError(f"archive member escapes destination: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported archive member type: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(destination, filter="data")
        else:
            tar.extractall(destination)


def validate_manifest_repo(manifest: dict[str, Any]) -> None:
    repo = manifest.get("repo")
    if repo != REPO:
        raise RuntimeError(f"manifest repo {repo!r} does not match expected repo {REPO!r}")


def prepare_existing_targets(manifest: dict[str, Any], force: bool) -> None:
    for path, entry in manifest_artifact_paths(manifest).items():
        if not path.exists():
            continue
        actual = sha256_file(path)
        if actual == entry["sha256"]:
            continue
        if not force:
            raise RuntimeError(
                f"existing artifact differs from manifest and --force was not set: {path} "
                f"(actual {actual}, expected {entry['sha256']})"
            )
        path.unlink()


def verify_outputs(manifest: dict[str, Any]) -> None:
    artifact_paths = manifest_artifact_paths(manifest)
    for path in EXPECTED_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"expected artifact missing after extraction: {path}")
    for path, entry in artifact_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"manifest artifact missing after extraction: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch for {path}: actual {actual}, expected {entry['sha256']}")


def verify_archive_sources(extract_dir: Path, manifest: dict[str, Any]) -> None:
    for entry in manifest["artifacts"].values():
        source = extract_dir / entry["path"]
        if not source.exists():
            raise FileNotFoundError(f"archive missing artifact: {entry['path']}")
        actual = sha256_file(source)
        if actual != entry["sha256"]:
            raise RuntimeError(f"archive checksum mismatch for {entry['path']}: actual {actual}, expected {entry['sha256']}")


def run_readiness() -> None:
    subprocess.run(
        [
            "python",
            "scripts/check_research_v1_ready.py",
            "--root",
            "runs/research_v1",
            "--checkpoint",
            "runs/research_v1/base_pretrain_s42/checkpoints/final.zip",
            "--eval-csv",
            "runs/research_v1/eval_base_pretrain/eval/heldout_random.csv",
            "--buffer",
            "runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl",
            "--min-failures",
            "1000",
            "--min-episodes",
            "100",
            "--min-success-rate",
            "0.10",
            "--min-route-completion",
            "0.35",
            "--max-timeout-rate",
            "0.95",
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    archive = Path(args.archive)
    manifest_path = Path(DEFAULT_MANIFEST)
    try:
        if not args.skip_download:
            download_release(args.release, archive)
        if not archive.exists():
            raise FileNotFoundError(f"archive not found: {archive}")

        with tempfile.TemporaryDirectory(prefix="research_v1_extract_") as tmp:
            extract_dir = Path(tmp)
            extract_archive(archive, extract_dir)
            extracted_manifest = extract_dir / DEFAULT_MANIFEST
            if not extracted_manifest.exists():
                if not manifest_path.exists():
                    raise FileNotFoundError(f"manifest not found in archive or working directory: {DEFAULT_MANIFEST}")
                extracted_manifest = manifest_path
            manifest = load_manifest(extracted_manifest)
            validate_manifest_repo(manifest)
            verify_archive_sources(extract_dir, manifest)
            prepare_existing_targets(manifest, args.force)
            for path, entry in manifest_artifact_paths(manifest).items():
                source = extract_dir / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, path)
            if extracted_manifest.resolve() != manifest_path.resolve():
                shutil.copy2(extracted_manifest, manifest_path)

        verify_outputs(load_manifest(manifest_path))
        run_readiness()
    except Exception as exc:
        print(f"FAIL Research V1 artifact download/validation: {exc}")
        return 1

    print("PASS Research V1 artifact download/validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
