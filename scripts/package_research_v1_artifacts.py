from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_base_checkpoint_quality import check_quality, read_metrics
from scripts.check_failure_buffer_quality import load_records, summarize


DEFAULT_CHECKPOINT = Path("runs/research_v1/base_pretrain_s42/checkpoints/final.zip")
DEFAULT_EVAL_CSV = Path("runs/research_v1/eval_base_pretrain/eval/heldout_random.csv")
DEFAULT_BUFFER = Path("runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl")
DEFAULT_RELEASE_NAME = "research-v1-foundation-v1"
REPO = "Ian747-tw/MetaDrive-Experiment-Framework"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_artifacts(
    checkpoint: Path,
    eval_csv: Path,
    buffer: Path,
    root: Path,
    min_failures: int,
    min_episodes: int,
    min_success_rate: float,
    min_route_completion: float,
    max_timeout_rate: float,
    max_unknown_fraction: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    for path in (checkpoint, eval_csv, buffer):
        if not path.exists():
            raise FileNotFoundError(f"missing artifact: {path}")

    metrics = read_metrics(eval_csv)
    metric_failures = check_quality(
        metrics,
        min_episodes=min_episodes,
        min_success_rate=min_success_rate,
        min_route_completion=min_route_completion,
        max_timeout_rate=max_timeout_rate,
    )
    if metric_failures:
        raise RuntimeError("base checkpoint quality failed: " + "; ".join(metric_failures))

    buffer_failures, buffer_warnings, buffer_summary = summarize(
        load_records(buffer),
        min_records=min_failures,
        max_unknown_fraction=max_unknown_fraction,
        require_multiple_modes=True,
    )
    if buffer_failures or buffer_warnings:
        messages = [*buffer_failures, *[f"warning: {warning}" for warning in buffer_warnings]]
        raise RuntimeError("failure buffer quality failed: " + "; ".join(messages))

    readiness_cmd = [
        "python",
        "scripts/check_research_v1_ready.py",
        "--root",
        root.as_posix(),
        "--checkpoint",
        checkpoint.as_posix(),
        "--eval-csv",
        eval_csv.as_posix(),
        "--buffer",
        buffer.as_posix(),
        "--min-failures",
        str(min_failures),
        "--min-episodes",
        str(min_episodes),
        "--min-success-rate",
        str(min_success_rate),
        "--min-route-completion",
        str(min_route_completion),
        "--max-timeout-rate",
        str(max_timeout_rate),
        "--max-unknown-fraction",
        str(max_unknown_fraction),
    ]
    subprocess.run(readiness_cmd, check=True)
    return metrics, buffer_summary


def build_manifest(
    release_name: str,
    checkpoint: Path,
    eval_csv: Path,
    buffer: Path,
    metrics: dict[str, float],
    buffer_summary: dict[str, Any],
) -> dict[str, Any]:
    failure_buffer_entry = artifact_entry(buffer)
    failure_buffer_entry["record_count"] = buffer_summary["record_count"]
    return {
        "release_name": release_name,
        "created_at": datetime.now(UTC).isoformat(),
        "repo": REPO,
        "artifacts": {
            "checkpoint": artifact_entry(checkpoint),
            "eval_csv": artifact_entry(eval_csv),
            "failure_buffer": failure_buffer_entry,
        },
        "base_checkpoint_metrics": {
            "success_rate": metrics["success_rate"],
            "route_completion_mean": metrics["route_completion_mean"],
            "timeout_rate": metrics["timeout_rate"],
        },
        "failure_buffer_quality": {
            "record_count": buffer_summary["record_count"],
            "distinct_seed_count": buffer_summary["distinct_seed_count"],
            "unknown_fraction": buffer_summary["unknown_fraction"],
            "failure_mode_counts": buffer_summary["failure_mode_counts"],
        },
    }


def write_release_notes(path: Path, manifest: dict[str, Any]) -> None:
    metrics = manifest["base_checkpoint_metrics"]
    buffer = manifest["failure_buffer_quality"]
    path.write_text(
        f"""# Research V1 Foundation Artifacts

Release: `{manifest["release_name"]}`

These assets provide the shared generated Research V1 foundation artifacts for comparable Axis 1-4 experiments.

## Assets

- `research_v1_foundation_artifacts.tar.gz`
- `research_v1_artifact_manifest.json`

## Extracted Paths

- `runs/research_v1/base_pretrain_s42/checkpoints/final.zip`
- `runs/research_v1/eval_base_pretrain/eval/heldout_random.csv`
- `runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl`

## Quality Metrics

- base success_rate: `{metrics["success_rate"]}`
- base route_completion_mean: `{metrics["route_completion_mean"]}`
- base timeout_rate: `{metrics["timeout_rate"]}`
- failure buffer records: `{buffer["record_count"]}`
- failure buffer distinct seeds: `{buffer["distinct_seed_count"]}`
- failure buffer unknown_fraction: `{buffer["unknown_fraction"]}`
- failure mode counts: `{buffer["failure_mode_counts"]}`

## Download And Validate

```bash
python scripts/download_research_v1_artifacts.py --release {manifest["release_name"]}
make validate-research-v1-artifacts
```

Manual extraction:

```bash
gh release download {manifest["release_name"]} --pattern "research_v1_foundation_artifacts.tar.gz"
tar -xzf research_v1_foundation_artifacts.tar.gz
make validate-research-v1-artifacts
```
""",
        encoding="utf-8",
    )


def package_archive(output: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="research_v1_artifacts_") as tmp:
        staging = Path(tmp)
        for entry in manifest["artifacts"].values():
            source = Path(entry["path"])
            destination = staging / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(manifest_path, staging / manifest_path.name)
        with tarfile.open(output, "w:gz") as tar:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(staging))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/research_v1")
    parser.add_argument("--output", default="dist/research_v1_foundation_artifacts.tar.gz")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--eval-csv", default=str(DEFAULT_EVAL_CSV))
    parser.add_argument("--buffer", default=str(DEFAULT_BUFFER))
    parser.add_argument("--release-name", default=DEFAULT_RELEASE_NAME)
    parser.add_argument("--min-failures", type=int, default=1000)
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--min-success-rate", type=float, default=0.10)
    parser.add_argument("--min-route-completion", type=float, default=0.35)
    parser.add_argument("--max-timeout-rate", type=float, default=0.95)
    parser.add_argument("--max-unknown-fraction", type=float, default=0.25)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    checkpoint = Path(args.checkpoint)
    eval_csv = Path(args.eval_csv)
    buffer = Path(args.buffer)
    metrics, buffer_summary = validate_artifacts(
        checkpoint,
        eval_csv,
        buffer,
        root,
        args.min_failures,
        args.min_episodes,
        args.min_success_rate,
        args.min_route_completion,
        args.max_timeout_rate,
        args.max_unknown_fraction,
    )
    manifest = build_manifest(args.release_name, checkpoint, eval_csv, buffer, metrics, buffer_summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.parent / "research_v1_artifact_manifest.json"
    release_notes_path = output.parent / "research_v1_foundation_release_notes.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_release_notes(release_notes_path, manifest)
    package_archive(output, manifest_path, manifest)

    print(f"Wrote archive: {output}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote release notes: {release_notes_path}")
    print()
    print("Create release if missing:")
    print(
        f"gh release create {args.release_name} {output} {manifest_path} "
        f'--title "Research V1 Foundation Artifacts" --notes-file {release_notes_path}'
    )
    print()
    print("Upload assets if release exists:")
    print(f"gh release upload {args.release_name} {output} {manifest_path} --clobber")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
