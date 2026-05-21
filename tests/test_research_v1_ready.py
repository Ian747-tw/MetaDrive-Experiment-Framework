from __future__ import annotations

from pathlib import Path

from scripts.check_research_v1_ready import commands


def test_research_v1_ready_commands_honor_root() -> None:
    root = Path("/tmp/shared/research_v1")
    checkpoint = root / "base_pretrain_s42/checkpoints/final.zip"
    eval_csv = root / "eval_base_pretrain/eval/heldout_random.csv"
    buffer = root / "base_explore_large/buffers/failure_buffer.jsonl"
    output = commands(root, checkpoint, eval_csv, buffer)
    assert "experiment.output_dir=/tmp/shared/research_v1/base_pretrain_s42" in output
    assert "experiment.output_dir=/tmp/shared/research_v1/eval_base_pretrain" in output
    assert "experiment.output_dir=/tmp/shared/research_v1/base_explore_large" in output
    assert "algorithm.checkpoint_path=/tmp/shared/research_v1/base_pretrain_s42/checkpoints/final.zip" in output
    assert "python scripts/check_research_v1_ready.py --root /tmp/shared/research_v1" in output
    assert "--buffer /tmp/shared/research_v1/base_explore_large/buffers/failure_buffer.jsonl" in output


def test_research_v1_ready_commands_handle_shallow_paths() -> None:
    root = Path("/tmp/shared/research_v1")
    output = commands(root, Path("final.zip"), Path("eval.csv"), Path("failure_buffer.jsonl"))
    assert "experiment.output_dir=/tmp/shared/research_v1/base_pretrain_s42" in output
    assert "experiment.output_dir=/tmp/shared/research_v1/eval_base_pretrain" in output
    assert "experiment.output_dir=/tmp/shared/research_v1/base_explore_large" in output
    assert "cp /tmp/shared/research_v1/base_pretrain_s42/checkpoints/final.zip final.zip" in output
    assert "--checkpoint final.zip" in output
    assert "--eval-csv eval.csv" in output
    assert "--buffer failure_buffer.jsonl" in output


def test_research_v1_ready_commands_handle_one_level_paths() -> None:
    root = Path("/tmp/shared/research_v1")
    output = commands(root, Path("ckpt/final.zip"), Path("evals/eval.csv"), Path("buffers/failure_buffer.jsonl"))
    assert "experiment.output_dir=ckpt" in output
    assert "experiment.output_dir=evals" in output
    assert "experiment.output_dir=buffers" in output
    assert "cp ckpt/checkpoints/final.zip ckpt/final.zip" in output
    assert "cp evals/eval/heldout_random.csv evals/eval.csv" in output
    assert "cp buffers/buffers/failure_buffer.jsonl buffers/failure_buffer.jsonl" in output
    assert "--checkpoint ckpt/final.zip" in output
    assert "--eval-csv evals/eval.csv" in output
    assert "--buffer buffers/failure_buffer.jsonl" in output


def test_research_v1_ready_commands_preserve_absolute_shallow_paths() -> None:
    root = Path("/tmp/shared/research_v1")
    output = commands(root, Path("/tmp/final.zip"), Path("/tmp/eval.csv"), Path("/tmp/failure_buffer.jsonl"))
    assert "experiment.output_dir=/tmp" in output
    assert "cp /tmp/checkpoints/final.zip /tmp/final.zip" in output
    assert "cp /tmp/eval/heldout_random.csv /tmp/eval.csv" in output
    assert "cp /tmp/buffers/failure_buffer.jsonl /tmp/failure_buffer.jsonl" in output
    assert "--checkpoint /tmp/final.zip" in output
    assert "--eval-csv /tmp/eval.csv" in output
    assert "--buffer /tmp/failure_buffer.jsonl" in output
