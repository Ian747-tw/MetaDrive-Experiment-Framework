from __future__ import annotations

from pathlib import Path

from scripts.check_research_v1_ready import commands


def test_research_v1_ready_commands_honor_root() -> None:
    root = Path("/tmp/shared/research_v1")
    output = commands(root)
    assert "experiment.output_dir=/tmp/shared/research_v1/base_pretrain_s42" in output
    assert "experiment.output_dir=/tmp/shared/research_v1/eval_base_pretrain" in output
    assert "experiment.output_dir=/tmp/shared/research_v1/base_explore" in output
    assert "algorithm.checkpoint_path=/tmp/shared/research_v1/base_pretrain_s42/checkpoints/final.zip" in output
    assert "python scripts/check_research_v1_ready.py --root /tmp/shared/research_v1" in output
