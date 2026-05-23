from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.select_best_checkpoint import discover_checkpoints, hard_reject, select_best
from scripts import select_best_checkpoint


def test_discover_checkpoints_orders_periodic_then_best_then_final(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for name in ("final.zip", "latest_300000_steps.zip", "latest_100000_steps.zip", "best_mean_reward.zip"):
        (checkpoint_dir / name).write_text("x", encoding="utf-8")

    assert [path.name for path in discover_checkpoints(tmp_path)] == [
        "latest_100000_steps.zip",
        "latest_300000_steps.zip",
        "best_mean_reward.zip",
        "final.zip",
    ]


def test_hard_reject_progress_screens() -> None:
    assert hard_reject({"success_rate": 0.19, "route_completion_mean": 0.5, "timeout_rate": 0.5}) == "success_rate < 0.20"
    assert hard_reject({"success_rate": 0.2, "route_completion_mean": 0.39, "timeout_rate": 0.5}) == "route_completion_mean < 0.40"
    assert hard_reject({"success_rate": 0.2, "route_completion_mean": 0.4, "timeout_rate": 0.81}) == "timeout_rate > 0.80"
    assert hard_reject({"success_rate": 0.2, "route_completion_mean": 0.4, "timeout_rate": 0.8}) == ""


def test_select_best_uses_metric_then_route_then_safety_tiebreak() -> None:
    rows = [
        {"checkpoint": "a", "hard_reject": True, "safety_efficiency_score": 1.0, "route_completion_mean": 1.0, "collision_rate": 0.0, "offroad_rate": 0.0},
        {"checkpoint": "b", "hard_reject": False, "safety_efficiency_score": 0.5, "route_completion_mean": 0.6, "collision_rate": 0.2, "offroad_rate": 0.2},
        {"checkpoint": "c", "hard_reject": False, "safety_efficiency_score": 0.5, "route_completion_mean": 0.7, "collision_rate": 0.5, "offroad_rate": 0.5},
    ]
    assert select_best(rows, "safety_efficiency_score")["checkpoint"] == "c"

    rows[1]["route_completion_mean"] = 0.7
    assert select_best(rows, "safety_efficiency_score")["checkpoint"] == "b"


def test_select_best_returns_none_when_all_rejected() -> None:
    assert select_best([{"checkpoint": "a", "hard_reject": True}], "safety_efficiency_score") is None


def test_main_exits_nonzero_when_all_checkpoints_rejected(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "final.zip").write_text("x", encoding="utf-8")
    output_dir = tmp_path / "selection"

    monkeypatch.setattr(
        select_best_checkpoint.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            run_dir=str(run_dir),
            eval_config="unused.yaml",
            output_dir=str(output_dir),
            metric="safety_efficiency_score",
            eval_start_seed=4500,
            eval_num_scenarios=100,
            eval_episodes=100,
            horizon=500,
            traffic_density=0.1,
        ),
    )
    monkeypatch.setattr(
        select_best_checkpoint,
        "evaluate_checkpoint",
        lambda checkpoint, eval_config, output_dir, args: {
            "n_episodes": 100,
            "success_rate": 0.0,
            "route_completion_mean": 0.0,
            "timeout_rate": 1.0,
            "collision_rate": 0.0,
            "offroad_rate": 0.0,
            "episode_cost_mean": 0.0,
            "safety_efficiency_score": -1.0,
        },
    )

    assert select_best_checkpoint.main() == 1
    assert not (checkpoint_dir / "selected_dev_best.zip").exists()
    assert (output_dir / "checkpoint_selection.csv").exists()


def test_evaluator_raises_for_missing_explicit_checkpoint(tmp_path):
    from omegaconf import OmegaConf

    from fasb.evaluation.evaluator import Evaluator

    cfg = OmegaConf.create({"algorithm": {"params": {"device": "cpu"}}})
    env = object()

    with pytest.raises(FileNotFoundError):
        Evaluator(cfg, tmp_path)._load_policy(tmp_path / "missing.zip", env)
