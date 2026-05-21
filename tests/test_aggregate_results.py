from __future__ import annotations

import pandas as pd
import pytest

from scripts.aggregate_results import aggregate


def test_aggregate_results_combines_eval_and_failure_mode(tmp_path) -> None:
    eval_dir = tmp_path / "eval_axis1_fasb_s42"
    (eval_dir / "eval").mkdir(parents=True)
    (eval_dir / "analysis").mkdir()
    pd.DataFrame(
        [
            {
                "success_rate": 0.2,
                "collision_rate": 0.1,
                "offroad_rate": 0.0,
                "timeout_rate": 0.4,
                "route_completion_mean": 0.5,
                "episode_cost_mean": 0.1,
                "cost_violation_rate": 0.2,
                "safety_efficiency_score": -0.1,
            }
        ]
    ).to_csv(eval_dir / "eval" / "heldout_random.csv", index=False)
    pd.DataFrame([{"failure_mode": "collision", "episodes": 3}]).to_csv(
        eval_dir / "analysis" / "failure_by_mode.csv", index=False
    )

    main_path, mode_path = aggregate(tmp_path)

    main = pd.read_csv(main_path)
    assert main.loc[0, "method"] == "axis1_fasb"
    assert main.loc[0, "eval_run"] == "eval_axis1_fasb_s42"
    assert mode_path is not None
    assert mode_path.exists()


def test_aggregate_results_fails_when_missing_eval_csvs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path)
