from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasb.core.config import load_config
from fasb.core.imports import instantiate_from_config
from fasb.core.validation import (
    validate_budget_output,
    validate_cost_output,
    validate_failure_label_output,
    validate_failure_score_output,
    validate_penalty_output,
    validate_scenario_sample,
)
from fasb.schemas.outputs import ScenarioSample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    episode = {
        "seed": 1042,
        "collision": True,
        "offroad": False,
        "timeout": False,
        "route_completion": 0.42,
        "success": False,
    }
    try:
        cost = instantiate_from_config(cfg.cost_function)
        cost_out = cost(None, None, 0.0, None, False, False, {"crash": True})
        validate_cost_output(cost_out)
        print(f"PASS cost_function: {cost.name}")

        scorer = instantiate_from_config(cfg.failure_scorer)
        score = scorer.score(episode)
        validate_failure_score_output(score)
        print(f"PASS failure_scorer: {scorer.name}")

        classifier = instantiate_from_config(cfg.failure_classifier)
        label = classifier.classify(episode, score)
        validate_failure_label_output(label)
        print(f"PASS failure_classifier: {classifier.name}")

        budgeter = instantiate_from_config(cfg.safety_budget)
        budget = budgeter.get_budget(episode, score, label)
        validate_budget_output(budget)
        print(f"PASS safety_budget: {budgeter.name}")

        scheduler = instantiate_from_config(cfg.penalty_scheduler)
        penalty = scheduler.get_penalty(episode, budget, score, label)
        validate_penalty_output(penalty)
        print(f"PASS penalty_scheduler: {scheduler.name}")

        sampler_cfg = cfg.get("sampler")
        if sampler_cfg:
            sampler = instantiate_from_config(sampler_cfg)
            sample = sampler.next() if hasattr(sampler, "next") else ScenarioSample(1, "synthetic", 1.0, {})
            validate_scenario_sample(sample)
            print(f"PASS sampler: {sampler.name}")
        return 0
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
