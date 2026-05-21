.PHONY: install validate smoke stress e2e-stress check-env check-env-metadrive benchmark-dry-run test research-v1-base-train research-v1-base-eval research-v1-build-buffer research-v1-check aggregate-results

install:
	./scripts/bootstrap.sh

validate:
	python scripts/validate_components.py --config configs/train/fasb_ppo.yaml

smoke:
	python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml

check-env:
	python scripts/check_env.py

check-env-metadrive:
	python scripts/check_env.py --require-metadrive

test:
	python -m pytest tests/test_component_validation.py tests/test_failure_buffer.py tests/test_plugin_loading.py tests/test_metrics.py tests/test_run_dir.py -q

stress:
	python scripts/explore_failures.py --config configs/explore/base_checkpoint.yaml eval.n_episodes=2 metadrive.config.horizon=50
	python scripts/train.py --config configs/train/fasb_ppo.yaml training.total_timesteps=32 algorithm.params.n_steps=16 algorithm.params.batch_size=16 metadrive.config.horizon=30
	python scripts/evaluate.py --config configs/eval/heldout_random.yaml --checkpoint runs/fasb_ppo/checkpoints/final.zip eval.n_episodes=2 metadrive.config.horizon=30
	python scripts/analyze_failures.py --run runs/heldout_random_eval

e2e-stress:
	python scripts/run_e2e_stress.py --clean-runs

benchmark-dry-run:
	python scripts/benchmark.py --config configs/benchmark/final.yaml --dry-run

research-v1-base-train:
	python scripts/train.py --config configs/research_v1/base_pretrain.yaml

research-v1-base-eval:
	python scripts/evaluate.py --config configs/research_v1/base_eval.yaml --checkpoint runs/research_v1/base_pretrain_s42/checkpoints/final.zip

research-v1-build-buffer:
	python scripts/explore_failures.py --config configs/research_v1/base_explore.yaml

research-v1-check:
	python scripts/check_research_v1_ready.py --min-failures 30

aggregate-results:
	python scripts/aggregate_results.py --root runs/research_v1
