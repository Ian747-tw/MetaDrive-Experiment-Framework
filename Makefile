.PHONY: install validate smoke stress e2e-stress check-env check-env-metadrive benchmark-dry-run test

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
