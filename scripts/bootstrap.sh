#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python3}"

"${python_bin}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cat <<'MSG'

Bootstrap complete.

Next:
  source .venv/bin/activate
  python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
  python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml

Customize:
  configs/train/fasb_ppo.yaml
  examples/custom_plugins/
MSG
