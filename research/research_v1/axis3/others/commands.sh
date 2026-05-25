#!/usr/bin/env bash
set -euo pipefail

# Reproduction helpers (paths assume ws7 layout; adjust python/venv as needed).
# Fixed wrapper:
#   /tmp2/b14902068/.venv/bin/python scripts/run_axis3_fixed.py 0.05
# Adaptive wrapper:
#   /tmp2/b14902068/.venv/bin/python scripts/run_axis3_adaptive.py 0.10
# Batch suite (10 reps):
#   /tmp2/b14902068/.venv/bin/python scripts/batch_axis3_suite.py --reps 10 --workers 6

