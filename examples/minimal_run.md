# Minimal Run

```bash
git clone git@github.com:Ian747-tw/MetaDrive-Experiment-Framework.git
cd MetaDrive-Experiment-Framework
./scripts/bootstrap.sh
source .venv/bin/activate
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml
python scripts/train.py --config configs/train/fasb_ppo.yaml training.total_timesteps=1000
python scripts/evaluate.py --config configs/eval/heldout_random.yaml --checkpoint runs/fasb_ppo/checkpoints/final.zip eval.n_episodes=5
```
