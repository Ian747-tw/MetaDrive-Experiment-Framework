import shutil
from pathlib import Path
import yaml
import re

def main():
    base_dir = Path(r"c:\Users\white\Desktop\hw\DRL\hw5\MetaDrive-Experiment-Framework")
    research_dir = base_dir / "research" / "research_v1" / "axis2"
    configs_dir = research_dir / "configs"
    
    templates_dir = configs_dir / "templates"
    resolved_train_dir = configs_dir / "resolved_train"
    resolved_eval_dir = configs_dir / "resolved_eval"

    templates_dir.mkdir(parents=True, exist_ok=True)
    resolved_train_dir.mkdir(parents=True, exist_ok=True)
    resolved_eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy Template YAMLs
    template_files = {
        base_dir / "configs" / "research_v1" / "axis2_sampler_mixed060_final.yaml": "axis2_sampler_mixed060_final.yaml",
        base_dir / "configs" / "research_v1" / "screening" / "axis2_sampler_mixed030.yaml": "axis2_sampler_mixed030.yaml",
        base_dir / "configs" / "research_v1" / "screening" / "axis2_sampler_mixed060.yaml": "axis2_sampler_mixed060.yaml",
        base_dir / "configs" / "research_v1" / "screening" / "axis2_sampler_mixed090.yaml": "axis2_sampler_mixed090.yaml",
        base_dir / "configs" / "research_v1" / "screening" / "axis2_sampler_mixed100.yaml": "axis2_sampler_mixed100.yaml",
        base_dir / "configs" / "research_v1" / "screening" / "axis2_sampler_uniform.yaml": "axis2_sampler_uniform.yaml",
        base_dir / "configs" / "research_v1" / "stabilization" / "fasb_stable_ratio005_penalty025.yaml": "axis2_sampler_mixed005.yaml"
    }

    for src, dest_name in template_files.items():
        if src.exists():
            shutil.copy2(src, templates_dir / dest_name)
            print(f"Copied template: {dest_name}")
        else:
            print(f"Warning: Template source {src} does not exist.")

    # 2. Generate Resolved Train and Eval YAMLs dynamically based on the multiseed runs
    # Methods: mixed005, mixed030, mixed060, mixed090
    # Seeds: 2000, 3000, 4000, 5000, 6000, 7000
    methods_ratios = {
        "mixed005": 0.05,
        "mixed030": 0.30,
        "mixed060": 0.60,
        "mixed090": 0.90
    }
    seeds = [2000, 3000, 4000, 5000, 6000, 7000]

    for method, ratio in methods_ratios.items():
        for seed in seeds:
            eval_run = f"eval_axis2_{method}_s{seed}"
            train_run = f"axis2_{method}_s{seed}"

            # --- Resolved Train Config ---
            train_config = {
                "experiment": {
                    "name": train_run,
                    "mode": "fasb_ppo",
                    "seed": seed,
                    "output_dir": f"runs/research_v1/{train_run}",
                    "save_resolved_config": True
                },
                "mode": "fasb_ppo",
                "metadrive": {
                    "env_class": "metadrive.envs.MetaDriveEnv",
                    "config": {
                        "start_seed": seed,
                        "num_scenarios": 500,
                        "traffic_density": 0.1,
                        "random_traffic": True,
                        "use_render": False,
                        "horizon": 500,
                        "log_level": 50,
                        "crash_vehicle_done": False,
                        "crash_object_done": False,
                        "out_of_road_done": False
                    }
                },
                "vec_env": {
                    "type": "dummy",
                    "n_envs": 1,
                    "start_method": "forkserver"
                },
                "algorithm": {
                    "backend": "sb3",
                    "name": "PPO",
                    "policy": "MlpPolicy",
                    "checkpoint_path": "runs/research_v1/base_pretrain_s42/checkpoints/final.zip",
                    "params": {
                        "learning_rate": 3.0e-05,
                        "n_steps": 128,
                        "batch_size": 64,
                        "n_epochs": 10,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_range": 0.2,
                        "ent_coef": 0.0,
                        "vf_coef": 0.5,
                        "max_grad_norm": 0.5,
                        "verbose": 1,
                        "device": "cpu",
                        "policy_kwargs": {
                            "net_arch": [256, 256]
                        }
                    }
                },
                "training": {
                    "total_timesteps": 300000,
                    "save_every_steps": 100000,
                    "eval_every_steps": 0
                },
                "failure_buffer": {
                    "path": "runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl",
                    "max_size": 5000
                },
                "cost_function": {
                    "_target_": "fasb.plugins.cost.DefaultDrivingCost"
                },
                "failure_scorer": {
                    "_target_": "fasb.plugins.failure_scorer.DefaultFailureScorer"
                },
                "failure_classifier": {
                    "_target_": "fasb.plugins.failure_classifier.DefaultFailureClassifier"
                },
                "safety_budget": {
                    "_target_": "fasb.plugins.safety_budget.AdaptiveSafetyBudget",
                    "d_min": 0.1,
                    "d_max": 0.3,
                    "timeout_budget": 0.3
                },
                "penalty_scheduler": {
                    "_target_": "fasb.plugins.penalty_scheduler.RiskPenaltyScheduler",
                    "lambda_min": 0.0,
                    "lambda_max": 0.25
                },
                "sampler": {
                    "_target_": "fasb.plugins.sampler.MixedFailureSampler",
                    "start_seed": seed,
                    "num_scenarios": 500,
                    "failure_ratio": ratio,
                    "alpha": 0.7,
                    "max_too_hard_ratio": 0.15
                },
                "eval": {
                    "n_episodes": 100
                }
            }

            # --- Resolved Eval Config ---
            eval_config = {
                "experiment": {
                    "name": eval_run,
                    "seed": 42,
                    "output_dir": f"runs/research_v1/{eval_run}"
                },
                "metadrive": {
                    "env_class": "metadrive.envs.MetaDriveEnv",
                    "config": {
                        "start_seed": 5000,
                        "num_scenarios": 200,
                        "traffic_density": 0.1,
                        "random_traffic": True,
                        "use_render": False,
                        "horizon": 500,
                        "log_level": 50,
                        "crash_vehicle_done": False,
                        "crash_object_done": False,
                        "out_of_road_done": False
                    }
                },
                "algorithm": {
                    "checkpoint_path": None
                },
                "cost_function": {
                    "_target_": "fasb.plugins.cost.DefaultDrivingCost"
                },
                "eval": {
                    "scenario_set": "heldout_random",
                    "n_episodes": 100,
                    "deterministic": True
                }
            }

            # Write YAMLs using standard formatting
            train_file = resolved_train_dir / f"{train_run}.yaml"
            with open(train_file, "w") as f:
                yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)

            eval_file = resolved_eval_dir / f"{eval_run}.yaml"
            with open(eval_file, "w") as f:
                yaml.dump(eval_config, f, default_flow_style=False, sort_keys=False)

    # 3. Write Seed 42 Sweep Resolved Train and Eval YAMLs
    # Single-seed sweep: ratio 0.00 to 1.00 (from sampler yaml configs)
    sampler_sweep = {
        "axis2_sampler_mixed000_s42": 0.00,
        "axis2_sampler_mixed010_s42": 0.10,
        "axis2_sampler_mixed020_s42": 0.20,
        "axis2_sampler_mixed030_s42": 0.30,
        "axis2_sampler_mixed040_s42": 0.40,
        "axis2_sampler_mixed050_s42": 0.50,
        "axis2_sampler_mixed060_final_s42": 0.60,
        "axis2_sampler_mixed070_s42": 0.70,
        "axis2_sampler_mixed080_s42": 0.80,
        "axis2_sampler_mixed090_s42": 0.90,
        "axis2_sampler_mixed095_s42": 0.95,
        "axis2_sampler_mixed099_s42": 0.99,
        "axis2_sampler_mixed100_s42": 1.00,
    }

    for train_run, ratio in sampler_sweep.items():
        eval_run = f"eval_{train_run}"
        seed = 42

        # Train config
        train_config = {
            "experiment": {
                "name": train_run,
                "mode": "fasb_ppo",
                "seed": seed,
                "output_dir": f"runs/research_v1/{train_run}",
                "save_resolved_config": True
            },
            "mode": "fasb_ppo",
            "metadrive": {
                "metadrive": None, # Will resolve to default env_class
                "config": {
                    "start_seed": 2000,
                    "num_scenarios": 500,
                    "traffic_density": 0.1,
                    "random_traffic": True,
                    "use_render": False,
                    "horizon": 500,
                    "log_level": 50,
                    "crash_vehicle_done": False,
                    "crash_object_done": False,
                    "out_of_road_done": False
                }
            },
            "vec_env": {
                "type": "dummy",
                "n_envs": 1,
                "start_method": "forkserver"
            },
            "algorithm": {
                "backend": "sb3",
                "name": "PPO",
                "policy": "MlpPolicy",
                "checkpoint_path": "runs/research_v1/base_pretrain_s42/checkpoints/final.zip",
                "params": {
                    "learning_rate": 3.0e-05,
                    "n_steps": 128,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "verbose": 1,
                    "device": "cpu",
                    "policy_kwargs": {
                        "net_arch": [256, 256]
                    }
                }
            },
            "training": {
                "total_timesteps": 300000,
                "save_every_steps": 100000,
                "eval_every_steps": 0
            },
            "failure_buffer": {
                "path": "runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl",
                "max_size": 5000
            },
            "cost_function": {
                "_target_": "fasb.plugins.cost.DefaultDrivingCost"
            },
            "failure_scorer": {
                "_target_": "fasb.plugins.failure_scorer.DefaultFailureScorer"
            },
            "failure_classifier": {
                "_target_": "fasb.plugins.failure_classifier.DefaultFailureClassifier"
            },
            "safety_budget": {
                "_target_": "fasb.plugins.safety_budget.AdaptiveSafetyBudget",
                "d_min": 0.1,
                "d_max": 0.3,
                "timeout_budget": 0.3
            },
            "penalty_scheduler": {
                "_target_": "fasb.plugins.penalty_scheduler.RiskPenaltyScheduler",
                "lambda_min": 0.0,
                "lambda_max": 0.25
            },
            "sampler": {
                "_target_": "fasb.plugins.sampler.MixedFailureSampler",
                "start_seed": 2000,
                "num_scenarios": 500,
                "failure_ratio": ratio,
                "alpha": 0.7,
                "max_too_hard_ratio": 0.15
            },
            "eval": {
                "n_episodes": 100
            }
        }

        # Eval config
        eval_config = {
            "experiment": {
                "name": eval_run,
                "seed": 42,
                "output_dir": f"runs/research_v1/{eval_run}"
            },
            "metadrive": {
                "env_class": "metadrive.envs.MetaDriveEnv",
                "config": {
                    "start_seed": 5000,
                    "num_scenarios": 200,
                    "traffic_density": 0.1,
                    "random_traffic": True,
                    "use_render": False,
                    "horizon": 500,
                    "log_level": 50,
                    "crash_vehicle_done": False,
                    "crash_object_done": False,
                    "out_of_road_done": False
                }
            },
            "algorithm": {
                "checkpoint_path": None
            },
            "cost_function": {
                "_target_": "fasb.plugins.cost.DefaultDrivingCost"
            },
            "eval": {
                "scenario_set": "heldout_random",
                "n_episodes": 100,
                "deterministic": True
            }
        }

        # Write YAMLs
        train_file = resolved_train_dir / f"{train_run}.yaml"
        with open(train_file, "w") as f:
            yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)

        eval_file = resolved_eval_dir / f"{eval_run}.yaml"
        with open(eval_file, "w") as f:
            yaml.dump(eval_config, f, default_flow_style=False, sort_keys=False)

    # 4. Uniform Sampler
    train_run = "axis2_sampler_uniform_s42"
    eval_run = "eval_axis2_sampler_uniform_s42"
    
    # Train config (Uniform)
    train_config = {
        "experiment": {
            "name": train_run,
            "mode": "fasb_ppo",
            "seed": 42,
            "output_dir": f"runs/research_v1/{train_run}",
            "save_resolved_config": True
        },
        "mode": "fasb_ppo",
        "metadrive": {
            "config": {
                "start_seed": 2000,
                "num_scenarios": 500,
                "traffic_density": 0.1,
                "random_traffic": True,
                "use_render": False,
                "horizon": 500,
                "log_level": 50,
                "crash_vehicle_done": False,
                "crash_object_done": False,
                "out_of_road_done": False
            }
        },
        "vec_env": {
            "type": "dummy",
            "n_envs": 1,
            "start_method": "forkserver"
        },
        "algorithm": {
            "backend": "sb3",
            "name": "PPO",
            "policy": "MlpPolicy",
            "checkpoint_path": "runs/research_v1/base_pretrain_s42/checkpoints/final.zip",
            "params": {
                "learning_rate": 3.0e-05,
                "n_steps": 128,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "verbose": 1,
                "device": "cpu",
                "policy_kwargs": {
                    "net_arch": [256, 256]
                }
            }
        },
        "training": {
            "total_timesteps": 300000,
            "save_every_steps": 100000,
            "eval_every_steps": 0
        },
        "failure_buffer": {
            "path": "runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl",
            "max_size": 5000
        },
        "cost_function": {
            "_target_": "fasb.plugins.cost.DefaultDrivingCost"
        },
        "failure_scorer": {
            "_target_": "fasb.plugins.failure_scorer.DefaultFailureScorer"
        },
        "failure_classifier": {
            "_target_": "fasb.plugins.failure_classifier.DefaultFailureClassifier"
        },
        "safety_budget": {
            "_target_": "fasb.plugins.safety_budget.AdaptiveSafetyBudget",
            "d_min": 0.1,
            "d_max": 0.3,
            "timeout_budget": 0.3
        },
        "penalty_scheduler": {
            "_target_": "fasb.plugins.penalty_scheduler.RiskPenaltyScheduler",
            "lambda_min": 0.0,
            "lambda_max": 0.25
        },
        "sampler": {
            "_target_": "fasb.plugins.sampler.UniformFailureSampler",
            "start_seed": 2000,
            "num_scenarios": 500,
            "alpha": 0.7,
            "max_too_hard_ratio": 0.15
        },
        "eval": {
            "n_episodes": 100
        }
    }

    # Write Uniform
    with open(resolved_train_dir / f"{train_run}.yaml", "w") as f:
        yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)

    # Eval config
    eval_config = {
        "experiment": {
            "name": eval_run,
            "seed": 42,
            "output_dir": f"runs/research_v1/{eval_run}"
        },
        "metadrive": {
            "env_class": "metadrive.envs.MetaDriveEnv",
            "config": {
                "start_seed": 5000,
                "num_scenarios": 200,
                "traffic_density": 0.1,
                "random_traffic": True,
                "use_render": False,
                "horizon": 500,
                "log_level": 50,
                "crash_vehicle_done": False,
                "crash_object_done": False,
                "out_of_road_done": False
            }
        },
        "algorithm": {
            "checkpoint_path": None
        },
        "cost_function": {
            "_target_": "fasb.plugins.cost.DefaultDrivingCost"
        },
        "eval": {
            "scenario_set": "heldout_random",
            "n_episodes": 100,
            "deterministic": True
        }
    }
    with open(resolved_eval_dir / f"{eval_run}.yaml", "w") as f:
        yaml.dump(eval_config, f, default_flow_style=False, sort_keys=False)

    print("All template and resolved train/eval configs generated successfully under research/research_v1/axis2/configs/!")

if __name__ == "__main__":
    main()
