#!/usr/bin/env python3
from dataclasses import dataclass
import os
import time

import optuna
import tyro

from cleanrl_utils.tuner import Tuner


@dataclass
class SweepArgs:
    num_trials: int = 20
    num_seeds: int = 2
    total_timesteps: int = 2_000_000
    metric_last_n_average_window: int = 20
    study_name: str = "ppo_atari_soap_rel_hp_sweep"
    storage: str = ""
    env_id: str = "ale_py:BreakoutNoFrameskip-v4"
    use_wandb: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None


def build_params_fn(args: SweepArgs):
    def params(trial: optuna.Trial):
        return {
            # Make exp-name unique per trial to avoid run_dir collisions.
            "exp-name": f"ppo_atari_soap_rel_hp_sweep_t{trial.number}",
            "total-timesteps": args.total_timesteps,
            # Fixed baselines
            "num-envs": 8,
            "num-steps": 128,
            "num-minibatches": 4,
            "update-epochs": 4,
            "clip-coef": 0.1,
            "ent-coef": 0.01,
            # Sweep targets
            "learning-rate": trial.suggest_float("learning-rate", 5e-4, 3e-3, log=True),
            "gae-lambda": trial.suggest_float("gae-lambda", 0.90, 0.98),
            "max-grad-norm": trial.suggest_float("max-grad-norm", 0.5, 5.0),
            "soap-precondition-frequency": trial.suggest_categorical(
                "soap-precondition-frequency", [8, 16, 32]
            ),
        }

    return params


def main():
    args = tyro.cli(SweepArgs)

    # Save under runs/soap_sweep/<run_name>
    os.environ["RUN_GROUP"] = "soap_sweep"

    # Generate a unique storage path if none is provided to avoid collisions.
    storage = args.storage or f"sqlite:///{args.study_name}_{int(time.time())}.db"

    wandb_kwargs = {}
    if args.use_wandb:
        wandb_kwargs = {
            "project": args.wandb_project_name,
            "entity": args.wandb_entity,
            "group": args.study_name,
        }

    tuner = Tuner(
        script="experiments/ppo_atari_soap_rel.py",
        metric="charts/episodic_return",
        target_scores={args.env_id: None},
        params_fn=build_params_fn(args),
        direction="maximize",
        aggregation_type="average",
        metric_last_n_average_window=args.metric_last_n_average_window,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        storage=storage,
        study_name=args.study_name,
        wandb_kwargs=wandb_kwargs,
    )
    tuner.tune(num_trials=args.num_trials, num_seeds=args.num_seeds)


if __name__ == "__main__":
    main()
