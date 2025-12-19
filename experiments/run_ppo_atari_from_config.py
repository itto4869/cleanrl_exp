#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PPO Atari SOAP/SOAP-Rel from a config JSON."
    )
    parser.add_argument(
        "--config",
        default="experiments/config.json",
        help="Path to config.json with soap, soap-rel, and envs.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Total timesteps per run.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma/space-separated list of seeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def parse_seeds(seeds_text):
    parts = [p for p in seeds_text.replace(",", " ").split() if p]
    return [int(p) for p in parts]


def require_keys(obj, keys, section):
    missing = [key for key in keys if key not in obj]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing keys in {section}: {missing_str}")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    require_keys(config, ["soap", "soap-rel", "envs"], "config root")
    if not config["envs"]:
        raise ValueError("config.json envs is empty")

    scripts = {
        "soap": "experiments/ppo_atari_soap.py",
        "soap-rel": "experiments/ppo_atari_soap_rel.py",
    }
    param_keys = [
        "learning-rate",
        "gae-lambda",
        "max-grad-norm",
        "soap-precondition-frequency",
    ]

    seeds = parse_seeds(args.seeds)
    run_env = os.environ.copy()
    base_run_group = run_env.setdefault("RUN_GROUP", "config_sweep")

    for key, script in scripts.items():
        params = config[key]
        require_keys(params, param_keys, key)
        param_args = [
            "--learning-rate",
            str(params["learning-rate"]),
            "--gae-lambda",
            str(params["gae-lambda"]),
            "--max-grad-norm",
            str(params["max-grad-norm"]),
            "--soap-precondition-frequency",
            str(params["soap-precondition-frequency"]),
            "--total-timesteps",
            str(args.total_timesteps),
        ]
        for env_id in config["envs"]:
            for seed in seeds:
                run_env["RUN_GROUP"] = os.path.join(base_run_group, key)
                cmd = [
                    sys.executable,
                    script,
                    "--env-id",
                    env_id,
                    "--seed",
                    str(seed),
                ] + param_args
                print(" ".join(cmd))
                if args.dry_run:
                    continue
                subprocess.run(cmd, check=True, env=run_env)


if __name__ == "__main__":
    main()
