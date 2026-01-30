#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PPO Atari SOAP-Rel from a config JSON."
    )
    parser.add_argument(
        "--config",
        default="experiments/soap_rel_config.json",
        help="Path to soap_rel_config.json with params and envs.",
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


def to_param_args(params):
    param_args = []
    for key, value in params.items():
        if key == "total-timesteps":
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            flag = f"--{key}"
            if value:
                param_args.append(flag)
            else:
                param_args.append(f"--no-{key}")
            continue
        param_args.extend([f"--{key}", str(value)])
    return param_args


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    require_keys(config, ["soap", "envs"], "config root")
    if not config["envs"]:
        raise ValueError("soap_rel_config.json envs is empty")

    script = "experiments/ppo_atari_soap_rel.py"
    soap_params = config["soap"]
    if not isinstance(soap_params, dict):
        raise ValueError("soap config section must be a JSON object")
    ppo_params = soap_params.get("ppo", {})
    if ppo_params and not isinstance(ppo_params, dict):
        raise ValueError("soap.ppo must be a JSON object")

    optimizer_params = dict(soap_params)
    optimizer_params.pop("ppo", None)
    merged_params = dict(ppo_params)
    merged_params.update(optimizer_params)

    param_args = to_param_args(merged_params)
    param_args += ["--total-timesteps", str(args.total_timesteps)]

    seeds = parse_seeds(args.seeds)
    run_env = os.environ.copy()
    base_run_group = run_env.setdefault("RUN_GROUP", "soap_rel_config_sweep")
    run_label = "soap-rel"

    for env_id in config["envs"]:
        for seed in seeds:
            run_env["RUN_GROUP"] = os.path.join(base_run_group, run_label)
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
