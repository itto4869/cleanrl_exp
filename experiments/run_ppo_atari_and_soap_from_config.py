#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PPO Atari (Adam) and PPO Atari SOAP from a base_config JSON."
    )
    parser.add_argument(
        "--config",
        default="experiments/base_config.json",
        help="Path to base_config.json with ppo, soap, and envs.",
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
        "--adam-key",
        default="adam",
        help="Config key for Adam params.",
    )
    parser.add_argument(
        "--ppo-key",
        default="ppo",
        help="Config key for shared PPO params (optional).",
    )
    parser.add_argument(
        "--soap-key",
        default="soap",
        help="Config key for SOAP params.",
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


def find_latest_prefixed_dir(base_dir, prefix):
    if not os.path.isdir(base_dir):
        return None
    candidates = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    return os.path.join(base_dir, candidates[-1])


def relocate_run_dir(base_run_group, label, run_name_prefix):
    src = find_latest_prefixed_dir("runs", run_name_prefix)
    if not src:
        return
    dst_root = os.path.join("runs", base_run_group, label)
    os.makedirs(dst_root, exist_ok=True)
    shutil.move(src, os.path.join(dst_root, os.path.basename(src)))


def relocate_video_dir(base_run_group, label, run_name_prefix):
    src = find_latest_prefixed_dir("videos", run_name_prefix)
    if not src:
        return
    dst_root = os.path.join("videos", base_run_group, label)
    os.makedirs(dst_root, exist_ok=True)
    shutil.move(src, os.path.join(dst_root, os.path.basename(src)))


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    require_keys(config, [args.adam_key, args.soap_key, "envs"], "config root")
    if not config["envs"]:
        raise ValueError("config.json envs is empty")
    ppo_params = config.get(args.ppo_key, {})
    if ppo_params and not isinstance(ppo_params, dict):
        raise ValueError("PPO config section must be a JSON object")

    scripts = [
        {
            "key": args.adam_key,
            "label": args.adam_key,
            "script": "cleanrl/ppo_atari.py",
            "param_keys": ["learning-rate"],
            "extra_args": [],
        },
        {
            "key": args.soap_key,
            "label": args.soap_key,
            "script": "experiments/ppo_atari_soap.py",
            "param_keys": [
                "learning-rate",
                "soap-precondition-frequency",
            ],
            "extra_args": ["--soap-precondition-frequency"],
        },
    ]

    seeds = parse_seeds(args.seeds)
    run_env = os.environ.copy()
    base_run_group = run_env.setdefault("RUN_GROUP", "base_config")

    for entry in scripts:
        params = config[entry["key"]]
        require_keys(params, entry["param_keys"], entry["key"])
        if not isinstance(params, dict):
            raise ValueError(f"{entry['key']} config section must be a JSON object")
        entry_ppo = params.get("ppo", {})
        if entry_ppo and not isinstance(entry_ppo, dict):
            raise ValueError(f"{entry['key']}.ppo must be a JSON object")
        optimizer_params = dict(params)
        optimizer_params.pop("ppo", None)
        merged_params = dict(ppo_params)
        merged_params.update(entry_ppo)
        merged_params.update(optimizer_params)
        if entry["label"] == "adam" and "exp-name" not in merged_params:
            merged_params["exp-name"] = "adam"
        param_args = []
        for key, value in merged_params.items():
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
        param_args += ["--total-timesteps", str(args.total_timesteps)]
        for env_id in config["envs"]:
            for seed in seeds:
                run_env["RUN_GROUP"] = os.path.join(base_run_group, entry["label"])
                cmd = [
                    sys.executable,
                    entry["script"],
                    "--env-id",
                    env_id,
                    "--seed",
                    str(seed),
                ] + param_args
                print(" ".join(cmd))
                if args.dry_run:
                    continue
                subprocess.run(cmd, check=True, env=run_env)
                if entry["label"] == "adam":
                    run_prefix = f"{env_id}__{merged_params['exp-name']}__{seed}__"
                    relocate_run_dir(base_run_group, entry["label"], run_prefix)
                    relocate_video_dir(base_run_group, entry["label"], run_prefix)


if __name__ == "__main__":
    main()
