#!/usr/bin/env python3
import runpy
import sys

BOOL_FLAGS = {"soap-normalize-grads"}
TRUTHY = {"1", "true", "yes", "y", "on"}
FALSY = {"0", "false", "no", "n", "off"}


def normalize_bool_flags(argv):
    normalized = [argv[0]]
    for arg in argv[1:]:
        if not arg.startswith("--") or "=" not in arg:
            normalized.append(arg)
            continue

        key, value = arg[2:].split("=", 1)
        if key in BOOL_FLAGS:
            lowered = value.strip().lower()
            if lowered in TRUTHY:
                normalized.append(f"--{key}")
                continue
            if lowered in FALSY:
                normalized.append(f"--no-{key}")
                continue

        normalized.append(arg)
    return normalized


def main():
    sys.argv = normalize_bool_flags(sys.argv)
    result = runpy.run_path("experiments/ppo_atari_soap_rollout.py", run_name="__main__")
    globals().update(result)


if __name__ == "__main__":
    main()
