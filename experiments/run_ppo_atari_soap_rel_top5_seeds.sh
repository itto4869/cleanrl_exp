#!/usr/bin/env bash
set -euo pipefail

# Run the top Optuna configs with extra seeds.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_GROUP="soap_top5_seed_sweep"
export RUN_GROUP

# JSON input: list of objects with trial_number and params (learning-rate, gae-lambda, max-grad-norm, soap-precondition-frequency)
JSON_PATH="${TOP5_JSON:-}"
if [[ -n "${1-}" ]]; then
  JSON_PATH="$1"
  shift
fi

if [[ -z "${JSON_PATH}" ]]; then
  echo "Usage: TOP5_JSON=path/to/top5.json bash $(basename "$0") [json_path] [extra args...]"
  exit 1
fi

if [[ ! -f "${JSON_PATH}" ]]; then
  echo "JSON file not found: ${JSON_PATH}"
  exit 1
fi

# Seeds can be overridden: SEEDS="0 1 2 3 4"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

# Baseline args used in the sweep.
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-2000000}
COMMON_ARGS=(
  --env-id ale_py:BreakoutNoFrameskip-v4
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --num-envs 8
  --num-steps 128
  --num-minibatches 4
  --update-epochs 4
  --clip-coef 0.1
  --ent-coef 0.01
)

# Parse JSON into "tag|args" lines.
mapfile -t CONFIGS < <(
  python - <<'PY' "${JSON_PATH}"
import json, sys
path = sys.argv[1]
with open(path, "r") as f:
    data = json.load(f)
for item in data:
    tn = item.get("trial_number")
    p = item.get("params", {})
    if tn is None or not p:
        continue
    tag = f"t{tn}"
    args = f"--learning-rate {p['learning-rate']} --gae-lambda {p['gae-lambda']} --max-grad-norm {p['max-grad-norm']} --soap-precondition-frequency {p['soap-precondition-frequency']}"
    print(f"{tag}|{args}")
PY
)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs parsed from ${JSON_PATH}"
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  IFS="|" read -r tag args <<<"$config"
  for seed in $SEEDS; do
    echo "Running ${tag} with seed=${seed}"
    python experiments/ppo_atari_soap_rel.py \
      --exp-name "ppo_atari_soap_rel_hp_sweep_${tag}" \
      --seed "${seed}" \
      $args \
      "${COMMON_ARGS[@]}" \
      "$@"
  done
done
