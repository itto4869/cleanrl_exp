#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Explicit log/video group for this sweep
RUN_GROUP="ppo_default_soap_precond_sweep"
export RUN_GROUP

# SOAP precondition frequency
FREQUENCIES=(10 32 48 50 64 80 96 100)

SEED="${SEED:-0}"
COMMON_ARGS=(
  --env-id ale_py:BreakoutNoFrameskip-v4
  --seed 0
)

for freq in "${FREQUENCIES[@]}"; do
  echo "Running ppo_atari_soap_rel with precondition_frequency=${freq}"
  python experiments/ppo_atari_soap_rel.py \
    --soap-precondition-frequency "${freq}" \
    --exp-name "ppo_atari_default_soap_rel_f${freq}" \
    --seed "${SEED}" \
    "${COMMON_ARGS[@]}" \
    "$@"
done
