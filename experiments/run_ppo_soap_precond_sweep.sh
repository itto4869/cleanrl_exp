#!/usr/bin/env bash
set -euo pipefail

# Frequencies to sweep. Edit this list to change the experiment points.
FREQUENCIES=(5 10 20 40)

# Common args shared by all runs; append extra flags when invoking this script.
COMMON_ARGS=()

# Optional: set SEED via env var when invoking (defaults to 1).
SEED="${SEED:-1}"

for freq in "${FREQUENCIES[@]}"; do
  echo "Running ppo_atari_soap_rel with precondition_frequency=${freq}"
  python experiments/ppo_atari_soap_rel.py \
    --soap-precondition-frequency "${freq}" \
    --exp-name "ppo_atari_soap_rel_f${freq}" \
    --seed "${SEED}" \
    "${COMMON_ARGS[@]}" \
    "$@"
done
