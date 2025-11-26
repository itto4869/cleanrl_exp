#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python experiments/ppo_atari_adam_rel.py \
  --env-id ale_py:BreakoutNoFrameskip-v4 \
  --seed 0 \
  --learning_rate 2.0e-3 \
  --gae-lambda 0.90 \
  --max_grad_norm 5.0 \
