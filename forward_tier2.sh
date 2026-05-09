#!/bin/bash
# BRTS Tier-1 + Tier-2 configuration:
#   - Multi-rollout teacher sampling with overlap-based selection
#   - When all Tier-1 rollouts fail, sample one ground-truth-guided rollout
#     (Tier-2 recovery) and retain it if its extracted answer is correct
#
# Run from the repository root:
#   bash forward_tier2.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEACHER_ROLLOUT_MULTI_FORWARD_SELECT_BY_OVERLAP=True \
TEACHER_ROLLOUT_TIER1_ONLY=False \
TEACHER_ROLLOUT_N_ROLLOUTS_HINT=1 \
bash "$SCRIPT_DIR/on_policy_distillation.sh"
