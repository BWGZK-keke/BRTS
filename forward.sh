#!/bin/bash
# BRTS Tier-1-only configuration:
#   - Multi-rollout teacher sampling with overlap-based selection
#   - No ground-truth-guided (Tier-2) recovery
#
# Run from the repository root:
#   bash forward.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEACHER_ROLLOUT_MULTI_FORWARD_SELECT_BY_OVERLAP=True \
TEACHER_ROLLOUT_TIER1_ONLY=True \
TEACHER_ROLLOUT_N_ROLLOUTS_HINT=0 \
bash "$SCRIPT_DIR/on_policy_distillation.sh"
