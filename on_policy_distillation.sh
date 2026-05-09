#!/bin/bash
# On-Policy Distillation training entrypoint (BRTS).
#
# Examples:
#   # Tier-1-only teacher selection, no hinted retries:
#   TEACHER_ROLLOUT_TIER1_ONLY=True TEACHER_ROLLOUT_N_ROLLOUTS_HINT=0 \
#       bash on_policy_distillation.sh
#
#   # Watch logs:
#   swanlab watch checkpoint/swanlab_log
#
# Convenience wrappers: forward.sh (Tier-1 only), forward_tier2.sh (with Tier-2).
#
# SLURM users: uncomment / adjust the SBATCH directives below to match your cluster.
# #SBATCH --job-name=brts
# #SBATCH --gres=gpu:8
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=64
# #SBATCH --mem=500G
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1

# Hugging Face credentials and cache.
# Set HF_TOKEN in your environment (e.g. `export HF_TOKEN=...` or via a secret
# manager) if the models you use require authentication. Do NOT commit a
# real token to this file.
# Optionally point HF_HOME at a shared cache directory; otherwise the default
# (~/.cache/huggingface) is used.
if [ -n "${HF_HOME:-}" ]; then
    export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
fi

# Optional explicit env activation for non-interactive/sbatch shells.
# Usage examples:
#   CONDA_ENV=verl bash on_policy_distillation.sh
#   VENV_PATH=/path/to/venv bash on_policy_distillation.sh
if [ -n "${CONDA_ENV:-}" ]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] CONDA_ENV is set but 'conda' is not available in PATH."
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
if [ -n "${VENV_PATH:-}" ]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] PYTHON_BIN '$PYTHON_BIN' is not found in PATH."
    exit 1
fi
echo "[INFO] Using python binary: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" - <<'PY'
import sys
print(f"[INFO] Python executable: {sys.executable}")
print(f"[INFO] Python version: {sys.version.split()[0]}")
PY

# Run from the script's own repository root so relative paths are stable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer repo-local verl package over any site-packages install.
export PYTHONPATH="$SCRIPT_DIR/verl${PYTHONPATH:+:$PYTHONPATH}"

# Work around databus/wandb protobuf ABI mismatch on environments with
# protobuf >= 4.x. Use fast C++ protobuf by default and only fallback when
# databus import fails.
if [ -z "${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-}" ]; then
    if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import databus
PY
    then
        export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
        echo "[WARN] databus import failed with current protobuf; set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python."
    fi
fi

set -x

# Normalize CUDA compiler env vars to avoid stale paths like
# /usr/local/cuda-12.8/bin/nvcc when that toolkit is not installed.
ensure_cuda_nvcc() {
    local selected_nvcc=""
    local candidate=""

    if [ -n "${CUDACXX:-}" ] && [ ! -x "${CUDACXX}" ]; then
        echo "[WARN] Ignoring invalid CUDACXX=${CUDACXX} (file not executable)."
    fi
    if [ -n "${CUDA_HOME:-}" ] && [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
        echo "[WARN] Ignoring invalid CUDA_HOME=${CUDA_HOME} (missing bin/nvcc)."
    fi

    if [ -n "${CUDACXX:-}" ] && [ -x "${CUDACXX}" ]; then
        selected_nvcc="${CUDACXX}"
    elif [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        selected_nvcc="${CUDA_HOME}/bin/nvcc"
    elif command -v nvcc >/dev/null 2>&1; then
        selected_nvcc="$(command -v nvcc)"
    else
        for candidate in \
            /usr/local/cuda/bin/nvcc \
            /usr/local/cuda-12/bin/nvcc \
            /usr/local/cuda-12.8/bin/nvcc \
            /usr/local/cuda-12.6/bin/nvcc
        do
            if [ -x "$candidate" ]; then
                selected_nvcc="$candidate"
                break
            fi
        done
    fi

    if [ -z "$selected_nvcc" ]; then
        echo "[WARN] nvcc not found. FlashInfer JIT compilation may fail."
        echo "[WARN] Set CUDA_HOME/CUDACXX to a valid CUDA toolkit if startup fails."
        return 0
    fi

    export CUDACXX="$selected_nvcc"
    export CUDA_HOME="$(dirname "$(dirname "$selected_nvcc")")"
    case ":$PATH:" in
        *":${CUDA_HOME}/bin:"*) ;;
        *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
    esac
    echo "[INFO] Using CUDA_HOME=${CUDA_HOME}"
    echo "[INFO] Using CUDACXX=${CUDACXX}"
}
ensure_cuda_nvcc

# vLLM worker import path relies on numba, which is incompatible with NumPy>=2.3
# for common cluster wheels. Fail fast with actionable guidance instead of
# crashing deep inside Ray worker startup.
if [ "${SKIP_VLLM_NUMBA_CHECK:-0}" != "1" ]; then
    if ! "$PYTHON_BIN" - <<'PY'
import re
import sys

try:
    import numpy as np
except Exception as exc:
    print(f"[ERROR] Failed to import numpy: {exc}")
    sys.exit(10)


def _major_minor(version: str):
    m = re.match(r"^(\d+)\.(\d+)", version)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


if _major_minor(np.__version__) > (2, 2):
    print(f"[ERROR] NumPy {np.__version__} is too new for this vLLM/numba stack.")
    sys.exit(11)

try:
    import numba
except Exception as exc:
    print(f"[ERROR] Failed to import numba with NumPy {np.__version__}: {exc}")
    sys.exit(12)

print(f"[INFO] NumPy/Numba check passed: numpy={np.__version__}, numba={numba.__version__}")
PY
    then
        echo "[ERROR] Incompatible NumPy/Numba environment for vLLM."
        echo "[ERROR] Run this once on the target server, then retry:"
        echo "        $PYTHON_BIN -m pip install --user --no-cache-dir --force-reinstall 'numpy<=2.2' 'numba>=0.61,<0.62'"
        echo "[ERROR] You can bypass this preflight check with SKIP_VLLM_NUMBA_CHECK=1 (not recommended)."
        exit 1
    fi
fi

# Configure logging when running outside SBATCH.
if [ -z "$SLURM_JOB_ID" ]; then
    # Create the log directory and file for local runs.
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    # Mirror output to both terminal and log file.
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

# Bypass compliance gateway for Ray's local HTTP traffic (dashboard/runtime-env agent).
# Ray binds to loopback and an fdbd:dc61:... IPv6 address; without this, local
# requests get intercepted and return 403 "Missing Destination-Service header".
# Note: httpx (used by huggingface_hub) parses no_proxy entries as URLs and
# cannot handle IPv6 CIDR, so list loopback hosts individually.
export no_proxy="${no_proxy},localhost,127.0.0.1,::1"
export NO_PROXY="$no_proxy"

# Pre-export Ray runtime_env vars in the shell so get_ppo_ray_runtime_env() pops
# them and Ray doesn't need to POST them via the runtime-env agent.
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_FLASHINFER_MOE_INT4=0
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=0
export VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=0
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_CUMEM_ENABLE=0
# In restricted network environments, disable per-actor Ray runtime_env HTTP setup
# and pass worker env vars via actor constructor kwargs instead.
export VERL_DISABLE_RAY_RUNTIME_ENV=1

run_ray_cli() {
    if command -v ray >/dev/null 2>&1; then
        ray "$@"
    else
        "$PYTHON_BIN" -m ray.scripts.scripts "$@"
    fi
}

run_ray_cli stop --force || true
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
# export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONUNBUFFERED=1
# Make token available for libraries that read either env var name.
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
export PROJECT_NAME='OnPolicyDistillation' # TODO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export ADV_ESTIMATOR=token_reward_direct
# export ADV_ESTIMATOR=token_reward_direct_plus_grpo
# export ADV_ESTIMATOR=token_grpo
# export ADV_ESTIMATOR=grpo
export GRPO_OUTCOME_WEIGHT=1.0
# export ADV_ESTIMATOR=token_grpo
# Swanlab setting used to continue exp  
# export SWANLAB_RESUME=must
# export SWANLAB_RUN_ID="jri5qia6iy67v7su0zjsv"


# DeepMath-103K
# SANITY_MODE values:
#   auto  -> enable sanity defaults when teacher rollout hints indicate smoke test
#   True  -> always use sanity defaults (unless explicitly overridden)
#   False -> use full training defaults
export SANITY_MODE=${SANITY_MODE:-auto}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}  # TODO: 31744 /15360 / 7168 / 3072 / 5120
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744} # TODO: 15360 / 7168 / 3072
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64} # TODO: 1 / 8 / 16 / 32 / 64 (default 64)
export TEMPERATURE=${TEMPERATURE:-1.0} # TODO: 0.6 / 0.8 / 1.0 / 1.2 (default 1.0)
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0} # Teacher logits temperature (default 1.0, no scaling)
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0} # TODO: 1.0 / 1.1 / 1.2 (default 1.0, no penalty)
export N_RESPONSES=${N_RESPONSES:-2} # TODO: 2 / 4 / 8 / 16 / 32 (default: 2 for speed)
export VAL_SAMPLES=${VAL_SAMPLES:-4} # validation rollout samples (default: 4 for speed)
export TEST_FREQ=${TEST_FREQ:-10} # run validation every N training steps (default: 10)
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16} # 0 represents no top-k sampling
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"} # "only_stu" or "only_tch" or "intersection" or "union" or "union-intersection"
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"} # "student_p" or "teacher_p" or "none"
export RM_MICRO_BATCH_SIZE_PER_GPU=${RM_MICRO_BATCH_SIZE_PER_GPU:-8}
# export LR=${LR:-1e-6}
# export LR_SCHEDULER=${LR_SCHEDULER:-constant}
export USE_KL=${USE_KL:-False} # TODO: True / False (default False)
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False} # TODO: True / False (default False)
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16} # H100-friendly default; can override to fp32 if needed
export IS_PLOT=${IS_PLOT:-False} # TODO: True / False (default False)
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"} # TODO: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean" / "seq-mean-token-sum-norm" (default "token-mean")

# Teacher rollout settings (on-policy distillation with teacher generation)
# When TEACHER_ROLLOUT_ENABLE=True the reward model generates N_TEACHER_ROLLOUTS
# responses per prompt; the best is selected by (correct answer + top-K overlap).
# Whether that selected rollout is applied to teacher forward/distillation context
# is controlled by TEACHER_ROLLOUT_APPLY_SELECTED_FOR_TEACHER_FORWARD.
# If TEACHER_ROLLOUT_MULTI_FORWARD_SELECT_BY_OVERLAP=True, teacher forward is
# run on both student context and selected context, and the higher-overlap trial
# is used for distillation (no extra generation pass).
export TEACHER_ROLLOUT_ENABLE=${TEACHER_ROLLOUT_ENABLE:-True}
export N_TEACHER_ROLLOUTS=${N_TEACHER_ROLLOUTS:-2}
if [ -z "${TEACHER_ROLLOUT_MAX_NEW_TOKENS+x}" ]; then
    TEACHER_MAX_NEW_TOKENS_FROM_ENV=0
else
    TEACHER_MAX_NEW_TOKENS_FROM_ENV=1
fi
export TEACHER_ROLLOUT_MAX_NEW_TOKENS=${TEACHER_ROLLOUT_MAX_NEW_TOKENS:-$MAX_RESP_LENGTH}
export TEACHER_ROLLOUT_GEN_TEMPERATURE=${TEACHER_ROLLOUT_GEN_TEMPERATURE:-0.7}
export TEACHER_ROLLOUT_TOP_P=${TEACHER_ROLLOUT_TOP_P:-0.95}
export TEACHER_ROLLOUT_FALLBACK=${TEACHER_ROLLOUT_FALLBACK:-most_similar} # most_similar | skip
export TEACHER_ROLLOUT_TIER1_ONLY=${TEACHER_ROLLOUT_TIER1_ONLY:-False}
export TEACHER_ROLLOUT_N_ROLLOUTS_HINT=${TEACHER_ROLLOUT_N_ROLLOUTS_HINT:-1}
export TEACHER_ROLLOUT_APPLY_SELECTED_FOR_TEACHER_FORWARD=${TEACHER_ROLLOUT_APPLY_SELECTED_FOR_TEACHER_FORWARD:-True}
export TEACHER_ROLLOUT_MULTI_FORWARD_SELECT_BY_OVERLAP=${TEACHER_ROLLOUT_MULTI_FORWARD_SELECT_BY_OVERLAP:-False}
# Auxiliary KD term on selected teacher context:
#   L_total = L_student_ctx + AUX_TEACHER_CTX_KD_COEF * L_teacher_ctx
export AUX_TEACHER_CTX_KD_ENABLE=${AUX_TEACHER_CTX_KD_ENABLE:-False}
export AUX_TEACHER_CTX_KD_COEF=${AUX_TEACHER_CTX_KD_COEF:-0.05}
export AUX_TEACHER_CTX_KD_WEIGHT_MODE=${AUX_TEACHER_CTX_KD_WEIGHT_MODE:-student_p} # student_p | teacher_p | none

# Prompt perturbation for the second teacher rollout (ablation study).
# Set TEACHER_PROMPT_DISTURB_ENABLE=True to append TEACHER_PROMPT_DISTURB_TEXT to
# the prompt of rollout index TEACHER_PROMPT_DISTURB_ROLLOUT_INDEX (1-indexed).
# Default: disabled.  Example override:
#   TEACHER_PROMPT_DISTURB_ENABLE=True bash on_policy_distillation.sh
export TEACHER_PROMPT_DISTURB_ENABLE=${TEACHER_PROMPT_DISTURB_ENABLE:-False}
export TEACHER_PROMPT_DISTURB_ROLLOUT_INDEX=${TEACHER_PROMPT_DISTURB_ROLLOUT_INDEX:-2}
export TEACHER_PROMPT_DISTURB_TEXT=${TEACHER_PROMPT_DISTURB_TEXT:-"Please reason step by step and rethink in detail before giving the final answer."}

# If tier1-only mode is enabled, force hint retries to 0.
if [ "$TEACHER_ROLLOUT_TIER1_ONLY" = "True" ]; then
    export TEACHER_ROLLOUT_N_ROLLOUTS_HINT=0
fi

# Remove-padding path requires flash_attn.bert_padding on CUDA.
# Keep default True when available, otherwise auto-disable to avoid runtime crash.
export USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
if [ "$USE_REMOVE_PADDING" = "True" ]; then
    if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from flash_attn.bert_padding import unpad_input  # noqa: F401
PY
    then
        echo "[WARN] flash_attn.bert_padding is unavailable; setting USE_REMOVE_PADDING=False."
        export USE_REMOVE_PADDING=False
    fi
fi

if [ "$SANITY_MODE" = "auto" ]; then
    if [[ "$N_TEACHER_ROLLOUTS" =~ ^[0-9]+$ && "$TEACHER_ROLLOUT_MAX_NEW_TOKENS" =~ ^[0-9]+$ \
        && "$N_TEACHER_ROLLOUTS" -le 1 && "$TEACHER_ROLLOUT_MAX_NEW_TOKENS" -le 512 ]]; then
        SANITY_MODE=True
    else
        SANITY_MODE=False
    fi
fi

if [ "$SANITY_MODE" = "True" ]; then
    export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH_SANITY:-256}
    export MAX_RESP_LENGTH=${MAX_RESP_LENGTH_SANITY:-512}
    export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH_SANITY:-512}
    export MINI_BATCH_SIZE=${MINI_BATCH_SIZE_SANITY:-4}
    export N_RESPONSES=${N_RESPONSES_SANITY:-1}
    export LOG_PROB_TOP_K=${LOG_PROB_TOP_K_SANITY:-4}
    export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU_SANITY:-4096}
    export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
    if [ "$TEACHER_MAX_NEW_TOKENS_FROM_ENV" -eq 0 ]; then
        export TEACHER_ROLLOUT_MAX_NEW_TOKENS=$MAX_RESP_LENGTH
    fi
    echo "[INFO] SANITY_MODE=True -> prompt=${MAX_PROMPT_LENGTH}, resp=${MAX_RESP_LENGTH}, val_resp=${MAX_VAL_RESP_LENGTH}, n=${N_RESPONSES}, mbs=${MINI_BATCH_SIZE}, top_k=${LOG_PROB_TOP_K}, ppo_max_tokens=${PPO_MAX_TOKEN_LEN_PER_GPU}, total_steps=${TOTAL_TRAINING_STEPS}, n_gpus=${N_GPUS_PER_NODE}"
fi

export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))

# TODO: qwen3_1p7b_base / qwen3_1p7b / llama31_8b_base / llama31_8b_inst / qwen3_8b_base / qwen3_8b / qwen25_1p5b_base / qwen25_1p5b_inst / qwen25_7b_base / qwen25_7b_inst / qwen25_math_7b_base / qwen25_math_7b_inst / qwen25_math_1p5b_base / qwen25_math_1p5b_inst / distill_r1_1p5b / olmo2_1124_7b_base / olmo2_1124_7b_sft / olmo2_1124_7b_inst / llama32_3b_inst
# export EXPERIMENT_NAME=grpo_${TASK}_llama31_tulu3_8b_sft_8k-T_${TEMPERATURE}-n_${N_RESPONSES}-kl_${USE_KL}-mbs_${MINI_BATCH_SIZE}-${REWARD_TYPE}-$(date +%Y-%m-%d_%H-%M-%S)

# export TRAIN_DATASET=datasets/DAPO-Math-17k/data/dapo-math-17k-10percent.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/OpenThoughts3_opd.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/sampled_complement_30k.parquet
# export TRAIN_DATASET=datasets/DeepMath-103K/verl_format/train_filtered_sampled.parquet
# Teacher-aligned prompt format: "{Question} ... \\boxed{}"
export TRAIN_DATASET=datasets/dapo-math-17k-processed.parquet
# export TRAIN_DATASET=datasets/Skywork-OR1-RL-Data/data/math-00000-of-00001.parquet
# export TRAIN_DATASET=datasets/Skywork-OR1-RL-Data/filtered/math-1p5b-filtered-diff-max8.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet
# export TRAIN_DATASET=datasets/skywork/train_7b_math.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k-Processed/DAPO-Math_part2.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/verl_format/train.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k-TeacherAligned
# export TRAIN_DATASET_NAME=POLARIS-4B-S1
# export TRAIN_DATASET_NAME=Skywork-OR1-RL-Data
# export TRAIN_DATASET_NAME=DAPO-Math-17k-1percent
# export TRAIN_DATASET_NAME=DeepMath-103K-filtered-sampled
# export TRAIN_DATASET_NAME=DAPO-Math-17k-10percent
# export TRAIN_DATASET_NAME=OpenThoughts3-1.2M-opd
# export TRAIN_DATASET_NAME=OpenThoughts3-1.2M-30k

export TEST_DATA_DIR=datasets/test_data
# TRAIN_DATASET=${TRAIN_FILE:-["$DATA_DIR/$TASK/train_${SAMPLE_SIZE}.parquet"]}
TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME24/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$DATA_DIR/AIME24/test.parquet","$DATA_DIR/AIME25/test.parquet","$DATA_DIR/AMC23/test.parquet","$DATA_DIR/MATH-500/test.parquet","$DATA_DIR/Minerva/test.parquet","$DATA_DIR/Olympiad-Bench/test.parquet"]}

# TODO:
# export ACTOR_MODEL_PATH=model/qwen3-1.7b-math-sft
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft-skywork
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft-ds-7b
# export ACTOR_MODEL_PATH=/workspace/model/Qwen3-1.7B-SFT-DAPO-4B-RL
# export ACTOR_MODEL_PATH=/workspace/model/Qwen3-1.7B-SFT-DAPO-4B
# export ACTOR_MODEL_PATH=model/Qwen2.5-Math-1.5B
DEFAULT_MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

resolve_model_ref() {
    local raw_ref="$1"
    local fallback_ref="$2"
    local ref_name="$3"

    # Treat local-looking refs (absolute, ./, ../, or model/...) as filesystem paths.
    if [[ "$raw_ref" == /* || "$raw_ref" == ./* || "$raw_ref" == ../* || "$raw_ref" == model/* ]]; then
        if [ -e "$raw_ref" ]; then
            echo "$raw_ref"
            return
        fi
        if [ -e "$SCRIPT_DIR/$raw_ref" ]; then
            echo "$SCRIPT_DIR/$raw_ref"
            return
        fi
        echo "[WARN] $ref_name '$raw_ref' does not exist on this node; fallback to '$fallback_ref'." >&2
        echo "$fallback_ref"
        return
    fi

    # Non-local refs are assumed to be Hugging Face repo ids.
    echo "$raw_ref"
}

if [ -z "${ACTOR_MODEL_PATH:-}" ]; then
    if [ -d "model/DeepSeek-R1-Distill-Qwen-1.5B" ]; then
        export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-1.5B
    else
        export ACTOR_MODEL_PATH=$DEFAULT_MODEL_REPO
    fi
fi
export ACTOR_MODEL_PATH="$(resolve_model_ref "$ACTOR_MODEL_PATH" "$DEFAULT_MODEL_REPO" "ACTOR_MODEL_PATH")"
# export ACTOR_MODEL_PATH=model/JustRL-DeepSeek-1.5B-step_0400
# export ACTOR_MODEL_PATH=model/JustRL-DeepSeek-1.5B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-SFT
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base-SFT-OpenThought3-4B/checkpoint-1800
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base-SFT-DeepMath-4B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-sft/checkpoint-6000
# export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
# export ACTOR_MODEL_PATH=model/DS-1.5B-SFT
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
# export REWARD_MODEL_PATH=model/Qwen3-4B
# export REWARD_MODEL_PATH=model/Qwen3-4B-grpo
# export REWARD_MODEL_PATH=model/Qwen3-1.7B
# export REWARD_MODEL_PATH=model/OpenMath-Nemotron-1.5B
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
# export REWARD_MODEL_PATH=model/Qwen3-4B-Non-Thinking-RL-Math
# export REWARD_MODEL_PATH=model/Skywork-OR1-Math-7B
# export REWARD_MODEL_PATH=model/Polaris-4B-Preview
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-14B
if [ -z "${REWARD_MODEL_PATH:-}" ]; then
    if [ -d "model/JustRL-DeepSeek-1.5B" ]; then
        export REWARD_MODEL_PATH=model/JustRL-DeepSeek-1.5B
    else
        # Fall back to the actor model when a local reward checkpoint is absent.
        export REWARD_MODEL_PATH="$ACTOR_MODEL_PATH"
    fi
fi
export REWARD_MODEL_PATH="$(resolve_model_ref "$REWARD_MODEL_PATH" "$ACTOR_MODEL_PATH" "REWARD_MODEL_PATH")"
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")
echo "Resolved ACTOR_MODEL_PATH=$ACTOR_MODEL_PATH"
echo "Resolved REWARD_MODEL_PATH=$REWARD_MODEL_PATH"

export PROJECT_PATH=checkpoint
export PARALLEL_SIZE=1
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export N_NODES=${N_NODES:-1}
export CKPT_PATH=${PROJECT_PATH}/${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)
OUTLINES_RUN_ID=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid)
export OUTLINES_CACHE_DIR="${HOME}/.cache/outlines/${OUTLINES_RUN_ID}"
export NCCL_DEBUG=WARN

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# sm_100a (Blackwell) requires CUDA 12.8+ to JIT-compile flashinfer kernels; fall back to PyTorch SDPA
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export HF_ATTN_IMPL=${HF_ATTN_IMPL:-sdpa}
export TOKENIZERS_PARALLELISM=true
SWANLAB_RUN_STAMP=$(date +%Y-%m-%d_%H-%M-%S)_$$
export SWANLAB_LOG_ROOT=${SWANLAB_LOG_ROOT:-${PROJECT_PATH}/swanlab_log}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${SWANLAB_LOG_ROOT}/${SWANLAB_RUN_STAMP}}
mkdir -p "${SWANLAB_LOG_DIR}"
export SWANLAB_MODE=local
export HYDRA_FULL_ERROR=1


export EXPERIMENT_NAME=${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
fi

PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( ((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) > 32768) ? (MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) : 32768))}
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"
TOTAL_TRAINING_STEPS_ARG=""
if [ -n "${TOTAL_TRAINING_STEPS:-}" ]; then
    TOTAL_TRAINING_STEPS_ARG="trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
fi

# Validate runtime GPU visibility before spending time in dataloading/init.
VISIBLE_GPUS=$("$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
echo "[INFO] Visible CUDA GPUs: ${VISIBLE_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
if [ "$VISIBLE_GPUS" -lt 1 ]; then
    echo "[ERROR] No CUDA GPUs are visible to this process."
    echo "[ERROR] Run on a GPU node / with sbatch gpu resources, or check CUDA_VISIBLE_DEVICES."
    exit 1
fi
if [ "$VISIBLE_GPUS" -lt "$N_GPUS_PER_NODE" ]; then
    echo "[ERROR] Visible GPUs (${VISIBLE_GPUS}) < requested N_GPUS_PER_NODE (${N_GPUS_PER_NODE})."
    echo "[ERROR] Either request more GPUs or run with N_GPUS_PER_NODE=${VISIBLE_GPUS}."
    exit 1
fi


run_ray_cli start --head --node-ip-address=127.0.0.1 --num-gpus="$N_GPUS_PER_NODE"
sleep 5


"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    ++algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    ++actor_rollout_ref.model.override_config.attn_implementation=$HF_ATTN_IMPL \
    actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    ++actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    ++actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    ++actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    ++actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    ++actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_SAMPLES \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    ++actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    ++reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.model.path=$REWARD_MODEL_PATH \
    ++reward_model.model.override_config.attn_implementation=$HF_ATTN_IMPL \
    ++reward_model.model.input_tokenizer=null \
    ++reward_model.model.use_remove_padding=$USE_REMOVE_PADDING \
    ++reward_model.model.fsdp_config.param_offload=False \
    ++reward_model.model.dtype=$MODEL_DTYPE \
    ++reward_model.micro_batch_size_per_gpu=$RM_MICRO_BATCH_SIZE_PER_GPU \
    ++reward_model.teacher_rollout.enable=$TEACHER_ROLLOUT_ENABLE \
    ++reward_model.teacher_rollout.n_rollouts=$N_TEACHER_ROLLOUTS \
    ++reward_model.teacher_rollout.max_new_tokens=$TEACHER_ROLLOUT_MAX_NEW_TOKENS \
    ++reward_model.teacher_rollout.temperature=$TEACHER_ROLLOUT_GEN_TEMPERATURE \
    ++reward_model.teacher_rollout.top_p=$TEACHER_ROLLOUT_TOP_P \
    ++reward_model.teacher_rollout.fallback=$TEACHER_ROLLOUT_FALLBACK \
    ++reward_model.teacher_rollout.n_rollouts_hint=$TEACHER_ROLLOUT_N_ROLLOUTS_HINT \
    ++reward_model.teacher_rollout.tier1_only=$TEACHER_ROLLOUT_TIER1_ONLY \
    ++reward_model.teacher_rollout.apply_selected_for_teacher_forward=$TEACHER_ROLLOUT_APPLY_SELECTED_FOR_TEACHER_FORWARD \
    ++reward_model.teacher_rollout.multi_forward_select_by_overlap=$TEACHER_ROLLOUT_MULTI_FORWARD_SELECT_BY_OVERLAP \
    ++reward_model.teacher_rollout.prompt_disturb_enable=$TEACHER_PROMPT_DISTURB_ENABLE \
    ++reward_model.teacher_rollout.prompt_disturb_rollout_index=$TEACHER_PROMPT_DISTURB_ROLLOUT_INDEX \
    "++reward_model.teacher_rollout.prompt_disturb_text=$TEACHER_PROMPT_DISTURB_TEXT" \
    ++actor_rollout_ref.rollout.teacher_rollout.enable=$TEACHER_ROLLOUT_ENABLE \
    ++actor_rollout_ref.rollout.teacher_rollout.n_rollouts=$N_TEACHER_ROLLOUTS \
    ++actor_rollout_ref.rollout.teacher_rollout.max_new_tokens=$TEACHER_ROLLOUT_MAX_NEW_TOKENS \
    ++actor_rollout_ref.rollout.teacher_rollout.temperature=$TEACHER_ROLLOUT_GEN_TEMPERATURE \
    ++actor_rollout_ref.rollout.teacher_rollout.top_p=$TEACHER_ROLLOUT_TOP_P \
    ++actor_rollout_ref.rollout.teacher_rollout.fallback=$TEACHER_ROLLOUT_FALLBACK \
    ++actor_rollout_ref.rollout.teacher_rollout.n_rollouts_hint=$TEACHER_ROLLOUT_N_ROLLOUTS_HINT \
    ++actor_rollout_ref.rollout.teacher_rollout.tier1_only=$TEACHER_ROLLOUT_TIER1_ONLY \
    ++actor_rollout_ref.rollout.teacher_rollout.apply_selected_for_teacher_forward=$TEACHER_ROLLOUT_APPLY_SELECTED_FOR_TEACHER_FORWARD \
    ++actor_rollout_ref.rollout.teacher_rollout.multi_forward_select_by_overlap=$TEACHER_ROLLOUT_MULTI_FORWARD_SELECT_BY_OVERLAP \
    ++actor_rollout_ref.rollout.aux_teacher_ctx_kd_enable=$AUX_TEACHER_CTX_KD_ENABLE \
    ++actor_rollout_ref.rollout.aux_teacher_ctx_kd_coef=$AUX_TEACHER_CTX_KD_COEF \
    ++actor_rollout_ref.rollout.aux_teacher_ctx_kd_weight_mode=$AUX_TEACHER_CTX_KD_WEIGHT_MODE \
    ++actor_rollout_ref.rollout.teacher_rollout.prompt_disturb_enable=$TEACHER_PROMPT_DISTURB_ENABLE \
    ++actor_rollout_ref.rollout.teacher_rollout.prompt_disturb_rollout_index=$TEACHER_PROMPT_DISTURB_ROLLOUT_INDEX \
    "++actor_rollout_ref.rollout.teacher_rollout.prompt_disturb_text=$TEACHER_PROMPT_DISTURB_TEXT" \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=20 \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=1 \
    $TOTAL_TRAINING_STEPS_ARG \
    trainer.default_local_dir="$CKPT_PATH" \
    ++trainer.is_plot=$IS_PLOT

# Log the end time for local runs.
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
