<div align="center">

# On-Policy Distillation with Best-of-N Teacher Rollout Selection

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#overview" style="text-decoration: none; font-weight: bold;">📖 Overview</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#repository-layout" style="text-decoration: none; font-weight: bold;">📁 Repository Layout</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div>

---

## 📖Overview

This repository contains the official implementation of **BRTS** (**B**est-of-N **R**ollout **T**eacher **S**election), a lightweight extension to on-policy distillation (OPD) that improves how teacher trajectories are selected and used as supervision.

Standard OPD supervises the student on its own sampled trajectories using a single stochastic teacher rollout per prompt. As a result, the supervision signal can be high-variance: the sampled teacher trajectory can be incorrect, uninformative, or poorly matched to the student's current reasoning behavior. BRTS addresses this with a **correctness- and alignment-aware** teacher-context branch:

1. **Tier 1 — Multi-rollout sampling.** For each prompt, draw `N` unconditioned teacher rollouts. If at least one is correct, select the correct rollout with the highest top-K token-level overlap with the student.
2. **Tier 2 — Ground-truth-guided recovery.** If all Tier-1 rollouts are wrong, re-prompt the teacher with the ground-truth answer silently injected as a validation signal, and retain the resulting rollout only if its extracted answer is correct.
3. **Tier 3 — Fallback.** If neither tier yields a correct rollout, fall back to the Tier-1 rollout with the highest student overlap.

The selected trajectory is then used in an auxiliary teacher-context distillation loss alongside the standard student-context OPD loss:

```
L_total = L_student-context + λ · L_teacher-context
```

Experiments on AIME 2024, AIME 2025, and AMC 2023 show that BRTS improves over standard OPD, with the largest gains on harder benchmarks where reliable teacher trajectories are sparse.

## ✨Getting Started

### Environment Setup

This codebase is built on top of [verl](https://github.com/volcengine/verl). To prepare the environment:

```bash
conda create -n verl python==3.12
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

Set Hugging Face credentials in your environment if your models require authentication:

```bash
export HF_TOKEN=<your_token>
# Optional: point HF_HOME at a shared cache directory
export HF_HOME=/path/to/hf_cache
```

### Training

#### BRTS (Tier-1 only)

Multi-rollout teacher sampling with overlap-based selection, no ground-truth-guided recovery:

```bash
bash forward.sh
```

This launches training with the following BRTS settings:

| Variable | Value | Description |
|----------|-------|-------------|
| `TEACHER_ROLLOUT_MULTI_FORWARD_SELECT_BY_OVERLAP` | `True` | Enable Best-of-N selection by top-K overlap |
| `TEACHER_ROLLOUT_TIER1_ONLY` | `True` | Disable Tier-2 ground-truth-guided recovery |
| `TEACHER_ROLLOUT_N_ROLLOUTS_HINT` | `0` | No hinted retries |

#### BRTS (Tier-1 + Tier-2)

Adds ground-truth-guided recovery for prompts where all Tier-1 rollouts fail:

```bash
bash forward_tier2.sh
```

This sets `TEACHER_ROLLOUT_TIER1_ONLY=False` and `TEACHER_ROLLOUT_N_ROLLOUTS_HINT=1`.

<details>
<summary><b>Key Parameters (override via env vars)</b></summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Models** |||
| `ACTOR_MODEL_PATH` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Path or HF repo id of the student (policy) model |
| `REWARD_MODEL_PATH` | `model/JustRL-DeepSeek-1.5B` (or actor model fallback) | Path of the teacher model that provides token-level distillation signals |
| **Distillation** |||
| `ADV_ESTIMATOR` | `token_reward_direct` | Required for OPD; do not change |
| `LOG_PROB_TOP_K` | `16` | Number of top-K tokens retained per position; `0` falls back to sampled-token OPD |
| `TOP_K_STRATEGY` | `only_stu` | Top-K candidate set source: `only_stu`, `only_tch`, `intersection`, `union`, `union-intersection` |
| `REWARD_WEIGHT_MODE` | `student_p` | Token-reward weighting: `student_p`, `teacher_p`, `none` |
| **Generation** |||
| `N_RESPONSES` | `2` | Student rollouts per prompt |
| `MAX_PROMPT_LENGTH` | `1024` | Max prompt length |
| `MAX_RESP_LENGTH` | `7168` | Max student response length during training |
| `MAX_VAL_RESP_LENGTH` | `31744` | Max response length during validation |
| **BRTS — Teacher Rollout** |||
| `TEACHER_ROLLOUT_ENABLE` | `True` | Enable teacher rollouts for distillation |
| `N_TEACHER_ROLLOUTS` | `2` | Number of Tier-1 unconditioned teacher candidates |
| `TEACHER_ROLLOUT_GEN_TEMPERATURE` | `0.7` | Teacher sampling temperature |
| `TEACHER_ROLLOUT_TOP_P` | `0.95` | Teacher nucleus sampling |
| `TEACHER_ROLLOUT_TIER1_ONLY` | `False` | Disable Tier-2 hinted recovery |
| `TEACHER_ROLLOUT_N_ROLLOUTS_HINT` | `1` | Number of Tier-2 ground-truth-guided rollouts |
| `TEACHER_ROLLOUT_FALLBACK` | `most_similar` | Tier-3 fallback: `most_similar` or `skip` |
| `TEACHER_ROLLOUT_MULTI_FORWARD_SELECT_BY_OVERLAP` | `False` | Run teacher forward on both student and selected contexts and pick by overlap |
| **BRTS — Auxiliary Teacher-Context KD** |||
| `AUX_TEACHER_CTX_KD_ENABLE` | `False` | Enable auxiliary teacher-context distillation loss |
| `AUX_TEACHER_CTX_KD_COEF` | `0.05` | Coefficient λ on the auxiliary teacher-context loss |
| `AUX_TEACHER_CTX_KD_WEIGHT_MODE` | `student_p` | Weighting for the auxiliary loss |
| **BRTS — Prompt Perturbation Ablation** |||
| `TEACHER_PROMPT_DISTURB_ENABLE` | `False` | Append a perturbation string to one teacher rollout's prompt |
| `TEACHER_PROMPT_DISTURB_ROLLOUT_INDEX` | `2` | Which rollout (1-indexed) to perturb |
| `TEACHER_PROMPT_DISTURB_TEXT` | `"Please reason step by step and rethink in detail before giving the final answer."` | Perturbation text |

</details>

> [!IMPORTANT]
> **Non-thinking Models:** When training a non-thinking model (e.g. `Qwen3-1.7B (Non-thinking)`), add `+data.apply_chat_template_kwargs.enable_thinking=False` to the training script.

### Validation

The evaluation pipeline is adapted from [JustRL](https://github.com/RyanLiu112/JustRL).

**Generation (optional):**

```bash
cd scripts/val/eval
python gen_vllm.py
```

Before running generation, set `MODEL_NAMES` in `gen_vllm.py` to the checkpoint(s) you want to evaluate, and set an appropriate `available_workers`.

**Grading:**

```bash
cd scripts/val/eval
python grade.py
```

The grading script processes all JSONL files in the output directory and produces `grading_results.json`. Enable the LLM-based verifier with:

```bash
python grade.py --enable_model_verifier
```

*All main experiments were conducted on 8 × NVIDIA B200 GPUs.*

## 📁Repository Layout

```
.
├── forward.sh                 # BRTS Tier-1-only launcher
├── forward_tier2.sh           # BRTS Tier-1 + Tier-2 launcher
├── on_policy_distillation.sh  # Main OPD/BRTS training script
├── grpo.sh                    # GRPO RL baseline
├── scripts/
│   ├── fetch_val_acc.py       # Swanlab log → accuracy table utility
│   ├── infer/
│   │   ├── vllm_rollout.py    # Teacher rollout generator
│   │   └── dedup_deepmath.py  # Deduplicate DeepMath against DAPO-Math-17k
│   └── val/eval/
│       ├── gen_vllm.py        # Validation generation
│       ├── grade.py           # Validation grading (with optional LLM verifier)
│       └── utils.py
└── verl/                      # verl framework with BRTS extensions
    └── verl/trainer/ppo/
        ├── teacher_selection.py     # ★ Tier-1/2/3 teacher rollout selection
        ├── rollout_corr_helper.py   # ★ Correctness checking & overlap utilities
        ├── ray_trainer.py           # PPO trainer (with BRTS hooks)
        └── ...
```

The core BRTS implementation lives in `verl/verl/trainer/ppo/teacher_selection.py`, which implements the 3-tier waterfall (Tier-1 unconditioned sampling → Tier-2 ground-truth-guided recovery → Tier-3 most-similar fallback).

## 🎈Citation

If you find this work helpful, please cite us:

```bibtex
@inproceedings{brts2026,
  title={On-Policy Distillation with Best-of-N Teacher Rollout Selection},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## Acknowledgements

This codebase is built on top of [verl](https://github.com/volcengine/verl), and the evaluation pipeline is adapted from [JustRL](https://github.com/RyanLiu112/JustRL). We thank the authors of these projects for their excellent open-source work.
