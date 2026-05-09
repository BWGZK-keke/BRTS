# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Teacher response selection for On-Policy Distillation with teacher rollout.

3-tier waterfall (per sample):
  Tier 1 — M1 unhinted teacher rollouts. If any are correct, pick argmax
           top-K overlap-to-student among the correct ones.
  Tier 2 — samples with no correct tier-1 rollout get M2 hinted rollouts
           (SILENT_VALIDATION_KEY in the prompt). Same correct + most-similar
           selection rule.
  Tier 3 — still no correct rollout: pick the tier-1 rollout most similar to
           the student (best-available trace).

Selection output is consumed by training: the selected teacher rollout is used
as teacher forward context for reward/distillation. Tier tags and selection
metrics are also logged for inspection.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from verl import DataProto


def _extract_ground_truths(batch: DataProto) -> List[str]:
    BS = batch.batch["input_ids"].shape[0]
    rm_info = batch.non_tensor_batch.get("reward_model", None)
    gts: List[str] = []
    for i in range(BS):
        gt = ""
        if rm_info is not None:
            entry = rm_info[i]
            if isinstance(entry, dict):
                gt = str(entry.get("ground_truth", ""))
            else:
                try:
                    gt = str(entry)
                except Exception:
                    gt = ""
        gts.append(gt)
    return gts


def _grade_rollouts(
    teacher_texts: List[str],
    ground_truths: List[str],
    BS: int,
    M: int,
) -> torch.Tensor:
    """Returns (BS, M) bool tensor of per-rollout correctness."""
    from verl.utils.reward_score.ttrl_math import extract_answer, grade

    correctness = torch.zeros(BS, M, dtype=torch.bool)
    for i in range(BS):
        gt = ground_truths[i]
        if not gt:
            continue
        for j in range(M):
            text = teacher_texts[i * M + j]
            pred = extract_answer(text)
            if pred is None:
                continue
            try:
                correctness[i, j] = bool(grade(pred, gt, fast=True))
            except Exception:
                correctness[i, j] = False
    return correctness


def _topk_overlap(
    teacher_ids_2d: torch.Tensor,  # (BS, M, L)
    student_topk: Optional[torch.Tensor],  # (BS, SeqLen, K) or None
    resp_len: int,
    pad_id: int,
) -> torch.Tensor:
    """Fraction of teacher tokens that fall inside student's top-K at each position."""
    BS, M, L = teacher_ids_2d.shape
    if student_topk is None:
        return torch.zeros(BS, M, dtype=torch.float32)

    s_topk = student_topk.cpu()
    seq_len_k = s_topk.shape[1]
    compare_len = min(L, resp_len, seq_len_k)
    t_slice = teacher_ids_2d[:, :, :compare_len]
    s_slice = s_topk[:, :compare_len, :]

    similarity = torch.zeros(BS, M, dtype=torch.float32)
    for i in range(BS):
        s_k_i = s_slice[i]
        for j in range(M):
            t_j = t_slice[i, j]
            valid = t_j != pad_id
            if valid.sum() == 0:
                continue
            in_topk = (t_j.unsqueeze(-1) == s_k_i).any(dim=-1)
            similarity[i, j] = (in_topk & valid).float().sum() / valid.float().sum()
    return similarity


def select_teacher_tier1(
    batch: DataProto,
    teacher_rollouts: DataProto,
    n_rollouts: int,
    tokenizer,
) -> Tuple[dict, dict]:
    """Score tier-1 teacher rollouts and identify samples that need a tier-2 retry.

    Returns:
        state: dict carrying tier-1 tensors for reuse in ``finalize_selection``.
        metrics: tier-1 logging scalars.
    """
    BS = batch.batch["input_ids"].shape[0]
    teacher_ids = teacher_rollouts.batch["teacher_response_ids"]  # (BS*M, L)
    M = n_rollouts
    L = teacher_ids.shape[1]
    teacher_ids_2d = teacher_ids.view(BS, M, L).cpu()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    ground_truths = _extract_ground_truths(batch)
    teacher_texts = tokenizer.batch_decode(teacher_ids.tolist(), skip_special_tokens=True)
    correctness = _grade_rollouts(teacher_texts, ground_truths, BS, M)

    similarity = _topk_overlap(
        teacher_ids_2d=teacher_ids_2d,
        student_topk=batch.batch.get("student_top_k_ids", None),
        resp_len=batch.batch["responses"].shape[1],
        pad_id=pad_id,
    )

    retry_indices = [i for i in range(BS) if not bool(correctness[i].any())]

    state = {
        "teacher_ids_2d": teacher_ids_2d,
        "correctness": correctness,
        "similarity": similarity,
        "retry_indices": torch.tensor(retry_indices, dtype=torch.long),
        "ground_truths": ground_truths,
        "pad_id": pad_id,
    }
    metrics = {
        "teacher_selection/tier1_any_correct_frac": float(correctness.any(dim=1).float().mean().item()),
        "teacher_selection/tier1_retry_frac": len(retry_indices) / BS,
    }
    return state, metrics


def finalize_selection(
    batch: DataProto,
    tier1_state: dict,
    tier2_rollouts: Optional[DataProto],
    n_rollouts_tier2: int,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Merge tier-1 picks with tier-2 hinted retries and tier-3 fallback.

    Returns:
        final_selected_ids: (BS, L_max) teacher tokens selected per sample.
        tier_tags: (BS,) int8 — 1 / 2 / 3.
        metrics: dict of per-tier fractions and mean similarities.
    """
    BS = batch.batch["input_ids"].shape[0]
    teacher_ids_2d = tier1_state["teacher_ids_2d"]  # (BS, M1, L1)
    M1, L1 = teacher_ids_2d.shape[1], teacher_ids_2d.shape[2]
    correctness1 = tier1_state["correctness"]
    similarity1 = tier1_state["similarity"]
    retry_indices = tier1_state["retry_indices"]
    ground_truths = tier1_state["ground_truths"]
    pad_id = tier1_state["pad_id"]
    n_retry = int(retry_indices.numel())

    # ------------------------------------------------------------------
    # Tier 2 scoring (only for samples that needed retry)
    # ------------------------------------------------------------------
    tier2_ids_2d = None
    tier2_correct = None
    tier2_sim = None
    L2 = 0
    if tier2_rollouts is not None and n_retry > 0:
        t2 = tier2_rollouts.batch["teacher_response_ids"]  # (n_retry*M2, L2) or (BS*M2, L2)
        M2 = n_rollouts_tier2
        L2 = t2.shape[1]
        if t2.shape[0] == n_retry * M2:
            # Preferred layout: only retry samples were generated.
            tier2_ids_2d = t2.view(n_retry, M2, L2).cpu()
        elif t2.shape[0] == BS * M2:
            # Backward-compatible layout: hinted rollouts were generated for all
            # samples; keep only retry rows.
            tier2_all = t2.view(BS, M2, L2).cpu()
            tier2_ids_2d = tier2_all[retry_indices]
        else:
            raise ValueError(
                "Unexpected tier-2 rollout shape: "
                f"got {t2.shape[0]} rows, expected {n_retry * M2} (retries only) "
                f"or {BS * M2} (full batch)."
            )

        retry_gts = [ground_truths[i] for i in retry_indices.tolist()]
        tier2_ids_flat = tier2_ids_2d.reshape(-1, L2)
        t2_texts = tokenizer.batch_decode(tier2_ids_flat.tolist(), skip_special_tokens=True)
        tier2_correct = _grade_rollouts(t2_texts, retry_gts, n_retry, M2)

        s_topk = batch.batch.get("student_top_k_ids", None)
        s_topk_retry = s_topk[retry_indices].cpu() if s_topk is not None else None
        tier2_sim = _topk_overlap(
            teacher_ids_2d=tier2_ids_2d,
            student_topk=s_topk_retry,
            resp_len=batch.batch["responses"].shape[1],
            pad_id=pad_id,
        )

    # ------------------------------------------------------------------
    # Assemble final selection (padded to max response length across tiers)
    # ------------------------------------------------------------------
    L_max = max(L1, L2) if L2 > 0 else L1
    final_selected_ids = torch.full(
        (BS, L_max), pad_id, dtype=teacher_ids_2d.dtype
    )
    tier_tags = torch.zeros(BS, dtype=torch.int8)

    retry_to_local = {int(idx): k for k, idx in enumerate(retry_indices.tolist())}

    n_tier = [0, 0, 0]
    sum_sim_tier = [0.0, 0.0, 0.0]

    def _pad_to_lmax(t: torch.Tensor, src_len: int) -> torch.Tensor:
        if src_len < L_max:
            return F.pad(t, (0, L_max - src_len), value=pad_id)
        return t

    for i in range(BS):
        if i not in retry_to_local:
            # Tier 1: at least one correct → most-similar among correct
            sim_masked = similarity1[i].clone()
            sim_masked[~correctness1[i]] = -1.0
            best_j = int(sim_masked.argmax())
            final_selected_ids[i] = _pad_to_lmax(teacher_ids_2d[i, best_j], L1)
            tier_tags[i] = 1
            n_tier[0] += 1
            sum_sim_tier[0] += float(similarity1[i, best_j])
            continue

        k = retry_to_local[i]
        if tier2_correct is not None and bool(tier2_correct[k].any()):
            # Tier 2: correct in hinted retry → most-similar among correct
            sim_masked = tier2_sim[k].clone()
            sim_masked[~tier2_correct[k]] = -1.0
            best_j = int(sim_masked.argmax())
            final_selected_ids[i] = _pad_to_lmax(tier2_ids_2d[k, best_j], L2)
            tier_tags[i] = 2
            n_tier[1] += 1
            sum_sim_tier[1] += float(tier2_sim[k, best_j])
        else:
            # Tier 3: pick most-similar tier-1 rollout (no correctness guarantee)
            best_j = int(similarity1[i].argmax())
            final_selected_ids[i] = _pad_to_lmax(teacher_ids_2d[i, best_j], L1)
            tier_tags[i] = 3
            n_tier[2] += 1
            sum_sim_tier[2] += float(similarity1[i, best_j])

    metrics = {
        "teacher_selection/frac_tier1": n_tier[0] / BS,
        "teacher_selection/frac_tier2": n_tier[1] / BS,
        "teacher_selection/frac_tier3": n_tier[2] / BS,
        "teacher_selection/avg_sim_tier1": sum_sim_tier[0] / max(1, n_tier[0]),
        "teacher_selection/avg_sim_tier2": sum_sim_tier[1] / max(1, n_tier[1]),
        "teacher_selection/avg_sim_tier3": sum_sim_tier[2] / max(1, n_tier[2]),
    }
    if tier2_correct is not None:
        metrics["teacher_selection/tier2_any_correct_frac_of_retries"] = float(
            tier2_correct.any(dim=1).float().mean().item()
        )
    return final_selected_ids, tier_tags, metrics


def finalize_selection_second_then_first(
    batch: DataProto,
    tier1_state: dict,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Fast 3-tier selection using only tier-1 rollouts with fixed index priority.

    Rule per sample:
      Tier 1: if the 2nd rollout (index=1) is correct, select it.
      Tier 2: else if the 1st rollout (index=0) is correct, select it.
      Tier 3: else fallback to most-similar tier-1 rollout.

    This mode avoids hinted retries and can reduce teacher time when paired with
    ``n_rollouts=2``.
    """
    BS = batch.batch["input_ids"].shape[0]
    teacher_ids_2d = tier1_state["teacher_ids_2d"]  # (BS, M1, L1)
    correctness1 = tier1_state["correctness"]  # (BS, M1)
    similarity1 = tier1_state["similarity"]  # (BS, M1)
    pad_id = tier1_state["pad_id"]

    M1 = int(teacher_ids_2d.shape[1])
    L1 = int(teacher_ids_2d.shape[2])
    if M1 < 2:
        raise ValueError(
            "selection_mode='second_then_first' requires n_rollouts >= 2, "
            f"but got n_rollouts={M1}."
        )

    final_selected_ids = torch.full((BS, L1), pad_id, dtype=teacher_ids_2d.dtype)
    tier_tags = torch.zeros(BS, dtype=torch.int8)

    n_tier = [0, 0, 0]
    sum_sim_tier = [0.0, 0.0, 0.0]

    for i in range(BS):
        if bool(correctness1[i, 1]):
            final_selected_ids[i] = teacher_ids_2d[i, 1]
            tier_tags[i] = 1
            n_tier[0] += 1
            sum_sim_tier[0] += float(similarity1[i, 1])
        elif bool(correctness1[i, 0]):
            final_selected_ids[i] = teacher_ids_2d[i, 0]
            tier_tags[i] = 2
            n_tier[1] += 1
            sum_sim_tier[1] += float(similarity1[i, 0])
        else:
            best_j = int(similarity1[i].argmax())
            final_selected_ids[i] = teacher_ids_2d[i, best_j]
            tier_tags[i] = 3
            n_tier[2] += 1
            sum_sim_tier[2] += float(similarity1[i, best_j])

    metrics = {
        "teacher_selection/frac_tier1": n_tier[0] / BS,
        "teacher_selection/frac_tier2": n_tier[1] / BS,
        "teacher_selection/frac_tier3": n_tier[2] / BS,
        "teacher_selection/avg_sim_tier1": sum_sim_tier[0] / max(1, n_tier[0]),
        "teacher_selection/avg_sim_tier2": sum_sim_tier[1] / max(1, n_tier[1]),
        "teacher_selection/avg_sim_tier3": sum_sim_tier[2] / max(1, n_tier[2]),
    }
    return final_selected_ids, tier_tags, metrics
