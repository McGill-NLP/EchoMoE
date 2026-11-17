"""The distillation objective: a coarsened forward KL against the teacher's cached top-M targets,
plus a small LM term on Dolci response tokens.

Per position, with the teacher's top-M ids `M`, its log-probabilities `log p` there and its
log-mass outside them `log P` (the tail bucket):

    KL = sum_{v in M} p(v) (log p(v) - log q(v)) + P (log P - log Q),    Q = q mass outside M

The student is normalized over the whole vocabulary. Both terms are computed a chunk of positions
at a time, so the full-vocabulary softmax never exists for the whole sequence. In training the
output projection runs inside the chunk too (`fused_loss_terms`).
"""

from __future__ import annotations

import functools
import math
from typing import Any

import torch
import torch.nn.functional as functional


def _chunk_terms(
    student_chunk: torch.Tensor,
    id_chunk: torch.Tensor,
    log_p: torch.Tensor,
    log_p_tail: torch.Tensor,
    target_chunk: torch.Tensor,
    accumulation_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_chunk = student_chunk.to(dtype=accumulation_dtype)
    log_p = log_p.to(dtype=accumulation_dtype)
    log_p_tail = log_p_tail.to(dtype=accumulation_dtype)
    carries_tail = log_p_tail > -math.inf
    log_q = functional.log_softmax(student_chunk, dim=-1)
    log_q_top = log_q.gather(1, id_chunk)
    # Summing outside the top-M rather than subtracting the top-M from one keeps log q_tail
    # exact once the student's mass there underflows the accumulation dtype.
    log_q_tail = log_q.scatter(1, id_chunk, -math.inf).logsumexp(dim=-1)
    kl = (log_p.exp() * (log_p - log_q_top)).sum(dim=-1) + torch.where(
        carries_tail, log_p_tail.exp() * (log_p_tail - log_q_tail), 0.0
    )
    cross_entropy = -log_q.gather(1, target_chunk.unsqueeze(-1)).squeeze(-1)
    return kl, cross_entropy


@functools.cache
def _compiled_chunk_terms():
    return torch.compile(_chunk_terms, dynamic=False, fullgraph=True)


def _chunk_terms_for(rows: int, position_chunk_size: int, compile_regions: bool):
    # Only the last chunk can be shorter. It stays eager so the compiled region meets one row
    # count per run and never recompiles.
    if compile_regions and rows == position_chunk_size:
        return _compiled_chunk_terms()
    return _chunk_terms


def _flat_targets(labels, target_ids, target_logprobs, target_tail_logprob, positions):
    return (
        target_ids.reshape(positions, -1).to(dtype=torch.int64),
        target_logprobs.detach().reshape(positions, -1),
        target_tail_logprob.detach().reshape(positions),
        labels.reshape(positions),
    )


def position_terms(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_ids: torch.Tensor,
    target_logprobs: torch.Tensor,
    target_tail_logprob: torch.Tensor,
    position_chunk_size: int,
    accumulation_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The coarsened KL and the gold-label cross entropy at every position, from full logits
    (validation)."""
    position_shape = student_logits.shape[:-1]
    positions, vocabulary = math.prod(position_shape), student_logits.shape[-1]
    ids, log_target, log_target_tail, targets = _flat_targets(
        labels, target_ids, target_logprobs, target_tail_logprob, positions
    )
    student = student_logits.reshape(positions, vocabulary)
    kl_parts, cross_entropy_parts = [], []
    for student_chunk, id_chunk, log_p, log_p_tail, target_chunk in zip(
        torch.split(student, position_chunk_size),
        torch.split(ids, position_chunk_size),
        torch.split(log_target, position_chunk_size),
        torch.split(log_target_tail, position_chunk_size),
        torch.split(targets, position_chunk_size),
        strict=True,
    ):
        kl, cross_entropy = _chunk_terms(
            student_chunk, id_chunk, log_p, log_p_tail, target_chunk, accumulation_dtype
        )
        kl_parts.append(kl)
        cross_entropy_parts.append(cross_entropy)
    return (
        torch.cat(kl_parts).reshape(position_shape),
        torch.cat(cross_entropy_parts).reshape(position_shape),
    )


def softcapped(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    """Gemma 4's final-logit softcap `cap * tanh(logits / cap)` in the logits' dtype, or the
    logits unchanged without a cap."""
    return logits if cap is None else cap * torch.tanh(logits / cap)


class _FusedHeadTerms(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        ids: torch.Tensor,
        log_target: torch.Tensor,
        log_target_tail: torch.Tensor,
        targets: torch.Tensor,
        position_chunk_size: int,
        accumulation_dtype: torch.dtype,
        compile_regions: bool,
        logit_softcap: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(hidden, weight, ids, log_target, log_target_tail, targets)
        ctx.position_chunk_size = position_chunk_size
        ctx.accumulation_dtype = accumulation_dtype
        ctx.compile_regions = compile_regions
        ctx.logit_softcap = logit_softcap
        positions = hidden.shape[0]
        kl = hidden.new_empty(positions, dtype=accumulation_dtype)
        cross_entropy = hidden.new_empty(positions, dtype=accumulation_dtype)
        for start in range(0, positions, position_chunk_size):
            end = min(start + position_chunk_size, positions)
            logits = softcapped(torch.mm(hidden[start:end], weight.t()), logit_softcap)
            terms = _chunk_terms_for(logits.shape[0], position_chunk_size, compile_regions)
            kl[start:end], cross_entropy[start:end] = terms(
                logits,
                ids[start:end],
                log_target[start:end],
                log_target_tail[start:end],
                targets[start:end],
                accumulation_dtype,
            )
        return kl, cross_entropy

    @staticmethod
    def backward(ctx: Any, grad_kl: torch.Tensor, grad_cross_entropy: torch.Tensor) -> tuple:
        hidden, weight, ids, log_target, log_target_tail, targets = ctx.saved_tensors
        chunk_size = ctx.position_chunk_size
        grad_hidden = torch.empty_like(hidden)
        # Chunk products are summed in the accumulation dtype. The weight gradient is rounded to
        # the weight's dtype once, at the end.
        grad_weight = torch.zeros(weight.shape, dtype=ctx.accumulation_dtype, device=weight.device)
        for start in range(0, hidden.shape[0], chunk_size):
            end = min(start + chunk_size, hidden.shape[0])
            hidden_chunk = hidden[start:end]
            with torch.enable_grad():
                projection = torch.mm(hidden_chunk, weight.t()).requires_grad_(True)
                logits = softcapped(projection, ctx.logit_softcap)
                terms = _chunk_terms_for(logits.shape[0], chunk_size, ctx.compile_regions)
                kl, cross_entropy = terms(
                    logits,
                    ids[start:end],
                    log_target[start:end],
                    log_target_tail[start:end],
                    targets[start:end],
                    ctx.accumulation_dtype,
                )
                (grad_logits,) = torch.autograd.grad(
                    (kl, cross_entropy),
                    (projection,),
                    (grad_kl[start:end], grad_cross_entropy[start:end]),
                )
            grad_hidden[start:end] = torch.mm(grad_logits, weight)
            grad_weight += torch.mm(grad_logits.t(), hidden_chunk)
        return grad_hidden, grad_weight.to(dtype=weight.dtype), *(None,) * 8


def fused_loss_terms(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    labels: torch.Tensor,
    prediction_mask: torch.Tensor,
    lm_mask: torch.Tensor,
    *,
    target_ids: torch.Tensor,
    target_logprobs: torch.Tensor,
    target_tail_logprob: torch.Tensor,
    position_chunk_size: int,
    accumulation_dtype: torch.dtype,
    compile_regions: bool = False,
    logit_softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's KL sum over the prediction mask and cross-entropy sum over the LM mask. Only
    one position chunk's logits exist at a time, and the backward recomputes them."""
    position_shape = hidden_states.shape[:-1]
    positions = math.prod(position_shape)
    ids, log_target, log_target_tail, targets = _flat_targets(
        labels, target_ids, target_logprobs, target_tail_logprob, positions
    )
    kl, cross_entropy = _FusedHeadTerms.apply(
        hidden_states.reshape(positions, hidden_states.shape[-1]),
        head_weight,
        ids,
        log_target,
        log_target_tail,
        targets,
        position_chunk_size,
        accumulation_dtype,
        compile_regions,
        logit_softcap,
    )
    kl, cross_entropy = kl.reshape(position_shape), cross_entropy.reshape(position_shape)
    return (
        (kl * prediction_mask.to(dtype=kl.dtype)).sum(),
        (cross_entropy * lm_mask.to(dtype=cross_entropy.dtype)).sum(),
    )


def local_objective(
    kl_sum: torch.Tensor,
    lm_sum: torch.Tensor,
    *,
    kl_tokens: int,
    lm_tokens: int,
    kl_weight: float,
    lm_weight: float,
) -> torch.Tensor:
    """This rank's share of `kl_weight * KL mean + lm_weight * LM mean`. It divides by the global
    token counts, so the summed gradients equal the global objective's under uneven packing."""
    kl = kl_sum * (float(kl_weight) / kl_tokens)
    lm = lm_sum * (0.0 if lm_tokens == 0 else float(lm_weight) / lm_tokens)
    return kl + lm
