"""Exact causal attention over one packed THD batch, for Gemma 4's global layers (head dim 512).

No fused kernel trains at head dim 512 on H100: cuDNN has no backend for it and FlashAttention
stops at 256. Transformer Engine's unfused path converts THD rows with the unpadded offsets, so it
reads the wrong rows when sequences are padded inside a pack (each to a multiple of 8). This takes
the pack as Megatron hands it, rows `[t, heads, dim]` at the padded offsets, and returns attention
at the same rows, zero at padding rows. Scores are taken over the whole pack in the inputs' dtype
with a block-diagonal causal mask, and the softmax in float32, as Transformer Engine's unfused path
computes them.
"""

from collections.abc import Sequence
from typing import Any

import torch
from megatron.bridge.models.gemma.modeling_gemma4 import _is_local_attn_layer


def packed_masks(
    cu_seqlens: torch.Tensor, cu_seqlens_padded: torch.Tensor, total: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The `[t, t]` attention mask and the `[t]` real-row mask of a pack of `total` rows. A real
    row attends to the real rows of its sequence at or before it. A padding row attends to itself
    alone, so no row of the mask is empty, and its output is zeroed."""
    positions = torch.arange(total, device=cu_seqlens.device)
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    starts = cu_seqlens_padded[:-1].long()
    sequences = lengths.numel()
    sequence = torch.searchsorted(cu_seqlens_padded[1:].long(), positions, right=True)
    clamped = sequence.clamp(max=sequences - 1)
    real = (sequence < sequences) & (positions - starts[clamped] < lengths[clamped])
    allowed = (sequence[:, None] == sequence[None, :]) & (positions[None, :] <= positions[:, None])
    allowed &= real[:, None] & real[None, :]
    allowed |= torch.eye(total, dtype=torch.bool, device=positions.device) & ~real[:, None]
    return allowed, real


def packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor,
    real: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """`[t, heads, dim]` attention of the pack, the key/value heads shared by query groups."""
    groups = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    softmax_dtype = torch.promote_types(query.dtype, torch.float32)
    scores = torch.einsum("qhd,khd->hqk", query, key).to(softmax_dtype) * scale
    scores = scores.masked_fill(~allowed, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(query.dtype)
    output = torch.einsum("hqk,khd->qhd", probabilities, value)
    return output * real[:, None, None].to(output.dtype)


def install_packed_attention(chunks: Sequence[Any]) -> None:
    """Run every Gemma 4 global layer's core attention on `packed_attention`, at the softmax scale
    of the Transformer Engine module it replaces. Only that module's forward changes, so the state
    dict and checkpoints stay the stock model's."""
    for chunk in chunks:
        for module in chunk.modules():
            if type(module).__name__ != "Gemma4SelfAttention" or _is_local_attn_layer(
                module.layer_number, module.config.interleaved_attn_pattern
            ):
                continue
            core_attention = module.core_attention
            scale = float(core_attention.unfused_attention.softmax_scale)

            def forward(query, key, value, *args, packed_seq_params, _scale=scale, **kwargs):
                allowed, real = packed_masks(
                    packed_seq_params.cu_seqlens_q,
                    packed_seq_params.cu_seqlens_q_padded,
                    query.shape[0],
                )
                output = packed_attention(query, key, value, allowed, real, _scale)
                return output.reshape(output.shape[0], -1)

            core_attention.forward = forward
