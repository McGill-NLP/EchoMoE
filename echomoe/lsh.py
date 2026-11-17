"""LSH-MoE (Nie et al., NeurIPS 2024), the baseline the report compares EchoMoE with.

In every MoE layer each source rank hashes the tokens it sends to an expert with cross-polytope
LSH: `num_hashes` random rotations, and under each the nearest of the `2 * projection_dim` signed
axes. Tokens that share a bucket form a cluster and only its centroid `c` (the members' mean) is
sent. Each member gets `E(c) + alpha_e (x_i - c)`, with one alpha per layer and expert learned from
0. The published compensation (alpha = 1) breaks the pretrained model: at the first MoE layers of
Qwen3-30B-A3B the offsets `x_i - c` are several times larger than `E(c)`. The offsets need no
communication, since each source holds its tokens and the centroids it sent.

A cluster depends only on its members' inputs, which the receiving rank already holds. Training
therefore clusters the received rows by (source rank, local expert, bucket), which gives the
numbers and gradients of hashing at the source with DeepEP's dispatch unchanged. It adds each
scaled offset on the expert's rank before the combine, which equals adding it at the source since
the combine is a weighted sum. Served without
expert parallelism, a rank is the only source of its batch. One set of rotations, drawn from the
seed, serves every rank and layer.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

# The export's file of the learned alpha, `[layers, experts]` in float32.
ALPHA_FILE = "lsh_alpha.safetensors"
# The byte price is DeepEP's at EP32 on 4 nodes, as `echomoe.communication_cost` prices EchoMoE:
# expert e sits on node e // 32 and a token's own node is uniform, so (N - 1) / N of its sends
# leave the node.
NODE_COUNT = 4


@dataclass(frozen=True)
class LshConfig:
    """`num_hashes` cross-polytope hashes, each onto `projection_dim` axes, rotated by `seed`."""

    projection_dim: int
    num_hashes: int
    seed: int

    @property
    def bucket_count(self) -> int:
        return (2 * self.projection_dim) ** self.num_hashes


def rotations(config: LshConfig, hidden_size: int, layer: int = 0) -> torch.Tensor:
    """The `[num_hashes, hidden_size, projection_dim]` random projections of one layer, on the
    host. Training and serving use layer 0's at every layer. The Table 16 analysis draws each
    layer's own."""
    projections = []
    for index in range(config.num_hashes):
        key = f"lsh-moe:{config.seed}:{layer}:{config.projection_dim}:{index}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        projections.append(
            torch.randn(
                (hidden_size, config.projection_dim),
                generator=torch.Generator(device="cpu").manual_seed(seed),
                dtype=torch.float32,
                device="cpu",
            )
        )
    return torch.stack(projections)


@contextmanager
def _exact_float32():
    # The trainer's `high` matmul precision would project in TF32 and move rows near a bucket
    # boundary into another bucket.
    allowed = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allowed


@torch.no_grad()
def buckets(rows: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Each row's bucket: under every rotation the nearest signed axis, the codes read as digits
    base `2 * projection_dim`. Rows are normalized first, so a bucket ignores scale."""
    with _exact_float32():
        unit = functional.normalize(rows.float(), dim=-1)
        projected = torch.einsum("nd,hdk->hnk", unit, rotations)
    codes = torch.cat((projected, -projected), dim=-1).argmax(dim=-1)
    base = 2 * rotations.shape[-1]
    digit_weights = base ** torch.arange(rotations.shape[0], device=rows.device, dtype=torch.int64)
    return (codes * digit_weights[:, None]).sum(dim=0)


def cluster_ids(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each key's cluster, numbered in key order, and the number of clusters on the device. A sort
    stands in for `torch.unique`, which waits on the device."""
    sorted_keys, order = torch.sort(keys)
    cluster_starts = torch.ones_like(sorted_keys, dtype=torch.bool)
    cluster_starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    sorted_ids = torch.cumsum(cluster_starts, dim=0) - 1
    return torch.empty_like(sorted_ids).scatter_(0, order, sorted_ids), cluster_starts.sum()


def centroids(rows: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Each row's cluster mean `c`, summed in at least float32 and returned in the rows' dtype.
    Every member gets 1/n of its centroid's gradient."""
    accumulate_dtype = torch.promote_types(rows.dtype, torch.float32)
    sums = rows.new_zeros(rows.shape, dtype=accumulate_dtype).index_add(
        0, ids, rows.to(accumulate_dtype)
    )
    sizes = torch.zeros(rows.shape[0], dtype=torch.int64, device=rows.device)
    sizes.index_add_(0, ids, torch.ones_like(ids))
    return (sums / sizes.clamp_min(1).unsqueeze(1)).to(rows.dtype).index_select(0, ids)


def received_centroids(
    rows: torch.Tensor,
    received: torch.Tensor,
    routing_map: torch.Tensor,
    rank_prefix_matrix: torch.Tensor,
    ep_rank: int,
    rotations: torch.Tensor,
    bucket_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The expert input `rows` of an expert-parallel rank, each replaced by its (source rank, local
    expert, bucket) centroid, each row's local expert, and the number of clusters. `received` holds
    the rows as DeepEP's
    intranode dispatch delivered them and `routing_map` their local experts. Megatron's permute
    lays `rows` out by expert, then by token. `rank_prefix_matrix[i, r]` counts the rows rank r
    receives from ranks 0..i, and each source's rows arrive as one block, in rank order."""
    expert, token = torch.nonzero_static(routing_map.T, size=rows.shape[0]).unbind(dim=1)
    prefix = rank_prefix_matrix[:, ep_rank].to(torch.int64)
    per_source = torch.diff(prefix, prepend=prefix.new_zeros(1))
    source = torch.repeat_interleave(
        torch.arange(per_source.shape[0], device=per_source.device),
        per_source,
        output_size=received.shape[0],
    )
    bucket = buckets(received, rotations)
    keys = (source[token] * routing_map.shape[1] + expert) * bucket_count + bucket[token]
    ids, clusters = cluster_ids(keys)
    return centroids(rows, ids), expert, clusters


def served_centroids(
    rows: torch.Tensor,
    topk_ids: torch.Tensor,
    rotations: torch.Tensor,
    bucket_count: int,
    real_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One rank's batch served without expert parallelism, the rank being its only source: every
    (token, slot) pair, token-major as vLLM's fused MoE numbers them, replaced by the centroid of
    the pairs that share its (expert, bucket), and the number of clusters of real pairs. A pair of
    a padding row, past `real_tokens`, keeps a negative key of its own, so it neither moves nor
    joins a cluster."""
    tokens, top_k = topk_ids.shape
    keys = topk_ids.to(torch.int64) * bucket_count + buckets(rows, rotations).unsqueeze(1)
    pair_index = torch.arange(tokens * top_k, device=keys.device).view(tokens, top_k)
    keys = torch.where(pair_index < real_tokens * top_k, keys, -1 - pair_index).view(-1)
    ids, clusters = cluster_ids(keys)
    padding_pairs = (tokens - real_tokens) * top_k
    return centroids(rows.repeat_interleave(top_k, dim=0), ids), clusters - padding_pairs


def residual_weights(
    weights: torch.Tensor, alpha: torch.Tensor, expert: torch.Tensor
) -> torch.Tensor:
    """Each pair's residual weight `w alpha_e`, in the routing weights' float32, so a bfloat16
    alpha's gradient sums its pairs there."""
    return weights * alpha.to(weights.dtype)[expert]


def compensate(output: torch.Tensor, offsets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Training: each row's weighted expert output `w E(c)` plus `w alpha_e (x - c)`, with the
    offsets `x - c` and the residual weights per row, before the combine sums a token's rows."""
    return output + (weights.unsqueeze(-1) * offsets).to(output.dtype)


def compensate_pairs(
    output: torch.Tensor, rows: torch.Tensor, pair_centroids: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Serving: each token's combined output plus `sum_k w_tk alpha_e (x_t - c_tk)` over its
    token-major pairs and their `[tokens, top_k]` residual weights, summed in float32."""
    tokens, top_k = weights.shape
    accumulate_dtype = torch.promote_types(output.dtype, torch.float32)
    offsets = (rows.repeat_interleave(top_k, dim=0) - pair_centroids).view(tokens, top_k, -1)
    added = torch.einsum("tk,tkh->th", weights.to(accumulate_dtype), offsets.to(accumulate_dtype))
    return (output.to(accumulate_dtype) + added).to(output.dtype)


def node_rows(routed: torch.Tensor) -> torch.Tensor:
    """DeepEP's rows for a `[tokens, experts]` routing mask: one per token and node that holds one
    of its experts."""
    tokens, experts = routed.shape
    return routed.view(tokens, NODE_COUNT, experts // NODE_COUNT).any(dim=-1).sum()


def rdma_volume(counts: list[list[float]]) -> dict[str, float]:
    """Remote rows per token per layer from each layer's tokens, DeepEP rows and LSH-MoE clusters:
    DeepEP's, LSH-MoE's centroids, and the RDMA reduction, DeepEP's bytes over LSH-MoE's. Both send
    full-width rows, and the few bytes of cluster index per token are not priced."""
    deepep_rows = [rows * (NODE_COUNT - 1) / NODE_COUNT / tokens for tokens, rows, _ in counts]
    lsh_rows = [
        clusters * (NODE_COUNT - 1) / NODE_COUNT / tokens for tokens, _, clusters in counts
    ]
    return {
        "deepep_remote_rows_per_token": sum(deepep_rows) / len(deepep_rows),
        "remote_centroids_per_token": sum(lsh_rows) / len(lsh_rows),
        "reduction": sum(deepep_rows) / sum(lsh_rows),
    }
