"""Both models over NCCL's all-gather and reduce-scatter, for runs where NCCL has no InfiniBand (TCP
sockets) and DeepEP, which needs InfiniBand GPU-initiated networking, cannot start.

The base model takes vLLM's all-gather + reduce-scatter MoE transport. EchoMoE's layers get a
GatherBuffer in place of DeepEP v2's buffer: its dispatch all-gathers every rank's rows (or codes)
with their routes and keeps those routed to this rank's experts, packed first in gathered order with
DeepEP's receive layout (source `origin * capacity + row`, routes as local expert ids, -1 elsewhere).
Its combine puts each kept row's code back at its gathered position and reduce-scatters. Shapes are
fixed, so decode steps still capture into CUDA graphs. It also runs Qwen3-30B-A3B's reference layers'
all-gather and reduce-scatter. A larger model's reference layer takes its experts' rows from the
cache's all-gather (`dispatch_gathered`).

With FP8 activations (Qwen3-30B-A3B) the base model's all-gather moves FP8 rows, cast per token as
DeepEP casts and cast back for its BF16 experts, and every combine sums in BF16 within a node and in
FP8 between nodes (kernels.reduce_scatter_fp8_nodes), as DeepEP's FP8 combine does.
"""

import itertools
from types import SimpleNamespace

import torch
import torch.distributed as dist
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

from .kernels import fp8_cast, fp8_cast_back, reduce_scatter_fp8_nodes


def pack(slot, values, rows, fill=0):
    """`values[i]` at row `slot[i]` of `rows` rows (`fill` elsewhere). Slot `rows` is dropped."""
    out = torch.full((rows + 1, *values.shape[1:]), fill, device=values.device, dtype=values.dtype)
    out.index_copy_(0, slot, values)
    return out[:rows]


class GatherBuffer:
    """DeepEP v2's dispatch (do_expand=False) and combine over one all-gather and one
    reduce-scatter of `group`. `fp8_combine`: the combine as kernels.reduce_scatter_fp8_nodes."""

    def __init__(self, group, fp8_combine):
        self.group, self.fp8_combine = group, fp8_combine
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)

    def get_theoretical_num_sms(self, *args, **kwargs):
        return 0

    def receive(self, parts, ids, weights, num_experts, capacity):
        """The rows of the gathered `parts` (each `[world * capacity, ...]`) routed to this rank's
        experts, their routes and weights, and the handle the combine reads."""
        rows, local = self.world * capacity, num_experts // self.world
        mine = (ids >= self.rank * local) & (ids < (self.rank + 1) * local)
        hit = mine.any(dim=1)
        slot = torch.where(hit, torch.cumsum(hit, 0) - 1, rows)
        local_ids = torch.where(mine, ids - self.rank * local, -1)
        total = hit.sum(dtype=torch.int64).reshape(1)
        received = tuple(
            pack(slot, part.view(torch.uint8), rows).view(part.dtype) for part in parts
        )
        source = pack(slot, torch.arange(rows, device=ids.device)[:, None], rows, fill=-1)
        handle = SimpleNamespace(
            recv_src_metadata=source,
            psum_num_recv_tokens_per_scaleup_rank=total,
            capacity=capacity,
        )
        return received, pack(slot, local_ids, rows, fill=-1), pack(slot, weights, rows), handle

    def dispatch_gathered(self, parts, ids, weights, num_experts, capacity):
        """The dispatch of rows every rank already holds (`parts`, a larger model's reference rows
        from the cache's all-gather): one all-gather of the routes, padded to `capacity` rows that
        route nowhere."""
        top_k = ids.shape[1]
        send = torch.zeros((capacity, 12 * top_k), device=ids.device, dtype=torch.uint8)
        send[:, : 8 * top_k].view(torch.int64).fill_(-1)
        send[: ids.shape[0], : 8 * top_k].view(torch.int64).copy_(ids)
        send[: ids.shape[0], 8 * top_k :].view(torch.float32).copy_(weights)
        gathered = torch.empty(
            (self.world * capacity, 12 * top_k), device=ids.device, dtype=torch.uint8
        )
        dist.all_gather_into_tensor(gathered, send, group=self.group)
        received, ids, weights, handle = self.receive(
            parts,
            gathered[:, : 8 * top_k].view(torch.int64),
            gathered[:, 8 * top_k :].view(torch.float32),
            num_experts,
            capacity,
        )
        handle.sent = capacity
        return received, ids, weights, handle

    def dispatch(self, x, topk_idx, topk_weights, num_experts, num_max_tokens_per_rank, **kwargs):
        """One all-gather of each row's `[x | ids | weights]` bytes. The rows are BF16 or DeepEP's
        FP8 pair (values, scales), received as the same."""
        capacity, sent_rows = num_max_tokens_per_rank, topk_idx.shape[0]
        parts = x if isinstance(x, tuple) else (x,)
        raw = [part.contiguous().view(torch.uint8) for part in parts]
        row_bytes, top_k = sum(part_bytes.shape[1] for part_bytes in raw), topk_idx.shape[1]
        send = torch.zeros(
            (capacity, row_bytes + 12 * top_k), device=topk_idx.device, dtype=torch.uint8
        )
        send[:sent_rows, :row_bytes].copy_(torch.cat(raw, dim=1))
        send[:, row_bytes : row_bytes + 8 * top_k].view(torch.int64).fill_(-1)
        send[:sent_rows, row_bytes : row_bytes + 8 * top_k].view(torch.int64).copy_(topk_idx)
        send[:sent_rows, row_bytes + 8 * top_k :].view(torch.float32).copy_(topk_weights)
        gathered = torch.empty(
            (self.world * capacity, send.shape[1]), device=send.device, dtype=torch.uint8
        )
        dist.all_gather_into_tensor(gathered, send, group=self.group)
        ends = itertools.accumulate(part_bytes.shape[1] for part_bytes in raw)
        columns = [
            gathered[:, end - part_bytes.shape[1] : end].view(part.dtype)
            for part, part_bytes, end in zip(parts, raw, ends)
        ]
        received, ids, weights, handle = self.receive(
            columns,
            gathered[:, row_bytes : row_bytes + 8 * top_k].view(torch.int64),
            gathered[:, row_bytes + 8 * top_k :].view(torch.float32),
            num_experts,
            capacity,
        )
        handle.sent = sent_rows
        return received if isinstance(x, tuple) else received[0], ids, weights, handle, None

    def combine(self, code, handle, topk_weights=None, **kwargs):
        """Each received row's code summed at its source. The first `handle.sent` rows."""
        rows = self.world * handle.capacity
        total = handle.psum_num_recv_tokens_per_scaleup_rank
        live = torch.arange(code.shape[0], device=code.device) < total
        position = torch.where(live, handle.recv_src_metadata[:, 0], rows)
        scattered = torch.zeros((rows + 1, code.shape[1]), device=code.device, dtype=code.dtype)
        scattered.index_copy_(0, position, code)
        if self.fp8_combine:
            combined = reduce_scatter_fp8_nodes(scattered[:rows], self.group, handle.capacity)
        else:
            combined = torch.empty(
                (handle.capacity, code.shape[1]), device=code.device, dtype=code.dtype
            )
            dist.reduce_scatter_tensor(
                combined, scattered[:rows], op=dist.ReduceOp.SUM, group=self.group
            )
        return combined[: handle.sent], None, None

    def all_gather(self, out, inp, group, capacity):
        dist.all_gather_into_tensor(out, inp, group=group)

    def reduce_scatter(self, code, group, capacity, fp8):
        if fp8:
            return reduce_scatter_fp8_nodes(code, group, capacity)
        out = torch.empty(capacity, code.shape[1], device=code.device, dtype=code.dtype)
        dist.reduce_scatter_tensor(out, code, op=dist.ReduceOp.SUM, group=group)
        return out


def install_tcp(fp8) -> None:
    """EchoMoE's layers on a GatherBuffer. With `fp8`, the base model's all-gather in FP8 and
    every combine as kernels.reduce_scatter_fp8_nodes. Before the engine is built."""
    AgRsAll2AllManager.get_handle = lambda self, kwargs: GatherBuffer(
        get_ep_group().device_group, fp8
    )
    # the CLI's cleanup reads the buffers the all2all manager cached
    AgRsAll2AllManager.handle_cache = SimpleNamespace(_cache={})
    if not fp8:
        return
    all_gather = AgRsAll2AllManager.dispatch

    def dispatch(
        self, hidden_states, topk_weights, topk_ids, is_sequence_parallel=False, extra_tensors=None
    ):
        assert extra_tensors is None
        payload, scale = fp8_cast(hidden_states)
        payload, topk_weights, topk_ids, (scale,) = all_gather(
            self, payload.view(torch.uint8), topk_weights, topk_ids, is_sequence_parallel, [scale]
        )
        count = torch.full((1,), payload.shape[0], device=payload.device, dtype=torch.int32)
        rows = fp8_cast_back(payload.view(torch.float8_e4m3fn), scale, count)
        return rows, topk_weights, topk_ids

    def combine(self, hidden_states, is_sequence_parallel=False):
        group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(hidden_states.shape[0] // group.world_size, group)
        if len(set(sizes)) == 1:
            return reduce_scatter_fp8_nodes(hidden_states, group.device_group, sizes[0])
        # unequal data-parallel chunks (eager prefill): each rank's rows padded to the largest
        size = max(sizes)
        padded = hidden_states.new_zeros((len(sizes) * size, hidden_states.shape[1]))
        for rank, (start, length) in enumerate(zip(itertools.accumulate([0, *sizes[:-1]]), sizes)):
            padded[rank * size : rank * size + length] = hidden_states[start : start + length]
        out = reduce_scatter_fp8_nodes(padded, group.device_group, size)
        return out[: sizes[group.rank_in_group]]

    AgRsAll2AllManager.dispatch = dispatch
    AgRsAll2AllManager.combine = combine
