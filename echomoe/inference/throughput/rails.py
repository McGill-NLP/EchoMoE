"""MoE dispatch and combine at expert parallelism over NCCL GIN rails, standing in for DeepEP v2's
ElasticBuffer in decode mode (do_expand=False, no CPU sync, CUDA-graph safe): RailBuffer, the
two-hop kernel (gin/rail_moe.cu), and StreamBuffer, the streaming kernel for prefill sizes
(gin/rail_stream.cu). Each token crosses InfiniBand once per node, to the rail peer there (the same
local rank), and moves on over NVLink. RailBuffer also runs an all-gather and a reduce-scatter over
the whole group, for EchoMoE's reference layers.

dispatch(x [n, w] bf16, topk_idx [n, k] int64 global expert ids or -1, topk_weights [n, k] fp32,
         num_max_tokens_per_rank T: n <= T <= max_tokens_per_rank, the same on every rank, num_sms)
    -> recv_x [R*T, w], recv_topk_idx [R*T, k] int64 (this rank's experts numbered from 0, other
       routes -1), recv_topk_weights [R*T, k] (as sent), handle
  Each token goes once to each rank holding one of its experts. Received rows are packed by source
  rank, then source token: the first handle.psum_num_recv_tokens_per_scaleup_rank[-1] rows are
  valid (inclusive prefix sum per source rank, int32 [R], as DeepEP's), the rest are not written.
  handle.recv_src[i] = source rank * T + source token (DeepEP's recv_src_metadata[:, 0]). x may
  also be DeepEP's FP8 pair (values [n, h] float8_e4m3fn, scales [n, ceil(h / 128)] float32),
  moved as bytes, and recv_x is then the same pair.
combine(y [>= received, w'] bf16 rows aligned with recv_x, handle, num_sms) -> [n, w'] bf16: each
  token's rows summed as DeepEP's combine sums them with topk_weights=None (its combine_reduce,
  combine_utils.cuh): each node's rows to one partial there, the partials to the output, both
  levels by the same rule (up to two slots add in native bf16, three or more in fp32 from 0 and
  round once), slots in DeepEP's order. Tokens routed nowhere are 0. With fp8_combine (w' a
  multiple of 128), deepep_fp8c's bits (fp8_scaleout_reduce): other nodes' partials cross
  InfiniBand in FP8, cast as kernels.fp8_cast casts but with the scale amax / 448 rounded, and the
  partials add in fp32 from 0 in slot order (the owner's in bf16), rounded once.
Dispatch and combine alternate on a buffer (one dispatch in flight), on every rank with the same T
and num_sms (CTAs per call). The widths w and w' may differ, multiples of 8 up to width_max.

RailBuffer.all_gather(out [R*T, B] uint8, inp [T, B] uint8): as dist.all_gather_into_tensor, B a
  multiple of 16 up to gather_bytes.
RailBuffer.reduce_scatter(out [T, c] bf16, inp [R*T, c] bf16): as dist.reduce_scatter_tensor (sum)
  in fp32, the local ranks' rows summed within each node in rank order, then the nodes in node
  order, c a multiple of 8 up to reduce_width. With fp8 (c a multiple of 128), as
  kernels.reduce_scatter_fp8_nodes: each node's sum rounded to bf16, other nodes' through FP8.
"""

import functools
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

SOURCES = Path(__file__).resolve().parent / "gin"


@functools.cache
def extension(name):
    """Built against the libnccl torch loaded and that library's headers, local rank 0 first, once
    per host. A preloaded NCCL build (the capped runs') wins over torch's, which is mapped too."""
    maps = Path("/proc/self/maps").read_text().splitlines()
    paths = {line.split()[-1] for line in maps if "libnccl" in line}
    preloaded = {
        os.path.realpath(path) for path in os.environ.get("LD_PRELOAD", "").split(":") if path
    }
    paths = (paths & preloaded) or paths
    assert len(paths) == 1, paths
    lib = Path(paths.pop())
    host = socket.gethostname() + ("-" + lib.parent.parent.name if preloaded else "")
    build = Path(os.environ["ECHOMOE_REPO"]) / "outputs" / "gin-jit" / host / name
    build.mkdir(parents=True, exist_ok=True)
    options = dict(
        name=name,
        sources=[str(SOURCES / f"{name}.cpp"), str(SOURCES / f"{name}.cu")],
        extra_include_paths=[str(lib.parent.parent / "include")],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[f"-L{lib.parent}", f"-l:{lib.name}", f"-Wl,-rpath,{lib.parent}"],
        build_directory=str(build),
    )
    built = load(**options) if int(os.environ["LOCAL_RANK"]) == 0 else None
    dist.barrier()
    return built or load(**options)


def communicator(group):
    # ProcessGroupNCCL creates its communicator lazily
    dist.all_reduce(torch.zeros(1, device="cuda"), group=group)
    return group._get_backend(torch.device("cuda"))._comm_ptr()


@dataclass
class Handle:
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor  # [R] int32, inclusive
    recv_src: torch.Tensor  # [R*T] int32
    comb_rows: torch.Tensor  # [T, k] int32: each token's rows in the combine receive area
    node_rows: torch.Tensor  # [8] int32: rows each node returns in the combine
    num_tokens: int
    num_max_tokens_per_rank: int
    num_recv_tokens_per_expert_list: list = field(default_factory=list)

    @property
    def recv_src_metadata(self):
        return self.recv_src.view(-1, 1)


def dispatch_outputs(ranks, topk, x, topk_idx, topk_weights, tokens):
    """The receive tensors and the handle of a dispatch of `x` (bf16 rows or the FP8 pair)."""
    rows = ranks * tokens
    x, scales = x if isinstance(x, tuple) else (x, x.new_empty(0, dtype=torch.float32))
    recv_x = x.new_empty(rows, x.shape[1])
    recv_scales = scales.new_empty(rows, scales.shape[1]) if scales.dim() == 2 else scales
    handle = Handle(
        x.new_empty(ranks, dtype=torch.int32),
        x.new_empty(rows, dtype=torch.int32),
        x.new_empty(tokens, topk, dtype=torch.int32),
        x.new_empty(8, dtype=torch.int32),
        x.shape[0],
        tokens,
    )
    return x, scales, recv_x, recv_scales, topk_idx.new_empty(rows, topk), handle


class RailBuffer:
    def __init__(
        self,
        group,
        max_tokens_per_rank,
        width_max,
        topk,
        num_experts,
        *,
        num_contexts,
        gather_bytes,
        reduce_width,
        fp8_combine,
    ):
        self.ext = extension("rail_moe")
        self.ranks, self.topk = dist.get_world_size(group), topk
        self.rail = self.ext.create(
            communicator(group),
            max_tokens_per_rank,
            width_max,
            topk,
            num_experts,
            num_contexts,
            gather_bytes,
            reduce_width,
            fp8_combine,
        )

    def dispatch(self, x, topk_idx, topk_weights, tokens, num_sms):
        x, scales, recv_x, recv_scales, recv_idx, handle = dispatch_outputs(
            self.ranks, self.topk, x, topk_idx, topk_weights, tokens
        )
        recv_weights = topk_weights.new_empty(recv_idx.shape)
        self.ext.dispatch(
            self.rail,
            x.contiguous(),
            scales.contiguous(),
            topk_idx.contiguous(),
            topk_weights.contiguous(),
            tokens,
            num_sms,
            recv_x,
            recv_scales,
            recv_idx,
            recv_weights,
            handle.recv_src,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.comb_rows,
            handle.node_rows,
        )
        received = (recv_x, recv_scales) if scales.dim() == 2 else recv_x
        return received, recv_idx, recv_weights, handle

    def combine(self, outputs, handle, num_sms):
        out = outputs.new_empty(handle.num_tokens, outputs.shape[1])
        self.ext.combine(
            self.rail,
            outputs.contiguous(),
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.comb_rows,
            handle.node_rows,
            handle.num_tokens,
            handle.num_max_tokens_per_rank,
            num_sms,
            out,
        )
        return out

    def all_gather(self, out, inp, num_sms):
        self.ext.all_gather(self.rail, inp.contiguous(), out, num_sms)

    def reduce_scatter(self, out, inp, num_sms, fp8):
        self.ext.reduce_scatter(self.rail, inp.contiguous(), out, num_sms, fp8)

    def destroy(self):
        """Collective."""
        self.ext.destroy(self.rail)


class StreamBuffer:
    """Up to 4096 tokens per rank. Remote transfers go as puts of at least chunk_bytes, spread over
    num_contexts GIN contexts by destination node. A call takes at least 2 CTAs (one relays)."""

    def __init__(
        self,
        group,
        max_tokens_per_rank,
        width_max,
        topk,
        num_experts,
        *,
        num_contexts,
        chunk_bytes,
        fp8_combine,
    ):
        self.ext = extension("rail_stream")
        self.ranks, self.topk = dist.get_world_size(group), topk
        self.chunk_bytes, self.fp8_combine = chunk_bytes, fp8_combine
        self.stream = self.ext.create(
            communicator(group), max_tokens_per_rank, width_max, topk, num_experts, num_contexts
        )

    def dispatch(self, x, topk_idx, topk_weights, tokens, num_sms):
        x, scales, recv_x, recv_scales, recv_idx, handle = dispatch_outputs(
            self.ranks, self.topk, x, topk_idx, topk_weights, tokens
        )
        recv_weights = topk_weights.new_empty(recv_idx.shape)
        self.ext.dispatch(
            self.stream,
            x.contiguous(),
            scales.contiguous(),
            topk_idx.contiguous(),
            topk_weights.contiguous(),
            tokens,
            num_sms,
            self.chunk_bytes,
            recv_x,
            recv_scales,
            recv_idx,
            recv_weights,
            handle.recv_src,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.comb_rows,
            handle.node_rows,
        )
        received = (recv_x, recv_scales) if scales.dim() == 2 else recv_x
        return received, recv_idx, recv_weights, handle

    def combine(self, outputs, handle, num_sms):
        out = outputs.new_empty(handle.num_tokens, outputs.shape[1])
        self.ext.combine(
            self.stream,
            outputs.contiguous(),
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.comb_rows,
            handle.node_rows,
            handle.num_tokens,
            handle.num_max_tokens_per_rank,
            num_sms,
            self.chunk_bytes,
            self.fp8_combine,
            out,
        )
        return out

    def destroy(self):
        """Collective."""
        self.ext.destroy(self.stream)
