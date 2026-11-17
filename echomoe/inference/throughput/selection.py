"""Each DeepEP v2 dispatch and combine, and EchoMoE's reference collectives, on the backend measured
fastest for the call: DeepEP at a chosen SM count, the two-hop kernel or the streaming kernel
(rails.py), or NCCL. The tables are per-call times measured on the 4-node cluster, by token limit
and row widths (configs/inference/throughput/costs-*.csv).
"""

import csv
import functools
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.all2all import DeepEPV2All2AllManager

from . import portable
from .checkpoint import FoldedExport, PortableExport
from .kernels import reduce_scatter_fp8_nodes, row_layout
from .rails import Handle, RailBuffer, StreamBuffer

# what vLLM's prepare-finalize waits on after a dispatch or combine of ours: nothing
NO_EVENT = SimpleNamespace(event=None, current_stream_wait=lambda: None)


class Costs:
    """A table's per-call times: for a call's token limit and row widths (rounded up to the measured
    grid, DeepEP's own choice beyond it), the backend with the lowest time. A dispatch and combine
    pair gets ("rail", CTAs), ("stream", CTAs) or DeepEP's SM count, priced by the rows of its
    precision: "bf16", "fp8" (FP8 dispatch and FP8 combine) or "fp8-dispatch" (FP8 dispatch, BF16
    combine). A reference collective gets the two-hop kernel's CTAs or 0 (NCCL). The streaming
    kernel's rows share one chunk size and context count."""

    def __init__(self, path):
        self.pairs, self.collectives, self.stream = {}, {}, None
        with open(path) as table:
            rows = list(csv.DictReader(table))
        for row in rows:
            tokens, width, strategy = int(row["tokens"]), int(row["width"]), row["strategy"]
            kernel = strategy.removesuffix("-fp8")
            if kernel in ("rail", "stream", "deepep-hybrid"):
                sms = int(row["sms"])
                backend = sms if kernel == "deepep-hybrid" else (kernel, sms)
                times = float(row["dispatch_us"]), float(row["combine_us"])
                self.pairs[backend, tokens, width, row["precision"]] = times
                if kernel == "stream":
                    # the kernels always combine as DeepEP's (rails.py), as these rows were timed
                    assert row["reduce"] == "1", row
                    self.stream = SimpleNamespace(
                        chunk=int(row["chunk"]), contexts=int(row["contexts"])
                    )
            else:
                self.collectives[strategy, int(row["sms"]), tokens, width] = float(
                    row["dispatch_us"]
                )
        self.tokens = sorted({tokens for _, tokens, _, _ in self.pairs})
        self.widths = sorted({row_width for _, _, row_width, _ in self.pairs})
        custom = [(key[0], tokens) for key, tokens, _, _ in self.pairs if isinstance(key, tuple)]
        self.rail_tokens = max(tokens for name, tokens in custom if name == "rail")
        self.stream.capacity = max(tokens for name, tokens in custom if name == "stream")
        self.pair = functools.cache(self._pair)
        self.gather = functools.cache(self._gather)
        self.scatter = functools.cache(self._scatter)

    @staticmethod
    def up(grid, value):
        return next((point for point in grid if point >= value), grid[-1])

    def _pair(self, tokens, dispatch, combine, precision):
        if tokens > self.tokens[-1] or max(dispatch, combine) > self.widths[-1]:
            return 0
        tokens, dispatch, combine = (
            self.up(self.tokens, tokens),
            self.up(self.widths, dispatch),
            self.up(self.widths, combine),
        )
        # In table order, so that every rank breaks a tie the same way: ranks that iterated a set
        # (its order follows the per-process string hash) ran different kernels for one call and hung.
        backends = dict.fromkeys(
            backend
            for backend, row_tokens, _, row_precision in self.pairs
            if (row_tokens, row_precision) == (tokens, precision)
        )
        times = {
            backend: self.pairs[backend, tokens, dispatch, precision][0]
            + self.pairs[backend, tokens, combine, precision][1]
            for backend in backends
        }
        return min(times, key=times.get) if times else 0

    def fastest(self, name, tokens, width):
        if tokens > self.rail_tokens:
            return 0
        tokens = self.up(self.tokens, tokens)
        times = {
            ctas: time
            for (strategy, ctas, row_tokens, row_width), time in self.collectives.items()
            if (row_tokens, row_width) == (tokens, width)
            and strategy in (f"rail-{name}", f"nccl-{name}")
        }
        return min(times, key=times.get) if times else 0

    def _gather(self, tokens):
        width = next(
            row_width
            for strategy, _, _, row_width in self.collectives
            if strategy == "rail-allgather"
        )
        return self.fastest("allgather", tokens, width)

    def _scatter(self, tokens, width, fp8):
        name = "reducescatter-fp8" if fp8 else "reducescatter"
        return self.fastest(name, tokens, self.up(self.widths, width))


class RailSwitch:
    """A DeepEP v2 buffer whose BF16 and FP8 (DeepEP's (values, scales) pair) dispatches go where
    `costs` puts them for their token limit (the same on every rank), dispatch width and combine
    `width`: (name, CTAs) on kernels[name], a number on DeepEP with that many SMs, 0 on DeepEP as
    asked. Each combine goes where its dispatch went. Everything else is DeepEP's. EchoMoE's buffer
    also runs its reference layers' all-gather and reduce-scatter: on the two-hop kernel at the CTAs
    `costs` picks, on NCCL where it picks 0. `fp8_combine`: install_selected's."""

    def __init__(self, deepep, kernels, width, costs, fp8_combine):
        self.deepep = deepep
        self.kernels = kernels
        self.width = width
        self.costs = costs
        self.fp8_combine = fp8_combine
        self.inflight = None  # the dispatch in flight's buffer and CTAs, for its combine

    def __getattr__(self, name):
        return getattr(self.deepep, name)

    def dispatch(
        self,
        x,
        *args,
        topk_idx=None,
        topk_weights=None,
        num_max_tokens_per_rank=None,
        do_expand=False,
        **kwargs,
    ):
        tokens, fp8 = num_max_tokens_per_rank, isinstance(x, tuple)
        rows = x[0] if fp8 else x
        where = 0
        if isinstance(rows, torch.Tensor) and not do_expand and tokens is not None:
            precision = ("fp8" if self.fp8_combine else "fp8-dispatch") if fp8 else "bf16"
            where = self.costs.pair(tokens, rows.shape[1], self.width, precision)
        if isinstance(where, tuple):
            self.inflight = self.kernels[where[0]], where[1]
            return (
                *self.inflight[0].dispatch(x, topk_idx, topk_weights, tokens, where[1]),
                NO_EVENT,
            )
        if where:
            kwargs["num_sms"] = where
        return self.deepep.dispatch(
            x,
            *args,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_max_tokens_per_rank=tokens,
            do_expand=do_expand,
            **kwargs,
        )

    def combine(self, x, handle, *args, **kwargs):
        if isinstance(handle, Handle):
            buffer, ctas = self.inflight
            return buffer.combine(x, handle, ctas), None, NO_EVENT
        return self.deepep.combine(x, handle, *args, **kwargs)

    def all_gather(self, out, inp, group, capacity):
        ctas = self.costs.gather(capacity)
        if ctas:
            self.kernels["rail"].all_gather(out, inp, ctas)
        else:
            dist.all_gather_into_tensor(out, inp, group=group)

    def reduce_scatter(self, code, group, capacity, fp8):
        """The reference layer's combine of `code` ([ranks * capacity, r_c]): FP8 between nodes with
        `fp8`, as kernels.reduce_scatter_fp8_nodes."""
        ctas = self.costs.scatter(capacity, code.shape[1], fp8)
        if not ctas and fp8:
            return reduce_scatter_fp8_nodes(code, group, capacity)
        out = torch.empty(capacity, code.shape[1], device=code.device, dtype=code.dtype)
        if ctas:
            self.kernels["rail"].reduce_scatter(out, code, ctas, fp8)
        else:
            dist.reduce_scatter_tensor(out, code, op=dist.ReduceOp.SUM, group=group)
        return out

    def destroy(self):
        """Collective: the kernels' buffers only. vLLM destroys DeepEP's."""
        for buffer in self.kernels.values():
            buffer.destroy()


@contextmanager
def gin_rate(rate):
    """NCCL_GIN_STATIC_RATE at `rate` (an ibv_rate value, None keeps it) while a buffer creates its
    GIN contexts: the capped runs' NCCL build reads it per context, so DeepEP's keep the process's
    rate."""
    if rate is None:
        yield
        return
    previous = os.environ["NCCL_GIN_STATIC_RATE"]
    os.environ["NCCL_GIN_STATIC_RATE"] = str(rate)
    try:
        yield
    finally:
        os.environ["NCCL_GIN_STATIC_RATE"] = previous


def install_selected(path, fp8_combine, *, rail_gin_rate=None, stream_gin_rate=None):
    """Every DeepEP v2 buffer vLLM makes (the base model's and EchoMoE's) gets a two-hop and a
    streaming buffer of the same width beside it (at most the table's and DeepEP's token limits),
    and a RailSwitch with the table at `path` in front of them. A buffer narrower than the model is
    EchoMoE's (Qwen3-30B-A3B), and its two-hop buffer also carries the reference rows' all-gather
    and reduce-scatter. A larger model's folded buffer also carries its reference layers' rows, so
    the kernels' buffers beside it are as wide as the model. `fp8_combine`: DeepEP is the copy
    whose combine crosses nodes in FP8, so the kernels' combines do too and FP8 calls are priced by
    the table's FP8 rows. `rail_gin_rate`, `stream_gin_rate`: the capped runs' rates (gin_rate)."""
    costs = Costs(path)
    get_handle = DeepEPV2All2AllManager.get_handle
    switches = {}

    def selected(self, kwargs):
        experts, width, top_k = (
            kwargs.get("num_experts", 256),
            kwargs["hidden"],
            kwargs["num_topk"],
        )
        token_limit = kwargs["num_max_tokens_per_rank"]
        buffer = get_handle(self, kwargs)
        if id(buffer) not in switches:
            vllm_config = get_current_vllm_config()
            config = vllm_config.model_config
            hidden = config.hf_text_config.hidden_size
            group = get_ep_group().device_group
            larger = vllm_config.load_config.load_format == portable.LOAD_FORMAT
            if larger:
                width = hidden
            folded = width < hidden
            kernels = {}
            with gin_rate(rail_gin_rate):
                kernels["rail"] = RailBuffer(
                    group,
                    min(costs.rail_tokens, token_limit),
                    width,
                    top_k,
                    experts,
                    num_contexts=1,
                    gather_bytes=row_layout(hidden, top_k)["row_bytes"] if folded else 0,
                    reduce_width=width if folded else 0,
                    fp8_combine=fp8_combine,
                )
            with gin_rate(stream_gin_rate):
                kernels["stream"] = StreamBuffer(
                    group,
                    min(costs.stream.capacity, token_limit),
                    width,
                    top_k,
                    experts,
                    num_contexts=costs.stream.contexts,
                    chunk_bytes=costs.stream.chunk,
                    fp8_combine=fp8_combine,
                )
            # EchoMoE's combine rows are narrower than its buffer where its dispatch rank is larger
            combine = width
            if larger:
                combine = PortableExport(Path(config.model).parent).recipe.combine_rank
            elif folded:
                combine = FoldedExport(config.model).folded["combine_rank"]
            switches[id(buffer)] = RailSwitch(buffer, kernels, combine, costs, fp8_combine)
            # registered with vLLM's buffers, so that its teardown destroys the kernels' buffers
            self.handle_cache._cache["rail", id(buffer)] = switches[id(buffer)]
        return switches[id(buffer)]

    DeepEPV2All2AllManager.get_handle = selected
