"""RDMA payload bytes per token for one compressor configuration.

A token costs, summed over the MoE layers of one forward pass:
    reference = d * sum_i P_s[i]          (every s-th layer publishes x_l once per remote node)
    dispatch  = r_d * sum_{l not ref} R_l  (non-reference layers send it once per remote node)
    combine   = r_c * sum_l R_l            (every layer returns the code from each remote node)
in elements of the side's precision, against the baseline (d + d) * sum_l R_l. `R_l` is the
number of distinct remote nodes holding one of the token's top-k experts at layer l, and `P_s[i]`
the same count over a reference and the s - 1 layers that read it. Both are measured from routing
(`routing_fanout.json`). DeepEP sends a token once per remote node and fans out over NVLink
inside it, so NVLink bytes, routing metadata and padding are not counted. `unit_fanout` counts
every send once, which gives the topology-free width ratio instead.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HIDDEN_SIZE = 2048
NUM_LAYERS = 48
NUM_EXPERTS = 128
ROUTER_TOPK = 8

# The priced grid: every fitted config, at 2, 4 and 8 nodes.
NODE_COUNTS = (2, 4, 8)
STRIDES = (4, 6, 8, 12)
DISPATCH_RANKS = (256, 512, 768, 1024, 1280)
PLAIN_DISPATCH_RANKS = (512, 768, 1024)
COMBINE_RANKS = (256, 512, 768, 1024, 1280)
COMBINE_OWNERS = ("direct", "bias", "biasgate")
CORRECTION_WIDTH = 128


@dataclass(frozen=True)
class Precision:
    label: str
    element_bytes: int
    scale_block: int | None = None
    scale_bytes: int = 0

    def bytes(self, elements: int) -> int:
        if self.scale_block is None:
            return elements * self.element_bytes
        scales = math.ceil(elements / self.scale_block) * self.scale_bytes
        return elements * self.element_bytes + scales


BF16 = Precision("bf16", 2)
FP8 = Precision("fp8-block128", 1, scale_block=128, scale_bytes=4)


@dataclass(frozen=True)
class Fanout:
    """Remote nodes a token reaches: `routed_nodes[layer]` is R, and `reference_nodes[stride][i]`
    is P_s at the i-th reference layer `i * stride`."""

    nodes: int
    routed_nodes: tuple[float, ...]
    reference_nodes: dict[int, tuple[float, ...]]
    source: str


def token_bytes(
    *,
    dispatch_rank: int | None,
    stride: int | None,
    combine_rank: int | None,
    reference_rank: int | None,
    fanout: Fanout,
    hidden: int = HIDDEN_SIZE,
    dispatch_precision: Precision = BF16,
    combine_precision: Precision = BF16,
) -> dict[str, float]:
    """Bytes per token per transfer, the full-width baseline and their ratio `reduction`. The
    baseline uses the same precisions, so `reduction` is the width reduction alone."""
    if dispatch_rank is None and stride is not None:
        raise ValueError("a full-width dispatch publishes no reference")
    layer_count = len(fanout.routed_nodes)
    reference_layers = () if stride is None else tuple(range(0, layer_count, stride))
    # bytes of one send of each payload
    reference_payload = dispatch_precision.bytes(
        hidden if reference_rank is None else reference_rank
    )
    dispatch_payload = dispatch_precision.bytes(hidden if dispatch_rank is None else dispatch_rank)
    combine_payload = combine_precision.bytes(hidden if combine_rank is None else combine_rank)
    # routed sends per token: R summed over every layer, and over the non-reference layers
    every_layer_sends = sum(fanout.routed_nodes)
    dispatch_sends = sum(
        fanout.routed_nodes[layer] for layer in range(layer_count) if layer not in reference_layers
    )
    reference = 0.0
    if stride is not None:
        reference = reference_payload * sum(
            fanout.reference_nodes[stride][: len(reference_layers)]
        )
    dispatch = dispatch_payload * dispatch_sends
    combine = combine_payload * every_layer_sends
    full_width_payload = dispatch_precision.bytes(hidden) + combine_precision.bytes(hidden)
    full_width = full_width_payload * every_layer_sends
    total = reference + dispatch + combine
    return {
        "reference": reference,
        "dispatch": dispatch,
        "combine": combine,
        "full_width": full_width,
        "total": total,
        "reduction": full_width / total,
    }


def load_fanout(path: str | Path, nodes: int) -> Fanout:
    measured = json.loads(Path(path).read_text())["nodes"][str(nodes)]
    return Fanout(
        nodes=nodes,
        routed_nodes=tuple(measured["remote"]["per_layer"]),
        reference_nodes={
            int(stride): tuple(entry["per_reference_layer"])
            for stride, entry in measured["publication"].items()
        },
        source=str(path),
    )


def missed_node_probability(nodes: int) -> float:
    experts_elsewhere = NUM_EXPERTS - NUM_EXPERTS // nodes
    return math.comb(experts_elsewhere, ROUTER_TOPK) / math.comb(NUM_EXPERTS, ROUTER_TOPK)


def uniform_fanout(nodes: int) -> Fanout:
    """R = (N-1)(1-q) and P_s = (N-1)(1-q^s), with q the chance that a node holds none of a
    token's experts, for routing independent across tokens and layers."""
    q = missed_node_probability(nodes)
    return Fanout(
        nodes=nodes,
        routed_nodes=(float((nodes - 1) * (1 - q)),) * NUM_LAYERS,
        reference_nodes={
            stride: (float((nodes - 1) * (1 - q**stride)),) * len(range(0, NUM_LAYERS, stride))
            for stride in STRIDES
        },
        source="uniform",
    )


def unit_fanout() -> Fanout:
    return Fanout(
        nodes=1,
        routed_nodes=(1.0,) * NUM_LAYERS,
        reference_nodes={stride: (1.0,) * len(range(0, NUM_LAYERS, stride)) for stride in STRIDES},
        source="width",
    )


def dispatch_label(rank: int | None, stride: int | None = None) -> str:
    if rank is None:
        return "dfull"
    if stride is None:
        return f"dplain{rank}"
    return f"dres{rank}-s{stride}"


def _configurations() -> list[tuple[str, str, int | None, int | None, int | None]]:
    configurations: list[tuple[str, str, int | None, int | None, int | None]] = []
    for stride in STRIDES:
        for rank in DISPATCH_RANKS:
            configurations.append(
                (f"{dispatch_label(rank, stride)}-cfull", "dispatch", rank, stride, None)
            )
    for rank in PLAIN_DISPATCH_RANKS:
        configurations.append((f"{dispatch_label(rank)}-cfull", "dispatch", rank, None, None))
    for rank in COMBINE_RANKS:
        for owner in COMBINE_OWNERS:
            gate = f"-q{CORRECTION_WIDTH}" if owner == "biasgate" else ""
            configurations.append((f"dfull-c{rank}-{owner}{gate}", "combine", None, None, rank))
    # the owner-side term sends nothing, so direct, bias and gated bias decoding price the same
    # and joint ids omit it
    for stride in (*STRIDES, None):
        for rank in DISPATCH_RANKS:
            for combine_rank in COMBINE_RANKS:
                label = f"{dispatch_label(rank, stride)}-c{combine_rank}"
                configurations.append((label, "joint", rank, stride, combine_rank))
    return configurations


def cost_table(fanout_path: str | Path) -> list[dict[str, Any]]:
    """Every grid configuration at every node count in bf16. Unprefixed columns use the measured
    fanout, `uniform_*` uniform routing and `width_*` the topology-free width ratio."""
    fanouts = {
        nodes: {
            "measured": load_fanout(fanout_path, nodes),
            "uniform": uniform_fanout(nodes),
            "width": unit_fanout(),
        }
        for nodes in NODE_COUNTS
    }
    rows = []
    for nodes in NODE_COUNTS:
        for config_id, group, dispatch_rank, stride, combine_rank in _configurations():
            row: dict[str, Any] = {
                "config_id": config_id,
                "group": group,
                "nodes": nodes,
                "dispatch_rank": dispatch_rank,
                "stride": stride,
                "combine_rank": combine_rank,
            }
            for kind, fanout in fanouts[nodes].items():
                costs = token_bytes(
                    dispatch_rank=dispatch_rank,
                    stride=stride,
                    combine_rank=combine_rank,
                    reference_rank=None,
                    fanout=fanout,
                )
                prefix = "" if kind == "measured" else f"{kind}_"
                for transfer in ("reference", "dispatch", "combine", "total", "full_width"):
                    row[f"{prefix}{transfer}_bytes"] = costs[transfer]
                row[f"{prefix}reduction"] = costs["reduction"]
            rows.append(row)
    return rows


def table_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fanout", type=Path, help="routing_fanout.json")
    parser.add_argument("output", type=Path, help="communication_cost.csv to write")
    args = parser.parse_args()
    rows = cost_table(args.fanout)
    args.output.write_text(table_csv(rows))
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
