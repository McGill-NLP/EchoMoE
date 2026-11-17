#!/usr/bin/env python3
"""Measure from a capture's router logits how many remote nodes a token reaches: `R` per layer
and `P_s` per reference layer at 2, 4 and 8 nodes, the fanout `communication_cost` prices with.

    python scripts/routing_fanout.py <capture root> results/communication/routing_fanout.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from echomoe.communication_cost import (
    NODE_COUNTS,
    NUM_EXPERTS,
    NUM_LAYERS,
    ROUTER_TOPK,
    STRIDES,
    uniform_fanout,
)
from echomoe.runs import atomic_write_json


def routing_fanout(root: Path) -> dict[str, Any]:
    # every part's meta.json, which must all share the model's experts, top-k and layers
    part_metas = {
        path.parent.name: json.loads(path.read_text())
        for path in sorted(root.glob("part-*/meta.json"))
    }
    (routing,) = {
        (meta["num_experts"], meta["topk"], meta["renormalize"], tuple(meta["layers"]))
        for meta in part_metas.values()
    }
    experts, topk, renormalize, layers = routing
    assert (experts, topk, layers) == (NUM_EXPERTS, ROUTER_TOPK, tuple(range(NUM_LAYERS))), routing

    # count, per token, the nodes holding one of its top-k experts at each layer, and at each
    # reference layer the nodes in the union over it and the layers that read it
    routed_nodes = {nodes: torch.zeros(NUM_LAYERS, dtype=torch.float64) for nodes in NODE_COUNTS}
    reference_nodes = {
        (nodes, stride): torch.zeros(NUM_LAYERS // stride, dtype=torch.float64)
        for nodes in NODE_COUNTS
        for stride in STRIDES
    }
    tokens = 0
    for part, meta in part_metas.items():
        # Every layer's file of one split and name holds the same tokens.
        for split, file_name in sorted(
            {(key.split("/")[0], key.split("/")[2]) for key in meta["files"]}
        ):
            top_experts = torch.stack(
                [
                    safe_open(root / part / split / f"layer-{layer:04d}" / file_name, "pt")
                    .get_slice("router_logits")[:]
                    .float()
                    .topk(topk, dim=1)
                    .indices
                    for layer in layers
                ]
            )
            tokens += top_experts.shape[1]
            for nodes in NODE_COUNTS:
                # which nodes hold one of each token's experts, per layer
                holds = torch.zeros(NUM_LAYERS, top_experts.shape[1], nodes, dtype=torch.bool)
                holds.scatter_(2, top_experts // (experts // nodes), True)
                routed_nodes[nodes] += holds.sum(dim=(1, 2), dtype=torch.float64)
                for stride in STRIDES:
                    for index, reference in enumerate(range(0, NUM_LAYERS, stride)):
                        group = holds[reference : reference + stride].any(dim=0)
                        reference_nodes[nodes, stride][index] += group.sum(dtype=torch.float64)

    # the counts per token, scaled to remote nodes, beside the uniform-routing values
    document: dict[str, Any] = {
        "tokens": tokens,
        "capture": str(root),
        "splits": "every fit and validation row of every part",
        "renormalize": renormalize,
        "routing": f"top-{topk} expert ids of router_logits per token per layer (renormalizing the weights does not change the ids)",
        "placement": "N nodes each hold a contiguous block of 128/N experts; expert e is on node e // (128/N)",
        "remote": "R at a layer = E[distinct nodes holding a chosen expert] * (N-1)/N: the token's home node is uniform over nodes and independent of routing, so E[distinct] - P(home among them) = E[distinct] (1 - 1/N)",
        "publication": "P_s at reference layer l = E[distinct nodes in the union over layers l..l+s-1] * (N-1)/N: the reference layer's experts read the publication (its routed rows are empty) and so do the s-1 consumer layers",
        "uniform": "R = (N-1)(1-q), P_s = (N-1)(1-q^s), q = C(128-128/N, 8)/C(128, 8)",
        "nodes": {},
    }
    for nodes in NODE_COUNTS:
        uniform = uniform_fanout(nodes)
        # from the nodes per token to the remote nodes per token
        remote_per_token = (nodes - 1) / nodes / tokens
        routed = (routed_nodes[nodes] * remote_per_token).tolist()
        entry: dict[str, Any] = {
            "remote": {
                "per_layer": routed,
                "mean": sum(routed) / NUM_LAYERS,
                "uniform": uniform.routed_nodes[0],
            },
            "publication": {},
        }
        for stride in STRIDES:
            reference = (reference_nodes[nodes, stride] * remote_per_token).tolist()
            entry["publication"][str(stride)] = {
                "per_reference_layer": reference,
                "mean": sum(reference) / len(reference),
                "uniform": uniform.reference_nodes[stride][0],
            }
        document["nodes"][str(nodes)] = entry
    return document


if __name__ == "__main__":
    atomic_write_json(Path(sys.argv[2]), routing_fanout(Path(sys.argv[1])))
