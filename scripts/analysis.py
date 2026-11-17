"""The MoE-input similarity behind Table 2 and the residual spectrum behind Table 4 and Figure 5.

    python scripts/analysis.py spectrum outputs/qwen3-30b-a3b/fit/moments.safetensors \
        outputs/analysis/residual_spectrum.json

For every MoE layer `spectrum` compares the input x_l with its residual x_l - a * x_{l-s} after a
reference s = 1, 2 or 3 layers back, the layers a stride-4 reference serves. Every statistic is a
ratio, so the moments' row normalization does not matter.

    python scripts/analysis.py similarity-capture <hf model> outputs/qwen3-30b-a3b/data/dolci \
        outputs/similarity/<model>
    python scripts/analysis.py similarity outputs/similarity/<model>/moments.safetensors \
        outputs/analysis/similarity/<model>.json

`similarity-capture` runs the model on GPUs over 256 Dolci validation documents and sums, in
float64, the moments of every MoE input and of every pair of MoE inputs up to 16 MoE layers apart.
The first 128 documents fit the predictor and the other 128 score it. `similarity` reads the sums
on CPU and averages over the pairs k = 1, 4, 8 and 16 MoE layers apart the cosine and the held-out
R^2 of a per-dimension affine predictor of the later input from the earlier one. DeepSeek-V3 and
GLM-5.3 are FP8 checkpoints, which this capture does not load. Their rows were captured by
dequantizing each weight block when it was used, with two documents per forward and part of the
weights in host memory, and GLM-5.3 needed transformers 5.17.

    python scripts/analysis.py similarity-capture --cross-token <hf model> \
        outputs/qwen3-30b-a3b/data/dolci outputs/similarity/<model>
    python scripts/analysis.py cross-token outputs/similarity/<model> \
        outputs/analysis/similarity/<model>-cross-token.json

With `--cross-token` the capture also writes every MoE input and the experts it is dispatched to,
one file pair per MoE layer. `cross-token` predicts a token's MoE input from another token sent to
the same expert at the same layer, with the same predictor and halves (Table 16). The partner is
the most similar token, in random pools of 100, 1,000 and 10,000 of the expert's tokens and in all
of them, or the token's LSH-MoE cluster medoid under 6 hashes onto 2 axes. The trained baseline in
`echomoe.lsh` sends the cluster mean instead and uses layer 0's projections at every layer. It runs
on a GPU when there is one, as the paper's numbers did. A CPU draws other random pools.
"""

import argparse
import json
import random
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import transformers
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.numpy import load_file
from safetensors.torch import save_file

from echomoe.lsh import LshConfig, buckets, rotations
from echomoe.runs import write_meta

REFERENCE_DISTANCES = (1, 2, 3)
CODE_WIDTHS = list(range(64, 2049, 64))
SEPARATIONS = (1, 4, 8, 16)
MAX_SEPARATION = 16
PANEL_DOCUMENTS = 256
MAX_DOCUMENT_TOKENS = 2048
DOLCI_RECIPE = Path(__file__).resolve().parents[1] / "configs/data/dolci.yaml"
POOL_SIZES = (100, 1000, 10000, None)
CROSS_TOKEN_LSH = LshConfig(projection_dim=2, num_hashes=6, seed=0)


def descending_eigenvalues(moment):
    return torch.linalg.eigvalsh((moment + moment.T) * 0.5).flip(0)


def spectra(path):
    """Per layer: (input eigenvalues, input trace, [(residual eigenvalues, residual trace)])."""
    layers = []
    with safe_open(path, framework="pt") as handle:
        input_moments = handle.get_slice("x_moment")
        cross_moments = handle.get_slice("cross_moment")
        gains = handle.get_tensor("gain").double()
        for layer in range(gains.shape[0]):
            input_moment = input_moments[layer : layer + 1][0].double()
            residuals = []
            for distance in REFERENCE_DISTANCES:
                if layer < distance:  # no reference that many layers back
                    continue
                reference = input_moments[layer - distance : layer - distance + 1][0].double()
                gain = gains[layer, distance - 1]
                cross_moment = cross_moments[layer : layer + 1, distance - 1 : distance][0, 0]
                scaled = gain[:, None] * cross_moment.double().T
                residual = (
                    input_moment - scaled - scaled.T + gain[:, None] * reference * gain[None, :]
                )
                residuals.append((descending_eigenvalues(residual), residual.diagonal().sum()))
            input_trace = input_moment.diagonal().sum()
            layers.append((descending_eigenvalues(input_moment), input_trace, residuals))
    return layers


def summary(layers):
    def mean(values):
        return float(torch.stack(list(values)).mean())

    def top(eigenvalues, width):
        return eigenvalues[:width].sum()

    def participation(eigenvalues):
        return eigenvalues.sum() ** 2 / (eigenvalues**2).sum()

    # one (input trace, residual eigenvalues, residual trace) per layer and reference distance
    residuals = [
        (input_trace, residual_eigenvalues, residual_trace)
        for _, input_trace, by_distance in layers
        for residual_eigenvalues, residual_trace in by_distance
    ]
    curves = {
        "raw_share": [
            mean(
                top(input_eigenvalues, width) / input_trace
                for input_eigenvalues, input_trace, _ in layers
            )
            for width in CODE_WIDTHS
        ],
        "residual_share": [
            mean(
                top(residual_eigenvalues, width) / residual_trace
                for _, residual_eigenvalues, residual_trace in residuals
            )
            for width in CODE_WIDTHS
        ],
        "plain_dropped": [
            mean(
                (input_trace - top(input_eigenvalues, width)) / input_trace
                for input_eigenvalues, input_trace, _ in layers
            )
            for width in CODE_WIDTHS
        ],
        "residual_dropped": [
            mean(
                (residual_trace - top(residual_eigenvalues, width)) / input_trace
                for input_trace, residual_eigenvalues, residual_trace in residuals
            )
            for width in CODE_WIDTHS
        ],
    }
    return {
        "residual_energy": mean(
            residual_trace / input_trace for input_trace, _, residual_trace in residuals
        ),
        "residual_energy_by_distance": [
            mean(
                by_distance[index][1] / input_trace
                for _, input_trace, by_distance in layers
                if len(by_distance) > index
            )
            for index in range(len(REFERENCE_DISTANCES))
        ],
        "effective_rank": {
            "raw": mean(participation(input_eigenvalues) for input_eigenvalues, _, _ in layers),
            "residual": mean(
                participation(residual_eigenvalues) for _, residual_eigenvalues, _ in residuals
            ),
        },
        "widths": CODE_WIDTHS,
        **curves,
    }


def documents(data):
    """The study's 256 Dolci validation conversations as (row id, plain text), fit half first."""
    table = pq.read_table(
        data / "split/rows.parquet",
        columns=["row_id", "source_file_id", "source_row"],
        filters=[("partition", "=", "validation")],
    ).to_pydict()
    location = dict(zip(table["row_id"], zip(table["source_file_id"], table["source_row"])))
    chosen = random.Random(0).sample(sorted(location), PANEL_DOCUMENTS)
    source = yaml.safe_load(DOLCI_RECIPE.read_text())["source"]
    pattern = f"data/{source['split']}-*.parquet"
    root = snapshot_download(
        source["dataset_id"],
        repo_type="dataset",
        revision=source["revision"],
        allow_patterns=pattern,
    )
    files = sorted(Path(root).glob(pattern))
    messages = {}
    for file in {location[row][0] for row in chosen}:
        column = pq.read_table(files[file], columns=["messages"]).column("messages")
        messages |= {
            row: column[location[row][1]].as_py() for row in chosen if location[row][0] == file
        }
    # no chat template, so every model reads the same characters
    return [
        (row, "\n\n".join(f"{m['role']}: {m['content']}" for m in messages[row])) for row in chosen
    ]


def fold(sums, inputs):
    """Add one document's MoE inputs to the sums of every layer and of every pair of layers."""
    recent = deque(maxlen=MAX_SEPARATION)
    pair = 0
    for layer, current in enumerate(inputs):
        current = current.double()
        norm = current.square().sum(1).sqrt()
        sums["sum"][layer] += current.sum(0)
        sums["sq"][layer] += current.square().sum(0)
        for earlier, earlier_norm in reversed(recent):
            product = current * earlier
            sums["cross"][pair] += product.sum(0)
            sums["cos_sum"][pair] += (product.sum(1) / (norm * earlier_norm)).sum()
            pair += 1
        recent.append((current, norm))


@torch.no_grad()
def capture(model_path, data, output, cross_token):
    """Write the MoE-input moments of the fit and the eval half to `output/moments.safetensors`,
    and with `cross_token` every MoE input and its experts to `output/layers/`."""
    panel = documents(data)
    config = transformers.AutoConfig.from_pretrained(model_path)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    # hook what dispatch sends: the MoE block's input, or the experts' input where the decoder
    # layer holds them inline (Gemma 4). Dense layers have neither.
    layers, inputs, routes = [], [], []
    for index, decoder_layer in enumerate(model.get_decoder().layers):
        block = next(
            (child for child in decoder_layer.children() if hasattr(child, "experts")),
            getattr(decoder_layer, "experts", None),
        )
        if block is not None:
            layers.append(index)
            # drop the first position, the attention sink, whose norm would own the moments
            block.register_forward_pre_hook(
                lambda module, args: inputs.append(
                    args[0].reshape(-1, args[0].shape[-1])[1:].to(model.device)
                )
            )
            # the expert ids the experts are called with, which are the ones dispatched
            getattr(block, "experts", block).register_forward_pre_hook(
                lambda module, args: routes.append(args[1][1:])
            )
    moe_layer_count, hidden_size = len(layers), config.get_text_config().hidden_size
    pairs = [
        (target, target - separation)
        for target in range(moe_layer_count)
        for separation in range(1, MAX_SEPARATION + 1)
        if separation <= target
    ]
    zeros = partial(torch.zeros, dtype=torch.float64, device=model.device)
    moments = {"layers": torch.tensor(layers), "pairs": torch.tensor(pairs)}
    output.mkdir(parents=True, exist_ok=True)
    row_documents = []
    if cross_token:
        (output / "layers").mkdir()
    halfway = PANEL_DOCUMENTS // 2
    for phase, half in ("fit", panel[:halfway]), ("eval", panel[halfway:]):
        sums = {
            "sum": zeros(moe_layer_count, hidden_size),
            "sq": zeros(moe_layer_count, hidden_size),
            "cross": zeros(len(pairs), hidden_size),
            "cos_sum": zeros(len(pairs)),
        }
        tokens = 0
        for row, text in half:
            ids = tokenizer(text, truncation=True, max_length=MAX_DOCUMENT_TOKENS)["input_ids"]
            ids = torch.tensor([ids], device=model.device)
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            assert len(inputs) == moe_layer_count, (
                f"{len(inputs)} MoE inputs, not {moe_layer_count}"
            )
            fold(sums, inputs)
            tokens += len(inputs[0])
            if cross_token:
                for position, (moe_input, route) in enumerate(zip(inputs, routes)):
                    assert len(route) == len(moe_input), (
                        f"{len(route)} routed rows, not {len(moe_input)}"
                    )
                    prefix = output / f"layers/layer-{position:03d}"
                    with Path(f"{prefix}.hidden.bf16").open("ab") as file:
                        moe_input.to(torch.bfloat16).cpu().view(torch.uint16).numpy().tofile(file)
                    with Path(f"{prefix}.experts.i16").open("ab") as file:
                        route.to(torch.int16).cpu().numpy().tofile(file)
                row_documents += [row] * len(inputs[0])
            inputs.clear()
            routes.clear()
        print(f"{phase} documents={len(half)} tokens={tokens}", flush=True)
        moments[f"{phase}/tokens"] = torch.tensor(tokens)
        moments |= {f"{phase}/{key}": total.cpu() for key, total in sums.items()}
    save_file(moments, output / "moments.safetensors")
    write_meta(output, inputs={"model": model_path, "data": data})
    if cross_token:
        splits = ["fit"] * halfway + ["eval"] * halfway
        manifest = {
            "layers": layers,
            "documents": [
                {"row_id": row, "split": split} for (row, _), split in zip(panel, splits)
            ],
            "row_document_ids": row_documents,
        }
        (output / "cross_token_manifest.json").write_text(json.dumps(manifest) + "\n")


def similarity(path):
    """Table 2's row: the mean cosine and held-out R^2 over the MoE-layer pairs at each
    separation."""
    moments = load_file(path)
    target, source = moments["pairs"].T
    fit_tokens, eval_tokens = float(moments["fit/tokens"]), float(moments["eval/tokens"])
    fit_target, fit_source = moments["fit/sum"][target], moments["fit/sum"][source]
    eval_target, eval_sq_target = moments["eval/sum"][target], moments["eval/sq"][target]
    # one scale and one shift per dimension, least squares on the fit half
    slope = (moments["fit/cross"] - fit_target * fit_source / fit_tokens) / (
        moments["fit/sq"][source] - fit_source**2 / fit_tokens
    )
    intercept = fit_target / fit_tokens - slope * (fit_source / fit_tokens)
    residual = (
        eval_sq_target
        - 2.0 * slope * moments["eval/cross"]
        - 2.0 * intercept * eval_target
        + slope**2 * moments["eval/sq"][source]
        + 2.0 * slope * intercept * moments["eval/sum"][source]
        + eval_tokens * intercept**2
    )
    total = (eval_sq_target - eval_target**2 / eval_tokens).sum(axis=1)
    predictability = 1.0 - residual.sum(axis=1) / total
    cosine = moments["eval/cos_sum"] / eval_tokens
    # MoE layers apart, which on Nemotron's hybrid stack is not the decoder-layer distance
    pair_separation = target - source
    return {
        "moe_layers": len(moments["layers"]),
        "tokens": {"fit": int(moments["fit/tokens"]), "eval": int(moments["eval/tokens"])},
        "separations": list(SEPARATIONS),
        "cosine": [float(cosine[pair_separation == k].mean()) for k in SEPARATIONS],
        "predictability": [
            float(predictability[pair_separation == k].mean()) for k in SEPARATIONS
        ],
    }


def nearest(inputs):
    """Each row's most similar other row by cosine, scored 4096 rows at a time."""
    unit = torch.nn.functional.normalize(inputs.float(), dim=1)
    partner = torch.empty(len(unit), dtype=torch.int64, device=unit.device)
    for start in range(0, len(unit), 4096):
        scores = unit[start : start + 4096] @ unit.T
        rows = torch.arange(len(scores), device=unit.device)
        scores[rows, rows + start] = -float("inf")
        partner[start : start + 4096] = scores.argmax(1)
    return partner


def add_pairs(totals, partners, tokens):
    """Add the pairs' sums of x, y, x^2, y^2 and xy, with x the partner and y the token, and their
    count to `totals`."""
    for start in range(0, len(partners), 4096):
        x = partners[start : start + 4096].double()
        y = tokens[start : start + 4096].double()
        totals["sums"] += torch.stack(
            [x.sum(0), y.sum(0), x.square().sum(0), y.square().sum(0), (x * y).sum(0)]
        )
    totals["pairs"] += len(partners)


def affine_r2(fit, held):
    """Held-out R^2 of the per-dimension affine predictor of y from x fitted on `fit`'s pairs."""
    (sum_x, sum_y, sum_xx, _, sum_xy), fit_pairs = fit["sums"], float(fit["pairs"])
    (held_x, held_y, held_xx, held_yy, held_xy), held_pairs = held["sums"], float(held["pairs"])
    slope = (sum_xy - sum_y * sum_x / fit_pairs) / (sum_xx - sum_x**2 / fit_pairs)
    intercept = sum_y / fit_pairs - slope * sum_x / fit_pairs
    residual = (
        held_yy
        - 2.0 * slope * held_xy
        - 2.0 * intercept * held_y
        + slope**2 * held_xx
        + 2.0 * slope * intercept * held_x
        + held_pairs * intercept**2
    )
    return float(1.0 - residual.sum() / (held_yy - held_y**2 / held_pairs).sum())


def representatives(inputs, routed, layer):
    """Each (token, expert) pair's representative under LSH-MoE's clustering by expert and
    bucket: the member nearest the cluster's mean, the lowest token on a tie."""
    (rows, hidden_size), topk, device = inputs.shape, routed.shape[1], inputs.device
    bucket = buckets(inputs, rotations(CROSS_TOKEN_LSH, hidden_size, layer).to(device))
    keys = routed.reshape(-1) * CROSS_TOKEN_LSH.bucket_count + bucket.repeat_interleave(topk)
    _, cluster, cluster_sizes = torch.unique(keys, return_inverse=True, return_counts=True)
    token = torch.arange(rows, device=device).repeat_interleave(topk)
    pair_inputs = inputs[token].float()
    clusters = len(cluster_sizes)
    cluster_sums = torch.zeros(clusters, hidden_size, device=device)
    cluster_sums.index_add_(0, cluster, pair_inputs)
    # in place, so a DeepSeek-V3 layer fits on one GPU
    distance = pair_inputs.sub_((cluster_sums / cluster_sizes[:, None])[cluster]).square_().sum(1)
    least = torch.full((clusters,), float("inf"), device=device)
    is_nearest = distance <= least.scatter_reduce_(0, cluster, distance, "amin")[cluster]
    medoid = torch.full((clusters,), rows, device=device)
    medoid = medoid.scatter_reduce_(0, cluster[is_nearest], token[is_nearest], "amin")
    return medoid[cluster], token


def cross_token_layer(hidden, experts, documents, fit, layer):
    """One layer's held-out R^2 from each token's best partner at each pool size and from its
    LSH-MoE representative, and the share of best partners in the same document."""
    hidden_size, device = hidden.shape[1], hidden.device

    def empty():
        sums = torch.zeros(5, hidden_size, dtype=torch.float64, device=device)
        return {"sums": sums, "pairs": 0, "same_document": 0}

    totals = {"lsh": {}} | {size: {} for size in POOL_SIZES}
    for half, rows in ("fit", fit), ("eval", ~fit):
        inputs, routed, row_documents = hidden[rows], experts[rows].long(), documents[rows]
        experts_seen = int(routed.max()) + 1
        pools = [
            (routed == expert).any(1).nonzero(as_tuple=True)[0] for expert in range(experts_seen)
        ]
        largest = max(len(pool) for pool in pools)
        by_size = {size: empty() for size in POOL_SIZES if size is None or size < largest}
        generator = torch.Generator(device=device).manual_seed(1000 * layer)
        for pool in (pool for pool in pools if len(pool) > 1):
            for size, size_totals in by_size.items():
                sample = pool
                if size is not None and len(pool) > size:
                    order = torch.randperm(len(pool), generator=generator, device=device)
                    sample = pool[order[:size]]
                sample_inputs = inputs[sample]
                partner = nearest(sample_inputs)
                # the original drew a uniform partner here too, and later pools follow its draws
                torch.randint(
                    0, len(sample) - 1, (len(sample),), generator=generator, device=device
                )
                add_pairs(size_totals, sample_inputs[partner], sample_inputs)
                size_totals["same_document"] += int(
                    (row_documents[sample] == row_documents[sample[partner]]).sum()
                )
        # a pool size above every pool is the whole pool
        for size in POOL_SIZES:
            totals[size][half] = by_size.get(size, by_size[None])
        medoid, token = representatives(inputs, routed, layer)
        replaced = medoid != token
        totals["lsh"][half] = empty()
        add_pairs(totals["lsh"][half], inputs[medoid[replaced]], inputs[token[replaced]])
    best_eval = totals[None]["eval"]
    return {
        "best": {
            size: affine_r2(totals[size]["fit"], totals[size]["eval"]) for size in POOL_SIZES
        },
        "same_document": best_eval["same_document"] / best_eval["pairs"],
        "lsh": affine_r2(totals["lsh"]["fit"], totals["lsh"]["eval"]),
    }


def cross_token(capture):
    """Table 16's rows 2 and 3, and the pool sizes and same-document share the text quotes, as
    means over the MoE layers."""
    manifest = json.loads((capture / "cross_token_manifest.json").read_text())
    split = {document["row_id"]: document["split"] for document in manifest["documents"]}
    row_ids = manifest["row_document_ids"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    documents = torch.tensor(row_ids, device=device)
    fit = torch.tensor([split[row_id] == "fit" for row_id in row_ids], device=device)
    per_layer = []
    for position, layer in enumerate(manifest["layers"]):
        prefix = capture / f"layers/layer-{position:03d}"
        hidden = np.fromfile(f"{prefix}.hidden.bf16", dtype=np.uint16).reshape(len(row_ids), -1)
        experts = np.fromfile(f"{prefix}.experts.i16", dtype=np.int16).reshape(len(row_ids), -1)
        hidden = torch.from_numpy(hidden).view(torch.bfloat16).to(device)
        experts = torch.from_numpy(experts).to(device)
        per_layer.append(cross_token_layer(hidden, experts, documents, fit, layer))
    return {
        "moe_layers": len(per_layer),
        "best_other_token": {
            str(size or "full"): sum(result["best"][size] for result in per_layer) / len(per_layer)
            for size in POOL_SIZES
        },
        "same_document": sum(result["same_document"] for result in per_layer) / len(per_layer),
        "lsh_representative": sum(result["lsh"] for result in per_layer) / len(per_layer),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("spectrum", "similarity", "cross-token"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
        command.add_argument("output", type=Path)
    command = commands.add_parser("similarity-capture")
    command.add_argument("model")
    command.add_argument("data", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--cross-token", action="store_true")
    args = parser.parse_args()
    if args.command == "similarity-capture":
        capture(args.model, args.data, args.output, args.cross_token)
    else:
        if args.command == "spectrum":
            result = summary(spectra(args.input))
        elif args.command == "similarity":
            result = similarity(args.input)
        else:
            result = cross_token(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + "\n")
