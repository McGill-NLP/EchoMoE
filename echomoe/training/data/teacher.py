"""Two vLLM passes over the frozen teacher: activation capture and target caching.

    python -m echomoe.training.data.teacher capture --config configs/fit/dolci-capture.yaml \
        --part-index I
    python -m echomoe.training.data.teacher targets --config configs/fit/<cache>.yaml \
        --part-index I

Each part is one eager TP=1 engine whose worker records through hooks on the forward: the MoE
inputs, outputs and router logits of the seeded capture selection (format in `capture.py`), or the
next-token targets of a layout's segments (format in `targets.py`). Paths inside the
configs are relative to the working directory.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.fused_moe import FusedMoE

from echomoe.fit import moe_layout
from echomoe.runs import write_meta

from .capture import SPLITS, ActivationCaptureWriter, partition_coordinates, select_coordinates
from .layout import load_layout, load_tokenized_dataset
from .targets import SegmentWriter, covered_segments, plan_part, top_m_targets

# Every stage config holds these keys and its stage's own, nothing else.
COMMON_FIELDS = {
    "model_export",
    "tokenized_dataset",
    "output_root",
    "part_count",
    "rows_per_file",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "enable_chunked_prefill",
}
CAPTURE_FIELDS = COMMON_FIELDS | {
    "fit_layout",
    "validation_layout",
    "selection_seed",
    "fit_rows",
    "validation_rows",
    "max_tokens_per_document",
}
TEACHER_FIELDS = COMMON_FIELDS | {
    "layout",
    "order_seed",
    "order_epochs",
    "token_budget",
    "top_m",
    "logits_chunk_rows",
}


def load_config(path, stage):
    config = yaml.safe_load(Path(path).read_text())
    fields = CAPTURE_FIELDS if stage == "capture" else TEACHER_FIELDS
    if set(config) != fields:
        raise ValueError(f"{path}: missing {fields - set(config)}, unknown {set(config) - fields}")
    return config


def instrument_forward(runner, session):
    # The private `_model_forward` is the one place both the prepared positions and the hidden
    # states are in hand, which is what maps scheduled rows back to layout positions.
    original_forward = runner._model_forward

    def instrumented(*args, **kwargs):
        batch = runner.input_batch
        count = batch.num_reqs
        starts = [int(value) for value in runner.query_start_loc.np[: count + 1]]
        session.begin_batch(
            request_ids=list(batch.req_ids[:count]),
            query_start_locations=starts,
            token_positions=kwargs["positions"][: starts[-1]].tolist(),
        )
        try:
            hidden_states = original_forward(*args, **kwargs)
            session.consume(hidden_states)
            return hidden_states
        finally:
            session.end_batch()

    runner._model_forward = instrumented
    return original_forward


def moe_blocks(runner, layers):
    """Every MoE block, keyed by MoE ordinal."""
    blocks = dict(
        enumerate(module for module in runner.model.modules() if isinstance(module, FusedMoE))
    )
    if sorted(blocks) != sorted(layers):
        raise RuntimeError(f"expected MoE layers {sorted(layers)}, model has {sorted(blocks)}")
    return blocks


class ActivationCaptureSession:
    def __init__(self, writer, requests):
        self.writer = writer
        self.requests = requests
        self.handles = []
        self.end_batch()

    def register_blocks(self, blocks):
        for layer, block in blocks.items():
            self.handles.append(
                block.register_forward_pre_hook(self._pre_hook(layer), with_kwargs=True)
            )
            self.handles.append(
                block.register_forward_hook(self._post_hook(layer), with_kwargs=True)
            )
            if block.runner.shared_experts is not None:
                self.handles.append(self._tap_routed_output(layer, block.runner))

    def _tap_routed_output(self, layer, runner):
        # DeepSeek-V3's block adds the shared experts' output to the routed output before it
        # returns, and the fit wants the routed output alone, scaled as the combine carries it.
        original = runner._maybe_apply_routed_scale_to_output

        def tapped(shared_output, fused_output):
            shared_output, fused_output = original(shared_output, fused_output)
            if self.rows:
                self.routed[layer] = self._select(fused_output).clone()
            return shared_output, fused_output

        runner._maybe_apply_routed_scale_to_output = tapped
        return _Untap(runner)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.end_batch()

    def begin_batch(self, *, request_ids, query_start_locations, token_positions):
        plan = []
        for index, request_id in enumerate(request_ids):
            selected = self.requests.get(request_id)
            if not selected:
                continue
            for row in range(query_start_locations[index], query_start_locations[index + 1]):
                coordinate = selected.get(int(token_positions[row]))
                if coordinate is not None:
                    plan.append((row, coordinate))
        plan.sort()
        self.rows = tuple(row for row, _ in plan)
        self.coordinates = tuple(coordinate for _, coordinate in plan)
        self.inputs, self.logits, self.routed = {}, {}, {}

    def end_batch(self):
        self.rows, self.coordinates = (), ()
        self.inputs, self.logits, self.routed = {}, {}, {}

    def consume(self, hidden_states):
        pass

    def _select(self, value):
        return value.index_select(
            0, torch.tensor(self.rows, dtype=torch.int64, device=value.device)
        )

    def _pre_hook(self, layer):
        def hook(module, args, kwargs):
            if not self.rows:
                return
            # The block may write its output into this same storage, so the selected input rows
            # are copied before the forward runs.
            selected = self._select(args[0] if args else kwargs["hidden_states"]).clone()
            self.inputs[layer] = selected
            # Qwen3 and DeepSeek-V3 run the gate inside FusedMoE and pass the hidden states in
            # the router_logits position. The gate is linear in each row, so applying it to the
            # captured rows reproduces the logits the model routes on, in the dtype it routes on.
            # Gemma 4 routes on the residual outside FusedMoE and passes the float32 logits.
            if module.runner.gate is None:
                logits = args[1] if len(args) > 1 else kwargs["router_logits"]
                self.logits[layer] = self._select(logits).detach()
            else:
                self.logits[layer] = module.runner.gate(selected)[0].detach()

        return hook

    def _post_hook(self, layer):
        def hook(module, args, kwargs, output):
            if not self.rows:
                return
            routed = self.routed.pop(layer, None)
            if routed is None:
                routed = self._select(output[0] if isinstance(output, tuple) else output)
            self.writer.add(
                layer, self.coordinates, self.inputs.pop(layer), routed, self.logits.pop(layer)
            )

        return hook


class _Untap:
    """A hook handle that drops the instance attribute shadowing the runner's method."""

    def __init__(self, runner):
        self.runner = runner

    def remove(self):
        del self.runner._maybe_apply_routed_scale_to_output


def _scored_spans(requests, segment_rows, *, request_ids, query_start_locations, token_positions):
    spans = []
    for index, request_id in enumerate(request_ids):
        segment = requests.get(request_id)
        if segment is None:
            continue
        start, stop = query_start_locations[index], query_start_locations[index + 1]
        kept = [row for row in range(start, stop) if token_positions[row] < segment_rows[segment]]
        if not kept:
            continue
        first = int(token_positions[kept[0]])
        positions = [int(token_positions[row]) for row in kept]
        if kept != list(range(kept[0], kept[0] + len(kept))) or positions != list(
            range(first, first + len(kept))
        ):
            raise ValueError(f"segment {segment} was scheduled at non-contiguous positions")
        spans.append((segment, first, kept[0], len(kept)))
    return spans


class TeacherTargetSession:
    def __init__(self, writer, compute_logits, requests, *, chunk_rows, top_m):
        self.writer, self.compute_logits = writer, compute_logits
        self.chunk_rows, self.top_m = chunk_rows, top_m
        self.requests = requests
        self.spans = []

    def begin_batch(self, **batch):
        self.spans = _scored_spans(self.requests, self.writer.segment_rows, **batch)

    def end_batch(self):
        self.spans = []

    def consume(self, hidden_states):
        # the LM head runs over the scored rows in chunks, so a full-vocabulary distribution never
        # leaves the GPU
        for segment, first, row, count in self.spans:
            for offset in range(0, count, self.chunk_rows):
                width = min(self.chunk_rows, count - offset)
                logits = self.compute_logits(hidden_states[row + offset : row + offset + width])
                self.writer.add(segment, first + offset, top_m_targets(logits, self.top_m))

    def remove_hooks(self):
        pass


class WorkerExtension:
    """Records one pass inside the vLLM worker, driven over collective RPC."""

    def start_capture(self, *, requests, output_path, expected_coordinates, rows_per_file, record):
        writer = ActivationCaptureWriter(
            output_path,
            expected_coordinates=expected_coordinates,
            rows_per_file=rows_per_file,
            record=record,
        )
        session = ActivationCaptureSession(writer, requests)
        session.register_blocks(moe_blocks(self.model_runner, record["layers"]))
        self._recording = (session, writer, instrument_forward(self.model_runner, session))

    def start_targets(
        self,
        *,
        requests,
        output_path,
        segments,
        segment_rows,
        rows_per_file,
        record,
        top_m,
        chunk_rows,
    ):
        runner = self.model_runner
        writer = SegmentWriter(
            output_path,
            segments=segments,
            segment_rows=segment_rows,
            rows_per_file=rows_per_file,
            record=record,
        )
        session = TeacherTargetSession(
            writer, runner.model.compute_logits, requests, chunk_rows=chunk_rows, top_m=top_m
        )
        self._recording = (session, writer, instrument_forward(runner, session))

    def finish_pass(self):
        self._recording[1].finish()

    def release_pass(self):
        if hasattr(self, "_recording"):
            session, _writer, original_forward = self._recording
            session.remove_hooks()
            self.model_runner._model_forward = original_forward
            del self._recording


def run_offline_pass(config, requests, start, **kwargs):
    # The recording hooks live in the worker, so the engine must stay in this process, which also
    # lets the payloads pass as Python objects. vLLM reads this variable when the engine starts.
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") not in (None, "0"):
        raise RuntimeError("VLLM_ENABLE_V1_MULTIPROCESSING must be unset or zero")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # A segment's prompt is at most 8,192 tokens, and vLLM counts the one token it generates.
    # Moonlight has 8,192 positions, which vLLM refuses below `max_model_len` 8,193 without this.
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    # The MoE kernel each model's reported captures and caches ran: vLLM's own choice at one engine
    # (FlashInfer's CUTLASS kernel) for Qwen3, its Triton kernel for the other models, which their
    # environment pinned as the served models do (`engine.py`). The two round differently.
    model = json.loads((Path(config["model_export"]) / "config.json").read_text())
    if model["model_type"] == "qwen3_moe":
        os.environ.pop("VLLM_USE_FLASHINFER_MOE_FP16", None)
    else:
        os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
    llm = LLM(
        model=str(config["model_export"]),
        tokenizer=str(config["model_export"]),
        dtype="bfloat16",
        quantization=None,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        enable_prefix_caching=False,
        speculative_config=None,
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enable_chunked_prefill=config["enable_chunked_prefill"],
        trust_remote_code=False,
        disable_log_stats=True,
        skip_tokenizer_init=True,
        generation_config="vllm",
        async_scheduling=False,
        enable_dbo=False,
        seed=0,
        worker_extension_cls="echomoe.training.data.teacher.WorkerExtension",
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
    request_ids = llm.enqueue([prompt for prompt, _ in requests], sampling, use_tqdm=False)
    payloads = {request_id: payload for request_id, (_, payload) in zip(request_ids, requests)}
    llm.collective_rpc(start, kwargs=kwargs | {"requests": payloads})
    try:
        llm.wait_for_completion(use_tqdm=False)
        llm.collective_rpc("finish_pass")
    finally:
        llm.collective_rpc("release_pass")


def capture_activations(config_path, part_index):
    config = load_config(config_path, "capture")
    output = Path(config["output_root"]) / f"part-{part_index:05d}"
    if output.exists():
        return
    # draw the whole selection, then take this part's documents
    dataset = load_tokenized_dataset(config["tokenized_dataset"])
    layouts = {split: load_layout(config[f"{split}_layout"]) for split in SPLITS}
    selection = select_coordinates(
        layouts["fit"],
        layouts["validation"],
        fit_rows=config["fit_rows"],
        validation_rows=config["validation_rows"],
        seed=config["selection_seed"],
        max_tokens_per_document=config["max_tokens_per_document"],
    )
    local = partition_coordinates(
        selection, index=part_index, count=config["part_count"], seed=config["selection_seed"]
    )
    # one shifted prompt per layout segment holding selected tokens, with their positions in it
    by_document = defaultdict(list)
    for coordinate in local:
        by_document[coordinate.split, coordinate.dataset_row].append(coordinate)
    requests = []
    for split, layout in layouts.items():
        arrays = (layout.dataset_rows, layout.token_starts, layout.raw_token_lengths)
        for dataset_row, start, length in zip(*(array.tolist() for array in arrays)):
            stop = start + length - 1
            chosen = [
                coordinate
                for coordinate in by_document[split, dataset_row]
                if start <= coordinate.token_index < stop
            ]
            if chosen:
                prompt = {"prompt_token_ids": dataset[dataset_row]["input_ids"][start:stop]}
                requests.append(
                    (prompt, {coordinate.token_index - start: coordinate for coordinate in chosen})
                )
    # one offline pass of the teacher, recorded inside its worker. The layers are MoE ordinals
    export = Path(config["model_export"])
    model = json.loads((export / "config.json").read_text())
    decoder_layers, experts = moe_layout(model)
    gemma = model["model_type"] == "gemma4_text"
    record = {
        "metadata": {"part_index": part_index, "part_count": config["part_count"]},
        "layers": list(range(len(decoder_layers))),
        "hidden_size": model["hidden_size"],
        "num_experts": experts,
        "topk": model["top_k_experts" if gemma else "num_experts_per_tok"],
        # Gemma 4 always renormalizes
        "renormalize": gemma or bool(model["norm_topk_prob"]),
    }
    weight_map = json.loads((export / "model.safetensors.index.json").read_text())["weight_map"]

    def per_moe_layer(name):
        values = []
        for layer in decoder_layers:
            with safe_open(export / weight_map[name.format(layer)], framework="pt") as handle:
                values.append(handle.get_tensor(name.format(layer)).double().tolist())
        return values

    # the fit routes as DeepSeek-V3 and Nemotron-H do: the top-k of sigmoid score plus correction
    # bias
    biases = {
        "deepseek_v3": "model.layers.{}.mlp.gate.e_score_correction_bias",
        "nemotron_h": "backbone.layers.{}.mixer.gate.e_score_correction_bias",
    }.get(model["model_type"])
    if biases is not None:
        record["router"] = {
            "routed_scaling_factor": model["routed_scaling_factor"],
            "expert_biases": per_moe_layer(biases),
        }
    if gemma:
        # and as Gemma 4 does, each renormalized weight times its expert's scale
        record["expert_scales"] = per_moe_layer("model.layers.{}.router.per_expert_scale")
    run_offline_pass(
        config,
        requests,
        "start_capture",
        output_path=output,
        expected_coordinates=local,
        rows_per_file=config["rows_per_file"],
        record=record,
    )
    write_meta(output, inputs={"config": config_path, "model_export": config["model_export"]})


def cache_targets(config_path, part_index):
    config = load_config(config_path, "targets")
    output = Path(config["output_root"]) / f"part-{part_index:05d}"
    dataset = load_tokenized_dataset(config["tokenized_dataset"])
    layout = load_layout(config["layout"])

    # the segments of this part that are not cached yet
    planned = plan_part(config, part_index, layout)
    published = covered_segments(output)
    pending = [int(segment) for segment in planned if segment not in published]
    if not pending:
        return

    # each pending segment is one shifted prompt whose every position is scored
    requests = []
    for segment in pending:
        start = int(layout.token_starts[segment])
        stop = start + int(layout.raw_token_lengths[segment]) - 1
        ids = dataset[int(layout.dataset_rows[segment])]["input_ids"]
        requests.append(({"prompt_token_ids": ids[start:stop]}, segment))

    # one offline pass of the teacher, recorded inside its worker
    run_offline_pass(
        config,
        requests,
        "start_targets",
        output_path=output,
        segments=planned.tolist(),
        segment_rows=(layout.raw_token_lengths[planned].astype(np.int64) - 1).tolist(),
        rows_per_file=config["rows_per_file"],
        record={"temperature": 1.0, "top_m": config["top_m"]},
        top_m=config["top_m"],
        chunk_rows=config["logits_chunk_rows"],
    )
    write_meta(output, inputs={"config": config_path, "model_export": config["model_export"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("capture", "targets"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--part-index", type=int, required=True)
    args = parser.parse_args()
    if args.stage == "capture":
        capture_activations(args.config, args.part_index)
    else:
        cache_targets(args.config, args.part_index)
