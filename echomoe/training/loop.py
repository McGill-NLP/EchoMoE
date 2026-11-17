"""Distill one compressed student against the teacher's cached targets.

`train` reads top to bottom: the data and the plan of every update, the student, the optimizer,
validation before the first update, then the updates, each written out as four numbered steps.
The helpers follow, in the order `train` uses them.
"""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from megatron.bridge.training.config import OptimizerConfigOverrideProvider, SchedulerConfig
from megatron.bridge.training.optim import setup_optimizer
from megatron.core import parallel_state
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer.optimizer_config import (
    OptimizerConfig as MegatronOptimizerConfig,
    ParamKey,
)
from megatron.core.packed_seq_params import PackedSeqParams

from echomoe.codec import (
    COMBINE_COMPRESSOR_MODULE_NAME,
    COMBINE_CORRECTION_MODULE_NAME,
    DISPATCH_COMPRESSOR_MODULE_NAME,
    DISPATCH_REFERENCE_MODULE_NAME,
)
from echomoe.lsh import LshConfig
from echomoe.runs import RunDirectory, atomic_write_bytes

from . import checkpoint
from .config import (
    SOURCES,
    OptimizerConfig,
    SchedulerConfig as RecipeSchedulerConfig,
    TrainingConfig,
    config_document,
)
from .data.batches import (
    SEQUENCE_KEYS,
    assemble_packed_batch,
    build_annealed_schedule,
    concatenate_packed_batches,
    derive_pack_order,
    packed_batch_to_device,
    source_queue,
)
from .data.layout import (
    ALIGNMENT,
    SEQUENCE_LENGTH,
    load_layout,
    load_tokenized_dataset,
    pad_token_id,
)
from .data.targets import load_target_cache
from .deepep import lsh_volume
from .loss import fused_loss_terms, local_objective, position_terms
from .model import COMPENSATION_MODULE_NAME, build_student, gather_parameters, init_distributed

# The learning-rate globs of the optimizer groups. The dense backbone takes the base rate. A
# folded student's down projection is the per-expert W_out,i (`factors`), which lives under
# `.experts.` beside its W_in,i (`readers`), so there the expert glob narrows to fc1.
GROUP_PATTERNS = {
    "router": "*.router.*",
    "experts": "*.experts.*",
    "dispatch_compressor": f"*.{DISPATCH_COMPRESSOR_MODULE_NAME}.*",
    "combine_compressor": f"*.{COMBINE_COMPRESSOR_MODULE_NAME}.*",
    "dispatch_reference": f"*.{DISPATCH_REFERENCE_MODULE_NAME}.*",
    "combine_correction": f"*.{COMBINE_CORRECTION_MODULE_NAME}.*",
    "lsh_compensation": f"*.{COMPENSATION_MODULE_NAME}.*",
}
FOLDED_GROUP_PATTERNS = {
    **GROUP_PATTERNS,
    "experts": "*.experts.linear_fc1.*",
    "expert_factor": "*.linear_fc2.factors",
}


def train(config: TrainingConfig) -> None:
    # the process group, every generator's seed
    init_distributed(config.runtime.distributed_timeout_seconds)
    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() != config.parallelism.expected_world_size:
        raise RuntimeError(f"the recipe expects {config.parallelism.expected_world_size} ranks")
    set_seeds(config.run.seed)
    torch.set_float32_matmul_precision(config.runtime.matmul_precision)

    # the two sources, the held-out sets (both sources' validation layouts), and the plan: the
    # source and pack of every slot of every update
    sources = [load_source(config, name) for name in SOURCES]
    pad_id = pad_token_id(Path(config.models.teacher) / "hf")
    held_out = [
        ValidationSource(
            source.name,
            source.kind,
            source.bank,
            source.validation_layout,
            source.validation_targets,
            config.evaluation.batches,
        )
        for source in sources
    ]
    mixing = config.data.mixing
    slot_source, slot_pack = build_annealed_schedule(
        sources[0].queue,
        sources[1].queue,
        packs_per_update=config.batch.global_batch_size,
        mix_seed=config.data.mix_seed,
        anneal_start=mixing.anneal_start,
        end_dolci_fraction=mixing.end_dolci_fraction,
    )
    total_steps = len(slot_source)

    # the student, from the resume checkpoint when there is one
    out_dir = Path(config.run.output_dir)
    resume = config.checkpoint.resume_from
    if resume is not None:
        resume = checkpoint.latest(out_dir) if resume == "latest" else Path(resume)
    chunks, attachment, fold = build_student(config, resume=resume)
    is_lsh = isinstance(config.compressor, LshConfig)
    (student,) = chunks
    head = next(
        module for name, module in student.named_modules() if name.endswith("output_layer")
    )
    mamba = any(type(module).__name__ == "HybridStack" for module in student.modules())
    # Gemma 4 ties its output layer to the input embedding and softcaps its logits
    language_model = next(m for m in student.modules() if getattr(m, "output_layer", None) is head)
    logit_softcap = getattr(head.config, "final_logit_softcapping", None)
    data_parallel_rank = parallel_state.get_data_parallel_rank(with_context_parallel=False)
    data_parallel_size = parallel_state.get_data_parallel_world_size(with_context_parallel=False)
    group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    context_parallel_size = config.parallelism.context_parallel_size

    # the optimizer and its schedule, restored on a resume
    optimizer, scheduler = build_optimizer(
        chunks,
        config.optimizer,
        config.scheduler,
        total_steps=total_steps,
        use_distributed_optimizer=config.parallelism.ddp.use_distributed_optimizer,
        patterns=FOLDED_GROUP_PATTERNS if config.models.student.folded else GROUP_PATTERNS,
    )
    start = (
        0 if resume is None else checkpoint.load(resume, optimizer, scheduler)["optimizer_step"]
    )

    # when checkpoints are saved and validation runs: every interval, at the update each source
    # runs out and at the last one (the end, or the bounded stop). Validation also runs at the end
    # of warmup, which the scheduler has checked comes before the end.
    stop = config.run.stop_after_step
    last = total_steps if stop is None else min(stop, total_steps)
    events = {int(np.flatnonzero((slot_source == index).any(axis=1))[-1]) + 1 for index in (0, 1)}
    events.add(last)
    interval = config.checkpoint.save_interval_steps
    save_steps = set(range(interval, total_steps + 1, interval)) | events
    interval = config.evaluation.interval_steps
    validation_steps = set(range(interval, total_steps + 1, interval)) | events
    if config.scheduler.warmup_steps > 0:
        validation_steps.add(config.scheduler.warmup_steps)

    # the run directory (rank 0). A resume truncates the metrics to its checkpoint.
    document = config_document(config)
    validate_now = resume is None and config.evaluation.at_start
    if rank == 0:
        if resume is None:
            inputs = {"teacher": config.models.teacher}
            if not is_lsh:
                inputs["initialization"] = config.models.student.compressor_initialization
            run = RunDirectory.create(
                out_dir,
                config=document,
                inputs=inputs,
                planned_updates=total_steps,
                planned_packs=int((slot_pack >= 0).sum()),
                planned_prediction_tokens=sum(
                    scored_tokens(source.train_layout, source.queue) for source in sources
                ),
                fold=fold,
            )
        else:
            run = RunDirectory.resume(out_dir, config=document)
            recorded = resume_metrics(out_dir, start)
            validate_now = start in validation_steps and not recorded
    decision = [validate_now]
    torch.distributed.broadcast_object_list(decision, src=0)
    metrics = Metrics(out_dir, config.logging.wandb, config=document, enabled=rank == 0)

    def validate(step: int) -> None:
        # A run's first forward is here, so gated bias decoding (`biasgate`) measures its code
        # scale here.
        gather_parameters(chunks)
        student.eval()
        record = {"step": step}
        with torch.no_grad():
            for source in held_out:
                scores = source_metrics(
                    source,
                    student,
                    config=config,
                    pad_id=pad_id,
                    rank=data_parallel_rank,
                    ranks=data_parallel_size,
                    group=group,
                    mamba=mamba,
                )
                record |= {
                    f"evaluation/{source.name}/{metric}": value for metric, value in scores.items()
                }
        student.train()
        # references published here must not be read by the next training forward
        attachment.reset_references()
        if is_lsh:
            record |= lsh_volume(attachment, "evaluation/lsh")
        metrics.log(record)

    if decision[0]:
        validate(start)

    # the updates, their host batches read ahead in one forked worker process (a thread holding
    # the GIL would slow the launch-bound step by about the read it hides)
    reader = iter(
        torch.utils.data.DataLoader(
            UpdateReader(
                sources,
                slot_source,
                slot_pack,
                range(start + 1, last + 1),
                top_m=config.loss.target_top_m,
                pad_id=pad_id,
                accumulation=config.batch.gradient_accumulation_steps,
                micro_batch=config.batch.micro_batch_size,
                rank=data_parallel_rank,
                ranks=data_parallel_size,
            ),
            batch_size=None,
            num_workers=1,
            prefetch_factor=2,
            multiprocessing_context="fork",
            persistent_workers=False,
        )
    )
    step = start
    for step in range(start + 1, last + 1):
        started = time.perf_counter()
        for chunk in chunks:
            chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # 1. this rank's packs of the update, with their cached teacher targets
        batches = [
            to_device(batch, context_parallel_size=context_parallel_size, mamba=mamba)
            for batch in next(reader)
        ]
        # the LM term counts only in an update that holds Dolci packs
        lm_weight = config.loss.lm_weight if (slot_source[step - 1] == 1).any() else 0.0

        # 2. the global token counts the loss is divided by, per micro-batch. A micro-batch with
        # no prediction on any rank is skipped.
        gather_parameters(chunks)
        counts = []
        for batch in batches:
            count = torch.stack(
                (
                    batch["prediction_mask"].sum(dtype=torch.int64),
                    batch["lm_mask"].sum(dtype=torch.int64),
                )
            )
            torch.distributed.all_reduce(count, group=group)
            counts.append(count)
        kl_tokens, lm_tokens = (int(value) for value in torch.stack(counts).sum(dim=0).tolist())
        scoring = [index for index, count in enumerate(counts) if int(count[0]) > 0]

        # 3. forward to the hidden states, the loss a chunk of positions at a time, backward
        kl_sum = lm_sum = 0.0
        for index in scoring:
            batch = batches[index]
            with accumulating(chunks, last=index == scoring[-1]):
                with head_returns_hidden_states(head):
                    hidden_states = student(**logit_kwargs(batch), fp32_output=False)
                kl, lm = fused_loss_terms(
                    hidden_states,
                    language_model.shared_embedding_or_output_weight()
                    if head.weight is None
                    else head.weight,
                    batch["labels"],
                    batch["prediction_mask"],
                    batch["lm_mask"],
                    target_ids=batch["target_ids"],
                    target_logprobs=batch["target_logprobs"],
                    target_tail_logprob=batch["target_tail_logprob"],
                    position_chunk_size=config.loss.position_chunk_size,
                    accumulation_dtype=torch.float32,
                    compile_regions=config.runtime.compile_regions,
                    logit_softcap=logit_softcap,
                )
                local_objective(
                    kl,
                    lm,
                    kl_tokens=kl_tokens,
                    lm_tokens=lm_tokens,
                    kl_weight=config.loss.kl_weight,
                    lm_weight=lm_weight,
                ).backward()
            kl_sum += float(kl.detach())
            lm_sum += float(lm.detach())
        finalize_model_grads(chunks, num_tokens=None)

        # 4. The scheduler advances before the update, so update j applies the rate of step j:
        # warmup ends exactly at the last warmup update and the final update lands on the floor.
        scheduler.step(1)
        rate = max(float(param_group["lr"]) for param_group in optimizer.param_groups)
        updated, grad_norm, _ = optimizer.step()
        if not updated:
            raise RuntimeError(f"the optimizer skipped update {step}")

        # the update's means over every rank, then the checkpoint and validation this step is due
        sums = torch.tensor([kl_sum, lm_sum], dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(sums, group=group)
        kl_mean = float(sums[0]) / kl_tokens
        lm_mean = float(sums[1]) / lm_tokens if lm_tokens else 0.0
        record = {
            "step": step,
            "train/forward_kl_mean": kl_mean,
            "train/response_nll_mean": lm_mean,
            "train/total_objective": config.loss.kl_weight * kl_mean + lm_weight * lm_mean,
            "train/prediction_tokens": kl_tokens,
            "optimizer/learning_rate": rate,  # the codec groups' (the highest)
            "optimizer/global_gradient_norm": float(grad_norm),
            "throughput/step_time_seconds": time.perf_counter() - started,
        }
        if is_lsh:
            record |= lsh_volume(attachment, "lsh")
        metrics.log(record)
        if step in save_steps:
            gather_parameters(chunks)
            # the save's collectives need the device memory the allocator holds as cache
            torch.cuda.empty_cache()
            checkpoint.save(
                out_dir / "checkpoints" / f"step-{step:07d}",
                chunks,
                optimizer,
                scheduler,
                step=step,
                progress={},
                keep_last=config.checkpoint.keep_last,
                codec_only=not (config.models.student.folded or is_lsh),
            )
        if step in validation_steps:
            validate(step)
    del reader

    # the result (rank 0)
    if rank == 0:
        results = {
            "status": "completed" if step == total_steps else "stopped",
            "completed_updates": step,
            "planned_updates": total_steps,
        }
        run.write_results(results)
    torch.distributed.barrier()


# --- the data of an update -----------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    name: str
    bank: Any
    train_layout: Any
    validation_layout: Any
    train_targets: Any
    validation_targets: Any
    queue: np.ndarray  # the pack ids the run consumes, in order

    @property
    def kind(self) -> str:
        return "sft" if self.name == "dolci" else "pretrain"


def load_source(config: TrainingConfig, name: str) -> Source:
    spec = getattr(config.data.sources, name)
    train, validation = load_layout(spec.train_layout), load_layout(spec.validation_layout)
    orders = derive_pack_order(config.data.order_seed, spec.passes, len(train))
    return Source(
        name=name,
        bank=load_tokenized_dataset(spec.tokenized_dataset),
        train_layout=train,
        validation_layout=validation,
        train_targets=load_target_cache(spec.train_teacher_targets, train),
        validation_targets=load_target_cache(spec.validation_teacher_targets, validation),
        queue=source_queue(train, orders, token_budget=spec.token_budget),
    )


class UpdateReader(torch.utils.data.IterableDataset):
    def __init__(self, sources, slot_source, slot_pack, steps, **settings):
        self.sources, self.slot_source, self.slot_pack = sources, slot_source, slot_pack
        self.steps, self.settings = steps, settings

    def __iter__(self):
        for step in self.steps:
            yield read_update(
                self.sources, self.slot_source[step - 1], self.slot_pack[step - 1], **self.settings
            )


def read_update(
    sources, slot_source, slot_pack, *, top_m, pad_id, accumulation, micro_batch, rank, ranks
):
    batches = []
    for offset in range(accumulation):
        packs = []
        for index in range(micro_batch):
            slot = (offset * micro_batch + index) * ranks + rank
            # a filler slot (-1) is an empty pack of the first source's layout
            source = sources[max(int(slot_source[slot]), 0)]
            pack = int(slot_pack[slot])
            batch = assemble_packed_batch(
                source.bank,
                source.train_layout,
                pack_id=None if pack < 0 else pack,
                pad_id=pad_id,
                sequence_length=SEQUENCE_LENGTH,
                alignment=ALIGNMENT,
            )
            batch = attach_teacher_targets(
                batch, source.train_targets, source.train_layout, top_m=top_m
            )
            batch["lm_mask"] = (
                batch["lm_mask"] & batch["prediction_mask"]
                if source.kind == "sft"
                else torch.zeros_like(batch["lm_mask"])
            )
            packs.append(batch)
        batches.append(concatenate_packed_batches(packs))
    return batches


def attach_teacher_targets(batch, cache, layout, *, top_m):
    # A cached row runs over the pack's segments with no alignment gaps, so the rows land 1:1 on
    # the prediction mask. Unscored positions carry an empty tail, which contributes nothing.
    prediction = batch["prediction_mask"]
    pack_id = int(batch["pack_id"])
    cached = None
    if pack_id >= 0:
        start, stop = int(layout.pack_offsets[pack_id]), int(layout.pack_offsets[pack_id + 1])
        cached = cache.read(range(start, stop))
    logprob_dtype = cached["logprobs"].dtype if cached is not None else torch.float32
    ids = torch.zeros((*prediction.shape, top_m), dtype=torch.int32)
    logprobs = torch.zeros((*prediction.shape, top_m), dtype=logprob_dtype)
    tail = torch.full(prediction.shape, -math.inf, dtype=torch.float32)
    if cached is not None:
        ids[prediction] = cached["ids"]
        logprobs[prediction] = cached["logprobs"]
        tail[prediction] = cached["tail"]
    return batch | {"target_ids": ids, "target_logprobs": logprobs, "target_tail_logprob": tail}


def to_device(batch, *, context_parallel_size, mamba=False):
    return partition_thd_batch(
        packed_batch_to_device(batch, "cuda"),
        context_parallel_size=context_parallel_size,
        mamba=mamba,
    )


def thd_batch_on_this_cp_rank(batch, cu_seqlens, cu_seqlens_padded, max_seqlen):
    # megatron-core 0.17's get_thd_batch_on_this_cp_rank, which 0.19 removed
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=int(max_seqlen[0].item()),
        max_seqlen_kv=int(max_seqlen[0].item()),
    )
    context_parallel_ranks = parallel_state.get_context_parallel_world_size()
    if context_parallel_ranks > 1:
        # importable only after transformer_engine, which megatron imports first
        import transformer_engine_torch

        index = transformer_engine_torch.thd_get_partitioned_indices(
            cu_seqlens_padded,
            batch["tokens"].size(1),
            context_parallel_ranks,
            parallel_state.get_context_parallel_rank(),
        )
        for key, data in batch.items():
            if key not in {"attention_mask", "cu_seqlens", "cu_seqlens_padded", "max_seqlen"}:
                batch[key] = data.index_select(1, index)
    return batch, packed_seq_params


def partition_thd_batch(batch, *, context_parallel_size, mamba=False):
    padded = batch["cu_seqlens_padded"]
    if int((padded[1:] - padded[:-1]).remainder(2 * context_parallel_size).count_nonzero()):
        raise ValueError("every padded segment length must be divisible by 2 * CP")
    keys = [name for name in SEQUENCE_KEYS if name in batch]
    payload = {name: batch[name] for name in keys} | {
        "attention_mask": None,
        "cu_seqlens": batch["cu_seqlens_unpadded"],
        "cu_seqlens_padded": padded,
        "max_seqlen": batch["max_seqlen"],
    }
    sliced, packed_seq_params = thd_batch_on_this_cp_rank(
        payload, batch["cu_seqlens_unpadded"], padded, batch["max_seqlen"]
    )
    # Transformer Engine otherwise makes this comparison on the device in every attention call, a
    # host synchronization per layer per forward.
    packed_seq_params.pad_between_seqs = not torch.equal(
        padded[:-1], batch["cu_seqlens_unpadded"][:-1]
    )
    if mamba:
        # Mamba layers (Nemotron-H) keep the packed documents apart by each token's sequence
        # index, which the params derive from the padded offsets once they know the pack length.
        assert context_parallel_size == 1, "the sequence index is built for one CP rank"
        packed_seq_params.total_tokens = int(sliced["tokens"].shape[-1])
        packed_seq_params.__post_init__()
    output = dict(batch) | {name: sliced[name] for name in keys}
    output["packed_seq_params"] = packed_seq_params
    return output


def logit_kwargs(batch):
    return {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": None,
        "labels": None,
        "padding_mask": batch["padding_mask"],
        "packed_seq_params": batch["packed_seq_params"],
    }


# --- the forward and backward of an update -------------------------------------------------------


def accumulating(chunks, *, last: bool):
    """Delay the data-parallel gradient reduction to the last micro-batch of an update."""
    stack = ExitStack()
    if not last:
        for chunk in chunks:
            stack.enter_context(chunk.no_sync())
    return stack


@contextmanager
def head_returns_hidden_states(head):
    # The module is still called, so the distributed optimizer's pre-hook still finishes the
    # weight's all-gather.
    head.forward = lambda input_, weight=None, runtime_gather_output=None: (input_, None)
    try:
        yield
    finally:
        del head.forward


# --- the optimizer -------------------------------------------------------------------------------


@dataclass
class GroupLearningRates(OptimizerConfigOverrideProvider):
    groups: dict
    patterns: dict

    def build_config_overrides(self, context):
        overrides = super().build_config_overrides(context) or {}
        for name, pattern in self.patterns.items():
            group = self.groups.get(name)
            if group is not None:
                overrides[ParamKey(name=pattern)] = {
                    "max_lr": group.learning_rate,
                    "min_lr": group.minimum_learning_rate,
                }
        return overrides


def build_optimizer(
    chunks,
    optimizer: OptimizerConfig,
    scheduler: RecipeSchedulerConfig,
    *,
    total_steps: int,
    use_distributed_optimizer: bool,
    patterns: dict | None,
):
    """Adam over the parameters that take gradient, with linear warmup and a cosine decay to each
    group's floor at `total_steps`. The base rate is the dense backbone's."""
    dense = optimizer.parameter_groups["dense_backbone"]
    optimizer_config = MegatronOptimizerConfig(
        optimizer="adam",
        lr=dense.learning_rate,
        min_lr=dense.minimum_learning_rate,
        weight_decay=optimizer.weight_decay,
        adam_beta1=optimizer.beta1,
        adam_beta2=optimizer.beta2,
        adam_eps=optimizer.epsilon,
        decoupled_weight_decay=optimizer.decoupled_weight_decay,
        bf16=True,
        use_distributed_optimizer=use_distributed_optimizer,
        clip_grad=optimizer.gradient_clip,
    )
    scheduler_config = SchedulerConfig(
        lr_decay_style=scheduler.family,
        lr_warmup_iters=scheduler.warmup_steps,
        lr_decay_iters=total_steps,
        start_weight_decay=optimizer.weight_decay,
        end_weight_decay=optimizer.weight_decay,
        weight_decay_incr_style="constant",
    )
    # The bridge fills these only in ConfigContainer.finalize(), in sample units. The scheduler
    # steps once per update here, so they are set in update units.
    scheduler_config.lr_decay_steps = total_steps
    scheduler_config.lr_warmup_steps = scheduler.warmup_steps
    scheduler_config.wd_incr_steps = total_steps
    overrides = None
    if patterns is not None:
        overrides = GroupLearningRates(optimizer.parameter_groups, patterns)
    return setup_optimizer(
        optimizer_config,
        scheduler_config,
        chunks,
        optimizer_config_override_provider=overrides,
    )


# --- validation ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationSource:
    name: str
    kind: str  # "pretrain" scores every prediction, "sft" the responses
    bank: Any
    layout: Any
    targets: Any
    batches: int | str  # packs per rank, or "all"


def source_metrics(source, student, *, config, pad_id, rank, ranks, group, mamba):
    rounds = max(1, math.ceil(len(source.layout) / ranks))
    if source.batches != "all":
        rounds = min(rounds, source.batches)
    sums = []
    for index in range(rounds):
        candidate = index * ranks + rank
        batch = assemble_packed_batch(
            source.bank,
            source.layout,
            pack_id=candidate if candidate < len(source.layout) else None,
            pad_id=pad_id,
            sequence_length=SEQUENCE_LENGTH,
            alignment=ALIGNMENT,
        )
        batch = attach_teacher_targets(
            batch, source.targets, source.layout, top_m=config.loss.target_top_m
        )
        batch = to_device(
            batch, context_parallel_size=config.parallelism.context_parallel_size, mamba=mamba
        )
        logits = student(**logit_kwargs(batch))
        sums.append(
            validation_sums(
                logits,
                batch,
                kind=source.kind,
                position_chunk_size=config.loss.position_chunk_size,
            )
        )
    totals = torch.stack(sums).sum(dim=0)
    torch.distributed.all_reduce(totals, group=group)
    kl_sum, predictions, scope_kl_sum, nll_sum, targets = totals.tolist()
    scope = "token" if source.kind == "pretrain" else "response"
    record = {
        "teacher_student_kl_mean": kl_sum / predictions,
        f"student_{scope}_nll": nll_sum / targets,
    }
    if source.kind == "sft":
        record["response_kl_mean"] = scope_kl_sum / targets
    return record


def validation_sums(logits, batch, *, kind, position_chunk_size):
    kl, cross_entropy = position_terms(
        logits,
        batch["labels"],
        target_ids=batch["target_ids"],
        target_logprobs=batch["target_logprobs"],
        target_tail_logprob=batch["target_tail_logprob"],
        position_chunk_size=position_chunk_size,
        accumulation_dtype=torch.float32,
    )
    prediction = batch["prediction_mask"]
    scope = prediction if kind == "pretrain" else batch["lm_mask"] & prediction
    kl, cross_entropy = kl.to(dtype=torch.float64), cross_entropy.to(dtype=torch.float64)
    return torch.stack(
        (
            (kl * prediction.to(dtype=torch.float64)).sum(),
            prediction.sum(dtype=torch.float64),
            (kl * scope.to(dtype=torch.float64)).sum(),
            (cross_entropy * scope.to(dtype=torch.float64)).sum(),
            scope.sum(dtype=torch.float64),
        )
    )


# --- the run's records ---------------------------------------------------------------------------


class Metrics:
    """`metrics.jsonl` on rank 0, mirrored to W&B when the recipe asks for it."""

    def __init__(
        self, directory: Path, wandb_config, *, config: dict, enabled: bool, name=None, tags=()
    ) -> None:
        self.enabled = enabled
        self.run = None
        if not enabled:
            return
        self.file = (directory / "metrics.jsonl").open("a")
        if wandb_config.mode != "disabled":
            try:
                self.run = wandb.init(
                    project=wandb_config.project,
                    group=wandb_config.group,
                    job_type=wandb_config.job_type,
                    name=name or directory.name,
                    tags=list(tags) or None,
                    mode=wandb_config.mode,
                    dir=str(directory),
                    config=config,
                )
            except Exception as error:  # a mirror only: the local record is the result
                print(f"W&B disabled: {error}")

    def log(self, record: dict) -> None:
        if not self.enabled:
            return
        line = json.dumps(record, allow_nan=True)
        self.file.write(line + "\n")
        self.file.flush()
        print(line, flush=True)
        if self.run is not None:
            self.run.log({k: v for k, v in record.items() if k != "step"}, step=record["step"])


def resume_metrics(directory: Path, step: int) -> bool:
    """Keep the records up to `step` and say whether its validation is among them."""
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    kept = [record for record in records if record["step"] <= step]
    atomic_write_bytes(
        directory / "metrics.jsonl", "".join(json.dumps(record) + "\n" for record in kept).encode()
    )
    return any(
        record["step"] == step and any(key.startswith("evaluation/") for key in record)
        for record in kept
    )


def scored_tokens(layout, packs: np.ndarray) -> int:
    offsets = np.asarray(layout.pack_offsets, dtype=np.int64)
    lengths = np.asarray(layout.raw_token_lengths, dtype=np.int64)
    return int(np.add.reduceat(lengths - 1, offsets[:-1])[packs].sum())


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - the module-level state
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
