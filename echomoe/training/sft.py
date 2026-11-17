"""Supervised fine-tuning of the teacher: the base model with the instruct chat template
(`echomoe.training.base_model`), on Dolci, with response-token cross entropy and nothing else.

    torchrun ... -m echomoe.training.sft --config configs/training/qwen3-30b-a3b/sft.yaml \
        [--resume-from latest] [--stop-after-step K]

Unlike the distillation trainer, the scheduler steps after the optimizer, so update 1 runs at a
learning rate of zero.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

# The CUDA allocator and Transformer Engine read these when torch loads: expandable segments, and
# no forced attention backend, so Transformer Engine picks one.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
for _name in ("NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN"):
    os.environ.pop(_name, None)

import torch  # noqa: E402
import yaml  # noqa: E402
from megatron.core import parallel_state  # noqa: E402
from megatron.core.distributed import finalize_model_grads  # noqa: E402

from echomoe.runs import RunDirectory  # noqa: E402

from . import checkpoint  # noqa: E402
from .config import (  # noqa: E402
    DDPConfig,
    OptimizerConfig,
    ParallelismConfig,
    ParameterGroup,
    RuntimeConfig,
    SchedulerConfig,
    WandbConfig,
)
from .data.batches import (  # noqa: E402
    assemble_packed_batch,
    concatenate_packed_batches,
    derive_pack_order,
    sft_pack_id,
    sft_total_microbatches,
)
from .data.layout import (  # noqa: E402
    ALIGNMENT,
    load_layout,
    load_tokenized_dataset,
    pad_token_id,
)
from .deepep import indices_to_multihot  # noqa: E402
from .loop import (  # noqa: E402
    Metrics,
    accumulating,
    build_optimizer,
    resume_metrics,
    set_seeds,
    to_device,
)
from .model import (  # noqa: E402
    build_model,
    forward_padding_flag_to_attention,
    gather_parameters,
    init_distributed,
    moe_layers,
)

# The keys each recipe section holds.
RECIPE = {
    "model": None,
    "data": {"tokenized_dataset", "layout", "validation_layout", "order_seed"},
    "parallelism": {
        "expert_parallel_size",
        "context_parallel_size",
        "rank_order",
        "context_parallel_communication",
        "expected_world_size",
        "expected_nodes",
        "gpus_per_node",
    },
    "batch": {"gradient_accumulation_steps", "global_batch_size", "sequence_length"},
    "dispatcher": {"type"},
    "schedule": {
        "epochs",
        "learning_rate",
        "minimum_learning_rate",
        "warmup_steps",
        "weight_decay",
    },
    "optimizer": {"beta1", "beta2", "epsilon", "decoupled_weight_decay"},
    "runtime": {
        "seed",
        "gradient_clip",
        "recompute",
        "recompute_num_layers",
        "apply_rope_fusion",
        "bias_activation_fusion",
        "keep_last_checkpoints",
        "moe_permute_fusion",
        "save_interval_steps",
        "evaluation_interval_steps",
        "output_dir",
        "wandb_mode",
        "wandb_project",
        "wandb_name",
        "wandb_group",
        "wandb_job_type",
        "wandb_tags",
    },
}
MICRO_BATCH_SIZE = 1
TIMEOUT_SECONDS = 600


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of the teacher.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", help='"latest", or a checkpoint directory')
    parser.add_argument("--stop-after-step", type=int, help="end after this update")
    args = parser.parse_args()
    recipe = load_recipe(args.config)
    data, parallel, batch, schedule = (
        recipe[name] for name in ("data", "parallelism", "batch", "schedule")
    )
    runtime, adam = recipe["runtime"], recipe["optimizer"]
    accumulation = batch["gradient_accumulation_steps"]
    context_parallel_size = parallel["context_parallel_size"]
    out_dir = Path(runtime["output_dir"])

    # the process group, every generator's seed
    init_distributed(TIMEOUT_SECONDS)
    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() != parallel["expected_world_size"]:
        raise RuntimeError(f"the recipe expects {parallel['expected_world_size']} ranks")
    set_seeds(runtime["seed"])

    # the token bank, its layouts and the pack order of every epoch
    bank = load_tokenized_dataset(data["tokenized_dataset"])
    pad_id = pad_token_id(recipe["model"])
    train_layout = load_layout(data["layout"])
    validation_layout = load_layout(data["validation_layout"])
    orders = derive_pack_order(data["order_seed"], schedule["epochs"], len(train_layout))
    ranks = parallel["expected_world_size"] // context_parallel_size
    total_steps = (
        sft_total_microbatches(
            orders,
            data_parallel_size=ranks,
            gradient_accumulation_steps=accumulation,
            micro_batch_size=MICRO_BATCH_SIZE,
        )
        // accumulation
    )
    total_packs = sum(len(order) for order in orders)

    # the model, with the settings the recipe leaves at their defaults written out
    parallelism = ParallelismConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=parallel["expert_parallel_size"],
        context_parallel_size=context_parallel_size,
        rank_order=parallel["rank_order"],
        context_parallel_communication=parallel["context_parallel_communication"],
        dispatcher=recipe["dispatcher"]["type"],
        expected_world_size=parallel["expected_world_size"],
        expected_nodes=parallel["expected_nodes"],
        gpus_per_node=parallel["gpus_per_node"],
        ddp=DDPConfig(
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=False,
            bucket_size=200_000_000,
            pad_buckets_for_high_nccl_busbw=True,
        ),
    )
    chunks = build_model(
        recipe["model"],
        parallelism,
        RuntimeConfig(
            recompute=runtime["recompute"],
            recompute_num_layers=runtime["recompute_num_layers"],
            attention_backend="fused",
            distributed_timeout_seconds=TIMEOUT_SECONDS,
            apply_rope_fusion=runtime["apply_rope_fusion"],
            bias_activation_fusion=runtime["bias_activation_fusion"],
            moe_router_fusion=True,
            moe_grouped_gemm=True,
            masked_softmax_fusion=True,
            moe_permute_fusion=runtime["moe_permute_fusion"],
            compile_regions=False,
        ),
        sequence_length=batch["sequence_length"],
        seed=runtime["seed"],
        moe_z_loss_coeff=0.0,
    )
    optimizer, scheduler = build_optimizer(
        chunks,
        OptimizerConfig(
            beta1=adam["beta1"],
            beta2=adam["beta2"],
            epsilon=adam["epsilon"],
            weight_decay=schedule["weight_decay"],
            gradient_clip=runtime["gradient_clip"],
            decoupled_weight_decay=adam["decoupled_weight_decay"],
            parameter_groups={
                "dense_backbone": ParameterGroup(
                    schedule["learning_rate"], schedule["minimum_learning_rate"]
                )
            },
        ),
        SchedulerConfig(family="cosine", warmup_steps=schedule["warmup_steps"]),
        total_steps=total_steps,
        use_distributed_optimizer=True,
        patterns=None,
    )
    # Padding positions route nowhere: the router zeroes their probabilities and drops their
    # routes, and DeepEP filters zero-probability routes out of its metadata.
    for chunk in chunks:
        for module in chunk.modules():
            if type(module).__name__.endswith("Router"):
                module.forward = _masked_router_forward(module.forward)
    for _layer, moe in moe_layers(chunks):
        manager = moe.token_dispatcher._comm_manager
        manager.capacity_factor = 0.0
        manager._indices_to_multihot = lambda indices, probs, _manager=manager: (
            indices_to_multihot(indices, probs, _manager.num_local_experts)
        )
    forward_padding_flag_to_attention(chunks)
    mamba = any(type(m).__name__ == "HybridStack" for c in chunks for m in c.modules())
    data_parallel_rank = parallel_state.get_data_parallel_rank(with_context_parallel=False)
    group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    # a resume loads the model over the conversion, then the optimizer, scheduler and RNG state
    start, totals = 0, {"prediction_tokens": 0, "response_tokens": 0}
    resume = args.resume_from
    if resume is not None:
        resume = checkpoint.latest(out_dir) if resume == "latest" else Path(resume)
        checkpoint.load_model(chunks, resume)
        state = checkpoint.load(resume, optimizer, scheduler)
        start, totals = state["optimizer_step"], state["progress"]

    # the run directory and metrics (rank 0)
    document = recipe | {"planned_updates": total_steps}
    if rank == 0:
        if resume is None:
            run = RunDirectory.create(out_dir, config=document, inputs={"model": recipe["model"]})
        else:
            run = RunDirectory.resume(out_dir, config=document)
            resume_metrics(out_dir, start)
    torch.distributed.barrier()
    wandb_config = WandbConfig(
        runtime["wandb_mode"],
        runtime["wandb_project"],
        runtime["wandb_group"],
        runtime["wandb_job_type"],
    )
    metrics = Metrics(
        out_dir,
        wandb_config,
        config=document,
        enabled=rank == 0,
        name=runtime["wandb_name"],
        tags=runtime["wandb_tags"],
    )
    held_out = {
        "pad_id": pad_id,
        "sequence_length": batch["sequence_length"],
        "context_parallel_size": context_parallel_size,
        "mamba": mamba,
        "rank": data_parallel_rank,
        "ranks": ranks,
    }

    def validate(step: int) -> float:
        numerator, denominator = evaluate(chunks, bank, validation_layout, group=group, **held_out)
        record = {
            "step": step,
            "validation/response_nll_sum": numerator,
            "validation/response_tokens": round(denominator),
            "validation/response_nll_mean": numerator / denominator,
        }
        metrics.log(record)
        return numerator / denominator

    validation_nll = None
    if resume is None:
        validation_nll = validate(0)

    # the updates, evaluating twenty times over the run and every `evaluation_interval_steps`
    last = total_steps if args.stop_after_step is None else min(args.stop_after_step, total_steps)
    for step in range(start + 1, last + 1):
        started = time.perf_counter()
        for chunk in chunks:
            chunk.zero_grad_buffer()
        optimizer.zero_grad()
        batches = []
        for offset in range(accumulation):
            microbatch = (step - 1) * accumulation + offset
            packs = [
                assemble_packed_batch(
                    bank,
                    train_layout,
                    pack_id=sft_pack_id(
                        orders,
                        microbatch_index=microbatch,
                        pack_index=index,
                        data_parallel_rank=data_parallel_rank,
                        data_parallel_size=ranks,
                        micro_batch_size=MICRO_BATCH_SIZE,
                    ),
                    pad_id=pad_id,
                    sequence_length=batch["sequence_length"],
                    alignment=ALIGNMENT,
                )
                for index in range(MICRO_BATCH_SIZE)
            ]
            batches.append(
                to_device(
                    concatenate_packed_batches(packs),
                    context_parallel_size=context_parallel_size,
                    mamba=mamba,
                )
            )
        counts = torch.tensor(
            [
                sum(int(packed["prediction_mask"].sum()) for packed in batches),
                sum(int(packed["lm_mask"].sum()) for packed in batches),
            ],
            dtype=torch.int64,
            device="cuda",
        )
        # raw response-loss sums, with the gradients divided once by the global token count
        numerator = denominator = None
        for index, micro in enumerate(batches):
            with accumulating(chunks, last=index == len(batches) - 1):
                loss, tokens = response_loss(chunks[0], micro)
                loss.backward()
            numerator = loss.detach() if numerator is None else numerator + loss.detach()
            denominator = tokens.detach() if denominator is None else denominator + tokens.detach()
        finalize_model_grads(chunks, num_tokens=denominator.clone())
        train_sum, train_tokens = reduce_pair(numerator, denominator, group)
        torch.distributed.all_reduce(counts, group=group)
        updated, grad_norm, _ = optimizer.step()
        if not updated:
            raise RuntimeError(f"the optimizer skipped update {step}")
        scheduler.step(1)
        totals["prediction_tokens"] += int(counts[0])
        totals["response_tokens"] += int(counts[1])
        metrics.log(
            {
                "step": step,
                "train/response_nll_sum": train_sum,
                "train/response_tokens": round(train_tokens),
                "train/response_nll_mean": train_sum / train_tokens,
                # read after the scheduler step: the rate of the next update
                "optimizer/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer/global_gradient_norm": float(grad_norm),
                "throughput/step_time_seconds": time.perf_counter() - started,
            }
        )
        consumed = min(step * accumulation * MICRO_BATCH_SIZE * ranks, total_packs)
        previous = min((step - 1) * accumulation * MICRO_BATCH_SIZE * ranks, total_packs)
        twentieth = total_packs / 20
        if (
            step % runtime["evaluation_interval_steps"] == 0
            or math.floor(consumed / twentieth) > math.floor(previous / twentieth)
            or consumed == total_packs
        ):
            validation_nll = validate(step)
        if step % runtime["save_interval_steps"] == 0 or step == last:
            gather_parameters(chunks)
            checkpoint.save(
                out_dir / "checkpoints" / f"step-{step:07d}",
                chunks,
                optimizer,
                scheduler,
                step=step,
                progress=totals,
                keep_last=runtime["keep_last_checkpoints"],
            )

    if rank == 0 and last == total_steps:
        run.write_results(
            {
                "completed_updates": total_steps,
                "processed_packs": total_packs,
                **totals,
                "final_validation_response_nll": validation_nll,
            }
        )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def load_recipe(path: str | Path) -> dict[str, Any]:
    recipe = yaml.safe_load(Path(path).read_text())
    if set(recipe) != set(RECIPE):
        raise ValueError(f"an SFT recipe has exactly the sections {sorted(RECIPE)}")
    for section, keys in RECIPE.items():
        if keys is not None and set(recipe[section]) != keys:
            raise ValueError(f"{section} must have exactly the keys {sorted(keys)}")
    return recipe


def response_loss(model, batch):
    losses = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=None,
        labels=batch["labels"],
        padding_mask=batch["padding_mask"],
        packed_seq_params=batch["packed_seq_params"],
    )
    mask = batch["lm_mask"]
    return (losses.reshape(mask.shape).float() * mask.float()).sum(), mask.sum(dtype=torch.float32)


def reduce_pair(numerator, denominator, group) -> tuple[float, float]:
    pair = torch.stack((numerator.detach().float(), denominator.detach().float()))
    torch.distributed.all_reduce(pair, group=group)
    return float(pair[0]), float(pair[1])


def evaluate(
    chunks,
    bank,
    layout,
    *,
    pad_id,
    sequence_length,
    context_parallel_size,
    mamba,
    rank,
    ranks,
    group,
):
    (model,) = chunks
    for chunk in chunks:
        chunk.eval()
    packs = len(layout)
    numerator = denominator = None
    with torch.no_grad():
        for index in range(max(1, math.ceil(packs / ranks))):
            candidate = index * ranks + rank
            batch = assemble_packed_batch(
                bank,
                layout,
                pack_id=candidate if candidate < packs else None,
                pad_id=pad_id,
                sequence_length=sequence_length,
                alignment=ALIGNMENT,
            )
            loss, tokens = response_loss(
                model,
                to_device(batch, context_parallel_size=context_parallel_size, mamba=mamba),
            )
            numerator = loss if numerator is None else numerator + loss
            denominator = tokens if denominator is None else denominator + tokens
    for chunk in chunks:
        chunk.train()
    return reduce_pair(numerator, denominator, group)


def _masked_router_forward(original):
    def forward(hidden_states, padding_mask=None):
        probabilities, routing_map = original(hidden_states, padding_mask)
        if padding_mask is None:
            return probabilities, routing_map
        padding = padding_mask.reshape(-1, 1)
        return probabilities.masked_fill(padding, 0.0), routing_map & ~padding

    return forward


if __name__ == "__main__":
    main()
