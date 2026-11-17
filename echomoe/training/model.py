"""Build the Megatron student: the teacher's HF export converted by Megatron-Bridge, the codec on
every MoE layer through the DeepEP adapter, the fit installed (or, for full-model adaptation,
folded into the experts in float64), and the zero-rate groups frozen. An LSH-MoE student
(`echomoe.lsh`) has LSH adapters and alpha modules in place of the codec, and no fit. MoE layers
are numbered by MoE ordinal, so DeepSeek-V3's leading dense layers are not counted.

    torchrun --nproc-per-node 8 -m echomoe.training.model --config <recipe>

writes the converted teacher backbone `<teacher>/megatron-ep8` that dev-mode (backbone frozen)
recipes load.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import re
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from megatron.bridge import AutoBridge
from megatron.bridge.models.gemma.gemma3_provider import Gemma3LanguageModelEmbedding
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.hybrid import hybrid_block
from megatron.core.transformer import transformer_block
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.mla_qk_norm_config import QKNormConfigResolver
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_pg_rank
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
from transformer_engine.pytorch.module.base import _2X_ACC_DGRAD, _2X_ACC_FPROP, _2X_ACC_WGRAD

from echomoe.codec import (
    COMBINE_COMPRESSOR_MODULE_NAME,
    COMBINE_CORRECTION_MODULE_NAME,
    DISPATCH_COMPRESSOR_MODULE_NAME,
    DISPATCH_REFERENCE_MODULE_NAME,
    MODULE_ATTRIBUTE_NAMES,
    CompressorConfig,
    ExpertWeights,
    compressor_config_document,
    fold_document,
    fold_student,
    require_exact_fold,
)
from echomoe.fit import install_fit, load_fit
from echomoe.lsh import LshConfig, rotations
from echomoe.runs import write_meta

from .checkpoint import load_model, save_model, step_meta
from .config import ParallelismConfig, RuntimeConfig, TrainingConfig, load_config
from .data.layout import SEQUENCE_LENGTH
from .deepep import Adapter, Attachment, LshAdapter
from .packed_attention import install_packed_attention

# The blocks a model's layers run in, and the module whose `checkpointed_forward` each calls.
BLOCK_MODULES = {"TransformerBlock": transformer_block, "HybridStack": hybrid_block}
MOE_MODULES = ("MoELayer", "Gemma4MoELayer")
# LSH-MoE's alpha modules (`CompensationScale`) and the factor between alpha and the parameter.
COMPENSATION_MODULE_NAME = "lsh_compensation"
ALPHA_SCALE = 0.01


def build_student(
    config: TrainingConfig, *, resume: Path | None, convert: bool = False
) -> tuple[list[Any], Attachment, dict[str, Any] | None]:
    """The compressed student, its attachment and, if folded at this start, the fold record. The
    backbone comes from the resume checkpoint, else the converted cache, else the HF conversion."""
    student, runtime = config.models.student, config.runtime
    teacher = Path(config.models.teacher)
    is_lsh = isinstance(config.compressor, LshConfig)
    converted = None if student.folded or is_lsh or convert else teacher / "megatron-ep8"
    folding = student.folded and resume is None
    fit = None
    if not is_lsh:
        fit = load_fit(
            student.compressor_initialization, compressor_config_document(config.compressor)
        )[1]
    # a group the recipe gives a zero rate keeps the uncompressed model's values
    groups = config.optimizer.parameter_groups
    frozen = [name for name, group in groups.items() if group.learning_rate == 0.0]
    built: dict[str, Any] = {}

    def prepare(chunks: list[Any]) -> list[Any]:
        # full recompute: chunk boundaries that keep each reference with the layers that read
        # it, and a gradient into every checkpointed chunk
        if runtime.recompute == "full" and not is_lsh:
            moe_count = len(moe_layers(chunks))
            config.compressor.validate_recompute_chunk(
                num_layers=moe_count, chunk_layers=runtime.recompute_num_layers
            )
            # dense layers come before the first MoE layer (DeepSeek-V3) or between MoE layers
            # (Nemotron-H's Mamba and attention layers)
            positions = moe_positions(chunks)
            if positions != list(range(moe_count)):
                align_recompute_chunks(chunks, positions, runtime.recompute_num_layers)

            # Megatron's reentrant checkpoint runs a chunk under `no_grad` and gives its outputs a
            # graph only when a tensor input requires grad, which a frozen backbone's hidden
            # states do not, and the codec modules inside would get no gradient. The input becomes
            # a detached alias that requires grad, so values and parameter gradients are unchanged.
            def require_grad(module, args, kwargs):
                hidden_states = kwargs["hidden_states"]
                if (
                    not (torch.is_grad_enabled() and module.training)
                    or hidden_states.requires_grad
                ):
                    return None
                return args, {**kwargs, "hidden_states": hidden_states.detach().requires_grad_()}

            for chunk in chunks:
                for module in chunk.modules():
                    if type(module).__name__ in BLOCK_MODULES:
                        module.register_forward_pre_hook(require_grad, with_kwargs=True)

        # the fold replaces the expert modules, so it reads their weights first
        captured = None
        if folding:
            captured = {
                layer: ExpertWeights(
                    gate_up=tuple(
                        getattr(moe.experts.linear_fc1, f"weight{index}").detach()
                        for index in range(int(moe.num_local_experts))
                    ),
                    down=tuple(
                        getattr(moe.experts.linear_fc2, f"weight{index}").detach()
                        for index in range(int(moe.num_local_experts))
                    ),
                )
                for layer, moe in moe_layers(chunks)
            }

        # the codec (or LSH-MoE) on every MoE layer, then the fit: folded into the experts, or
        # installed
        if is_lsh:
            attachment = attach_lsh(chunks, config.compressor)
        else:
            attachment = attach(chunks, config.compressor, folded=student.folded)
        built["attachment"] = attachment
        if captured is not None:
            built["folds"] = fold_student(
                attachment=attachment, weights=captured, tensors=fit, seed=config.run.seed
            )
        elif not student.folded and not is_lsh:
            install_fit(fit, attachment.modules())

        # frozen before the DDP wrap, which keeps the parameters out of the gradient buckets
        for chunk in chunks:
            for name, parameter in chunk.named_parameters():
                if parameter_group(name) in frozen:
                    parameter.requires_grad_(False)
        return chunks

    chunks = build_model(
        str(teacher / "hf"),
        config.parallelism,
        runtime,
        sequence_length=SEQUENCE_LENGTH,
        seed=config.run.seed,
        moe_z_loss_coeff=None,
        pre_wrap_hook=prepare,
        load_weights=resume is None and converted is None,
    )
    forward_padding_flag_to_attention(chunks)
    if converted is not None:
        load_model(chunks, converted, backbone_only=True)
    if resume is not None:
        load_model(chunks, resume, codec_only=step_meta(resume)["codec_only"])
    # The load's read buffers stay in the caching allocator, and NCCL allocates outside it (a
    # resumed four-node run died at its first update on a 128-byte communicator allocation).
    torch.cuda.empty_cache()
    record = None
    if folding:
        # every rank folds its own experts, so the gate reads the worst error over the ranks
        record = fold_document(built["folds"], seed=config.run.seed)
        require_exact_fold(record)
    return chunks, built["attachment"], record


def attach(chunks: Sequence[Any], compressor: CompressorConfig, *, folded: bool) -> Attachment:
    layers = moe_layers(chunks)
    _layer, first = layers[0]
    compressor.validate_model(
        hidden_size=int(first.config.hidden_size),
        num_layers=len(layers),
        payload_dtype=next(first.parameters()).dtype,
    )
    attachment = Attachment(group=first.token_dispatcher._comm_manager.group)
    roles = {layer: compressor.dispatch_role(layer) for layer, _moe in layers}
    for layer in sorted(roles):  # a reference is dropped after the last layer that reads it
        if roles[layer] == "consumer":
            attachment.last_consumer[compressor.reference.reference_layer_for(layer)] = layer
    for layer, moe in layers:
        modules = add_codec_modules(moe, layer, compressor, folded=folded)
        if "combine_correction" in modules:
            modules["combine_correction"].reduce_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True
            )
        if folded:
            attachment.folded_experts[layer] = moe.experts.linear_fc2
            if roles[layer] in ("plain", "consumer"):
                attachment.folded_readers[layer] = moe.experts.linear_fc1
        adapter = Adapter(
            moe.token_dispatcher,
            modules,
            layer=layer,
            dispatch_role=roles[layer],
            reference_layer=(
                compressor.reference.reference_layer_for(layer)
                if roles[layer] == "consumer"
                else None
            ),
            attachment=attachment,
            ep_rank=torch.distributed.get_rank(attachment.group),
            hidden_size=int(moe.config.hidden_size),
            folded=folded,
        )
        adapter.install()
        attachment.adapters[layer] = adapter
    return attachment


def attach_lsh(chunks: Sequence[Any], config: LshConfig) -> Attachment:
    layers = moe_layers(chunks)
    first = layers[0][1]
    attachment = Attachment(group=first.token_dispatcher._comm_manager.group)
    projections = rotations(config, int(first.config.hidden_size)).cuda()
    add_compensation_scales(chunks)
    for layer, moe in layers:
        adapter = LshAdapter(
            moe.token_dispatcher,
            {},
            layer=layer,
            dispatch_role="off",
            reference_layer=None,
            attachment=attachment,
            ep_rank=torch.distributed.get_rank(attachment.group),
            hidden_size=int(moe.config.hidden_size),
            folded=False,
            rotations=projections,
            bucket_count=config.bucket_count,
            scale=getattr(moe, COMPENSATION_MODULE_NAME),
        )
        adapter.install()
        attachment.adapters[layer] = adapter
    return attachment


def add_compensation_scales(chunks: Sequence[Any]) -> Sequence[Any]:
    for _layer, moe in moe_layers(chunks):
        setattr(moe, COMPENSATION_MODULE_NAME, CompensationScale(moe))
    return chunks


class CompensationScale(nn.Module):
    """One MoE layer's LSH-MoE alpha, one per local expert, from 0. The parameter is
    `alpha / ALPHA_SCALE`: Adam moves alpha as it would unscaled, while its gradient adds
    ALPHA_SCALE times less to the global norm every update is clipped by, so it barely changes the
    other weights' clipped updates. Expert parallel like the expert weights, and checkpointed as
    one tensor tiled over the experts, which every rank of an expert-parallel group would
    otherwise write under one key."""

    def __init__(self, moe: Any) -> None:
        super().__init__()
        like = moe.experts.linear_fc1.weight0
        self.raw = nn.Parameter(
            torch.zeros(int(moe.num_local_experts), dtype=like.dtype, device=like.device)
        )
        self.raw.allreduce = getattr(like, "allreduce", True)
        self.local_expert_offset = int(moe.local_expert_indices[0])
        self.num_global_experts = int(moe.config.num_moe_experts)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        key = f"{prefix}raw"
        local = self.raw.shape[0]
        tensor = ShardedTensor.from_rank_offsets(
            key,
            self.raw,
            *sharded_offsets,
            (
                len(sharded_offsets),
                self.local_expert_offset // local,
                self.num_global_experts // local,
            ),
            replica_id=(
                0,
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_expert_data_parallel_rank(),
            ),
            prepend_axis_num=len(sharded_offsets),
        )
        return {key: tensor}

    @property
    def alpha(self) -> torch.Tensor:
        return ALPHA_SCALE * self.raw


def add_codec_modules(
    moe: Any, layer: int, compressor: CompressorConfig, *, folded: bool
) -> dict[str, nn.Module]:
    """Put one MoE layer's codec modules on it and return them by side name. A folded student's
    expert slots also get zero banks, which the fold or a checkpoint fills."""
    hidden_size, like = int(moe.config.hidden_size), next(moe.parameters())
    built = compressor.build_modules(
        layer=layer,
        hidden_size=hidden_size,
        num_experts=int(moe.config.num_moe_experts),
        folded=folded,
    )
    modules = {}
    for side, module in zip(MODULE_ATTRIBUTE_NAMES, built, strict=True):
        if module is None:
            continue
        module.to(device=like.device, dtype=like.dtype)
        # replicated on every rank: reduced over the whole data-parallel group, not the
        # expert-data-parallel one
        for parameter in module.parameters():
            parameter.allreduce = True
        setattr(moe, MODULE_ATTRIBUTE_NAMES[side], module)
        modules[side] = module
    if folded:
        # the experts write the combine code through `factors` (W_out,i). A non-reference layer
        # reads `[z, c_d]` and a plain layer its code through `readers` (W_in,i). A reference
        # layer keeps its gate/up weight.
        experts, role = moe.experts, compressor.dispatch_role(layer)
        experts.linear_fc2 = ExpertBank(
            "factors",
            rows=modules["combine"].rank,
            columns=int(experts.linear_fc2.weight0.shape[1]),
            replaces=experts.linear_fc2,
            moe=moe,
        )
        if role in ("plain", "consumer"):
            experts.linear_fc1 = ExpertBank(
                "readers",
                rows=int(experts.linear_fc1.weight0.shape[0]),
                columns=modules["dispatch"].rank + (hidden_size if role == "consumer" else 0),
                replaces=experts.linear_fc1,
                moe=moe,
            )
        key_expert_bank_by_layer(moe)
    return modules


class ExpertBank(nn.Module):
    """A folded layer's `readers` `W_in,i = W_i [G, D_d]` (gate rows first) for `linear_fc1` or
    combine `factors` `W_out,i = C_c W_down,i` for `linear_fc2`, as one `(experts, rows, columns)`
    tensor."""

    def __init__(self, name: str, *, rows: int, columns: int, replaces: Any, moe: Any) -> None:
        super().__init__()
        like = replaces.weight0
        count = int(moe.num_local_experts)
        # One parameter owned by this module, because with `overlap_param_gather` Megatron's
        # pre-hook waits only on the direct parameters of the module that runs.
        bank = nn.Parameter(
            torch.zeros(count, rows, columns, dtype=like.dtype, device=like.device)
        )
        bank.allreduce = getattr(like, "allreduce", True)  # expert-parallel, as the weights were
        self.register_parameter(name, bank)
        self.name = name
        self.num_local_experts = count
        self.local_expert_offset = int(moe.local_expert_indices[0])
        self.num_global_experts = int(moe.config.num_moe_experts)
        self.pg_collection = getattr(replaces, "_pg_collection", None)

    def forward(self, rows: torch.Tensor, m_splits: Sequence[int]) -> tuple[torch.Tensor, None]:
        """`cat_i(rows_i W_i^T)`, rows grouped by `m_splits`, through Transformer Engine's grouped
        GEMM, the kernel the unfolded experts run."""
        return _GroupedExpertGemm.apply(rows, getattr(self, self.name), list(m_splits)), None

    def backward_dw(self) -> None:
        """TEGroupedMLP calls this unconditionally, and the bank has no delayed weight gradient."""

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """One key per bank, tiled over the expert axis. Without it every expert-parallel rank
        would write the same key and all but one bank would be lost."""
        local = self.num_local_experts
        key = f"{prefix}{self.name}"
        entry = make_sharded_tensors_for_checkpoint(
            {self.name: getattr(self, self.name)},
            prefix,
            {},
            sharded_offsets,
            dp_cp_group=(metadata or {}).get("dp_cp_group"),
        )[key]
        replica_id = tuple(entry.replica_id)
        tensor = ShardedTensor.from_rank_offsets(
            key,
            entry.data,
            *sharded_offsets,
            (
                len(sharded_offsets),
                self.local_expert_offset // local,
                self.num_global_experts // local,
            ),
            replica_id=(*replica_id[:2], int(get_pg_rank(self.pg_collection.expt_dp))),
            prepend_axis_num=len(sharded_offsets),
        )
        return {key: tensor}


class _GroupedExpertGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, rows: torch.Tensor, weight: torch.Tensor, splits: list[int]):
        rows = rows.contiguous()
        out = torch.empty((rows.shape[0], weight.shape[1]), dtype=rows.dtype, device=rows.device)
        general_grouped_gemm(
            list(weight.unbind(0)),
            list(torch.split(rows, splits)),
            [out],
            [None] * len(splits),
            rows.dtype,
            single_output=True,
            m_splits=splits,
            use_split_accumulator=_2X_ACC_FPROP,
        )
        ctx.save_for_backward(rows, weight)
        ctx.splits = splits
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        rows, weight = ctx.saved_tensors
        splits = ctx.splits
        grads = list(torch.split(grad_output.contiguous(), splits))
        unquantized: list[Any] = [None] * len(splits)
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.empty_like(rows)
            general_grouped_gemm(
                list(weight.unbind(0)),
                grads,
                [grad_input],
                unquantized,
                rows.dtype,
                single_output=True,
                layout="NN",
                m_splits=splits,
                grad=True,
                use_split_accumulator=_2X_ACC_DGRAD,
            )
        grad_weight = None
        if ctx.needs_input_grad[1]:
            # Zeros, not empty: an expert that received no rows may be left unwritten.
            grad_weight = torch.zeros_like(weight)
            general_grouped_gemm(
                list(torch.split(rows, splits)),
                grads,
                list(grad_weight.unbind(0)),
                unquantized,
                rows.dtype,
                layout="NT",
                m_splits=splits,
                grad=True,
                use_split_accumulator=_2X_ACC_WGRAD,
            )
        return grad_input, grad_weight, None


_LAYER_INDEX = re.compile(r"\.layers\.\d+\.")


def key_expert_bank_by_layer(moe: Any) -> None:
    # A folded first linear is the readers on some layers and the gate/up weight on others, so a
    # stack along Megatron's layer axis would have holes.
    experts = moe.experts
    bank_state_dict = experts.sharded_state_dict

    def sharded_state_dict(prefix="", sharded_offsets=(), metadata=None):
        entry = bank_state_dict(prefix=prefix, sharded_offsets=(), metadata=metadata)
        # A block of unlike layers (DeepSeek-V3's dense layers first) passes no layer offset and
        # already names each layer by its own index.
        if sharded_offsets:
            layer = int(sharded_offsets[0][1])
            for value in entry.values():
                value.key = _LAYER_INDEX.sub(rf"\g<0>{layer}.", value.key, count=1)
        return entry

    experts.sharded_state_dict = sharded_state_dict


class _Chunk:
    """Layers `start` to `end` of a transformer block, as a block that is one recompute chunk."""

    def __init__(self, block: Any, start: int, end: int) -> None:
        self.block, self.layers = block, block.layers[start:end]
        self.num_layers_per_pipeline_rank = end - start
        self.config = copy.copy(block.config)
        self.config.recompute_num_layers = end - start

    def __getattr__(self, name: str) -> Any:
        return getattr(self.block, name)


def align_recompute_chunks(chunks: Sequence[Any], positions: list[int], chunk_layers: int) -> None:
    """Recompute a block whose MoE layers sit at `positions` in chunks that each end after a group
    of `chunk_layers` MoE layers, the layers before the first MoE layer joining the first chunk and
    those after the last the last (Kanana-2: layers 0-4, 5-8, ..., 45-47, Nemotron-H: 0-8, 9-17,
    ..., 46-51). Megatron's uniform chunks count every layer and would split references from their
    consumers. Each chunk runs Megatron's own checkpointed forward."""
    (block,) = [
        module
        for chunk in chunks
        for module in chunk.modules()
        if type(module).__name__ in BLOCK_MODULES
    ]
    ends = [
        positions[min(index + chunk_layers, len(positions)) - 1] + 1
        for index in range(0, len(positions), chunk_layers)
    ]
    ends[-1] = block.num_layers_per_pipeline_rank
    views = [(start, _Chunk(block, start, end)) for start, end in zip([0, *ends[:-1]], ends)]
    module = BLOCK_MODULES[type(block).__name__]
    original = module.checkpointed_forward

    def checkpointed_forward(_block, *, hidden_states, layer_offset=0, **kwargs):
        for start, view in views:
            hidden_states = original(
                view, hidden_states=hidden_states, layer_offset=layer_offset + start, **kwargs
            )
        return hidden_states

    module.checkpointed_forward = checkpointed_forward


def build_model(
    checkpoint: str,
    parallel: ParallelismConfig,
    runtime: RuntimeConfig,
    *,
    sequence_length: int,
    seed: int,
    moe_z_loss_coeff: float | None,
    pre_wrap_hook: Callable[[list[Any]], list[Any]] | None = None,
    load_weights: bool = True,
) -> list[Any]:
    """Convert an HF checkpoint into DDP-wrapped Megatron chunks. Without `load_weights` the
    conversion and the random initialization are both skipped and the caller loads every tensor."""
    bridge = AutoBridge.from_hf_pretrained(checkpoint)
    provider = bridge.to_megatron_provider(load_weights=load_weights)
    if not load_weights:
        provider.perform_initialization = False
    provider.tensor_model_parallel_size = parallel.tensor_parallel_size
    provider.expert_model_parallel_size = parallel.expert_parallel_size
    provider.pipeline_model_parallel_size = parallel.pipeline_parallel_size
    provider.context_parallel_size = parallel.context_parallel_size
    provider.seq_length = sequence_length
    provider.sequence_parallel = parallel.tensor_parallel_size > 1
    provider.bf16 = True
    provider.gradient_accumulation_fusion = False
    provider.moe_aux_loss_coeff = 0.0
    provider.apply_rope_fusion = runtime.apply_rope_fusion
    provider.bias_activation_fusion = runtime.bias_activation_fusion
    provider.moe_router_fusion = runtime.moe_router_fusion
    provider.moe_grouped_gemm = runtime.moe_grouped_gemm
    provider.masked_softmax_fusion = runtime.masked_softmax_fusion
    provider.moe_permute_fusion = runtime.moe_permute_fusion
    provider.moe_deepep_num_sms = 20  # SMs DeepEP's kernels take from compute
    provider.overlap_moe_expert_parallel_comm = False
    provider.cp_comm_type = parallel.context_parallel_communication
    provider.calculate_per_token_loss = True
    provider.cross_entropy_loss_fusion = False
    provider.moe_token_dispatcher_type = "flex"
    provider.moe_flex_dispatcher_backend = "deepep"
    provider.moe_enable_deepep = False
    provider.moe_router_dtype = "fp32"
    provider.moe_shared_expert_overlap = False
    provider.moe_expert_capacity_factor = None
    # The router computes a z-loss whenever the coefficient is not None, even at zero.
    provider.moe_z_loss_coeff = moe_z_loss_coeff
    # The recipes do no load balancing, so a router's correction bias (DeepSeek-V3) keeps its
    # pretrained value. Megatron moves it by this rate after every update.
    if provider.moe_router_enable_expert_bias:
        provider.moe_router_bias_update_rate = 0.0
    if runtime.recompute == "full":  # the distillation, and Kanana-2's SFT
        provider.recompute_granularity = "full"
        provider.recompute_method = "uniform"
        provider.recompute_num_layers = runtime.recompute_num_layers
    apply_flex_dispatcher_backend(provider, "deepep")
    provider.attention_backend = AttnBackend[runtime.attention_backend]
    provider.finalize()
    if pre_wrap_hook is not None:
        provider.register_pre_wrap_hook(pre_wrap_hook)
    if not parallel_state.is_initialized():
        provider.initialize_model_parallel(seed=seed, order=parallel.rank_order)
    chunks = provider.provide_distributed_model(
        ddp_config=DistributedDataParallelConfig(**dataclasses.asdict(parallel.ddp)),
        bf16=True,
        wrap_with_ddp=True,
    )
    if type(provider).__name__ == "Gemma4ModelProvider":
        assert parallel.context_parallel_size == 1, "Gemma 4's global attention has no CP"
        install_packed_attention(chunks)
    return chunks


def init_distributed(timeout_seconds: int) -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=timeout_seconds))
    # NCCL opens point-to-point connections at their first use, which would otherwise be the
    # first checkpoint save, when the model and optimizer already fill the GPU (a step-100 save
    # failed with a CUDA out-of-memory inside NCCL). Open them while memory is free.
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    torch.distributed.gather_object(rank, [None] * world if rank == 0 else None, dst=0)


def moe_positions(chunks: Sequence[Any]) -> list[int]:
    """Every MoE layer's index in its block, in order."""
    return [
        int(name.split(".")[name.split(".").index("layers") + 1])
        for chunk in chunks
        for name, module in chunk.named_modules()
        if type(module).__name__ in MOE_MODULES
    ]


def moe_layers(chunks: Sequence[Any]) -> list[tuple[int, Any]]:
    """Every MoE layer and its MoE ordinal, in order."""
    found = [
        module
        for chunk in chunks
        for module in chunk.modules()
        if type(module).__name__ in MOE_MODULES
    ]
    return list(enumerate(found))


def parameter_group(name: str) -> str:
    if f".{DISPATCH_COMPRESSOR_MODULE_NAME}." in name:
        return "dispatch_compressor"
    if f".{COMBINE_COMPRESSOR_MODULE_NAME}." in name:
        return "combine_compressor"
    if f".{DISPATCH_REFERENCE_MODULE_NAME}." in name:
        return "dispatch_reference"
    if f".{COMBINE_CORRECTION_MODULE_NAME}." in name:
        return "combine_correction"
    # Precedes `.experts.`, which would otherwise swallow the expert factors.
    if name.endswith(".linear_fc2.factors"):
        return "expert_factor"
    if ".experts." in name:
        return "experts"
    if ".router." in name:
        return "router"
    return "dense_backbone"


def gather_parameters(chunks: Sequence[Any]) -> None:
    """All-gather the distributed optimizer's updated shards. Megatron gathers a parameter in a
    pre-hook of its module, and the codec's modules never run (`functional.linear`)."""
    for chunk in chunks:
        if chunk.ddp_config.use_distributed_optimizer:
            chunk.start_param_sync(force_sync=True)


def forward_padding_flag_to_attention(chunks: Sequence[Any]) -> None:
    """Let Transformer Engine attention read `pad_between_seqs` from the packed params instead of
    comparing the padded and unpadded cu_seqlens on the device at every call."""
    for chunk in chunks:
        for module in chunk.modules():
            kept = getattr(module, "kept_packed_seq_params", None)
            if isinstance(kept, set):
                kept.add("pad_between_seqs")


def _fix_megatron_core() -> None:
    """Two megatron-core 0.19.2 bugs that DeepSeek-V3 models meet and one megatron-bridge 0.6.2
    bug that Gemma 4 meets, fixed where they are.

    A router with a correction bias counts its tokens per expert with the `[tokens]` padding mask
    applied to the `[tokens, experts]` routing map, which broadcasts only when the two counts are
    equal. The count feeds only the bias update, whose rate is zero here.

    Without a query latent (Kanana-2, Moonlight) MLA projects the input layernorm's output to the
    queries directly, as transformers does and megatron-bridge maps it. 0.19.2 fuses a norm the
    model does not have into that projection.

    Gemma 4 scales its input embeddings by sqrt(hidden size) rounded to bf16, 53.0 at 2816, as
    transformers does. megatron-bridge multiplies by the exact root.
    """
    count = TopKRouter._apply_expert_bias
    TopKRouter._apply_expert_bias = lambda self, routing_map, padding_mask=None: count(
        self, routing_map, None if padding_mask is None else padding_mask[:, None]
    )
    resolve = QKNormConfigResolver._resolve_mla_qk_layernorm

    def resolve_mla(self: Any) -> dict[str, Any]:
        resolved = resolve(self)
        if not self.has_q_lora:
            resolved["linear_q_proj"] = self.submodules.linear_q_proj or self.linear_impl
        return resolved

    QKNormConfigResolver._resolve_mla_qk_layernorm = resolve_mla

    def scaled_embedding(self, input_ids, position_ids, tokentype_ids=None):
        scale = float(torch.tensor(self.config.hidden_size**0.5).to(self.config.params_dtype))
        return LanguageModelEmbedding.forward(self, input_ids, position_ids, tokentype_ids) * scale

    Gemma3LanguageModelEmbedding.forward = scaled_embedding


_fix_megatron_core()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache a recipe's converted teacher backbone.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    init_distributed(config.runtime.distributed_timeout_seconds)
    chunks, _attachment, _record = build_student(config, resume=None, convert=True)
    teacher = Path(config.models.teacher)
    save_model(chunks, teacher / "megatron-ep8", backbone_only=True)
    if torch.distributed.get_rank() == 0:
        write_meta(teacher / "megatron-ep8", inputs={"teacher": str(teacher / "hf")})
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
