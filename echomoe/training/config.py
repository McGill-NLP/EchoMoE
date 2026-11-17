"""The distillation recipe: one YAML file per run, read into frozen dataclasses.

Every key without a default here is required and unknown keys are refused, so a recipe states the
whole run. The `compressor` section is the codec's own document (`echomoe.codec`), or
`{lsh: ...}` for the LSH-MoE baseline (`echomoe.lsh`).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from echomoe.codec import CompressorConfig, compressor_config_document, parse_compressor_config
from echomoe.lsh import LshConfig

SOURCES = ("dolmino", "dolci")


@dataclass(frozen=True)
class RunConfig:
    kind: str
    seed: int
    output_dir: str
    stop_after_step: int | None = None


@dataclass(frozen=True)
class StudentConfig:
    compressor_initialization: str | None  # the fit, or null for LSH-MoE, which has none
    # Fold the fit into the experts at the start and train the folded student (full-model
    # adaptation).
    folded: bool = False


@dataclass(frozen=True)
class ModelsConfig:
    # The teacher's run directory. Its HF export `hf/` is the model the student starts from, and a
    # dev-mode start loads the backbone already converted to Megatron's layout in `megatron-ep8/`.
    teacher: str
    student: StudentConfig


@dataclass(frozen=True)
class SourceConfig:
    tokenized_dataset: str
    train_layout: str
    train_teacher_targets: str
    validation_layout: str
    validation_teacher_targets: str
    token_budget: int | None  # scored tokens taken from the source's pack stream, or None for all
    passes: int  # epochs of the pack order


@dataclass(frozen=True)
class SourcesConfig:
    dolmino: SourceConfig
    dolci: SourceConfig


@dataclass(frozen=True)
class MixingConfig:
    kind: str
    anneal_start: float
    end_dolci_fraction: float


@dataclass(frozen=True)
class DataConfig:
    order_seed: int
    mix_seed: int
    mixing: MixingConfig
    sources: SourcesConfig


@dataclass(frozen=True)
class LossConfig:
    target_top_m: int
    kl_weight: float
    lm_weight: float
    position_chunk_size: int


@dataclass(frozen=True)
class ParameterGroup:
    learning_rate: float
    minimum_learning_rate: float


@dataclass(frozen=True)
class OptimizerConfig:
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    gradient_clip: float
    decoupled_weight_decay: bool
    parameter_groups: dict[str, ParameterGroup]


@dataclass(frozen=True)
class SchedulerConfig:
    family: str
    warmup_steps: int


@dataclass(frozen=True)
class BatchConfig:
    micro_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int


@dataclass(frozen=True)
class DDPConfig:
    use_distributed_optimizer: bool
    grad_reduce_in_fp32: bool
    overlap_grad_reduce: bool
    overlap_param_gather: bool
    average_in_collective: bool
    bucket_size: int
    pad_buckets_for_high_nccl_busbw: bool


@dataclass(frozen=True)
class ParallelismConfig:
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    context_parallel_size: int
    rank_order: str
    context_parallel_communication: str | None
    dispatcher: str
    expected_world_size: int
    expected_nodes: int
    gpus_per_node: int
    ddp: DDPConfig


@dataclass(frozen=True)
class RuntimeConfig:
    recompute: str
    attention_backend: str
    distributed_timeout_seconds: int
    apply_rope_fusion: bool
    bias_activation_fusion: bool
    moe_router_fusion: bool
    moe_grouped_gemm: bool
    masked_softmax_fusion: bool
    moe_permute_fusion: bool
    compile_regions: bool  # torch.compile the loss's per-chunk terms
    recompute_num_layers: int | None = None
    matmul_precision: str = "highest"


@dataclass(frozen=True)
class CheckpointConfig:
    save_interval_steps: int
    keep_last: int  # optimizer states kept (every model checkpoint is kept)
    resume_from: str | None


@dataclass(frozen=True)
class EvaluationConfig:
    interval_steps: int
    batches: int | str  # validation packs per rank, or "all"
    at_start: bool = True


@dataclass(frozen=True)
class WandbConfig:
    mode: str
    project: str | None
    group: str | None
    job_type: str | None


@dataclass(frozen=True)
class LoggingConfig:
    wandb: WandbConfig


@dataclass(frozen=True)
class TrainingConfig:
    run: RunConfig
    models: ModelsConfig
    data: DataConfig
    loss: LossConfig
    compressor: CompressorConfig | LshConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    batch: BatchConfig
    parallelism: ParallelismConfig
    runtime: RuntimeConfig
    checkpoint: CheckpointConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig


def parameter_group_names(compressor: CompressorConfig | LshConfig, *, folded: bool) -> list[str]:
    names = ["dense_backbone", "router", "experts"]
    if isinstance(compressor, LshConfig):
        return [*names, "lsh_compensation"]
    if folded:
        names.append("expert_factor")
    if compressor.dispatch_rank is not None:
        names.append("dispatch_compressor")
    if compressor.reference is not None:
        names.append("dispatch_reference")
    if compressor.combine_rank is not None:
        names.append("combine_compressor")
    if compressor.combine_owner in ("bias", "biasgate"):
        names.append("combine_correction")
    return names


def load_config(
    path: str | Path, *, resume_from: str | None = None, stop_after_step: int | None = None
) -> TrainingConfig:
    config = _build(TrainingConfig, read_recipe(path), "recipe")
    if resume_from is not None:
        config = replace(config, checkpoint=replace(config.checkpoint, resume_from=resume_from))
    if stop_after_step is not None:
        config = replace(config, run=replace(config.run, stop_after_step=stop_after_step))

    # the checks a recipe edit can break
    student, parallel, batch = config.models.student, config.parallelism, config.batch
    backbone = ("dense_backbone", "router", "experts")
    groups = config.optimizer.parameter_groups
    trains_codec = isinstance(config.compressor, CompressorConfig)
    if (
        trains_codec
        and not student.folded
        and any(groups[name].learning_rate for name in backbone)
    ):
        raise ValueError(
            "a dev-mode recipe trains only the codec. Its checkpoints omit the backbone"
        )
    data_parallel = parallel.expected_world_size // (
        parallel.tensor_parallel_size
        * parallel.pipeline_parallel_size
        * parallel.context_parallel_size
    )
    if batch.global_batch_size != (
        batch.micro_batch_size * batch.gradient_accumulation_steps * data_parallel
    ):
        raise ValueError("global_batch_size must be micro batch * accumulation * data parallel")
    if parallel.ddp.average_in_collective:
        raise ValueError("average_in_collective must be false: the loss is normalized globally")
    expected_groups = parameter_group_names(config.compressor, folded=student.folded)
    if set(config.optimizer.parameter_groups) != set(expected_groups):
        raise ValueError(f"optimizer.parameter_groups must be exactly {expected_groups}")
    return config


def read_recipe(path: str | Path) -> dict[str, Any]:
    """A recipe with every key its `base` file sets and it does not."""
    recipe = yaml.safe_load(Path(path).read_text())
    if "base" in recipe:
        base = yaml.safe_load((Path(path).parent / recipe.pop("base")).read_text())
        recipe = _merge(base, recipe)
    return recipe


def config_document(config: TrainingConfig) -> dict[str, Any]:
    """The resolved recipe a run directory records and a resume compares. The resume target and
    the bounded stop may differ between attempts of one run."""
    document = dataclasses.asdict(config)
    if isinstance(config.compressor, LshConfig):
        document["compressor"] = {"lsh": document["compressor"]}
    else:
        document["compressor"] = compressor_config_document(config.compressor)
    del document["checkpoint"]["resume_from"], document["run"]["stop_after_step"]
    return document


def _merge(base: dict, recipe: dict) -> dict:
    merged = dict(base)
    for key, value in recipe.items():
        both = isinstance(value, dict) and isinstance(base.get(key), dict)
        merged[key] = _merge(base[key], value) if both else value
    return merged


def _build(kind: Any, value: Any, name: str) -> Any:
    if kind is CompressorConfig:
        return parse_compressor_config(value)
    if kind is LshConfig:  # written `{lsh: {...}}`
        if not isinstance(value, dict) or set(value) != {"lsh"}:
            raise ValueError(f"{name} must be a mapping with the one key lsh")
        value, name = value["lsh"], f"{name}.lsh"
    if dataclasses.is_dataclass(kind):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        fields = {field.name: field for field in dataclasses.fields(kind)}
        required = {key for key, field in fields.items() if field.default is dataclasses.MISSING}
        if not required <= set(value) <= set(fields):
            raise ValueError(
                f"{name}: missing {sorted(required - set(value))}, "
                f"unknown {sorted(set(value) - set(fields))}"
            )
        hints = typing.get_type_hints(kind)
        return kind(
            **{key: _build(hints[key], item, f"{name}.{key}") for key, item in value.items()}
        )
    origin, arguments = typing.get_origin(kind), typing.get_args(kind)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        errors = []
        for option in (item for item in arguments if item is not type(None)):
            try:
                return _build(option, value, name)
            except ValueError as error:
                errors.append(str(error))
        raise ValueError(". ".join(errors))
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a list")
        return tuple(
            _build(arguments[0], item, f"{name}[{index}]") for index, item in enumerate(value)
        )
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        return {key: _build(arguments[1], item, f"{name}.{key}") for key, item in value.items()}
    if kind is float and type(value) in (int, float):
        return float(value)
    if type(value) is not kind:
        raise ValueError(f"{name} must be {kind.__name__}, got {value!r}")
    return value
