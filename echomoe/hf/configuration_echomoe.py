"""The folded EchoMoE students' configs for transformers remote code
(`AutoConfig.from_pretrained(..., trust_remote_code=True)`). Each is its family's own config, with
the student's codec layout under `folded` as the export wrote it. vLLM reads the same `config.json`
through the family's own config class. Imports only transformers, as remote code must."""

from transformers import DeepseekV3Config, Gemma4TextConfig, NemotronHConfig, Qwen3MoeConfig


class Qwen3MoeFoldedConfig(Qwen3MoeConfig):
    """A folded Qwen3-MoE student's config."""


class DeepseekV3FoldedConfig(DeepseekV3Config):
    """A folded DeepSeek-V3 student's config (Kanana-2, Moonlight)."""


class Gemma4FoldedConfig(Gemma4TextConfig):
    """A folded Gemma 4 student's config."""


class NemotronHFoldedConfig(NemotronHConfig):
    """A folded Nemotron-H student's config."""
