"""The folded students' transformers remote code (`configuration_echomoe.py`,
`modeling_echomoe.py`), which each folded export carries beside its weights so that
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` loads it without this package.
`config.json` keeps the vLLM architecture name and adds `auto_map` for transformers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

REMOTE_CODE_FILES = ("configuration_echomoe.py", "modeling_echomoe.py")
# The remote classes by model type: `<name>Config` and `<name>ForCausalLM`.
REMOTE_CLASSES = {
    "qwen3_moe": "Qwen3MoeFolded",
    "deepseek_v3": "DeepseekV3Folded",
    "gemma4_text": "Gemma4Folded",
    "nemotron_h": "NemotronHFolded",
}


def add_remote_code(directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Copy the remote code into `directory` and return `config` with the `auto_map` naming it."""
    name = REMOTE_CLASSES[config["model_type"]]
    for file in REMOTE_CODE_FILES:
        shutil.copyfile(Path(__file__).parent / file, Path(directory) / file)
    auto_map = {
        "AutoConfig": f"configuration_echomoe.{name}Config",
        "AutoModelForCausalLM": f"modeling_echomoe.{name}ForCausalLM",
    }
    return config | {"auto_map": auto_map}
