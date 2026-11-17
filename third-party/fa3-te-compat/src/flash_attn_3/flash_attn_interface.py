"""Compat shim: the FA3 hopper/setup.py (mainline) installs the Python API
as top-level modules (`flash_attn_interface`, `flash_attn_config`) plus a
`flash_attn_3._C` CUDA extension, but Transformer Engine imports
`from flash_attn_3.flash_attn_interface import ...`. TE's own install
instructions (utils.py:136-141) explicitly tell users to `cp
flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py`
after installing FA3; we ship the same fix as a re-export module so it's
applied automatically by `uv pip sync` rather than as a post-install step."""

from flash_attn_interface import *  # noqa: F403
from flash_attn_interface import (  # noqa: F401
    _flash_attn_backward,
    _flash_attn_forward,
)
