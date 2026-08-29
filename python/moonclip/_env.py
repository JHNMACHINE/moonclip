"""
Auto-detection of distributed training environment.

Imports of torch are optional and local, so this module is safe to reach from
anywhere in the package; it lives under `moonclip.pytorch` regardless, which
is the layer allowed to know about torch. `import moonclip` does not pull in
either.

Detecting is not the same as adopting. Since 0.0.9 `CheckpointManager` calls
this to decide whether the topology question is ambiguous, and adopts the
answer only when asked to — see `_resolve_topology`.
"""

from __future__ import annotations

import os
from typing import Tuple


def _detect_distributed_env() -> Tuple[int, int]:
    """
    Auto-detect world_size and rank from the distributed environment.

    Detection priority:
    1. torch.distributed (if already initialized)
    2. torchrun env vars: RANK, WORLD_SIZE
    3. DeepSpeed / other launchers: same env vars
    4. Defaults: world_size=1, rank=0
    """
    # 1. torch.distributed already initialized
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
    except Exception:
        pass

    # 2. Environment variables (torchrun, deepspeed, etc.)
    env_world_size = os.environ.get("WORLD_SIZE")
    env_rank = os.environ.get("RANK")

    if env_world_size is not None and env_rank is not None:
        try:
            return int(env_world_size), int(env_rank)
        except ValueError:
            pass

    # 3. Defaults
    return 1, 0