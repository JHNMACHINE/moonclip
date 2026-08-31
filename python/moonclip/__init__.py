from typing import TYPE_CHECKING

from moonclip.moonclip import MoonclipManager, __version__  # noqa: F401

if TYPE_CHECKING:
    # Declared for type checkers only. At runtime these are resolved lazily by
    # __getattr__ below, so that `import moonclip` does not pull in torch;
    # without this block they are invisible to static analysis and every name
    # in __all__ but not in the module body is reported as an error.
    from moonclip.pytorch import (
        CheckpointManager,
        TensorStub,
        describe_state_dict,
        flatten_state_dict,
        unflatten_state_dict,
    )


def __getattr__(name):
    if name == "CheckpointManager":
        from moonclip.pytorch import CheckpointManager
        return CheckpointManager
    if name == "flatten_state_dict":
        from moonclip.pytorch import flatten_state_dict
        return flatten_state_dict
    if name == "unflatten_state_dict":
        from moonclip.pytorch import unflatten_state_dict
        return unflatten_state_dict
    if name == "describe_state_dict":
        from moonclip.pytorch import describe_state_dict
        return describe_state_dict
    if name == "TensorStub":
        from moonclip.pytorch import TensorStub
        return TensorStub
    raise AttributeError(f"module 'moonclip' has no attribute {name!r}")


__all__ = [
    "MoonclipManager",
    "CheckpointManager",
    "flatten_state_dict",
    "unflatten_state_dict",
    "describe_state_dict",
    "TensorStub",
    "__version__",
]
