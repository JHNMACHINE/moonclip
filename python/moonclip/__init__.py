from moonclip.moonclip import MoonclipManager, __version__  # noqa: F401


def __getattr__(name):
    if name == "CheckpointManager":
        from moonclip.pytorch import CheckpointManager
        return CheckpointManager
    if name == "flatten_state_dict":
        from moonclip.pytorch import flatten_state_dict
        return flatten_state_dict
    raise AttributeError(f"module 'moonclip' has no attribute {name!r}")


__all__ = ["MoonclipManager", "CheckpointManager", "flatten_state_dict", "__version__"]
