from revolver.revolver import RevolverManager, __version__  # noqa: F401


def __getattr__(name):
    if name == "CheckpointManager":
        from revolver.pytorch import CheckpointManager
        return CheckpointManager
    if name == "flatten_state_dict":
        from revolver.pytorch import flatten_state_dict
        return flatten_state_dict
    raise AttributeError(f"module 'revolver' has no attribute {name!r}")


__all__ = ["RevolverManager", "CheckpointManager", "flatten_state_dict", "__version__"]
