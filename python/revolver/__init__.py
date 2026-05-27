from revolver.revolver import RevolverManager, __version__  # noqa: F401


def __getattr__(name):
    if name == "CheckpointManager":
        from revolver.pytorch import CheckpointManager
        return CheckpointManager
    raise AttributeError(f"module 'revolver' has no attribute {name!r}")


__all__ = ["RevolverManager", "CheckpointManager", "__version__"]
