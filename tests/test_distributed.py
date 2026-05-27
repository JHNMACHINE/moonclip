"""
Tests for torchrun env var auto-detection.
Does NOT require torch — tests the detection logic in isolation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


class TestDistributedEnvDetection:
    """Test _detect_distributed_env without importing torch."""

    def test_defaults_without_env(self, monkeypatch):
        """Without any env vars, defaults to world_size=1, rank=0."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)

        from revolver.pytorch import _detect_distributed_env
        ws, r = _detect_distributed_env()
        assert ws == 1
        assert r == 0

    def test_torchrun_env_vars(self, monkeypatch):
        """Picks up RANK and WORLD_SIZE from torchrun."""
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "8")

        from revolver.pytorch import _detect_distributed_env
        ws, r = _detect_distributed_env()
        assert ws == 8
        assert r == 3

    def test_partial_env_ignored(self, monkeypatch):
        """If only RANK is set (no WORLD_SIZE), falls back to defaults."""
        monkeypatch.setenv("RANK", "2")
        monkeypatch.delenv("WORLD_SIZE", raising=False)

        from revolver.pytorch import _detect_distributed_env
        ws, r = _detect_distributed_env()
        assert ws == 1
        assert r == 0

    def test_invalid_env_ignored(self, monkeypatch):
        """Non-integer values fall back to defaults."""
        monkeypatch.setenv("RANK", "abc")
        monkeypatch.setenv("WORLD_SIZE", "xyz")

        from revolver.pytorch import _detect_distributed_env
        ws, r = _detect_distributed_env()
        assert ws == 1
        assert r == 0

    def test_zero_rank(self, monkeypatch):
        """Rank 0 is valid and should not be treated as missing."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")

        from revolver.pytorch import _detect_distributed_env
        ws, r = _detect_distributed_env()
        assert ws == 4
        assert r == 0

    def test_single_gpu_torchrun(self, monkeypatch):
        """torchrun --nproc_per_node=1 sets WORLD_SIZE=1, RANK=0."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")

        from revolver.pytorch import _detect_distributed_env
        ws, r = _detect_distributed_env()
        assert ws == 1
        assert r == 0


class TestCheckpointManagerAutoDetect:
    """Test that CheckpointManager uses auto-detection correctly."""

    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def test_auto_detect_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")

        from revolver import CheckpointManager
        mgr = CheckpointManager(storage_root=str(tmp_path))
        assert mgr.world_size == 4
        assert mgr.rank == 2

    def test_explicit_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "5")
        monkeypatch.setenv("WORLD_SIZE", "8")

        from revolver import CheckpointManager
        mgr = CheckpointManager(storage_root=str(tmp_path), world_size=2, rank=1)
        assert mgr.world_size == 2
        assert mgr.rank == 1

    def test_defaults_without_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)

        from revolver import CheckpointManager
        mgr = CheckpointManager(storage_root=str(tmp_path))
        assert mgr.world_size == 1
        assert mgr.rank == 0
