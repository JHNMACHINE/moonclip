"""
PyTorch-specific integration tests for Revolver.
Requires: pip install torch
Skipped automatically if torch is not installed.
"""

import os
import sys
import struct
import pickle

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

torch = pytest.importorskip("torch")

from revolver import CheckpointManager


class TestCheckpointManager:
    def test_model_save_load(self, tmp_path):
        mgr = CheckpointManager(storage_root=str(tmp_path))

        model = torch.nn.Linear(32, 16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Run one step so optimizer has state
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        original_weight = model.weight.data.clone()
        original_bias = model.bias.data.clone()

        snap_id = mgr.save(step=1, model=model, optimizer=optimizer, metadata={"loss": "0.5"})

        # Corrupt model
        model.weight.data.zero_()
        model.bias.data.zero_()

        mgr.load(snap_id, model=model, optimizer=optimizer)

        assert torch.allclose(model.weight.data, original_weight)
        assert torch.allclose(model.bias.data, original_bias)

    def test_load_latest(self, tmp_path):
        mgr = CheckpointManager(storage_root=str(tmp_path))
        model = torch.nn.Linear(8, 4)

        model.weight.data.fill_(1.0)
        mgr.save(step=100, model=model)

        model.weight.data.fill_(2.0)
        mgr.save(step=200, model=model)

        model.weight.data.zero_()
        mgr.load_latest(model=model)
        assert torch.allclose(model.weight.data, torch.full_like(model.weight, 2.0))

    def test_scheduler_and_scaler(self, tmp_path):
        mgr = CheckpointManager(storage_root=str(tmp_path))
        model = torch.nn.Linear(8, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        scheduler.step()
        scheduler.step()
        original_lr = optimizer.param_groups[0]["lr"]

        mgr.save(step=1, model=model, optimizer=optimizer, scheduler=scheduler)

        # Reset
        optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1)

        _, _ = mgr.load_latest(model=model, optimizer=optimizer2, scheduler=scheduler2)
        assert optimizer2.param_groups[0]["lr"] == pytest.approx(original_lr)

    def test_delta_between_training_steps(self, tmp_path):
        """After a few optimizer steps, most tensors change slightly — perfect for delta."""
        mgr = CheckpointManager(storage_root=str(tmp_path), full_every_steps=100000)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Save initial
        mgr.save(step=0, model=model, optimizer=optimizer)

        # One training step
        x = torch.randn(8, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        snap_id = mgr.save(step=1, model=model, optimizer=optimizer)

        snaps = mgr.list_snapshots()
        assert len(snaps) == 2
        # Second save should have some skipped tensors (optimizer state not all changed)
        # and delta tensors
        info = snaps[1]
        assert info["is_delta"] is True
        assert info["skipped_tensors"] > 0 or info["delta_tensors"] > 0

    def test_non_tensor_state(self, tmp_path):
        """Optimizer state_dict contains non-tensor data (step counts, etc)."""
        mgr = CheckpointManager(storage_root=str(tmp_path))

        class CustomObj:
            def state_dict(self):
                return {"count": 42, "name": "test", "nested": {"a": 1}}
            def load_state_dict(self, sd):
                self.loaded = sd

        obj = CustomObj()
        snap_id = mgr.save(step=1, model=obj)

        obj2 = CustomObj()
        mgr.load(snap_id, model=obj2)
        assert obj2.loaded["count"] == 42
        assert obj2.loaded["name"] == "test"
        assert obj2.loaded["nested"]["a"] == 1

    def test_many_layers_model(self, tmp_path):
        """Model with 20 layers — tests many-tensor handling."""
        mgr = CheckpointManager(storage_root=str(tmp_path), full_every_steps=100000)
        model = torch.nn.Sequential(*[torch.nn.Linear(32, 32) for _ in range(20)])

        mgr.save(step=0, model=model)

        # Change one layer's weights
        model[10].weight.data += 0.001

        snap_id = mgr.save(step=1, model=model)
        snaps = mgr.list_snapshots()
        info = snaps[1]

        # Most layers unchanged — should see many skips
        assert info["skipped_tensors"] > 30  # 20 layers × 2 params = 40, most skipped

        # Roundtrip
        model2 = torch.nn.Sequential(*[torch.nn.Linear(32, 32) for _ in range(20)])
        mgr.load(snap_id, model=model2)
        assert torch.allclose(model[10].weight.data, model2[10].weight.data)
