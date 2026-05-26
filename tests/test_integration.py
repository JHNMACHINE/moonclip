"""Integration tests for Revolver v1.0 — per-tensor, rank-aware."""

import os
import sys
import struct
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from revolver import RevolverManager


def make_tensors(seed=0, size=8192):
    """Create test tensors in the format (shape, dtype, bytes)."""
    return {
        "model.weight": ([64, 32], "float32", bytes([seed] * size)),
        "model.bias": ([64], "float32", bytes([seed + 1] * 256)),
        "optimizer.exp_avg": ([64, 32], "float32", bytes([0] * size)),
    }


def test_save_load_roundtrip(tmp_path):
    mgr = RevolverManager(storage_root=str(tmp_path), full_every_steps=100000)

    tensors = make_tensors(42)
    snap_id = mgr.save_tensors(step=1000, tensors=tensors, metadata={"loss": "0.634"})

    assert len(snap_id) == 36  # UUID

    loaded = mgr.load(snap_id)
    assert loaded["model.weight"] == tensors["model.weight"][2]
    assert loaded["model.bias"] == tensors["model.bias"][2]
    assert loaded["optimizer.exp_avg"] == tensors["optimizer.exp_avg"][2]


def test_per_tensor_skip(tmp_path):
    """Unchanged tensors should be skipped (zero I/O)."""
    mgr = RevolverManager(storage_root=str(tmp_path), full_every_steps=100000)

    tensors_v1 = make_tensors(0)
    mgr.save_tensors(step=100, tensors=tensors_v1)

    # Only change model.weight, keep bias and exp_avg identical
    tensors_v2 = make_tensors(0)
    data = bytearray(tensors_v2["model.weight"][2])
    data[0] = 1
    tensors_v2["model.weight"] = ([64, 32], "float32", bytes(data))

    snap_id = mgr.save_tensors(step=200, tensors=tensors_v2)

    snaps = mgr.list_snapshots()
    assert len(snaps) == 2
    info = snaps[1]
    assert info["is_delta"] is True
    assert info["skipped_tensors"] == 2, f"Expected 2 skipped, got {info['skipped_tensors']}"

    # Verify data integrity
    loaded = mgr.load(snap_id)
    assert loaded["model.weight"] == tensors_v2["model.weight"][2]
    assert loaded["model.bias"] == tensors_v1["model.bias"][2]


def test_per_tensor_delta(tmp_path):
    """Changed tensors should use XOR delta compression."""
    mgr = RevolverManager(storage_root=str(tmp_path), full_every_steps=100000)

    # Large tensor to trigger delta (>4096 bytes)
    big = bytes(100_000)
    tensors_v1 = {"big_tensor": ([100000], "uint8", big)}
    mgr.save_tensors(step=100, tensors=tensors_v1)

    # Change 2 bytes
    big_v2 = bytearray(big)
    big_v2[0] = 1
    big_v2[50000] = 2
    tensors_v2 = {"big_tensor": ([100000], "uint8", bytes(big_v2))}
    snap_id = mgr.save_tensors(step=200, tensors=tensors_v2)

    snaps = mgr.list_snapshots()
    info = snaps[1]
    assert info["delta_tensors"] == 1
    assert info["total_compressed"] < 1000, f"Delta should be tiny, got {info['total_compressed']}"

    loaded = mgr.load(snap_id)
    assert loaded["big_tensor"] == bytes(big_v2)


def test_load_latest(tmp_path):
    mgr = RevolverManager(storage_root=str(tmp_path), full_every_steps=100000)

    mgr.save_tensors(step=100, tensors=make_tensors(1))
    mgr.save_tensors(step=200, tensors=make_tensors(2))

    snap_id, loaded = mgr.load_latest()
    assert loaded["model.weight"][0] == 2


def test_integrity_check(tmp_path):
    mgr = RevolverManager(storage_root=str(tmp_path), full_every_steps=100000)
    snap_id = mgr.save_tensors(step=1, tensors=make_tensors(42))

    # Corrupt ALL shard files
    snaps_dir = os.path.join(str(tmp_path), "snapshots")
    corrupted = 0
    for root, dirs, files in os.walk(snaps_dir):
        for f in files:
            if f.endswith(".bin"):
                path = os.path.join(root, f)
                with open(path, "r+b") as fh:
                    fh.seek(0)
                    fh.write(b"\xFF\xFF\xFF\xFF")
                corrupted += 1

    assert corrupted > 0, "No .bin files found to corrupt"

    with pytest.raises(RuntimeError) as excinfo:
        mgr.load(snap_id)
    assert "Integrity" in str(excinfo.value) or "Compression" in str(excinfo.value)


def test_snapshot_stats(tmp_path):
    """list_snapshots returns per-tensor breakdown."""
    mgr = RevolverManager(storage_root=str(tmp_path), full_every_steps=100000)

    mgr.save_tensors(step=100, tensors=make_tensors(0), metadata={"loss": "1.0"})

    snaps = mgr.list_snapshots()
    assert len(snaps) == 1
    info = snaps[0]
    assert info["step"] == 100
    assert info["metadata"]["loss"] == "1.0"
    assert info["full_tensors"] == 3
    assert info["ranks"] == 1


def test_multi_rank(tmp_path):
    """Multi-rank save/load flow."""
    # Rank 0 creates snapshot
    mgr_r0 = RevolverManager(storage_root=str(tmp_path), world_size=2, rank=0, full_every_steps=100000)
    snap_id = mgr_r0.create_snapshot(step=1000, metadata={"loss": "0.5"})

    # Both ranks save their shards
    mgr_r0.save_rank(snap_id, tensors={
        "shard.weight": ([256, 512], "float32", bytes([1] * 256 * 512 * 4)),
    })

    mgr_r1 = RevolverManager(storage_root=str(tmp_path), world_size=2, rank=1, full_every_steps=100000)
    mgr_r1.save_rank(snap_id, tensors={
        "shard.weight": ([256, 512], "float32", bytes([2] * 256 * 512 * 4)),
    })

    # Rank 0 finalizes
    mgr_r0_final = RevolverManager(storage_root=str(tmp_path), world_size=2, rank=0, full_every_steps=100000)
    mgr_r0_final.finalize_snapshot(snap_id)

    # Each rank loads its shard
    loaded_r0 = RevolverManager(storage_root=str(tmp_path), world_size=2, rank=0).load(snap_id)
    loaded_r1 = RevolverManager(storage_root=str(tmp_path), world_size=2, rank=1).load(snap_id)

    assert loaded_r0["shard.weight"][0] == 1
    assert loaded_r1["shard.weight"][0] == 2


def test_compression_ratio(tmp_path):
    """Measure compression on realistic patterns."""
    mgr = RevolverManager(storage_root=str(tmp_path), compression_level=3, full_every_steps=100000)

    # Optimizer momentum: repetitive float data
    raw = struct.pack(f"{25000}f", *([0.0001] * 25000))
    tensors = {"optimizer.momentum": ([25000], "float32", raw)}
    mgr.save_tensors(step=1, tensors=tensors)

    snaps = mgr.list_snapshots()
    ratio = snaps[0]["total_compressed"] / snaps[0]["total_raw"]
    assert ratio < 0.1


# ─── Production Battle Tests ──────────────────────────────────────────

def test_extreme_dimensions(tmp_path):
    """Test saving/loading tensors with extreme shapes (0D, 1D, 5D)."""
    mgr = RevolverManager(storage_root=str(tmp_path), full_every_steps=100)

    tensors = {
        "scalar": ([], "float32", struct.pack("f", 3.14)),
        "single": ([1], "int32", struct.pack("i", 42)),
        "large_dim": ([2, 2, 2, 2, 2], "uint8", bytes([1] * 32)),
    }
    snap_id = mgr.save_tensors(step=1, tensors=tensors)
    loaded = mgr.load(snap_id)

    assert loaded["scalar"] == tensors["scalar"][2]
    assert loaded["single"] == tensors["single"][2]
    assert loaded["large_dim"] == tensors["large_dim"][2]


def test_cleanup_and_retention(tmp_path):
    """Verify that old snapshot directories and files are deleted according to retention policy."""
    mgr = RevolverManager(
        storage_root=str(tmp_path),
        max_full_snapshots=2,
        max_deltas_per_full=2,
        full_every_steps=5,
    )

    tensors = {"weight": ([100], "uint8", bytes([1] * 100))}

    # Save 9 snapshots to trigger multiple rounds of retention
    for step in range(1, 10):
        mgr.save_tensors(step=step, tensors=tensors)

    snaps = mgr.list_snapshots()
    # At most 2 full snapshots + their deltas should exist
    assert len(snaps) <= 6

    # Verify no files remain on disk for the pruned snapshots
    all_snap_dirs = os.listdir(os.path.join(str(tmp_path), "snapshots"))
    active_ids = {s["id"] for s in snaps}
    for d in all_snap_dirs:
        if d not in active_ids:
            d_path = os.path.join(str(tmp_path), "snapshots", d)
            files_left = []
            for root, dirs, files in os.walk(d_path):
                for f in files:
                    files_left.append(f)
            assert len(files_left) == 0, f"Snapshot {d} was pruned but still has files: {files_left}"


def test_concurrent_saves(tmp_path):
    """Verify that multiple threads can call save_tensors concurrently without crashes."""
    import threading
    mgr = RevolverManager(storage_root=str(tmp_path), full_every_steps=100)
    
    errors = []
    
    def worker(step):
        try:
            tensors = {f"thread_{step}": ([10], "uint8", bytes([step] * 10))}
            mgr.save_tensors(step=step, tensors=tensors)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 11)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
    snaps = mgr.list_snapshots()
    # Since they saved concurrently, they should all be in the manifest
    assert len(snaps) == 10


def test_pytorch_non_standard_types(tmp_path):
    """Verify CheckpointManager can save/load custom classes, empty dicts, and non-standard types."""
    from revolver import CheckpointManager
    mgr = CheckpointManager(storage_root=str(tmp_path))

    class CustomStateObject:
        def __init__(self, val):
            self.val = val
        def state_dict(self):
            return {"value": self.val, "empty_dict": {}, "nested": {"list": [1, 2, 3]}}
        def load_state_dict(self, sd):
            self.val = sd["value"]
            self._sd = sd

    obj = CustomStateObject(42)
    snap_id = mgr.save(step=1, model=obj)

    restored = CustomStateObject(0)
    mgr.load(snap_id, model=restored)

    assert restored.val == 42
    assert restored._sd["empty_dict"] == {}
    assert restored._sd["nested"]["list"] == [1, 2, 3]
