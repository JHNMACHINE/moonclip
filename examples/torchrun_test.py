"""
Test: Multi-rank checkpoint save/load via torchrun.

Run with:
    torchrun --nproc_per_node=2 examples/torchrun_test.py

This verifies that:
1. CheckpointManager auto-detects RANK and WORLD_SIZE from torchrun env
2. Each rank saves its own shard
3. Checkpoint survives resume across all ranks
4. Per-tensor delta tracking works in distributed mode
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from revolver import CheckpointManager


def main():
    # torchrun initializes these env vars:
    #   RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    # CheckpointManager should auto-detect them.

    # Initialize process group
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}/{world_size}] Started")

    # Use a shared temp directory (all ranks must see the same path)
    if rank == 0:
        ckpt_dir = tempfile.mkdtemp(prefix="revolver_torchrun_test_")
    else:
        ckpt_dir = ""

    # Broadcast ckpt_dir from rank 0
    objects = [ckpt_dir]
    dist.broadcast_object_list(objects, src=0)
    ckpt_dir = objects[0]
    print(f"[Rank {rank}] Checkpoint dir: {ckpt_dir}")

    # Create model + DDP
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    # ─── Test 1: Auto-detection ──────────────────────────────────────
    # CheckpointManager should pick up RANK and WORLD_SIZE automatically
    mgr = CheckpointManager(storage_root=ckpt_dir)
    assert mgr.world_size == world_size, f"Expected world_size={world_size}, got {mgr.world_size}"
    assert mgr.rank == rank, f"Expected rank={rank}, got {mgr.rank}"
    print(f"[Rank {rank}] ✓ Auto-detected world_size={mgr.world_size}, rank={mgr.rank}")

    # ─── Test 2: Save checkpoint ─────────────────────────────────────
    # Do a training step so optimizer has state
    x = torch.randn(4, 64)
    loss = ddp_model(x).sum()
    loss.backward()
    optimizer.step()

    # Save original weights for comparison
    original_weights = {
        name: param.data.clone() for name, param in model.named_parameters()
    }

    snap_id = mgr.save(step=100, model=ddp_model, optimizer=optimizer, metadata={"loss": f"{loss.item():.4f}"})
    dist.barrier()

    print(f"[Rank {rank}] ✓ Saved checkpoint {snap_id[:8]}...")

    # ─── Test 3: Verify snapshot info ────────────────────────────────
    if rank == 0:
        snaps = mgr.list_snapshots()
        assert len(snaps) == 1
        info = snaps[0]
        assert info["ranks"] == world_size
        assert info["step"] == 100
        print(f"[Rank 0] ✓ Snapshot info: {info['full_tensors']} full, "
              f"{info['total_compressed']//1024}KB compressed, "
              f"ranks={info['ranks']}")
    dist.barrier()

    # ─── Test 4: Corrupt and reload ──────────────────────────────────
    # Zero out model weights
    for param in model.parameters():
        param.data.zero_()

    # Load checkpoint
    mgr2 = CheckpointManager(storage_root=ckpt_dir)
    mgr2.load_latest(model=ddp_model, optimizer=optimizer)
    dist.barrier()

    # Verify weights match
    for name, param in model.named_parameters():
        assert torch.allclose(param.data, original_weights[name], atol=1e-6), \
            f"[Rank {rank}] Weight mismatch for {name}"
    print(f"[Rank {rank}] ✓ Weights restored correctly after load")

    # ─── Test 5: Delta checkpoint ────────────────────────────────────
    # Do another step
    x = torch.randn(4, 64)
    loss = ddp_model(x).sum()
    loss.backward()
    optimizer.step()

    snap_id2 = mgr.save(step=200, model=ddp_model, optimizer=optimizer)
    dist.barrier()

    if rank == 0:
        snaps = mgr.list_snapshots()
        info = snaps[-1]
        assert info["is_delta"] is True, "Second save should be delta"
        print(f"[Rank 0] ✓ Delta checkpoint: skip={info['skipped_tensors']}, "
              f"delta={info['delta_tensors']}, full={info['full_tensors']}")

    dist.barrier()

    # ─── Cleanup ─────────────────────────────────────────────────────
    if rank == 0:
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    dist.destroy_process_group()
    print(f"[Rank {rank}] ✓ All tests passed!")


if __name__ == "__main__":
    main()
