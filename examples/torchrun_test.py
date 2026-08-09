"""
Test: Multi-rank checkpoint save/load via torchrun.

Usage:
    torchrun --nproc_per_node=2 examples/torchrun_test.py

On Windows (if torchrun has libuv issues):
    python examples/torchrun_test.py --spawn

Verifies:
1. CheckpointManager auto-detects RANK and WORLD_SIZE
2. Each rank saves its own shard
3. Checkpoint survives resume
4. Per-tensor delta tracking works in distributed mode
"""

import argparse
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from moonclip import CheckpointManager


def run_test(rank=None, world_size=None, ckpt_dir=None):
    """Main test logic — works both with torchrun and mp.spawn."""

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}/{world_size}] Started")

    # Share ckpt_dir from rank 0
    if ckpt_dir is None:
        if rank == 0:
            ckpt_dir = tempfile.mkdtemp(prefix="moonclip_torchrun_")
        else:
            ckpt_dir = ""
        objects = [ckpt_dir]
        dist.broadcast_object_list(objects, src=0)
        ckpt_dir = objects[0]

    # ─── Test 1: Auto-detection ──────────────────────────────────────
    mgr = CheckpointManager(storage_root=ckpt_dir)
    assert mgr.world_size == world_size, f"Expected ws={world_size}, got {mgr.world_size}"
    assert mgr.rank == rank, f"Expected rank={rank}, got {mgr.rank}"
    print(f"[Rank {rank}] ✓ Auto-detected world_size={mgr.world_size}, rank={mgr.rank}")

    # ─── Test 2: Save ────────────────────────────────────────────────
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    x = torch.randn(4, 64)
    loss = ddp_model(x).sum()
    loss.backward()
    optimizer.step()

    original = {n: p.data.clone() for n, p in model.named_parameters()}

    snap_id = mgr.save(step=100, model=ddp_model, optimizer=optimizer,
                       metadata={"loss": f"{loss.item():.4f}"})
    dist.barrier()
    print(f"[Rank {rank}] ✓ Saved {snap_id[:8]}...")

    # ─── Test 3: Snapshot info ───────────────────────────────────────
    if rank == 0:
        snaps = mgr.list_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["ranks"] == world_size
        print(f"[Rank 0] ✓ Snapshot: {snaps[0]['full_tensors']} full, ranks={snaps[0]['ranks']}")
    dist.barrier()

    # ─── Test 4: Corrupt and reload ──────────────────────────────────
    for p in model.parameters():
        p.data.zero_()

    mgr2 = CheckpointManager(storage_root=ckpt_dir)
    mgr2.load_latest(model=ddp_model, optimizer=optimizer)
    dist.barrier()

    for n, p in model.named_parameters():
        assert torch.allclose(p.data, original[n], atol=1e-6), f"Mismatch in {n}!"
    print(f"[Rank {rank}] ✓ Weights restored correctly")

    # ─── Test 5: Delta checkpoint ────────────────────────────────────
    loss = ddp_model(torch.randn(4, 64)).sum()
    loss.backward()
    optimizer.step()

    mgr.save(step=200, model=ddp_model, optimizer=optimizer)
    dist.barrier()

    if rank == 0:
        info = mgr.list_snapshots()[-1]
        assert info["is_delta"], "Second save should be delta"
        print(f"[Rank 0] ✓ Delta: skip={info['skipped_tensors']}, "
              f"delta={info['delta_tensors']}, full={info['full_tensors']}")
    dist.barrier()

    # ─── Cleanup ─────────────────────────────────────────────────────
    if rank == 0:
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    dist.destroy_process_group()
    print(f"[Rank {rank}] ✓ All tests passed!")


def spawn_worker(rank, world_size, ckpt_dir):
    """Entry point for mp.spawn."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    run_test(ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", action="store_true",
                        help="Use mp.spawn instead of torchrun (Windows fallback)")
    parser.add_argument("--nproc", type=int, default=2)
    args = parser.parse_args()

    if args.spawn:
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(spawn_worker, args=(args.nproc, d), nprocs=args.nproc, join=True) # type: ignore
    else:
        run_test()
