"""One rank of the multi-rank checkpoint test. Launched by test_multirank.py.

A standalone script rather than a spawned function, on purpose:
``torch.multiprocessing.spawn`` has to re-import the parent module by name,
which is fragile under pytest, and ``torchrun`` brings its own launcher
problems on Windows. Plain subprocesses handed the environment torchrun would
have set are the portable common denominator, and they exercise the same
auto-detection path.

Every assertion here runs inside the rank. A failure exits non-zero and the
parent test reports that rank's output.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from moonclip import CheckpointManager


def build():
    """Same architecture and seed on every rank, as DDP requires."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    return model, ddp_model, optimizer


def train_one_step(ddp_model, optimizer):
    loss = ddp_model(torch.randn(4, 64)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 1. The manager takes the topology from the environment, the way it would
    #    under torchrun. Asked for by name since 0.0.9: it used to happen on
    #    its own, which meant the same code behaved differently depending on
    #    the launcher, and nothing said so.
    manager = CheckpointManager(storage_root=args.dir, world_size="auto", rank="auto")
    assert manager.world_size == world_size, (
        f"world_size: got {manager.world_size}, expected {world_size}"
    )
    assert manager.rank == rank, f"rank: got {manager.rank}, expected {rank}"

    model, ddp_model, optimizer = build()
    train_one_step(ddp_model, optimizer)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}

    # 2. Multi-rank save: create_snapshot on rank 0, save_rank on each rank,
    #    finalize on rank 0. Every rank has to take part.
    manager.save(step=100, model=ddp_model, optimizer=optimizer, metadata={"n": "1"})
    dist.barrier()

    if rank == 0:
        snapshots = manager.list_snapshots()
        assert len(snapshots) == 1, f"expected one snapshot, got {len(snapshots)}"
        assert snapshots[0]["ranks"] == world_size, (
            f"snapshot records {snapshots[0]['ranks']} ranks, expected {world_size}"
        )
    dist.barrier()

    # 3. Wipe the weights and reload from the checkpoint. A fresh manager, as a
    #    restarted process would build.
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    reloaded = CheckpointManager(storage_root=args.dir, world_size="auto", rank="auto")
    reloaded.load_latest(model=ddp_model, optimizer=optimizer)
    dist.barrier()

    for name, param in model.named_parameters():
        assert torch.allclose(param, original[name], atol=1e-6), (
            f"{name} did not come back"
        )

    # 4. A second save from an almost unchanged model must be a delta, not a
    #    full copy - the whole point of the engine, and it has to hold in
    #    multi-rank mode too.
    train_one_step(ddp_model, optimizer)
    manager.save(step=200, model=ddp_model, optimizer=optimizer)
    dist.barrier()

    if rank == 0:
        latest = manager.list_snapshots()[-1]
        assert latest["is_delta"], "the second multi-rank save should be a delta"
        assert latest["skipped_tensors"] + latest["delta_tensors"] > 0, (
            "delta snapshot stored nothing incrementally"
        )
    dist.barrier()

    dist.destroy_process_group()
    print(f"rank {rank}/{world_size} ok (pid {os.getpid()})")


if __name__ == "__main__":
    main()
