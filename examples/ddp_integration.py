"""
Example: integrating Revolver into a PyTorch Distributed Data Parallel (DDP) training loop.

This example spawns a 2-process distributed group, initializes a toy DDP model,
and demonstrates how to save and load checkpoints using CheckpointManager in a multi-rank setting.
"""

import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from revolver import CheckpointManager


# A simple model to use with DDP
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.net(x)


def run_worker(rank, world_size, storage_root):
    # Initialize distributed process group using Gloo backend (supports CPU)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Initialize model and wrap it in DDP
    model = ToyModel()
    ddp_model = DDP(model)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    # Create the CheckpointManager with multi-rank parameters
    manager = CheckpointManager(
        storage_root=storage_root,
        world_size=world_size,
        rank=rank,
        compression_level=3,
        max_full_snapshots=3,
    )

    print(f"[Rank {rank}] Initialized and ready.")

    # ─── 1. Run a dummy training step ─────────────────────────────────
    inputs = torch.randn(8, 128)
    outputs = ddp_model(inputs)
    loss = outputs.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Save parameters for verification later
    local_weights_before_save = [p.clone().detach() for p in ddp_model.parameters()]
    print(f"[Rank {rank}] Completed step 1. Loss = {loss.item():.4f}")

    # ─── 2. Multi-rank Save ──────────────────────────────────────────
    # Behind the scenes:
    # - Rank 0 creates the snapshot entry.
    # - The snapshot ID is broadcasted to all ranks.
    # - Each rank saves its own model/optimizer state dictionary shards.
    # - dist.barrier() coordinates the process.
    # - Rank 0 finalizes the snapshot.
    snap_id = manager.save(
        step=1,
        model=ddp_model,
        optimizer=optimizer,
        metadata={"loss": f"{loss.item():.4f}"}
    )
    print(f"[Rank {rank}] Successfully saved snapshot {snap_id[:8]}...")

    # ─── 3. Mutate parameters (simulating further training) ───────────
    # Let's perform another step to mutate parameters and optimizer state
    outputs = ddp_model(inputs)
    loss = outputs.sum()
    loss.backward()
    optimizer.step()
    
    local_weights_mutated = [p.clone().detach() for p in ddp_model.parameters()]
    # Assert that weights have indeed changed
    assert not all(torch.equal(w1, w2) for w1, w2 in zip(local_weights_before_save, local_weights_mutated))
    print(f"[Rank {rank}] Mutated weights in step 2.")

    # ─── 4. Resume / Load latest ──────────────────────────────────────
    # Behind the scenes:
    # - Rank 0 looks up the latest finalized snapshot ID.
    # - The ID is broadcasted to all ranks.
    # - Each rank loads and applies its local model and optimizer shards.
    manager.load_latest(model=ddp_model, optimizer=optimizer)
    print(f"[Rank {rank}] Resumed from latest checkpoint.")

    # ─── 5. Verify restored state ─────────────────────────────────────
    local_weights_after_load = [p.clone().detach() for p in ddp_model.parameters()]
    for w_before, w_after in zip(local_weights_before_save, local_weights_after_load):
        assert torch.equal(w_before, w_after), f"[Rank {rank}] Weight restoration mismatch!"
        
    print(f"[Rank {rank}] Verification passed. Weights match checkpoint perfectly!")

    # Clean up
    dist.destroy_process_group()


def main():
    world_size = 2
    # Create temporary directory for checkpoint storage
    with tempfile.TemporaryDirectory() as storage_root:
        print(f"Using storage root: {storage_root}")
        mp.spawn( # type: ignore
            run_worker,
            args=(world_size, storage_root),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()
