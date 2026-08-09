"""
Example: training a small Transformer on CPU with Moonclip checkpointing.

Demonstrates using CheckpointManager to save and restore model, optimizer,
and scheduler states during real training, with background async saves.
"""

import time
import concurrent.futures
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from moonclip import CheckpointManager, flatten_state_dict


# ─── Simple Transformer Model ────────────────────────────────────────

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out = self.transformer(embedded)
        return self.fc_out(out)


# ─── Training Loop ──────────────────────────────────────────────────

def train_transformer():
    print("Initializing toy Transformer training on CPU...")
    vocab_size = 32
    seq_len = 16
    batch_size = 16

    # 1. Create Model, Optimizer, and Scheduler
    model = SmallTransformer(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 2. Setup CheckpointManager
    manager = CheckpointManager(
        storage_root="./checkpoints",
        compression_level=3,
        max_full_snapshots=10,
        max_deltas_per_full=10,
        full_every_steps=100,
    )

    # 3. Create synthetic sequence dataset
    x_data = torch.randint(0, vocab_size, (256, seq_len))
    y_data = torch.randint(0, vocab_size, (256, seq_len))
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    # ─── Background save setup ──────────────────────────────────────
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    active_future: Optional[concurrent.futures.Future] = None

    def save_bg(step_num: int):
        nonlocal active_future
        if active_future is not None:
            active_future.result()

        # Serialize tensors in main thread (safe from race conditions)
        all_tensors: dict = {}
        for prefix, obj in [("model", model), ("optimizer", optimizer), ("scheduler", scheduler)]:
            sd = obj.state_dict()
            tensors, _ = flatten_state_dict(sd, prefix)
            all_tensors.update(tensors)

        # Submit to background thread
        active_future = executor.submit(
            manager.save_raw,
            step=step_num,
            tensors=all_tensors,
            metadata={"step": str(step_num)},
        )
        print(f"  [ckpt] Submitted background save for step {step_num}")

    # ─── Training ────────────────────────────────────────────────────
    step = 0
    checkpoint_steps: dict = {}

    model.train()
    for epoch in range(2):
        for x, y in dataloader:
            outputs = model(x)
            loss = loss_fn(outputs.view(-1, vocab_size), y.view(-1))

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            print(f"  Step {step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            # Background save at step 5
            if step == 5:
                save_bg(step)

            # Sync save at step 10
            elif step == 10:
                if active_future is not None:
                    active_future.result()
                snap_id = manager.save(
                    step=step, model=model, optimizer=optimizer, scheduler=scheduler,
                )
                checkpoint_steps[step] = snap_id
                print(f"  [ckpt] Sync save step {step} → {snap_id[:8]}...")
                break
        if step >= 10:
            break

    # Wait for background save to complete
    if active_future is not None:
        snap_id_bg = active_future.result()
        checkpoint_steps[5] = snap_id_bg
        print(f"  [ckpt] Background save step 5 → {snap_id_bg[:8]}...")

    # Save weights at step 10 for later verification
    weights_at_step_10 = {n: p.clone() for n, p in model.named_parameters()}

    # ─── Resume from step 5 ──────────────────────────────────────────
    print("\nResuming from checkpoint at step 5...")
    manager.load(checkpoint_steps[5], model=model, optimizer=optimizer, scheduler=scheduler)

    weights_at_step_5 = {n: p.clone() for n, p in model.named_parameters()}
    assert not all(
        torch.equal(weights_at_step_10[n], weights_at_step_5[n])
        for n in weights_at_step_10
    ), "Rollback should have changed weights"

    outputs = model(x_data[:batch_size])
    loss_val = loss_fn(outputs.view(-1, vocab_size), y_data[:batch_size].view(-1))
    print(f"  Rolled back to step 5. Loss: {loss_val.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # ─── Resume from step 10 ─────────────────────────────────────────
    print("\nResuming from checkpoint at step 10...")
    manager.load(checkpoint_steps[10], model=model, optimizer=optimizer, scheduler=scheduler)

    for n, p in model.named_parameters():
        assert torch.equal(p.data, weights_at_step_10[n]), f"Mismatch in {n}!"
    print("  ✓ Weights match step 10 perfectly!")

    manager.print_stats()
    executor.shutdown(wait=True)


if __name__ == "__main__":
    train_transformer()
