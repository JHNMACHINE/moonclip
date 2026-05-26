"""
Example: training a small Transformer on CPU with Revolver checkpointing.

Demonstrates using CheckpointManager to save and restore model, optimizer,
and scheduler states during real training.
"""

import time
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from revolver import CheckpointManager


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
    # Keep last 3 full checkpoints and up to 5 deltas
    manager = CheckpointManager(
        storage_root="./checkpoints",
        compression_level=3,
        max_full_snapshots=10,
        max_deltas_per_full=10,
        full_every_steps=100,
    )

    # 3. Create synthetic sequence dataset: sequence copying task
    x_data = torch.randint(0, vocab_size, (256, seq_len))
    y_data = torch.randint(0, vocab_size, (256, seq_len))
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    # ─── 4. Run Training Steps and Save Checkpoints ─────────────────────
    step = 0
    checkpoint_steps = {} # step -> snap_id

    # For thread pool / background saving:
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    active_future = None

    def save_bg(step_num):
        nonlocal active_future
        if active_future is not None:
            active_future.result() # Wait for previous save to finish
        
        # Copy state_dicts in main thread
        from revolver.pytorch import _flatten_and_extract_tensors
        import pickle
        all_tensors = {}

        def _add(prefix: str, obj: Any):
            if obj is None:
                return
            sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
            meta = _flatten_and_extract_tensors(sd, prefix, all_tensors)
            all_tensors[f"{prefix}._metadata"] = ([], "uint8", pickle.dumps(meta))

        _add("model", model)
        _add("optimizer", optimizer)
        _add("scheduler", scheduler)

        # Submit saving to background executor
        active_future = executor.submit(
            manager._mgr.save_tensors,
            step=step_num,
            tensors=all_tensors,
            metadata={"step": str(step_num)}
        )
        print(f"[Transformer Train] Submitted background save for step {step_num}")

    model.train()
    for epoch in range(2):
        for x, y in dataloader:
            outputs = model(x)
            # Reshape for cross entropy
            loss = loss_fn(outputs.view(-1, vocab_size), y.view(-1))

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            print(f"Step {step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            # Checkpoint at step 5 (background) and step 10 (synchronous)
            if step == 5:
                save_bg(step)
            elif step == 10:
                # Sync save
                if active_future is not None:
                    active_future.result() # Wait for any pending saves
                snap_id = manager.save(step=step, model=model, optimizer=optimizer, scheduler=scheduler)
                checkpoint_steps[step] = snap_id
                print(f"[Transformer Train] Saved synchronous checkpoint for step {step} -> {snap_id[:8]}...")
                break
        if step >= 10:
            break

    # Wait for the background save (step 5) to finish if it hasn't already
    if active_future is not None:
        snap_id_5 = active_future.result()
        checkpoint_steps[5] = snap_id_5
        print(f"[Transformer Train] Background save for step 5 completed -> {snap_id_5[:8]}...")

    # Let's save the current model/optimizer states for verification
    weights_before_resume = [p.clone().detach() for p in model.parameters()]
    lr_before_resume = scheduler.get_last_lr()[0]

    # ─── 5. Resume from Checkpoint at Step 5 ────────────────────────────
    print("\nResuming from checkpoint at step 5...")
    snap_id = checkpoint_steps[5]
    manager.load(snap_id, model=model, optimizer=optimizer, scheduler=scheduler)

    # Let's verify that the model has indeed rolled back
    weights_after_resume = [p.clone().detach() for p in model.parameters()]
    # Check that they differ from the mutated weights before resume
    assert not all(torch.equal(w1, w2) for w1, w2 in zip(weights_before_resume, weights_after_resume))
    
    # Run a forward pass on the rolled-back model
    outputs = model(x_data[:batch_size])
    loss_val = loss_fn(outputs.view(-1, vocab_size), y_data[:batch_size].view(-1))
    print(f"Rolled back model successfully. Validation loss at step 5 state: {loss_val.item():.4f}")
    print(f"Restored Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    print("\nResuming from checkpoint at step 10...")
    snap_id = checkpoint_steps[10]
    manager.load(snap_id, model=model, optimizer=optimizer, scheduler=scheduler)
    
    weights_step_10 = [p.clone().detach() for p in model.parameters()]
    for w_before, w_after in zip(weights_before_resume, weights_step_10):
        assert torch.equal(w_before, w_after), "Restored weights at step 10 do not match original weights!"
    print("Verification passed. Weights match step 10 checkpoint perfectly!")


if __name__ == "__main__":
    train_transformer()
