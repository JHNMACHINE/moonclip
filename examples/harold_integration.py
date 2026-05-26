"""
Example: integrating Revolver into Harold's training loop.

Drop-in replacement for the torch.save/torch.load pattern in checkpoint.py.
Now with background (async) saves that don't block the training loop.
"""

import time
import pickle
import concurrent.futures
from typing import Any
from revolver import CheckpointManager

# ─── Setup (once, in train.py or setup.py) ────────────────────────

manager = CheckpointManager(
    storage_root="./checkpoints",   # or /workspace/checkpoints on Vast.ai
    compression_level=3,            # zstd level, 1=fast 9=small
    max_full_snapshots=3,           # keep last 3 full snapshots
    max_deltas_per_full=10,         # up to 10 deltas between each full
    full_every_steps=5000,          # force full snapshot every 5k steps
    delta_threshold=0.5,            # if >50% bytes changed, save full
)

# Thread pool with 1 worker to execute saves sequentially in the background
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
active_future = None


# ─── Save checkpoint (async — doesn't block training) ─────────────

def save_checkpoint(step, model, optimizer, scheduler=None, scaler=None, **extra_metadata):
    """
    Save a training checkpoint asynchronously.

    The bytes are copied and handed to a background thread that handles
    delta computation, compression, and disk I/O. Returns immediately
    so the next training step can start.

    If the previous save is still in-flight, blocks until it completes
    before submitting the new one (at most 1 save in-flight).
    """
    global active_future

    # If the previous save is still in-flight, block until it completes
    if active_future is not None:
        active_future.result()

    # Extract state_dicts and convert to tensor bytes in the main thread
    # to avoid race conditions as training continues.
    from revolver.pytorch import _flatten_and_extract_tensors
    all_tensors = {}

    def _add(prefix: str, obj: Any):
        if obj is None:
            return
        sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
        meta = _flatten_and_extract_tensors(sd, prefix, all_tensors)
        meta_bytes = pickle.dumps(meta)
        all_tensors[f"{prefix}._metadata"] = ([], "uint8", meta_bytes)

    _add("model", model)
    _add("optimizer", optimizer)
    _add("scheduler", scheduler)
    _add("scaler", scaler)

    metadata = {"step": str(step)}
    metadata.update({k: str(v) for k, v in extra_metadata.items()})

    # Submit the actual saving (delta, compression, writing) to the background thread
    active_future = executor.submit(
        manager._mgr.save_tensors,
        step=step,
        tensors=all_tensors,
        metadata=metadata
    )
    print(f"[Revolver] Submitted step {step} (background)")


def save_checkpoint_sync(step, model, optimizer, **kwargs):
    """Blocking version — use at end of training or when you need the save done NOW."""
    global active_future
    # Wait for any in-flight background save to complete
    if active_future is not None:
        active_future.result()

    t0 = time.perf_counter()
    snap_id = manager.save(
        step=step,
        model=model,
        optimizer=optimizer,
        metadata={k: str(v) for k, v in kwargs.items()}
    )
    elapsed = time.perf_counter() - t0
    print(f"[Revolver] Saved step {step} -> {snap_id[:8]}... ({elapsed:.2f}s sync)")
    return snap_id


# ─── Load checkpoint ──────────────────────────────────────────────

def load_checkpoint(model, optimizer, scheduler=None, scaler=None, snap_id=None):
    """
    Load a checkpoint. If snap_id is None, loads the latest.
    Waits for any in-flight background save to complete first.
    """
    global active_future
    if active_future is not None:
        active_future.result()

    t0 = time.perf_counter()

    if snap_id is None:
        snap_id, _ = manager.load_latest(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )
    else:
        manager.load(
            snap_id,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )

    elapsed = time.perf_counter() - t0
    print(f"[Revolver] Loaded {snap_id[:8]}... ({elapsed:.2f}s)")
    return snap_id


def wait_for_bg_save():
    """Wait for any active background save to complete."""
    global active_future
    if active_future is not None:
        snap_id = active_future.result()
        active_future = None
        return snap_id
    return None


# ─── Training loop example ────────────────────────────────────────

def training_loop_example():
    import torch
    import torch.nn as nn

    model = nn.Linear(512, 512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Resume if checkpoints exist
    snaps = manager.list_snapshots()
    start_step = 0
    if snaps:
        snap_id = load_checkpoint(model, optimizer)
        start_step = int(snaps[-1]["step"]) + 1
        print(f"Resumed from step {start_step}")

    for step in range(start_step, 100000):  # Run 5 steps for demonstration
        # ... forward, backward, optimizer.step() ...
        loss = 0.5

        # Async save every 2 steps for demonstration
        if step > 0 and step % 100 == 0:
            save_checkpoint(
                step=step,
                model=model,
                optimizer=optimizer,
                loss=f"{loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    # Final save: sync to guarantee it's done before exit
    save_checkpoint_sync(step=100000, model=model, optimizer=optimizer)

    # Check what the background saver did
    snap_id = wait_for_bg_save()
    if snap_id:
        print(f"Last bg save completed: {snap_id[:8]}...")


if __name__ == "__main__":
    training_loop_example()
