"""
Example: integrating Revolver into Harold's training loop.

Drop-in replacement for checkpoint.py with background async saves.
Uses the public flatten_state_dict + save_raw API for zero-block saves.
"""

import time
import concurrent.futures
from typing import Any, Optional, Dict

from revolver import CheckpointManager, flatten_state_dict


# ─── Setup ───────────────────────────────────────────────────────────

manager = CheckpointManager(
    storage_root="./checkpoints",
    compression_level=3,
    max_full_snapshots=3,
    max_deltas_per_full=10,
    full_every_steps=5000,
    delta_threshold=0.5,
    save_dtype="bf16",
)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
active_future: Optional[concurrent.futures.Future] = None


# ─── Save (async) ───────────────────────────────────────────────────

def save_checkpoint(step: int, model: Any, optimizer: Any, scheduler: Any = None, **extra_metadata):
    """
    Save a training checkpoint asynchronously.

    Tensors are serialized in the main thread (safe, no race conditions),
    then delta/compression/write happens in a background thread.
    """
    global active_future

    if active_future is not None:
        active_future.result()

    # Serialize tensors in main thread (must happen before next training step)
    all_tensors: dict = {}

    def _add(prefix: str, obj: Any):
        if obj is None:
            return
        sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
        tensors, _ = flatten_state_dict(sd, prefix)
        all_tensors.update(tensors)

    _add("model", model)
    _add("optimizer", optimizer)
    _add("scheduler", scheduler)

    metadata = {"step": str(step)}
    metadata.update({k: str(v) for k, v in extra_metadata.items()})

    # Submit save to background thread
    active_future = executor.submit(
        manager.save_raw,
        step=step,
        tensors=all_tensors,
        metadata=metadata,
    )
    print(f"[Revolver] Submitted step {step} (background)")


def save_checkpoint_sync(step: int, model: Any, optimizer: Any, **kwargs):
    """Blocking save — use at end of training."""
    global active_future
    if active_future is not None:
        active_future.result()

    t0 = time.perf_counter()
    snap_id = manager.save(
        step=step,
        model=model,
        optimizer=optimizer,
        metadata={k: str(v) for k, v in kwargs.items()},
    )
    elapsed = time.perf_counter() - t0
    print(f"[Revolver] Saved step {step} → {snap_id[:8]}... ({elapsed:.2f}s sync)")
    return snap_id


# ─── Load / Resume ───────────────────────────────────────────────────

def load_checkpoint(model: Any, optimizer: Any, scheduler: Any = None):
    """Load the latest checkpoint. Returns next step (0 if none)."""
    global active_future
    if active_future is not None:
        active_future.result()

    return manager.resume(model=model, optimizer=optimizer, scheduler=scheduler)


def wait_for_bg_save():
    """Wait for any active background save to complete."""
    global active_future
    if active_future is not None:
        result = active_future.result()
        active_future = None
        return result
    return None


# ─── Training loop ───────────────────────────────────────────────────

def training_loop_example():
    import torch
    import torch.nn as nn

    model = nn.Linear(512, 512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Auto-resume
    start_step = load_checkpoint(model, optimizer)
    if start_step > 0:
        print(f"Resumed from step {start_step}")

    for step in range(start_step, 100_000):
        # ... forward, backward, optimizer.step() ...
        loss = 0.5

        if step > 0 and step % 1000 == 0:
            save_checkpoint(
                step=step,
                model=model,
                optimizer=optimizer,
                loss=f"{loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    # Final save
    save_checkpoint_sync(step=99_999, model=model, optimizer=optimizer)
    wait_for_bg_save()
    manager.print_stats()


if __name__ == "__main__":
    training_loop_example()
