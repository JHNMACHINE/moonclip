"""Validate the pinned staging on a real GPU: faster, and still correct.

Two questions, and the second is the one that can hurt:

1. How much of the checkpoint stall does the pinned path remove, end to end
   rather than on a synthetic copy?
2. Do the weights come back exactly? `non_blocking` copies are queued, not
   done. A missing synchronise does not raise — it stores whatever the reused
   buffer held before, which is the previous checkpoint's weights: plausible,
   and wrong. Correctness is checked first for that reason.
"""

import argparse
import shutil
import statistics
import tempfile
import time

import torch
import torch.nn as nn

import moonclip
from moonclip.pytorch import _PinnedStaging


def human_bytes(count):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(count) < 1024:
            return f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TiB"


def build(target_bytes, hidden=4096):
    per = hidden * hidden * 4
    layers = max(2, round(target_bytes / per))
    model = nn.Sequential(
        *[m for _ in range(layers) for m in (nn.Linear(hidden, hidden), nn.GELU())]
    ).cuda()
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return model, total, hidden


def stall(model, pinned, repeats):
    """Median blocking time of a save: flatten + handoff, as training sees it."""
    root = tempfile.mkdtemp(prefix="moonclip_gpu_")
    try:
        mgr = moonclip.CheckpointManager(
            storage_root=root, world_size=1, rank=0, async_save=True,
            pin_device_copies=pinned,
        )
        samples = []
        for step in range(repeats):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p), alpha=1e-4)
            mgr.flush()
            torch.cuda.synchronize()
            started = time.perf_counter()
            mgr.save(step=step, model=model)
            samples.append(time.perf_counter() - started)
        mgr.flush()
        return statistics.median(samples)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_roundtrip(model, pinned):
    """Save, wipe, reload: the weights must come back bit-identical."""
    root = tempfile.mkdtemp(prefix="moonclip_rt_")
    try:
        mgr = moonclip.CheckpointManager(
            storage_root=root, world_size=1, rank=0, async_save=True,
            pin_device_copies=pinned,
        )
        # Two saves: the second reuses the pinned buffers the first filled,
        # which is where a missing synchronise would show up.
        mgr.save(step=0, model=model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p), alpha=1e-3)
        expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
        snap = mgr.save(step=1, model=model)
        mgr.flush()

        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        mgr.load(snap, model=model)

        bad = [
            k for k, v in model.state_dict().items()
            if not torch.equal(v, expected[k])
        ]
        return bad
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=3.8)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device.")
        return

    print(f"device: {torch.cuda.get_device_name(0)}")
    model, total, _ = build(args.gib * (1 << 30))
    print(f"model:  {human_bytes(total)} of fp32 weights on the device\n")

    print("correctness first")
    for pinned in (True, False):
        bad = check_roundtrip(model, pinned)
        label = "pinned" if pinned else "pageable"
        print(f"  {label:<10}{'OK' if not bad else f'{len(bad)} TENSORS WRONG: {bad[:3]}'}")
    if any(check_roundtrip(model, p) for p in (True, False)):
        print("\nrefusing to report timings for a path that loses data")
        return

    print("\nblocking time of a save (what the training loop pays)")
    results = {}
    for pinned in (True, False):
        label = "pinned" if pinned else "pageable (today)"
        results[label] = stall(model, pinned, args.repeats)

    for label, seconds in results.items():
        print(f"  {label:<20}{seconds * 1000:>8.0f} ms{total / seconds / 1e9:>9.1f} GB/s")

    slow = results["pageable (today)"]
    fast = results["pinned"]
    print(f"\n{slow / fast:.2f}x less blocking, {(slow - fast) * 1000:.0f} ms saved per checkpoint")


if __name__ == "__main__":
    main()
