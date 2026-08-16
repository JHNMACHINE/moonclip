"""What a checkpoint costs the training loop before the writer takes over.

The stall a save imposes is three things, and only the first two block:

    GPU -> CPU  ->  flatten the state tree  ->  copy into Rust  ||  write

This benchmark isolates the last two — the CPU-side stretch — so each number
belongs to exactly one thing. It needs no GPU, which is the point: the work it
measures is the work that has nothing to do with the device.

Two rules make the numbers mean something, and both were learned by getting
them wrong:

**Time each writer on an idle machine.** Moonclip's background save is a zstd
pass that saturates every core, and `submit` blocks on the previous one. Timing
a save while another is draining measures the drain. Every timer below starts
on a flushed manager.

**Perturb the tensors between saves.** A delta engine handed identical bytes
skips everything and reports a throughput that no real run will ever see.

    python bench/handoff_bench.py --gib 3.8

Reference numbers, 16-core CPU, 3.8 GiB of fp32 weights, before and after the
handoff work of 2026-08-16:

    flatten (bytes)   560 ms      copy into Rust (bytes)    832 -> 207 ms
    flatten (tensors) 0.7 ms      copy into Rust (tensors)         226 ms

    blocking, byte path    1393 -> 674 ms
    blocking, tensor path            227 ms
"""

import argparse
import shutil
import statistics
import tempfile
import time

import torch

import moonclip


def human_bytes(count):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(count) < 1024:
            return f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TiB"


def build_state(target_bytes, hidden=4096):
    """CPU tensors shaped like a Linear stack, already contiguous.

    Contiguous on purpose: this measures the handoff, and a non-contiguous
    tensor would smuggle a torch-side copy into the number.
    """
    per = hidden * hidden * 4
    layers = max(1, round(target_bytes / per))
    state = {}
    for i in range(layers):
        state[f"layer{i}.weight"] = torch.randn(hidden, hidden)
        state[f"layer{i}.bias"] = torch.randn(hidden)
    total = sum(t.numel() * t.element_size() for t in state.values())
    return state, total


def perturb(state):
    """Touch every tensor, so the delta engine has real work — as Adam does."""
    for t in state.values():
        t.add_(torch.randn_like(t), alpha=1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=3.8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    state, total = build_state(args.gib * (1 << 30))
    print(f"state: {len(state)} tensors, {human_bytes(total)} on CPU")
    print(
        f"moonclip {moonclip.__version__}, torch {torch.__version__}, "
        f"{torch.get_num_threads()} torch threads"
    )

    root = tempfile.mkdtemp(prefix="moonclip_handoff_")
    timings = {
        k: []
        for k in (
            "flatten/bytes",
            "copy-in/bytes",
            "write/bytes",
            "flatten/tensors",
            "copy-in/tensors",
            "write/tensors",
        )
    }
    try:
        bytes_mgr = moonclip.CheckpointManager(
            storage_root=f"{root}/bytes", world_size=1, rank=0, async_save=True
        )
        tensor_mgr = moonclip.CheckpointManager(
            storage_root=f"{root}/tensors", world_size=1, rank=0, async_save=True
        )

        for step in range(args.repeats):
            perturb(state)

            # ── byte path: .tobytes() here, then a copy into Rust ─────
            bytes_mgr.flush()
            started = time.perf_counter()
            flat, _ = moonclip.flatten_state_dict(state, "model")
            timings["flatten/bytes"].append(time.perf_counter() - started)

            started = time.perf_counter()
            bytes_mgr.save_raw(step=step, tensors=flat)
            timings["copy-in/bytes"].append(time.perf_counter() - started)

            started = time.perf_counter()
            bytes_mgr.flush()
            timings["write/bytes"].append(time.perf_counter() - started)
            del flat

            # ── tensor path: one copy, in Rust, GIL released ──────────
            tensor_mgr.flush()
            started = time.perf_counter()
            flat, _ = moonclip.flatten_state_dict(state, "model", as_tensors=True)
            timings["flatten/tensors"].append(time.perf_counter() - started)

            started = time.perf_counter()
            tensor_mgr.save_raw(step=step, tensors=flat)
            timings["copy-in/tensors"].append(time.perf_counter() - started)

            started = time.perf_counter()
            tensor_mgr.flush()
            timings["write/tensors"].append(time.perf_counter() - started)
            del flat

        print(f"\n{'phase':<20}{'median':>12}{'rate':>12}")
        for name, samples in timings.items():
            median = statistics.median(samples)
            print(
                f"{name:<20}{median * 1000:>9.1f} ms"
                f"{total / median / 1e9:>9.1f} GB/s"
            )

        def blocking(path):
            return statistics.median(timings[f"flatten/{path}"]) + statistics.median(
                timings[f"copy-in/{path}"]
            )

        print(f"\nblocks training, byte path:   {blocking('bytes') * 1000:8.1f} ms")
        print(f"blocks training, tensor path: {blocking('tensors') * 1000:8.1f} ms")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
