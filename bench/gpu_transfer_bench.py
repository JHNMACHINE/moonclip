"""What the GPU-to-host copy actually costs, and what pinned memory buys.

The checkpoint stall on an RTX 3090 was measured at 1589 ms, of which 600 ms
was the device-to-host copy at 6.7 GB/s. PCIe 4.0 x16 is around 25 GB/s, so
that number is not a transfer limit — it is what a copy into *pageable* host
memory costs, because CUDA has to stage it through an internal pinned buffer
and cannot overlap it with anything.

The CPU-side work that used to dominate is gone (1393 ms -> 227 ms), so this
copy is now roughly three quarters of what remains. The obvious fix is a
pinned staging buffer. Obvious is not the same as measured, and on this project
obvious has lost three times in a row, so this benchmark decides the design
instead of confirming it.

Four ways to get the same bytes to the host:

    pageable                what the code does today
    pinned, blocking        pinned destination, synchronous copy
    pinned, non_blocking    same, async on the default stream
    pinned, side stream     async on a separate stream, which is the only
                            variant that can overlap with compute

Costs the benchmark also reports, because they decide whether the fix is worth
it at all: allocating pinned memory is slow and the allocation is unswappable,
so a staging buffer the size of the training state is not free.

    python bench/gpu_transfer_bench.py --gib 3.8

Run it on the GPU you care about. On a machine with no CUDA it exits saying so
rather than reporting numbers that mean nothing.
"""

import argparse
import statistics
import time

import torch


def human_bytes(count):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(count) < 1024:
            return f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TiB"


def build_state(target_bytes, device, hidden=4096):
    """Tensors shaped like a Linear stack, on the device.

    Shaped rather than one flat buffer on purpose: the real copy is a few
    hundred separate transfers, and per-tensor launch overhead is part of what
    is being measured.
    """
    per = hidden * hidden * 4
    layers = max(1, round(target_bytes / per))
    tensors = []
    for _ in range(layers):
        tensors.append(torch.randn(hidden, hidden, device=device))
        tensors.append(torch.randn(hidden, device=device))
    total = sum(t.numel() * t.element_size() for t in tensors)
    return tensors, total


def timed(fn, repeats=3):
    """Median wall time of `fn`, with the device quiesced around each run."""
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=3.8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device: this benchmark measures the PCIe copy and has "
              "nothing to say without one.")
        return

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")

    tensors, total = build_state(args.gib * (1 << 30), device)
    print(f"state:  {len(tensors)} tensors, {human_bytes(total)} on the device\n")

    results = {}

    # ── 1. What the code does today ─────────────────────────────────
    def pageable():
        return [t.detach().to("cpu", copy=True) for t in tensors]

    results["pageable (today)"] = timed(pageable, args.repeats)

    # ── 2. Pinned destination ───────────────────────────────────────
    # Allocated once and reused: pinning is a kernel operation and doing it
    # per save would cost more than the copy it speeds up. That cost is
    # reported separately below.
    alloc_started = time.perf_counter()
    staging = [
        torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tensors
    ]
    alloc_seconds = time.perf_counter() - alloc_started

    def pinned_blocking():
        for dst, src in zip(staging, tensors):
            dst.copy_(src)

    results["pinned, blocking"] = timed(pinned_blocking, args.repeats)

    def pinned_non_blocking():
        for dst, src in zip(staging, tensors):
            dst.copy_(src, non_blocking=True)

    results["pinned, non_blocking"] = timed(pinned_non_blocking, args.repeats)

    side = torch.cuda.Stream()

    def pinned_side_stream():
        with torch.cuda.stream(side):
            for dst, src in zip(staging, tensors):
                dst.copy_(src, non_blocking=True)
        side.synchronize()

    results["pinned, side stream"] = timed(pinned_side_stream, args.repeats)

    # ── Report ──────────────────────────────────────────────────────
    print(f"{'path':<24}{'time':>10}{'bandwidth':>14}{'vs today':>12}")
    baseline = results["pageable (today)"]
    for name, seconds in results.items():
        speedup = "" if name.startswith("pageable") else f"{baseline / seconds:.2f}x"
        print(f"{name:<24}{seconds * 1000:>7.0f} ms"
              f"{total / seconds / 1e9:>11.1f} GB/s{speedup:>12}")

    print(f"\npinning {human_bytes(total)} of host memory took "
          f"{alloc_seconds * 1000:.0f} ms — paid once if the buffer is reused, "
          f"paid every save if it is not")
    print("pinned memory is page-locked and cannot be swapped, so a staging "
          "buffer this size is memory the machine can never reclaim")

    best = min(results.values())
    saved = baseline - best
    print(f"\nbest case saves {saved * 1000:.0f} ms of the copy. Against a "
          f"CPU-side cost of ~227 ms on 3.8 GiB, that is the number that "
          f"decides whether a staging buffer is worth its memory.")


if __name__ == "__main__":
    main()
