#!/usr/bin/env bash
# Everything the rented GPU box has to do, in one command.
#
#   bash bench/gpu_session.sh          # the measurement that decides the design
#   bash bench/gpu_session.sh --full   # also build Moonclip and measure a real save
#
# Run from the repository root on a machine that already has torch with CUDA —
# a vast.ai PyTorch template does. The meter is running the whole time, so the
# default does the cheap decisive thing and stops.
#
# The first benchmark needs *only torch*. It measures the PCIe copy, which is
# what the remaining checkpoint stall is made of, and nothing in it touches
# Moonclip. That matters: building the extension needs a Rust toolchain the
# template does not ship, which is minutes of billed time for a number this
# does not need. Build only if the first result says the rest is worth it.

set -euo pipefail

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

echo "=== environment ==="
python - <<'PY'
import torch
print("torch     ", torch.__version__)
print("cuda      ", torch.version.cuda)
print("devices   ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}, {p.total_memory / (1<<30):.0f} GiB")
PY

echo
echo "=== device-to-host copy: what pinned memory buys ==="
# 3.8 GiB matches the model the original 600 ms / 6.7 GB/s figure came from,
# so the numbers are comparable to the ones in the plan.
python bench/gpu_transfer_bench.py --gib 3.8

if [ "$FULL" -eq 0 ]; then
    cat <<'EOF'

Stopping here. That table is what decides whether a pinned staging buffer is
worth its memory; nothing below it changes the answer.

Re-run with --full to build Moonclip on this box and measure a real checkpoint
end to end. That needs a Rust toolchain, so budget a few minutes of meter.

Destroy the instance when done — stopping it still bills storage.
EOF
    exit 0
fi

echo
echo "=== building Moonclip (needs Rust; this is the slow part) ==="
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.98.1 --profile minimal --component clippy
    . "$HOME/.cargo/env"
fi
pip install --quiet maturin
maturin develop --release

echo
echo "=== CPU-side handoff on this box's CPU ==="
# Note what this does and does not cover: handoff_bench builds its state on the
# CPU, so it measures the copy into Rust and not the device transfer above. It
# is here because the 227 ms figure was measured on a different CPU, and a
# rented box is not that machine.
#
# For the whole stall in one table — device copy, handoff and background write
# together — run ravex's own breakdown from the ravex checkout:
#   python bench/checkpoint_bench.py --params 1e9 --breakdown
python bench/handoff_bench.py --gib 3.8

echo
echo "Done. Destroy the instance — stopping it still bills storage."
