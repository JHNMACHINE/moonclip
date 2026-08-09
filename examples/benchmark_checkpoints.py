"""
Checkpoint benchmark: Moonclip vs torch.save vs safetensors vs Accelerate vs Lightning.

Measures save time, load time, and disk usage across multiple checkpoint saves
during a real training run. Each method uses its natural retention policy to
show real-world disk footprint.

Moonclip saves run asynchronously by default: save() returns as soon as the
tensor data has been copied, while hashing, delta detection, compression and
the disk write overlap with the next training steps. On top of that it keeps
cumulative disk savings over many saves via per-tensor delta tracking, zstd
compression, and built-in retention.

Usage:
    pip install moonclip torch safetensors accelerate pytorch-lightning
    python benchmark_checkpoints.py
    python benchmark_checkpoints.py --n-saves 20 --model-size large
    python benchmark_checkpoints.py --methods moonclip torch safetensors
"""

from __future__ import annotations

import argparse
import gc
import glob
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Model ───────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    mask: torch.Tensor

    def __init__(
        self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_seq_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.token_emb.weight
        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


MODEL_CONFIGS = {
    "small": dict(vocab_size=256, d_model=256, n_heads=4, n_layers=4, max_seq_len=128),
    "medium": dict(
        vocab_size=32000, d_model=512, n_heads=8, n_layers=8, max_seq_len=256
    ),
    "large": dict(
        vocab_size=32000, d_model=1024, n_heads=16, n_layers=12, max_seq_len=512
    ),
}


# ─── Result types ────────────────────────────────────────────────────


@dataclass
class SaveResult:
    step: int
    save_time_s: float
    disk_bytes: int
    n_accessible: int


@dataclass
class BenchmarkResult:
    method: str
    model_params: int
    saves: List[SaveResult] = field(default_factory=list)
    load_time_s: float = 0.0
    weights_match: bool = False
    max_diff: float = 0.0
    error: Optional[str] = None

    @property
    def first_save_time(self) -> float:
        return self.saves[0].save_time_s if self.saves else 0.0

    @property
    def avg_save_time(self) -> float:
        return (
            sum(s.save_time_s for s in self.saves) / len(self.saves)
            if self.saves
            else 0.0
        )

    @property
    def final_disk_bytes(self) -> int:
        return self.saves[-1].disk_bytes if self.saves else 0

    @property
    def single_save_bytes(self) -> int:
        return self.saves[0].disk_bytes if self.saves else 0


# ─── Helpers ─────────────────────────────────────────────────────────


def dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def fresh_dir(path: str) -> str:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def fake_training_step(model: nn.Module, optimizer: torch.optim.Optimizer, device: str):
    seq_len = min(model.max_seq_len, 32)
    vocab = model.token_emb.num_embeddings
    x = torch.randint(0, vocab, (4, seq_len), device=device)
    logits = model(x)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def verify_weights(
    model_a: nn.Module, model_b: nn.Module, atol: float = 1e-5
) -> Tuple[bool, float]:
    max_diff = 0.0
    for (_, p1), (_, p2) in zip(model_a.named_parameters(), model_b.named_parameters()):
        diff = (p1.data.cpu() - p2.data.cpu()).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff < atol, max_diff


def rotate_files(pattern_dir: str, ext: str, keep: int):
    files = sorted(glob.glob(os.path.join(pattern_dir, f"*{ext}")))
    for f in files[:-keep]:
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)


def rotate_dirs(parent: str, keep: int):
    dirs = sorted(
        [d for d in glob.glob(os.path.join(parent, "step_*")) if os.path.isdir(d)],
        key=lambda d: int(os.path.basename(d).split("_")[1]),
    )
    for d in dirs[:-keep]:
        shutil.rmtree(d)


# ─── Method: torch.save ─────────────────────────────────────────────


def bench_torch_save(
    model_cfg: dict,
    n_saves: int,
    steps_between: int,
    base_dir: str,
    device: str,
    keep: int,
) -> BenchmarkResult:
    ckpt_dir = fresh_dir(os.path.join(base_dir, "torch_save"))
    model = MiniGPT(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    result = BenchmarkResult(method="torch.save", model_params=n_params)

    for i in range(n_saves):
        step = (i + 1) * steps_between
        for _ in range(steps_between):
            fake_training_step(model, optimizer, device)

        path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
        t0 = time.perf_counter()
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path
        )
        save_time = time.perf_counter() - t0

        rotate_files(ckpt_dir, ".pt", keep)
        n_on_disk = len(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        result.saves.append(
            SaveResult(
                step=step,
                save_time_s=save_time,
                disk_bytes=dir_size(ckpt_dir),
                n_accessible=n_on_disk,
            )
        )

    model2 = MiniGPT(**model_cfg).to(device)
    last = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))[-1]
    t0 = time.perf_counter()
    ckpt = torch.load(last, map_location=device, weights_only=True)
    model2.load_state_dict(ckpt["model"])
    result.load_time_s = time.perf_counter() - t0
    result.weights_match, result.max_diff = verify_weights(model, model2)
    return result


# ─── Method: safetensors ─────────────────────────────────────────────


def bench_safetensors(
    model_cfg: dict,
    n_saves: int,
    steps_between: int,
    base_dir: str,
    device: str,
    keep: int,
) -> BenchmarkResult:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        r = BenchmarkResult(method="safetensors", model_params=0)
        r.error = "not installed (pip install safetensors)"
        return r

    ckpt_dir = fresh_dir(os.path.join(base_dir, "safetensors"))
    cfg = {**model_cfg, "tie_weights": False}
    model = MiniGPT(**cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    result = BenchmarkResult(method="safetensors*", model_params=n_params)

    for i in range(n_saves):
        step = (i + 1) * steps_between
        for _ in range(steps_between):
            fake_training_step(model, optimizer, device)

        path = os.path.join(ckpt_dir, f"step_{step:06d}.safetensors")
        sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
        t0 = time.perf_counter()
        save_file(sd, path)
        save_time = time.perf_counter() - t0

        rotate_files(ckpt_dir, ".safetensors", keep)
        n_on_disk = len(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
        result.saves.append(
            SaveResult(
                step=step,
                save_time_s=save_time,
                disk_bytes=dir_size(ckpt_dir),
                n_accessible=n_on_disk,
            )
        )

    model2 = MiniGPT(**cfg).to(device)
    last = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))[-1]
    t0 = time.perf_counter()
    state = load_file(last, device=device)
    model2.load_state_dict(state)
    result.load_time_s = time.perf_counter() - t0
    result.weights_match, result.max_diff = verify_weights(model, model2)
    return result


# ─── Method: Accelerate ──────────────────────────────────────────────


def bench_accelerate(
    model_cfg: dict,
    n_saves: int,
    steps_between: int,
    base_dir: str,
    device: str,
    keep: int,
) -> BenchmarkResult:
    try:
        from accelerate import Accelerator
    except ImportError:
        r = BenchmarkResult(method="accelerate", model_params=0)
        r.error = "not installed (pip install accelerate)"
        return r

    ckpt_dir = fresh_dir(os.path.join(base_dir, "accelerate"))
    cfg = {**model_cfg, "tie_weights": False}
    model = MiniGPT(**cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    result = BenchmarkResult(method="accelerate", model_params=n_params)

    accelerator = Accelerator(cpu=(device == "cpu"))
    model, optimizer = accelerator.prepare(model, optimizer)

    for i in range(n_saves):
        step = (i + 1) * steps_between
        for _ in range(steps_between):
            seq_len = min(model_cfg.get("max_seq_len", 128), 32)
            vocab = model_cfg.get("vocab_size", 256)
            x = torch.randint(0, vocab, (4, seq_len), device=accelerator.device)
            unwrapped = accelerator.unwrap_model(model)
            logits = unwrapped(x)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
            )
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        save_path = os.path.join(ckpt_dir, f"step_{step:06d}")
        t0 = time.perf_counter()
        accelerator.save_state(save_path)
        save_time = time.perf_counter() - t0

        rotate_dirs(ckpt_dir, keep)
        n_on_disk = len(
            [d for d in glob.glob(os.path.join(ckpt_dir, "step_*")) if os.path.isdir(d)]
        )
        result.saves.append(
            SaveResult(
                step=step,
                save_time_s=save_time,
                disk_bytes=dir_size(ckpt_dir),
                n_accessible=n_on_disk,
            )
        )

    last_path = sorted(
        [d for d in glob.glob(os.path.join(ckpt_dir, "step_*")) if os.path.isdir(d)],
        key=lambda d: int(os.path.basename(d).split("_")[1]),
    )[-1]
    model2 = MiniGPT(**cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    accelerator2 = Accelerator(cpu=(device == "cpu"))
    model2, optimizer2 = accelerator2.prepare(model2, optimizer2)
    t0 = time.perf_counter()
    accelerator2.load_state(last_path)
    result.load_time_s = time.perf_counter() - t0
    result.weights_match, result.max_diff = verify_weights(
        accelerator.unwrap_model(model),
        accelerator2.unwrap_model(model2),
    )
    return result


# ─── Method: Lightning ───────────────────────────────────────────────


def bench_lightning(
    model_cfg: dict,
    n_saves: int,
    steps_between: int,
    base_dir: str,
    device: str,
    keep: int,
) -> BenchmarkResult:
    ckpt_dir = fresh_dir(os.path.join(base_dir, "lightning"))
    model = MiniGPT(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    result = BenchmarkResult(method="lightning", model_params=n_params)

    for i in range(n_saves):
        step = (i + 1) * steps_between
        for _ in range(steps_between):
            fake_training_step(model, optimizer, device)

        path = os.path.join(ckpt_dir, f"step_{step:06d}.ckpt")
        ckpt_data = {
            "state_dict": model.state_dict(),
            "optimizer_states": [optimizer.state_dict()],
            "global_step": step,
            "epoch": 0,
        }
        t0 = time.perf_counter()
        torch.save(ckpt_data, path)
        save_time = time.perf_counter() - t0

        rotate_files(ckpt_dir, ".ckpt", keep)
        n_on_disk = len(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        result.saves.append(
            SaveResult(
                step=step,
                save_time_s=save_time,
                disk_bytes=dir_size(ckpt_dir),
                n_accessible=n_on_disk,
            )
        )

    model2 = MiniGPT(**model_cfg).to(device)
    last = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))[-1]
    t0 = time.perf_counter()
    ckpt = torch.load(last, map_location=device, weights_only=True)
    model2.load_state_dict(ckpt["state_dict"])
    result.load_time_s = time.perf_counter() - t0
    result.weights_match, result.max_diff = verify_weights(model, model2)
    return result


# ─── Method: Moonclip ────────────────────────────────────────────────


def bench_moonclip(
    model_cfg: dict,
    n_saves: int,
    steps_between: int,
    base_dir: str,
    device: str,
    keep: int,
) -> BenchmarkResult:
    try:
        from moonclip import CheckpointManager
    except ImportError:
        r = BenchmarkResult(method="moonclip", model_params=0)
        r.error = "not installed (pip install moonclip)"
        return r

    ckpt_dir = fresh_dir(os.path.join(base_dir, "moonclip"))
    model = MiniGPT(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    result = BenchmarkResult(method="moonclip", model_params=n_params)

    mgr = CheckpointManager(
        storage_root=ckpt_dir,
        compression_level=3,
        max_full_snapshots=keep,
        max_deltas_per_full=50,
        full_every_steps=keep * steps_between,
        delta_max_ratio=0.95,
    )

    for i in range(n_saves):
        step = (i + 1) * steps_between
        for _ in range(steps_between):
            fake_training_step(model, optimizer, device)

        t0 = time.perf_counter()
        mgr.save(step=step, model=model, optimizer=optimizer)
        save_time = time.perf_counter() - t0

        n_accessible = len(mgr.list_snapshots())
        result.saves.append(
            SaveResult(
                step=step,
                save_time_s=save_time,
                disk_bytes=dir_size(ckpt_dir),
                n_accessible=n_accessible,
            )
        )

    model2 = MiniGPT(**model_cfg).to(device)
    t0 = time.perf_counter()
    mgr.load_latest(model=model2)
    result.load_time_s = time.perf_counter() - t0
    result.weights_match, result.max_diff = verify_weights(model, model2)
    return result


# ─── Report ──────────────────────────────────────────────────────────


def fmt_bytes(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GB"
    elif b >= 1 << 20:
        return f"{b / (1 << 20):.1f} MB"
    elif b >= 1 << 10:
        return f"{b / (1 << 10):.1f} KB"
    return f"{b} B"


def fmt_time(t: float) -> str:
    if t < 0.001:
        return f"{t * 1e6:.0f}µs"
    elif t < 1.0:
        return f"{t * 1e3:.1f}ms"
    return f"{t:.2f}s"


def print_report(results: List[BenchmarkResult], n_saves: int, keep: int):
    valid = [r for r in results if r.error is None]
    errored = [r for r in results if r.error is not None]

    if not valid:
        print("No valid results.")
        return

    ref = next((r for r in valid if r.method == "torch.save"), valid[0])
    ref_disk = ref.final_disk_bytes or 1

    w = 94
    print(f"\n{'━' * w}")
    print("  CHECKPOINT BENCHMARK RESULTS")
    print(
        f"  Model: {valid[0].model_params:,} params | "
        f"Saves: {n_saves} | Retention: keep {keep} | "
        f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
    )
    print(f"{'━' * w}")

    print(
        f"\n  {'Method':<14} {'1st Save':>10} {'Avg Save':>10} "
        f"{'Load':>10} {'Final Disk':>12} {'vs torch':>10} {'History':>8} {'OK':>4}"
    )
    print(
        f"  {'─' * 14} {'─' * 10} {'─' * 10} "
        f"{'─' * 10} {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 4}"
    )

    for r in valid:
        ratio = r.final_disk_bytes / ref_disk if ref_disk > 0 else 0
        check = "✓" if r.weights_match else "✗"
        last_accessible = r.saves[-1].n_accessible if r.saves else 0
        print(
            f"  {r.method:<14} "
            f"{fmt_time(r.first_save_time):>10} "
            f"{fmt_time(r.avg_save_time):>10} "
            f"{fmt_time(r.load_time_s):>10} "
            f"{fmt_bytes(r.final_disk_bytes):>12} "
            f"{ratio:>9.2f}× "
            f"{last_accessible:>8} "
            f"{check:>4}"
        )

    if errored:
        print()
        for r in errored:
            print(f"  {r.method:<14} skipped: {r.error}")

    # Note about safetensors
    sf = next((r for r in valid if r.method == "safetensors*"), None)
    if sf:
        print("\n  * safetensors: model-only (no optimizer), weight tying disabled")

    # Moonclip insight
    rev = next((r for r in valid if r.method == "moonclip"), None)
    tr = next((r for r in valid if r.method == "torch.save"), None)
    if rev and tr:
        disk_pct = (
            (1 - rev.final_disk_bytes / tr.final_disk_bytes) * 100
            if tr.final_disk_bytes
            else 0
        )
        rev_hist = rev.saves[-1].n_accessible if rev.saves else 0
        tr_hist = tr.saves[-1].n_accessible if tr.saves else 0
        print(f"\n  {'─' * (w - 4)}")
        if disk_pct > 0:
            print(
                f"  Moonclip: {fmt_bytes(rev.final_disk_bytes)} on disk "
                f"({disk_pct:.1f}% less than torch.save)"
            )
        else:
            print(
                f"  Moonclip: {fmt_bytes(rev.final_disk_bytes)} on disk "
                f"vs {fmt_bytes(tr.final_disk_bytes)} for torch.save"
            )
        if rev_hist != tr_hist:
            print(
                f"  Moonclip keeps {rev_hist} accessible checkpoints "
                f"vs {tr_hist} for torch.save"
            )
        print("  Per-tensor delta tracking + zstd compression + built-in retention")

    print(f"\n{'━' * w}")

    # Disk growth table
    if n_saves > 1 and len(valid) > 1:
        print(f"\n  Cumulative disk usage (retention: keep {keep}):\n")
        col_w = max(14, max(len(r.method) for r in valid) + 2)
        print(f"  {'Save':>6}", end="")
        for r in valid:
            print(f"  {r.method:>{col_w}}", end="")
        print()
        print(f"  {'─' * 6}", end="")
        for _ in valid:
            print(f"  {'─' * col_w}", end="")
        print()

        for i in range(n_saves):
            print(f"  {i + 1:>6}", end="")
            for r in valid:
                if i < len(r.saves):
                    print(f"  {fmt_bytes(r.saves[i].disk_bytes):>{col_w}}", end="")
                else:
                    print(f"  {'—':>{col_w}}", end="")
            print()
        print()


# ─── Main ────────────────────────────────────────────────────────────

ALL_METHODS = {
    "torch": bench_torch_save,
    "safetensors": bench_safetensors,
    "accelerate": bench_accelerate,
    "lightning": bench_lightning,
    "moonclip": bench_moonclip,
}


def main():
    # The report uses box-drawing characters; Windows consoles may default
    # to a legacy codepage (cp1252) that can't encode them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Benchmark checkpoint methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default="medium",
        help="small (~3M), medium (~42M), large (~163M). Default: medium",
    )
    parser.add_argument(
        "--n-saves", type=int, default=10, help="Number of saves. Default: 10"
    )
    parser.add_argument(
        "--steps-between",
        type=int,
        default=50,
        help="Training steps between saves. Default: 50",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Retention: how many checkpoints each method keeps. Default: 3",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS.keys()),
        choices=list(ALL_METHODS.keys()),
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", type=str, default="./bench_checkpoints")
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model_size]
    n_params = sum(p.numel() for p in MiniGPT(**model_cfg).parameters())

    print(f"\n{'=' * 60}")
    print("  Checkpoint Benchmark")
    print(f"{'=' * 60}")
    print(f"  Model:      MiniGPT {args.model_size} ({n_params:,} params)")
    print(f"  Saves:      {args.n_saves} (every {args.steps_between} steps)")
    print(f"  Retention:  keep {args.keep} checkpoints")
    print(f"  Device:     {args.device}")
    print(f"  Methods:    {', '.join(args.methods)}")
    print(f"{'=' * 60}\n")

    results = []
    for name in args.methods:
        bench_fn = ALL_METHODS[name]
        print(f"  [{name}] running...", end=" ", flush=True)
        gc.collect()
        torch.manual_seed(42)

        try:
            r = bench_fn(
                model_cfg=model_cfg,
                n_saves=args.n_saves,
                steps_between=args.steps_between,
                base_dir=args.output_dir,
                device=args.device,
                keep=args.keep,
            )
            if r.error:
                print(f"skipped ({r.error})")
            else:
                total_t = sum(s.save_time_s for s in r.saves)
                print(
                    f"done ({fmt_time(total_t)} total, {r.saves[-1].n_accessible} ckpts on disk)"
                )
            results.append(r)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()
            r = BenchmarkResult(method=name, model_params=n_params)
            r.error = str(e)
            results.append(r)

    print_report(results, args.n_saves, args.keep)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"  Cleaned up {args.output_dir}\n")


if __name__ == "__main__":
    main()
