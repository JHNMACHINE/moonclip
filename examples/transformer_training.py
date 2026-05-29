"""
Real transformer training with Revolver checkpointing.

Trains a small GPT-style causal language model on synthetic text data,
demonstrating checkpoint save/resume with per-tensor delta tracking.

Auto-resumes from existing checkpoints if the directory is populated.
Also auto-detects save_dtype from checkpoint metadata on resume.

Usage:
    python transformer_training.py                        # Train (auto-resumes if ckpts exist)
    python transformer_training.py --save-dtype bf16      # Save checkpoints in bf16
    python transformer_training.py --fresh-start          # Ignore existing ckpts, start over

Requirements:
    pip install torch revolver
"""

import argparse
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from revolver import CheckpointManager


# ─── Model ───────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    mask: torch.Tensor
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
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
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
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

        y = attn @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
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
    """A small GPT-style causal LM. ~2.5M parameters with default config."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.max_seq_len = max_seq_len
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"MiniGPT: {n_params:,} parameters")

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
        x = self.ln_f(x)
        return self.head(x)


# ─── Dataset ─────────────────────────────────────────────────────────

class SyntheticTextDataset(Dataset):
    """Synthetic byte-level text with learnable incrementing patterns."""

    def __init__(self, n_samples: int = 10000, seq_len: int = 128):
        self.data = []
        for i in range(n_samples):
            start = i % 200
            seq = [(start + j) % 256 for j in range(seq_len + 1)]
            self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


# ─── Training ────────────────────────────────────────────────────────

def train(
    ckpt_dir: str = "./revolver_training_ckpts",
    fresh_start: bool = False,
    n_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    save_every_steps: int = 50,
    save_dtype: str = "fp32",
    device: str = "cpu",
):
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model — always trains in fp32 for stability (especially on CPU)
    model = MiniGPT(vocab_size=256, d_model=256, n_heads=4, n_layers=4, max_seq_len=128)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs * 312)

    dataset = SyntheticTextDataset(n_samples=10000, seq_len=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # The cast is handled entirely by Rust — just pass save_dtype
    ckpt = CheckpointManager(
        storage_root=ckpt_dir,
        compression_level=3,
        max_full_snapshots=3,
        full_every_steps=200,
        delta_threshold=0.5,
        save_dtype=save_dtype,
    )

    # Auto-resume
    if fresh_start:
        print("Fresh start requested, ignoring existing checkpoints")
        start_step = 0
    else:
        start_step = ckpt.resume(model=model, optimizer=optimizer, scheduler=scheduler)
        if start_step > 0:
            print(f"Resumed from step {start_step}")
        else:
            print("No existing checkpoints, starting from scratch")

    # Training loop
    global_step = start_step
    model.train()

    print(f"\n{'='*70}")
    print(f"Training MiniGPT | {n_epochs} epochs | device={device} | ckpt dtype={save_dtype}")
    print(f"Checkpoint: save every {save_every_steps} steps to {ckpt_dir}")
    print(f"{'='*70}\n")

    t_start = time.perf_counter()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, 256), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Checkpoint
            if global_step % save_every_steps == 0:
                t0 = time.perf_counter()
                snap_id = ckpt.save(
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata={
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "epoch": str(epoch),
                    },
                )
                t_ckpt = time.perf_counter() - t0

                snaps = ckpt.list_snapshots()
                info = snaps[-1]
                raw_kb = info["total_raw"] / 1024
                comp_kb = info["total_compressed"] / 1024
                ratio_pct = (1.0 - comp_kb / raw_kb) * 100 if raw_kb > 0 else 0
                print(
                    f"  [ckpt] step={global_step} | {t_ckpt:.3f}s | "
                    f"full={info['full_tensors']} Δ={info['delta_tensors']} "
                    f"skip={info['skipped_tensors']} | "
                    f"{comp_kb:.1f}KB / {raw_kb:.1f}KB raw "
                    f"({ratio_pct:.1f}% saved)"
                )

            # Log
            if batch_idx % 50 == 0:
                avg_loss = epoch_loss / max(epoch_steps, 1)
                elapsed = time.perf_counter() - t_start
                print(
                    f"  epoch {epoch+1}/{n_epochs} | "
                    f"step {global_step} | "
                    f"loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
                    f"lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed:.1f}s"
                )

        avg_loss = epoch_loss / epoch_steps
        print(f"  → Epoch {epoch+1} done | avg loss: {avg_loss:.4f}")

    # Final checkpoint
    snap_id = ckpt.save(
        step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata={"loss": f"{avg_loss:.4f}", "final": "true"},
    )
    ckpt.merge_now()

    total_time = time.perf_counter() - t_start

    print(f"\n{'='*70}")
    print(f"Training complete in {total_time:.1f}s | dtype={save_dtype}")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Final checkpoint: {snap_id[:8]}...")

    # Print aggregate stats — shows how much delta/skip saved vs naive torch.save
    ckpt.print_stats()

    # Verify resume
    print(f"\n--- Verifying resume ---")
    model2 = MiniGPT(vocab_size=256, d_model=256, n_heads=4, n_layers=4, max_seq_len=128)
    model2 = model2.to(device)

    ckpt2 = CheckpointManager(storage_root=ckpt_dir, save_dtype=save_dtype)
    ckpt2.load_latest(model=model2)

    use_bf16 = save_dtype in ("bf16", "bfloat16")
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        diff = (p1.data - p2.data).abs().max().item()
        max_diff = max(max_diff, diff)
        atol = 0.01 if use_bf16 else 1e-6
        if not torch.allclose(p1.data, p2.data, atol=atol, rtol=0.01):
            print(f"  MISMATCH: {n1} (max diff: {diff:.6f})")
            break
    else:
        print(f"  ✓ All weights match after resume (max diff: {max_diff:.6f})")

    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniGPT with Revolver checkpointing")
    parser.add_argument("--fresh-start", action="store_true",
                        help="Ignore existing checkpoints and start from scratch")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--save-dtype", type=str, default="fp32", choices=["fp32", "bf16"],
                        help="Dtype for checkpoint storage. Training is always fp32.")
    parser.add_argument("--ckpt-dir", type=str, default="./revolver_training_ckpts")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(
        ckpt_dir=args.ckpt_dir,
        fresh_start=args.fresh_start,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every_steps=args.save_every,
        save_dtype=args.save_dtype,
        device=args.device,
    )
