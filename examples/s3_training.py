"""
Transformer training with S3-backed checkpoints.

Checkpoints are saved locally (fast, SSD-aligned) and synced to S3
in batches for durability. If the training node dies, checkpoints
survive on S3 and can be resumed from any machine.

Usage with MinIO (local S3):
    # Start MinIO:
    docker run -p 9000:9000 -p 9001:9001 \
        -e MINIO_ROOT_USER=minioadmin \
        -e MINIO_ROOT_PASSWORD=minioadmin \
        minio/minio server /data --console-address ":9001"

    # Create bucket (via MinIO console at http://localhost:9001 or mc cli):
    mc alias set local http://localhost:9000 minioadmin minioadmin
    mc mb local/revolver-test

    # Train:
    python examples/s3_training.py \
        --s3-endpoint http://localhost:9000 \
        --s3-bucket revolver-test \
        --s3-access-key minioadmin \
        --s3-secret-key minioadmin

Usage with AWS S3:
    python examples/s3_training.py \
        --s3-bucket my-checkpoints \
        --s3-region us-east-1 \
        --s3-access-key AKIA... \
        --s3-secret-key ...

Usage with Cloudflare R2:
    python examples/s3_training.py \
        --s3-endpoint https://<account_id>.r2.cloudflarestorage.com \
        --s3-bucket my-checkpoints \
        --s3-access-key ... \
        --s3-secret-key ...

Usage with environment variables (recommended for CI/production):
    export S3_ENDPOINT=http://localhost:9000
    export S3_BUCKET=revolver-test
    export S3_ACCESS_KEY=minioadmin
    export S3_SECRET_KEY=minioadmin
    python examples/s3_training.py

Requirements:
    pip install torch revolver
    # For MinIO: docker
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from revolver import CheckpointManager


# ─── Model (same MiniGPT from transformer_training.py) ──────────────

class CausalSelfAttention(nn.Module):
    mask: torch.Tensor
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        y = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, n_heads=4, n_layers=4, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.max_seq_len = max_seq_len
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        print(f"MiniGPT: {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class SyntheticTextDataset(Dataset):
    def __init__(self, n_samples=10000, seq_len=128):
        self.data = [torch.tensor([(i % 200 + j) % 256 for j in range(seq_len + 1)], dtype=torch.long) for i in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][:-1], self.data[idx][1:]


# ─── Training ────────────────────────────────────────────────────────

def train(args):
    device = args.device
    os.makedirs(args.ckpt_dir, exist_ok=True)

    model = MiniGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * 312)
    loader = DataLoader(SyntheticTextDataset(), batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ─── S3-backed CheckpointManager ─────────────────────────────────
    #
    # How it works:
    # 1. Checkpoints are saved locally first (fast, SSD page-aligned)
    # 2. Every `sync_every_n_saves` saves, a background thread syncs
    #    new files to S3 (batched, doesn't block training)
    # 3. At end of training, `sync_now()` forces a final sync
    #
    # If the node dies between syncs, you lose at most
    # `sync_every_n_saves` checkpoints. The rest are safe on S3.

    s3_kwargs = {}
    if args.s3_bucket:
        s3_kwargs = {
            "s3_bucket": args.s3_bucket,
            "s3_region": args.s3_region,
            "s3_prefix": args.s3_prefix,
            "s3_access_key": args.s3_access_key,
            "s3_secret_key": args.s3_secret_key,
            "s3_path_style": args.s3_path_style,
            "sync_every_n_saves": args.sync_every,
        }
        if args.s3_endpoint:
            s3_kwargs["s3_endpoint"] = args.s3_endpoint

        print(f"\n[S3] bucket={args.s3_bucket}, prefix={args.s3_prefix}")
        print(f"[S3] endpoint={args.s3_endpoint or 'AWS default'}")
        print(f"[S3] sync every {args.sync_every} saves")
    else:
        print("\n[S3] Not configured — local-only checkpoints")

    ckpt = CheckpointManager(
        storage_root=args.ckpt_dir,
        compression_level=3,
        max_full_snapshots=3,
        full_every_steps=200,
        delta_threshold=0.5,
        save_dtype=args.save_dtype,
        **s3_kwargs,
    )

    # Auto-resume
    start_step = ckpt.resume(model=model, optimizer=optimizer, scheduler=scheduler)
    if start_step > 0:
        print(f"Resumed from step {start_step}")

    global_step = start_step
    model.train()

    print(f"\n{'='*70}")
    print(f"Training MiniGPT | {args.epochs} epochs | device={device} | dtype={args.save_dtype}")
    print(f"Checkpoint: save every {args.save_every} steps")
    if args.s3_bucket:
        print(f"S3 sync: every {args.sync_every} saves to s3://{args.s3_bucket}/{args.s3_prefix}")
    print(f"{'='*70}\n")

    t_start = time.perf_counter()

    epoch_loss = 0.0
    epoch_steps = 0

    for epoch in range(args.epochs):
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

            if global_step % args.save_every == 0:
                t0 = time.perf_counter()
                snap_id = ckpt.save(
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata={"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"},
                )
                t_ckpt = time.perf_counter() - t0
                info = ckpt.list_snapshots()[-1]
                comp_kb = info["total_compressed"] / 1024
                raw_kb = info["total_raw"] / 1024
                pct = (1.0 - comp_kb / raw_kb) * 100 if raw_kb > 0 else 0
                print(f"  [ckpt] step={global_step} | {t_ckpt:.3f}s | "
                      f"full={info['full_tensors']} Δ={info['delta_tensors']} skip={info['skipped_tensors']} | "
                      f"{comp_kb:.1f}KB / {raw_kb:.1f}KB ({pct:.1f}% saved)")

            if batch_idx % 50 == 0:
                print(f"  epoch {epoch+1}/{args.epochs} | step {global_step} | "
                      f"loss {loss.item():.4f} (avg {epoch_loss/max(epoch_steps,1):.4f}) | "
                      f"{time.perf_counter()-t_start:.1f}s")

        print(f"  → Epoch {epoch+1} done | avg loss: {epoch_loss/epoch_steps:.4f}")

    # Final save and sync
    ckpt.save(step=global_step, model=model, optimizer=optimizer, scheduler=scheduler,
              metadata={"loss": f"{epoch_loss/epoch_steps:.4f}", "final": "true"})

    # Force sync to S3 before exit
    if args.s3_bucket:
        print("\n[S3] Final sync...")
        t0 = time.perf_counter()
        ckpt.sync_now()
        print(f"[S3] Sync complete in {time.perf_counter()-t0:.1f}s")

    ckpt.merge_now()
    ckpt.print_stats()

    # Verify resume
    print("--- Verifying resume ---")
    model2 = MiniGPT().to(device)
    ckpt2 = CheckpointManager(storage_root=args.ckpt_dir, save_dtype=args.save_dtype, **s3_kwargs)
    ckpt2.load_latest(model=model2)

    max_diff = max((p1.data - p2.data).abs().max().item()
                   for (_, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()))
    atol = 0.01 if args.save_dtype in ("bf16", "bfloat16") else 1e-6
    status = "✓" if max_diff < atol * 10 else "✗"
    print(f"  {status} All weights match (max diff: {max_diff:.6f})")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniGPT with S3-backed Revolver checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--save-dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--ckpt-dir", type=str, default="./revolver_s3_ckpts")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # S3 config (also reads from env vars)
    parser.add_argument("--s3-bucket", type=str, default=os.environ.get("S3_BUCKET"))
    parser.add_argument("--s3-region", type=str, default=os.environ.get("S3_REGION", "us-east-1"))
    parser.add_argument("--s3-prefix", type=str, default=os.environ.get("S3_PREFIX", "revolver-training/"))
    parser.add_argument("--s3-endpoint", type=str, default=os.environ.get("S3_ENDPOINT"))
    parser.add_argument("--s3-access-key", type=str, default=os.environ.get("S3_ACCESS_KEY"))
    parser.add_argument("--s3-secret-key", type=str, default=os.environ.get("S3_SECRET_KEY"))
    parser.add_argument("--s3-path-style", action="store_true", default=os.environ.get("S3_PATH_STYLE", "").lower() in ("1", "true"))
    parser.add_argument("--sync-every", type=int, default=3, help="Sync to S3 every N checkpoint saves")

    args = parser.parse_args()

    if args.s3_bucket and not args.s3_access_key:
        parser.error("--s3-access-key (or S3_ACCESS_KEY env) required when using S3")
    if args.s3_bucket and not args.s3_secret_key:
        parser.error("--s3-secret-key (or S3_SECRET_KEY env) required when using S3")

    train(args)
