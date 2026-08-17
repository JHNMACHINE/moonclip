"""
Integration tests for Moonclip v1.0.
Tests the Rust MoonclipManager directly (no PyTorch dependency needed).
PyTorch-specific tests are in test_pytorch.py (requires torch).
"""

import os
import sys
import random
import struct
import threading
import json
import time
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from moonclip import MoonclipManager


# ─── Helpers ─────────────────────────────────────────────────────────

def make_tensors(seed=0, size=8192):
    """Create test tensors: {name: (shape, dtype, bytes)}.

    The three payloads must stay distinct: identical tensors are now
    deduplicated within a snapshot, which would change the full/skipped
    counts these tests assert on.
    """
    return {
        "model.weight": ([64, 32], "float32", bytes([seed % 256] * size)),
        "model.bias": ([64], "float32", bytes([(seed + 1) % 256] * 256)),
        "optimizer.exp_avg": ([64, 32], "float32", bytes([(seed + 128) % 256] * size)),
    }


# ─── Basic save/load ────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        tensors = make_tensors(42)
        snap_id = mgr.save_tensors(step=1000, tensors=tensors, metadata={"loss": "0.634"})

        assert len(snap_id) == 36
        loaded = mgr.load(snap_id)
        assert loaded["model.weight"] == tensors["model.weight"][2]
        assert loaded["model.bias"] == tensors["model.bias"][2]
        assert loaded["optimizer.exp_avg"] == tensors["optimizer.exp_avg"][2]

    def test_load_latest(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        mgr.save_tensors(step=100, tensors=make_tensors(1))
        mgr.save_tensors(step=200, tensors=make_tensors(2))
        snap_id, loaded = mgr.load_latest()
        assert loaded["model.weight"][0] == 2

    def test_load_nonexistent_errors(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        with pytest.raises(RuntimeError, match="Not found|Snapshot"):
            mgr.load("00000000-0000-0000-0000-000000000000")

    def test_load_invalid_uuid_errors(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path))
        with pytest.raises(ValueError):
            mgr.load("not-a-uuid")

    def test_metadata_preserved(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        mgr.save_tensors(step=100, tensors=make_tensors(0), metadata={"loss": "1.234", "lr": "3e-4"})
        snaps = mgr.list_snapshots()
        assert snaps[0]["metadata"]["loss"] == "1.234"
        assert snaps[0]["metadata"]["lr"] == "3e-4"

    def test_save_no_metadata(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        snap_id = mgr.save_tensors(step=1, tensors=make_tensors(0))
        loaded = mgr.load(snap_id)
        assert len(loaded) == 3

    def test_single_tensor(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        data = bytes(range(256))
        snap_id = mgr.save_tensors(step=1, tensors={"only": ([256], "uint8", data)})
        assert mgr.load(snap_id)["only"] == data

    def test_large_tensor(self, tmp_path):
        """1MB tensor roundtrip."""
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        data = bytes(range(256)) * 4096  # 1MB
        snap_id = mgr.save_tensors(step=1, tensors={"big": ([1048576], "uint8", data)})
        assert mgr.load(snap_id)["big"] == data


# ─── Per-tensor delta tracking ──────────────────────────────────────

class TestDelta:
    def test_unchanged_tensors_skipped(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        tensors_v1 = make_tensors(0)
        mgr.save_tensors(step=100, tensors=tensors_v1)

        tensors_v2 = make_tensors(0)
        data = bytearray(tensors_v2["model.weight"][2])
        data[0] = 1
        tensors_v2["model.weight"] = ([64, 32], "float32", bytes(data))
        snap_id = mgr.save_tensors(step=200, tensors=tensors_v2)

        info = mgr.list_snapshots()[1]
        assert info["is_delta"] is True
        assert info["skipped_tensors"] == 2

        loaded = mgr.load(snap_id)
        assert loaded["model.bias"] == tensors_v1["model.bias"][2]
        assert loaded["optimizer.exp_avg"] == tensors_v1["optimizer.exp_avg"][2]

    def test_xor_delta_small_change(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        # Incompressible, like real weights: a run of zeros would compress
        # to nothing on its own, so a delta against it would save nothing
        # and is correctly rejected.
        big = random.Random(1234).randbytes(100_000)
        mgr.save_tensors(step=100, tensors={"big": ([100000], "uint8", big)})

        big_v2 = bytearray(big)
        big_v2[0] ^= 1
        big_v2[50000] ^= 2
        snap_id = mgr.save_tensors(step=200, tensors={"big": ([100000], "uint8", bytes(big_v2))})

        info = mgr.list_snapshots()[1]
        assert info["delta_tensors"] == 1
        assert info["total_compressed"] < 1000
        assert mgr.load(snap_id)["big"] == bytes(big_v2)

    def test_tied_weights_stored_once(self, tmp_path):
        """Identical tensors share one copy on disk and still load back."""
        shared = random.Random(7).randbytes(200_000)
        other = random.Random(8).randbytes(200_000)
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        snap_id = mgr.save_tensors(
            step=100,
            tensors={
                "token_emb.weight": ([50000], "float32", shared),
                "qkv.weight": ([50000], "float32", other),
                "head.weight": ([50000], "float32", shared),
            },
        )
        mgr.flush()

        on_disk = sum(
            os.path.getsize(os.path.join(d, f))
            for d, _, fs in os.walk(tmp_path)
            for f in fs
        )
        # Two distinct 200 KB tensors of incompressible bytes. A third copy
        # of the shared one would push this well past 400 KB.
        assert on_disk < 460_000, f"tied weight stored twice? {on_disk} bytes"

        loaded = mgr.load(snap_id)
        assert loaded["token_emb.weight"] == shared
        assert loaded["head.weight"] == shared
        assert loaded["qkv.weight"] == other

    def test_alias_survives_delta_chain(self, tmp_path):
        """An aliased tensor keeps resolving across later snapshots."""
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        rng = random.Random(11)
        for step in (100, 200, 300):
            shared = rng.randbytes(200_000)
            snap_id = mgr.save_tensors(
                step=step,
                tensors={
                    "emb": ([50000], "float32", shared),
                    "head": ([50000], "float32", shared),
                },
            )
            loaded = mgr.load(snap_id)
            assert loaded["emb"] == shared
            assert loaded["head"] == shared

    def test_all_tensors_changed_no_skip(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        t1 = {"a": ([100], "uint8", bytes([0] * 100)), "b": ([100], "uint8", bytes([0] * 100))}
        mgr.save_tensors(step=100, tensors=t1)
        t2 = {"a": ([100], "uint8", bytes([1] * 100)), "b": ([100], "uint8", bytes([2] * 100))}
        mgr.save_tensors(step=200, tensors=t2)

        info = mgr.list_snapshots()[1]
        assert info["skipped_tensors"] == 0, f"All tensors changed, expected 0 skipped, got {info['skipped_tensors']}"

    def test_delta_chain_correctness(self, tmp_path):
        """Multiple delta steps maintain data integrity."""
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000, max_deltas_per_full=100)
        base = bytes(50_000)
        mgr.save_tensors(step=100, tensors={"w": ([50000], "uint8", base)})

        current = bytearray(base)
        for step in range(200, 600, 100):
            current[step % len(current)] = step % 256
            mgr.save_tensors(step=step, tensors={"w": ([50000], "uint8", bytes(current))})

        _, loaded = mgr.load_latest()
        assert loaded["w"] == bytes(current)


# ─── Compression ────────────────────────────────────────────────────

class TestCompression:
    def test_repetitive_data_compresses_well(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), compression_level=3, full_every_steps=100000)
        raw = struct.pack(f"{25000}f", *([0.0001] * 25000))
        mgr.save_tensors(step=1, tensors={"momentum": ([25000], "float32", raw)})
        info = mgr.list_snapshots()[0]
        ratio = info["total_compressed"] / info["total_raw"]
        assert ratio < 0.01

    def test_no_compression(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), compression_level=0, full_every_steps=100000)
        data = bytes(range(256)) * 40
        mgr.save_tensors(step=1, tensors={"data": ([10240], "uint8", data)})
        info = mgr.list_snapshots()[0]
        assert info["total_compressed"] >= info["total_raw"]

    def test_compression_levels(self, tmp_path):
        """Higher compression level = smaller output."""
        data = bytes([42] * 100_000)
        sizes = {}
        for level in [1, 3, 9]:
            d = str(tmp_path / f"level_{level}")
            mgr = MoonclipManager(storage_root=d, compression_level=level, full_every_steps=100000)
            mgr.save_tensors(step=1, tensors={"w": ([100000], "uint8", data)})
            sizes[level] = mgr.list_snapshots()[0]["total_compressed"]
        # level 9 should be <= level 1 (may be equal for trivial data)
        assert sizes[9] <= sizes[1]


# ─── Integrity ──────────────────────────────────────────────────────

class TestIntegrity:
    def test_corruption_detected(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        snap_id = mgr.save_tensors(step=1, tensors=make_tensors(42))
        mgr.flush()  # wait for the background save before touching files

        # Past the pack header, which is where tensor data starts.
        #
        # This used to write at offset 0, back when a pack was nothing but
        # concatenated tensor blobs. A pack now opens with a 32-byte header
        # describing its own contents, so corrupting offset 0 damages the
        # description and leaves every tensor intact — the load then succeeds
        # and the test proves nothing. The load path reads tensors at the
        # offsets the manifest records and never looks at the header, so the
        # bytes that matter here are the ones after it.
        PACK_HEADER_LEN = 32

        corrupted = 0
        for root, dirs, files in os.walk(str(tmp_path / "snapshots")):
            for f in files:
                if f.endswith(".pack"):
                    with open(os.path.join(root, f), "r+b") as fh:
                        fh.seek(PACK_HEADER_LEN)
                        fh.write(b"\xFF\xFF\xFF\xFF")
                    corrupted += 1
        assert corrupted > 0

        with pytest.raises(RuntimeError, match=r"Integrity|Compression"):
            mgr.load(snap_id)


# ─── Snapshot info ──────────────────────────────────────────────────

class TestSnapshotInfo:
    def test_full_save_stats(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        mgr.save_tensors(step=100, tensors=make_tensors(0))
        info = mgr.list_snapshots()[0]
        assert info["step"] == 100
        assert info["full_tensors"] == 3
        assert info["delta_tensors"] == 0
        assert info["skipped_tensors"] == 0
        assert info["ranks"] == 1
        assert info["total_raw"] > 0
        assert info["total_compressed"] > 0

    def test_delta_save_stats(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        mgr.save_tensors(step=100, tensors=make_tensors(0))
        t2 = make_tensors(0)
        d = bytearray(t2["model.weight"][2]); d[0] = 1
        t2["model.weight"] = ([64, 32], "float32", bytes(d))
        mgr.save_tensors(step=200, tensors=t2)

        info = mgr.list_snapshots()[1]
        assert info["is_delta"] is True
        total = info["skipped_tensors"] + info["delta_tensors"] + info["full_tensors"]
        assert total == 3

    def test_empty_list_on_fresh_manager(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path))
        assert mgr.list_snapshots() == []


# ─── Multi-rank ─────────────────────────────────────────────────────

class TestMultiRank:
    def test_two_rank_save_load(self, tmp_path):
        p = str(tmp_path)
        mgr_r0 = MoonclipManager(storage_root=p, world_size=2, rank=0, full_every_steps=100000)
        snap_id = mgr_r0.create_snapshot(step=1000, metadata={"loss": "0.5"})

        mgr_r0.save_rank(snap_id, tensors={
            "shard": ([100], "float32", bytes([1] * 400)),
        })
        mgr_r1 = MoonclipManager(storage_root=p, world_size=2, rank=1, full_every_steps=100000)
        mgr_r1.save_rank(snap_id, tensors={
            "shard": ([100], "float32", bytes([2] * 400)),
        })

        MoonclipManager(storage_root=p, world_size=2, rank=0, full_every_steps=100000).finalize_snapshot(snap_id)

        assert MoonclipManager(storage_root=p, world_size=2, rank=0).load(snap_id)["shard"][0] == 1
        assert MoonclipManager(storage_root=p, world_size=2, rank=1).load(snap_id)["shard"][0] == 2

    def test_incomplete_finalize_fails(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), world_size=2, rank=0, full_every_steps=100000)
        snap_id = mgr.create_snapshot(step=1000)
        mgr.save_rank(snap_id, tensors={"s": ([10], "uint8", bytes(10))})

        with pytest.raises(RuntimeError, match="1/2"):
            mgr.finalize_snapshot(snap_id)


# ─── Page alignment ────────────────────────────────────────────────

class TestPageAlignment:
    def test_pack_files_aligned(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        mgr.save_tensors(step=1, tensors=make_tensors(42))
        mgr.flush()
        checked = 0
        for root, _, files in os.walk(str(tmp_path / "snapshots")):
            for f in files:
                if f.endswith(".pack"):
                    size = os.path.getsize(os.path.join(root, f))
                    assert size % 4096 == 0, f"{f} is {size} bytes"
                    checked += 1
        assert checked > 0

    def test_manifest_aligned(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        mgr.save_tensors(step=1, tensors=make_tensors(0))
        mgr.flush()
        assert os.path.getsize(str(tmp_path / "manifest.json")) % 4096 == 0

    def test_data_survives_padding(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        data = bytes(range(100))
        snap_id = mgr.save_tensors(step=1, tensors={"tiny": ([100], "uint8", data)})
        assert mgr.load(snap_id)["tiny"] == data


# ─── Retention policy ───────────────────────────────────────────────

class TestRetention:
    def test_old_snapshots_cleaned(self, tmp_path):
        mgr = MoonclipManager(
            storage_root=str(tmp_path), max_full_snapshots=2,
            max_deltas_per_full=2, full_every_steps=5,
        )
        for step in range(1, 10):
            mgr.save_tensors(step=step, tensors={"w": ([100], "uint8", bytes([step] * 100))})
        snaps = mgr.list_snapshots()
        assert len(snaps) <= 6

    def test_data_still_loadable_after_retention(self, tmp_path):
        mgr = MoonclipManager(
            storage_root=str(tmp_path), max_full_snapshots=2,
            full_every_steps=1,
        )
        ids = []
        for step in range(1, 6):
            ids.append(mgr.save_tensors(step=step, tensors={"w": ([10], "uint8", bytes([step] * 10))}))

        # Latest should always be loadable
        _, loaded = mgr.load_latest()
        assert loaded["w"][0] == 5


# ─── Edge cases ─────────────────────────────────────────────────────

class TestEdgeCases:
    def test_extreme_dimensions(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        tensors = {
            "scalar": ([], "float32", struct.pack("f", 3.14)),
            "single": ([1], "int32", struct.pack("i", 42)),
            "5d": ([2, 2, 2, 2, 2], "uint8", bytes([1] * 32)),
        }
        snap_id = mgr.save_tensors(step=1, tensors=tensors)
        loaded = mgr.load(snap_id)
        assert loaded["scalar"] == tensors["scalar"][2]
        assert loaded["single"] == tensors["single"][2]
        assert loaded["5d"] == tensors["5d"][2]

    def test_empty_tensor(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        snap_id = mgr.save_tensors(step=1, tensors={"empty": ([0], "float32", b"")})
        assert mgr.load(snap_id)["empty"] == b""

    def test_many_small_tensors(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        tensors = {f"t_{i}": ([4], "float32", struct.pack("4f", 0.0, 0.0, 0.0, float(i))) for i in range(100)}
        snap_id = mgr.save_tensors(step=1, tensors=tensors)
        loaded = mgr.load(snap_id)
        assert len(loaded) == 100
        assert loaded["t_99"] == tensors["t_99"][2]

    def test_special_chars_in_name(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        tensors = {
            "module.layers.0.self_attn.q_proj.weight": ([4], "float32", bytes(16)),
            "encoder/block_0/layer_0": ([4], "float32", bytes(16)),
        }
        snap_id = mgr.save_tensors(step=1, tensors=tensors)
        loaded = mgr.load(snap_id)
        assert len(loaded) == 2

    def test_persist_across_manager_instances(self, tmp_path):
        """Data saved by one manager instance is loadable by another."""
        p = str(tmp_path)
        mgr1 = MoonclipManager(storage_root=p, full_every_steps=100000)
        snap_id = mgr1.save_tensors(step=42, tensors={"w": ([10], "uint8", bytes(10))})
        del mgr1

        mgr2 = MoonclipManager(storage_root=p, full_every_steps=100000)
        loaded = mgr2.load(snap_id)
        assert loaded["w"] == bytes(10)
        assert mgr2.list_snapshots()[0]["step"] == 42


# ─── Casting ────────────────────────────────────────────────────────

class TestSaveDtype:
    def test_bf16_does_not_turn_nan_into_infinity(self, tmp_path):
        """A NaN whose payload sits below bit 16 used to round up to +Inf.

        bf16 keeps the top 7 mantissa bits, so 0x7F800001 arrived at the
        truncation with a zero mantissa under an all-ones exponent — which is
        infinity. A checkpoint that quietly makes a diverged run look finite is
        worse than one that fails.
        """
        mgr = MoonclipManager(
            storage_root=str(tmp_path), save_dtype="bf16", full_every_steps=100000
        )
        payload = b"".join(
            struct.pack("<I", bits) for bits in (0x7F800001, 0xFF800001, 0x7FC00000)
        )
        snap_id = mgr.save_tensors(step=1, tensors={"w": ([3], "float32", payload)})
        mgr.flush()

        values = struct.unpack("<3f", mgr.load(snap_id)["w"])
        assert all(v != v for v in values), f"NaN did not survive the cast: {values}"

    def test_bf16_keeps_infinity(self, tmp_path):
        mgr = MoonclipManager(
            storage_root=str(tmp_path), save_dtype="bf16", full_every_steps=100000
        )
        payload = struct.pack("<2f", float("inf"), float("-inf"))
        snap_id = mgr.save_tensors(step=1, tensors={"w": ([2], "float32", payload)})
        mgr.flush()

        hi, lo = struct.unpack("<2f", mgr.load(snap_id)["w"])
        assert hi == float("inf") and lo == float("-inf")


# ─── Warnings ───────────────────────────────────────────────────────

class TestMergerWarning:
    def test_merge_stride_warns_about_the_known_defect(self, tmp_path):
        """Merging drops tensors added after the base snapshot (fixed in 0.0.6).

        Whoever turns it on has to hear that before the run, not after.
        """
        with pytest.warns(UserWarning, match="merge_stride"):
            MoonclipManager(storage_root=str(tmp_path), merge_stride=3)

    def test_the_default_is_silent(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MoonclipManager(storage_root=str(tmp_path))


# ─── Concurrency ────────────────────────────────────────────────────

class TestConcurrency:
    def test_concurrent_saves(self, tmp_path):
        mgr = MoonclipManager(storage_root=str(tmp_path), full_every_steps=100000)
        errors = []

        def worker(step):
            try:
                mgr.save_tensors(step=step, tensors={f"t_{step}": ([10], "uint8", bytes([step % 256] * 10))})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 11)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrency errors: {errors}"
        # save_tensors returns once the job is queued, not once it is written,
        # so the last save can still be in flight here.
        mgr.flush()
        assert len(mgr.list_snapshots()) == 10


# ─── S3 backend validation ──────────────────────────────────────────

class TestS3Validation:
    def test_missing_credentials_errors(self, tmp_path):
        with pytest.raises(ValueError, match="s3_access_key"):
            MoonclipManager(storage_root=str(tmp_path), s3_bucket="test-bucket")

    def test_missing_secret_errors(self, tmp_path):
        with pytest.raises(ValueError, match="s3_secret_key"):
            MoonclipManager(storage_root=str(tmp_path), s3_bucket="test", s3_access_key="AK")

    def test_s3_config_accepted(self, tmp_path):
        """S3 config with local primary + remote sync should construct without error."""
        mgr = MoonclipManager(
            storage_root=str(tmp_path),
            s3_bucket="bucket",
            s3_access_key="AK",
            s3_secret_key="SK",
            s3_endpoint="http://localhost:19999",
            s3_path_style=True,
            sync_every_n_saves=10,
        )
        # Should be able to save locally even though S3 is unreachable
        snap_id = mgr.save_tensors(step=1, tensors={"w": ([10], "uint8", bytes(10))})
        assert len(snap_id) == 36
