"""
PyTorch-specific integration tests for Moonclip.
Requires: pip install torch
Skipped automatically if torch is not installed.
"""

import os
import sys
import struct
import pickle

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

torch = pytest.importorskip("torch")

from moonclip import CheckpointManager


class TestCheckpointManager:
    def test_model_save_load(self, tmp_path):
        mgr = CheckpointManager(storage_root=str(tmp_path))

        model = torch.nn.Linear(32, 16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Run one step so optimizer has state
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        original_weight = model.weight.data.clone()
        original_bias = model.bias.data.clone()

        snap_id = mgr.save(step=1, model=model, optimizer=optimizer, metadata={"loss": "0.5"})

        # Corrupt model
        model.weight.data.zero_()
        model.bias.data.zero_()

        mgr.load(snap_id, model=model, optimizer=optimizer)

        assert torch.allclose(model.weight.data, original_weight)
        assert torch.allclose(model.bias.data, original_bias)

    def test_load_latest(self, tmp_path):
        mgr = CheckpointManager(storage_root=str(tmp_path))
        model = torch.nn.Linear(8, 4)

        model.weight.data.fill_(1.0)
        mgr.save(step=100, model=model)

        model.weight.data.fill_(2.0)
        mgr.save(step=200, model=model)

        model.weight.data.zero_()
        mgr.load_latest(model=model)
        assert torch.allclose(model.weight.data, torch.full_like(model.weight, 2.0))

    def test_scheduler_and_scaler(self, tmp_path):
        mgr = CheckpointManager(storage_root=str(tmp_path))
        model = torch.nn.Linear(8, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

       # Need a dummy forward/backward before scheduler.step()
        x = torch.randn(1, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.step()
        scheduler.step()
        original_lr = optimizer.param_groups[0]["lr"]

        mgr.save(step=1, model=model, optimizer=optimizer, scheduler=scheduler)

        # Reset
        optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1)

        _, _ = mgr.load_latest(model=model, optimizer=optimizer2, scheduler=scheduler2)
        assert optimizer2.param_groups[0]["lr"] == pytest.approx(original_lr)

    def test_delta_between_training_steps(self, tmp_path):
        """After a few optimizer steps, most tensors change slightly — perfect for delta."""
        mgr = CheckpointManager(storage_root=str(tmp_path), full_every_steps=100000)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Save initial
        mgr.save(step=0, model=model, optimizer=optimizer)

        # One training step
        x = torch.randn(8, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        snap_id = mgr.save(step=1, model=model, optimizer=optimizer)

        snaps = mgr.list_snapshots()
        assert len(snaps) == 2
        # Second save should have some skipped tensors (optimizer state not all changed)
        # and delta tensors
        info = snaps[1]
        assert info["is_delta"] is True
        assert info["skipped_tensors"] > 0 or info["delta_tensors"] > 0

    def test_non_tensor_state(self, tmp_path):
        """Optimizer state_dict contains non-tensor data (step counts, etc)."""
        mgr = CheckpointManager(storage_root=str(tmp_path))

        class CustomObj:
            def state_dict(self):
                return {"count": 42, "name": "test", "nested": {"a": 1}}
            def load_state_dict(self, sd):
                self.loaded = sd

        obj = CustomObj()
        snap_id = mgr.save(step=1, model=obj)

        obj2 = CustomObj()
        mgr.load(snap_id, model=obj2)
        assert obj2.loaded["count"] == 42
        assert obj2.loaded["name"] == "test"
        assert obj2.loaded["nested"]["a"] == 1

    def test_many_layers_model(self, tmp_path):
        """Model with 20 layers — tests many-tensor handling."""
        mgr = CheckpointManager(storage_root=str(tmp_path), full_every_steps=100000)
        model = torch.nn.Sequential(*[torch.nn.Linear(32, 32) for _ in range(20)])

        mgr.save(step=0, model=model)

        # Change one layer's weights
        model[10].weight.data += 0.001

        snap_id = mgr.save(step=1, model=model)
        snaps = mgr.list_snapshots()
        info = snaps[1]

        # Most layers unchanged — should see many skips
        assert info["skipped_tensors"] > 30  # 20 layers × 2 params = 40, most skipped

        # Roundtrip
        model2 = torch.nn.Sequential(*[torch.nn.Linear(32, 32) for _ in range(20)])
        mgr.load(snap_id, model=model2)
        assert torch.allclose(model[10].weight.data, model2[10].weight.data)


class TestDeltaRoundTrip:
    """A delta snapshot has to restore the exact weights, byte for byte.

    Deltas are stored byte-shuffled and XOR-ed against the base, so a
    checkpoint restored from one passes through two transforms that are
    invisible until they are wrong — and when they are wrong they do not
    raise, they hand back plausible-looking noise.
    """

    def test_weights_survive_a_delta_snapshot_exactly(self, tmp_path):
        mgr = CheckpointManager(storage_root=str(tmp_path), full_every_steps=100000)
        model = torch.nn.Sequential(
            torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 64)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        mgr.save(step=0, model=model, optimizer=optimizer)

        # Real steps: gradient descent leaves most of each float's bits alone,
        # which is the case the delta path and the shuffle are built for.
        for _ in range(3):
            loss = model(torch.randn(8, 256)).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        expected = {k: v.clone() for k, v in model.state_dict().items()}
        snap_id = mgr.save(step=1, model=model, optimizer=optimizer)

        assert mgr.list_snapshots()[1]["is_delta"] is True
        assert mgr.list_snapshots()[1]["delta_tensors"] > 0, "no delta, nothing tested"

        restored = torch.nn.Sequential(
            torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 64)
        )
        mgr.load(snap_id, model=restored)

        for key, want in expected.items():
            assert torch.equal(restored.state_dict()[key], want), f"{key} differs"


class TestFlattenAsTensors:
    """`flatten_state_dict(as_tensors=True)` hands tensors to Rust directly.

    The copy that used to happen twice on the calling thread — `.tobytes()`
    here, then again on the way into Rust — happens once inside `save_raw`,
    in parallel with the GIL released. What must not change is the result.
    """

    def test_matches_the_byte_path_exactly(self, tmp_path):
        from moonclip import flatten_state_dict

        model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 8))
        state = model.state_dict()

        by_bytes = CheckpointManager(storage_root=str(tmp_path / "bytes"))
        by_tensors = CheckpointManager(storage_root=str(tmp_path / "tensors"))

        flat_bytes, _ = flatten_state_dict(state, "model")
        flat_tensors, _ = flatten_state_dict(state, "model", as_tensors=True)

        # Same names, so per-tensor delta tracking keys on the same things.
        assert set(flat_bytes) == set(flat_tensors)

        loaded_bytes = by_bytes.load(by_bytes.save_raw(step=0, tensors=flat_bytes))
        loaded_tensors = by_tensors.load(
            by_tensors.save_raw(step=0, tensors=flat_tensors)
        )

        for key, original in state.items():
            assert torch.equal(loaded_bytes["model"][key], original)
            assert torch.equal(loaded_tensors["model"][key], original)

    def test_unchanged_tensors_are_still_skipped(self, tmp_path):
        """Names must stay stable across steps or every save is a full one."""
        from moonclip import flatten_state_dict

        mgr = CheckpointManager(storage_root=str(tmp_path), full_every_steps=100000)
        model = torch.nn.Sequential(*[torch.nn.Linear(32, 32) for _ in range(10)])

        flat, _ = flatten_state_dict(model.state_dict(), "model", as_tensors=True)
        mgr.save_raw(step=0, tensors=flat)

        model[3].weight.data += 0.001
        flat, _ = flatten_state_dict(model.state_dict(), "model", as_tensors=True)
        mgr.save_raw(step=1, tensors=flat)

        info = mgr.list_snapshots()[1]
        assert info["is_delta"] is True
        assert info["skipped_tensors"] > 15  # 10 layers x 2 params, one touched

    def test_exotic_dtypes_stay_on_the_byte_path(self):
        """The extension cannot address every dtype; those must not be handed
        to it as tensors."""
        from moonclip import flatten_state_dict

        state = {"weight": torch.randn(4, 4), "spectrum": torch.randn(4, dtype=torch.complex64)}
        flat, _ = flatten_state_dict(state, "model", as_tensors=True)

        assert torch.is_tensor(flat["model/weight"])
        assert isinstance(flat["model/spectrum"], tuple)


class TestRawTensorGuards:
    """`save_tensors` reads the bytes at `data_ptr()`. Anything that makes
    those bytes not be the tensor has to be refused, not stored wrong."""

    def test_non_contiguous_is_refused(self, tmp_path):
        from moonclip import MoonclipManager

        mgr = MoonclipManager(storage_root=str(tmp_path))
        transposed = torch.randn(8, 8).t()
        assert not transposed.is_contiguous()

        with pytest.raises(ValueError, match="contiguous"):
            mgr.save_tensors(step=0, tensors={"x": transposed})

    def test_empty_tensor_roundtrips(self, tmp_path):
        """A zero-element tensor's data_ptr may be null."""
        from moonclip import MoonclipManager

        mgr = MoonclipManager(storage_root=str(tmp_path))
        snap_id = mgr.save_tensors(
            step=0, tensors={"empty": torch.empty(0), "real": torch.ones(4)}
        )
        loaded = mgr.load(snap_id)

        assert len(loaded["empty"]) == 0
        assert torch.equal(
            torch.frombuffer(loaded["real"], dtype=torch.float32), torch.ones(4)
        )


class TestReportingNeverKillsTheRun:
    """Reporting runs inside the training loop. A status line that raises takes
    the whole run down — and on a Windows console it did: `print_stats` drew a
    rule out of U+2500, which cp1252 cannot encode."""

    @staticmethod
    def _cp1252_stdout(monkeypatch):
        """Replace stdout with a strict cp1252 stream, as a Windows console is."""
        import io

        stream = io.TextIOWrapper(
            io.BytesIO(), encoding="cp1252", errors="strict", newline=""
        )
        monkeypatch.setattr(sys, "stdout", stream)
        return stream

    @staticmethod
    def _text(stream):
        stream.flush()
        return stream.buffer.getvalue().decode("cp1252")

    def test_print_stats_survives_a_cp1252_console(self, tmp_path, monkeypatch):
        mgr = CheckpointManager(storage_root=str(tmp_path))
        mgr.save(step=0, model=torch.nn.Linear(8, 4))

        stream = self._cp1252_stdout(monkeypatch)
        mgr.print_stats()  # must not raise

        out = self._text(stream)
        assert "Moonclip Checkpoint Stats" in out
        assert "─" not in out, "the box-drawing rule must not reach a cp1252 console"

    def test_print_stats_with_no_checkpoints_survives_too(self, tmp_path, monkeypatch):
        mgr = CheckpointManager(storage_root=str(tmp_path))
        stream = self._cp1252_stdout(monkeypatch)
        mgr.print_stats()
        assert "No checkpoints yet" in self._text(stream)

    def test_an_unencodable_path_does_not_raise(self, tmp_path, monkeypatch):
        """The rule is ours to choose; a path is not. `save_to_pt` prints the
        destination, and nothing stops it holding a character the console has
        no room for."""
        from moonclip.pytorch import _emit

        stream = self._cp1252_stdout(monkeypatch)
        _emit("[Moonclip] Exported to /tmp/中文/model.pt (1.0 MB)")

        out = self._text(stream)
        assert "[Moonclip] Exported to" in out
        assert "model.pt" in out


class TestFloat8:
    """Float8 arrives two ways and they are not the same thing.

    A tensor that is *already* float8 — which is what FSDP2 and torchao hand
    over — is stored byte for byte and comes back byte for byte. Until 0.0.9
    it could not be stored at all: `save_tensors` raised
    `ValueError: unsupported dtype torch.float8_e4m3fn`, and the only way
    forward was converting every parameter back to bf16 by hand.

    `save_dtype="fp8"` is the other way, and is lossy on purpose: fp32 goes in
    and one byte per element comes out, against a per-tensor scale.
    """

    FORMATS = ["float8_e4m3fn", "float8_e5m2"]

    @pytest.mark.parametrize("dtype_name", FORMATS)
    def test_a_native_float8_tensor_round_trips_bit_exact(self, tmp_path, dtype_name):
        from moonclip import MoonclipManager

        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            pytest.skip(f"this torch has no {dtype_name}")

        torch.manual_seed(0)
        # Scaled past ±1 so the exponent range is exercised, not just the
        # mantissa around 1.0.
        original = (torch.randn(128, 64) * 12).to(dtype)

        mgr = MoonclipManager(storage_root=str(tmp_path), async_save=False)
        snap_id = mgr.save_tensors(step=0, tensors={"w": original})
        loaded = mgr.load(snap_id)

        assert len(loaded["w"]) == original.numel(), "float8 is one byte per element"
        back = torch.frombuffer(bytes(loaded["w"]), dtype=dtype).reshape(original.shape)
        assert torch.equal(back.view(torch.uint8), original.view(torch.uint8)), (
            "a float8 tensor must come back with the identical bits — nothing "
            "on this path is allowed to reinterpret it"
        )

    @pytest.mark.parametrize("dtype_name", FORMATS)
    def test_a_native_float8_tensor_ignores_save_dtype(self, tmp_path, dtype_name):
        """`save_dtype` names what fp32 is cast *down* to. Applying it to a
        tensor that is already float8 would cast it back *up* — doubling the
        size of the one dtype chosen to make things smaller."""
        from moonclip import MoonclipManager

        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            pytest.skip(f"this torch has no {dtype_name}")

        original = (torch.randn(128, 64) * 12).to(dtype)
        mgr = MoonclipManager(
            storage_root=str(tmp_path), async_save=False, save_dtype="bf16"
        )
        snap_id = mgr.save_tensors(step=0, tensors={"w": original})
        loaded = mgr.load(snap_id)

        assert len(loaded["w"]) == original.numel(), "stored at bf16 width, not float8"
        back = torch.frombuffer(bytes(loaded["w"]), dtype=dtype).reshape(original.shape)
        assert torch.equal(back.view(torch.uint8), original.view(torch.uint8))

    def test_every_float8_dtype_this_torch_has_can_be_stored(self, tmp_path):
        """The list in `get_element_size` is the gate. A name missing from it
        is not a degraded save, it is a hard `ValueError` in the middle of a
        training run — so the test enumerates whatever this torch actually
        exposes rather than the two anyone uses."""
        from moonclip import MoonclipManager

        names = [n for n in dir(torch) if n.startswith("float8_")]
        assert names, "this torch exposes no float8 dtypes at all"

        mgr = MoonclipManager(storage_root=str(tmp_path), async_save=False)
        for name in names:
            dtype = getattr(torch, name)
            if not isinstance(dtype, torch.dtype):
                continue
            raw = torch.zeros(64, dtype=torch.uint8).view(dtype)
            snap_id = mgr.save_tensors(step=0, tensors={name: raw})
            assert len(mgr.load(snap_id)[name]) == 64, name

    def test_fp8_save_dtype_quantizes_and_hands_back_the_original_dtype(self, tmp_path):
        """Training gets fp32 buffers back whatever the checkpoint holds, or
        `load_state_dict` fails on dtype — the same contract bf16 has."""
        from moonclip import MoonclipManager

        torch.manual_seed(0)
        # Well outside float8's own ±448: only a correctly applied scale
        # brings these back at all.
        original = torch.randn(256, 128) * 3000

        mgr = MoonclipManager(
            storage_root=str(tmp_path), async_save=False, save_dtype="fp8"
        )
        snap_id = mgr.save_tensors(step=0, tensors={"w": original})
        loaded = mgr.load(snap_id)

        assert len(loaded["w"]) == original.numel() * 4, "fp32 width on the way out"
        back = torch.frombuffer(bytes(loaded["w"]), dtype=torch.float32).reshape(
            original.shape
        )

        amax = original.abs().max().item()
        worst = (back - original).abs().max().item() / amax
        # e4m3 keeps four significant bits. Bounded on both sides: a test that
        # only checked the error was small would still pass if quantization
        # silently stopped happening.
        assert worst < 0.05, f"relative-to-amax error {worst} is too large for e4m3"
        assert worst > 0.0, "nothing was quantized"

    def test_fp8_is_a_quarter_of_fp32_on_disk(self, tmp_path):
        """The reason to accept the precision loss. Measured on the pack file,
        because that is what the disk bill is."""
        from moonclip import MoonclipManager

        torch.manual_seed(0)
        # Incompressible, so the comparison is the dtype and not zstd.
        original = torch.randn(512, 512)

        sizes = {}
        for label, save_dtype in (("fp32", "none"), ("fp8", "fp8")):
            root = tmp_path / label
            mgr = MoonclipManager(
                storage_root=str(root), async_save=False, save_dtype=save_dtype
            )
            mgr.save_tensors(step=0, tensors={"w": original})
            mgr.flush()
            sizes[label] = sum(
                p.stat().st_size for p in root.rglob("*.pack")
            )

        assert sizes["fp32"] > 0 and sizes["fp8"] > 0, sizes
        ratio = sizes["fp8"] / sizes["fp32"]
        assert ratio < 0.35, f"fp8 pack is {ratio:.2f} of the fp32 one: {sizes}"

    def test_an_unknown_float8_spelling_is_still_refused(self, tmp_path):
        """Adding float8 to `save_dtype` must not have turned the setting into
        one that accepts anything: a typo there used to produce checkpoints at
        twice the expected size and say nothing."""
        from moonclip import MoonclipManager

        with pytest.raises(Exception, match="fp8"):
            MoonclipManager(storage_root=str(tmp_path), save_dtype="fp8_e3m4")
