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


class TestTopologyIsStated:
    """Which topology a manager is for is answered by the caller, not guessed.

    It used to be guessed, and guessing could only ever take capability away:
    `Coordinator::save` refuses when `world_size != 1`, so adopting a detected
    world size turned the single-rank `save()` into an error decided by how
    the process happened to be launched. Nothing in the code said so.
    """

    @staticmethod
    def _launcher(monkeypatch, rank, world_size):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", str(world_size))

    def test_one_process_is_still_one_store(self, tmp_path, monkeypatch):
        """The unchanged case, and the common one: nothing to disagree with."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)

        mgr = CheckpointManager(storage_root=str(tmp_path))
        assert (mgr.world_size, mgr.rank) == (1, 0)

    def test_an_unanswered_launcher_is_refused_not_adopted(self, tmp_path, monkeypatch):
        self._launcher(monkeypatch, 3, 8)

        with pytest.raises(ValueError) as caught:
            CheckpointManager(storage_root=str(tmp_path))

        message = str(caught.value)
        # The numbers it saw, so nobody has to go looking for them.
        assert "rank 3 of 8" in message
        # And every way out, because an error that only says "no" makes the
        # reader guess, which is what this whole change is against.
        assert "world_size=1, rank=0" in message
        assert "world_size=8, rank=3" in message
        assert 'world_size="auto"' in message

    def test_auto_still_reads_the_launcher(self, tmp_path, monkeypatch):
        """The old behaviour is still reachable — it just has to be asked for."""
        self._launcher(monkeypatch, 3, 8)

        mgr = CheckpointManager(
            storage_root=str(tmp_path), world_size="auto", rank="auto"
        )
        assert (mgr.world_size, mgr.rank) == (8, 3)

    def test_stated_values_beat_the_launcher(self, tmp_path, monkeypatch):
        """The case Ravex depends on: every rank writes to its own directory,
        so each store is single-rank however many processes there are."""
        self._launcher(monkeypatch, 3, 8)

        mgr = CheckpointManager(storage_root=str(tmp_path), world_size=1, rank=0)
        assert (mgr.world_size, mgr.rank) == (1, 0)

        # And the single-rank save API works, which is the whole point: under
        # the old detection this raised "Multi-rank save requires explicit
        # create_snapshot/save_rank/finalize flow".
        mgr.save(step=0, model=torch.nn.Linear(8, 4))
        mgr.flush()
        assert mgr.list_snapshots()

    def test_half_an_answer_is_still_an_answer(self, tmp_path, monkeypatch):
        """Stating one and detecting the other is legal; it is stating neither
        that is ambiguous."""
        self._launcher(monkeypatch, 3, 8)

        mgr = CheckpointManager(storage_root=str(tmp_path), world_size="auto", rank=0)
        assert (mgr.world_size, mgr.rank) == (8, 0)


class TestUnflattenStateDict:
    """`flatten_state_dict` is public; until 0.0.9 its inverse was not.

    The only implementation lived inside `CheckpointManager._apply_loaded`,
    welded to the step that calls `load_state_dict` on live objects — so a
    caller holding the bytes but no objects had to reimplement it or take the
    applying it did not want.
    """

    def _trained(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(16, 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.randn(4, 16)).sum().backward()
        optimizer.step()
        return model, optimizer

    def test_it_is_reachable_from_the_package(self):
        import moonclip

        assert callable(moonclip.unflatten_state_dict)
        assert "unflatten_state_dict" in moonclip.__all__

    def test_it_undoes_a_save_without_touching_any_object(self, tmp_path):
        from moonclip import MoonclipManager, unflatten_state_dict

        model, optimizer = self._trained()
        mgr = CheckpointManager(storage_root=str(tmp_path), async_save=False)
        mgr.save(step=1, model=model, optimizer=optimizer)
        mgr.flush()

        # Read through the low-level manager, which returns the flat map and
        # applies nothing — the position Ravex is in.
        raw = MoonclipManager(storage_root=str(tmp_path)).load_latest()[1]
        rebuilt = unflatten_state_dict(raw)

        assert set(rebuilt) == {"model", "optimizer"}
        for name, value in model.state_dict().items():
            assert torch.equal(rebuilt["model"][name], value), name
        # The optimizer travels as one pickled blob rather than per-tensor, so
        # this also covers the second of the two shapes that get written.
        assert rebuilt["optimizer"]["param_groups"][0]["lr"] == 1e-3

    def test_it_agrees_with_the_applying_path(self, tmp_path):
        """Same bytes, same answer, whether or not objects were handed over.
        The applying path is the one everything else already trusts."""
        from moonclip import MoonclipManager, unflatten_state_dict

        model, optimizer = self._trained()
        mgr = CheckpointManager(storage_root=str(tmp_path), async_save=False)
        mgr.save(step=1, model=model, optimizer=optimizer)
        mgr.flush()

        _, applied = CheckpointManager(storage_root=str(tmp_path)).load_latest()
        raw = MoonclipManager(storage_root=str(tmp_path)).load_latest()[1]
        rebuilt = unflatten_state_dict(raw)

        assert set(applied) == set(rebuilt)
        for name, value in applied["model"].items():
            assert torch.equal(rebuilt["model"][name], value), name

    def test_names_it_cannot_rebuild_are_left_out(self):
        """Not guessed at, and not an exception either: a store may hold
        tensors this never wrote, and reporting on them is not its job."""
        from moonclip import unflatten_state_dict

        assert unflatten_state_dict({}) == {}
        assert unflatten_state_dict({"loose/tensor": b"\x00\x01"}) == {}


class TestPerComponentSaveDtype:
    """`save_dtype` chosen per tensor name.

    The reason it exists: on a real run the optimizer moments are ~85% of
    the bytes written and delta almost not at all, and they are also the
    part that tolerates the least precision. One dtype for the whole
    snapshot cannot express that.
    """

    def _run(self, tmp_path, save_dtype, steps=2):
        """Train two steps, checkpoint each, reload into a fresh pair.

        Returns the largest absolute error on the weights and on `exp_avg`,
        which is what tells a cast component from an uncast one.
        """
        torch.manual_seed(0)
        model = torch.nn.Linear(128, 128, bias=False)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        mgr = CheckpointManager(
            storage_root=str(tmp_path), async_save=False, save_dtype=save_dtype
        )

        def step():
            model(torch.randn(8, 128)).sum().backward()
            opt.step()
            opt.zero_grad()

        for i in range(steps):
            step()
            mgr.save(step=i + 1, model=model, optimizer=opt)
        mgr.flush()

        want_w = model.weight.detach().clone()
        want_m = opt.state[model.weight]["exp_avg"].clone()

        torch.manual_seed(0)
        model2 = torch.nn.Linear(128, 128, bias=False)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=0.1)
        model2(torch.randn(8, 128)).sum().backward()
        opt2.step()
        opt2.zero_grad()
        CheckpointManager(storage_root=str(tmp_path)).load_latest(
            model=model2, optimizer=opt2
        )

        return (
            (model2.weight.detach() - want_w).abs().max().item(),
            (opt2.state[model2.weight]["exp_avg"] - want_m).abs().max().item(),
        )

    def test_a_string_still_casts_everything(self, tmp_path):
        """The form every existing configuration uses must not change
        meaning."""
        weight_err, moment_err = self._run(tmp_path, "bf16")
        assert weight_err > 0
        assert moment_err > 0

    def test_a_pattern_casts_only_what_it_matches(self, tmp_path):
        """The headline case: moments to bf16, weights untouched.

        The weights must come back *bit* exact — anything else means the
        cast leaked past its pattern.
        """
        weight_err, moment_err = self._run(tmp_path, {"optimizer/*": "bf16"})
        assert weight_err == 0.0
        assert moment_err > 0

    def test_an_exception_is_written_by_putting_it_first(self, tmp_path):
        weight_err, moment_err = self._run(
            tmp_path, {"model/*": "none", "*": "bf16"}
        )
        assert weight_err == 0.0
        assert moment_err > 0

    def test_the_order_of_the_rules_is_the_precedence(self, tmp_path):
        """First match wins, so a catch-all written first shadows what
        follows. Reversing the previous test has to reverse its outcome —
        otherwise the ordering is not doing anything and the documented rule
        is a fiction."""
        weight_err, _ = self._run(tmp_path, {"*": "bf16", "model/*": "none"})
        assert weight_err > 0

    def test_no_save_dtype_casts_nothing(self, tmp_path):
        weight_err, moment_err = self._run(tmp_path, None)
        assert weight_err == 0.0
        assert moment_err == 0.0

    def test_a_delta_after_a_partial_cast_still_reconstructs(self, tmp_path):
        """The failure mode the per-tensor `retain_base` rule avoids.

        With one component cast and the other not, the base kept in memory
        for the next delta is partial. Four steps is enough for the second
        save onward to be deltas against it.
        """
        weight_err, moment_err = self._run(
            tmp_path, {"optimizer/*": "bf16"}, steps=4
        )
        assert weight_err == 0.0
        # bf16 keeps 8 significant bits. An error this small is rounding; a
        # delta applied to the wrong base would be nothing like it.
        assert moment_err < 0.05

    def test_integer_state_is_never_cast(self, tmp_path):
        """`save_dtype` names a target for floats. A step counter is not a
        float and has no business being touched because one was set."""
        torch.manual_seed(0)
        model = torch.nn.Linear(64, 64, bias=False)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        mgr = CheckpointManager(
            storage_root=str(tmp_path), async_save=False, save_dtype="bf16"
        )
        counter = torch.arange(1024, dtype=torch.int64)
        mask = torch.zeros(1024, dtype=torch.bool)
        mask[::3] = True

        model(torch.randn(4, 64)).sum().backward()
        opt.step()
        mgr.save(
            step=1,
            model=model,
            optimizer=opt,
            extra={"counters": {"seen": counter, "mask": mask}},
        )
        mgr.flush()

        _, state = CheckpointManager(storage_root=str(tmp_path)).load_latest()
        back = state["extra/counters"]
        assert back["seen"].dtype == torch.int64
        assert torch.equal(back["seen"], counter)
        assert back["mask"].dtype == torch.bool
        assert torch.equal(back["mask"], mask)

    def test_fp64_is_a_target_that_loses_nothing(self, tmp_path):
        """It only ever widens — it recovers no precision the source did not
        have — but it must not drop any either."""
        weight_err, moment_err = self._run(tmp_path, "fp64")
        assert weight_err == 0.0
        assert moment_err == 0.0

    def test_a_pattern_that_matches_nothing_is_reported(self, tmp_path, capfd):
        """The quiet failure of a pattern API. `{"weight": ...}` reads as if
        it did something: names arrive as `model/weight`, so it matches
        nothing, casts nothing, and would otherwise say nothing."""
        self._run(tmp_path, {"weight": "bf16"}, steps=1)
        err = capfd.readouterr().err
        assert "save_dtype pattern(s) weight matched none" in err

    @pytest.mark.parametrize(
        "bad,exc",
        [
            ({"optimizer/*": "bfloat"}, RuntimeError),
            ({"optimizer/*": 16}, TypeError),
            ({"": "bf16"}, RuntimeError),
            ("bfloat", RuntimeError),
            (16, TypeError),
        ],
    )
    def test_a_bad_save_dtype_is_refused_not_ignored(self, tmp_path, bad, exc):
        """Same rule as the scalar form has always had: a value that cannot
        be understood stops the run, because mapping it to "do not cast"
        makes a typo into a silently doubled checkpoint."""
        with pytest.raises(exc):
            CheckpointManager(storage_root=str(tmp_path), save_dtype=bad)



class TestDescribeWithoutLoading:
    """Reading a checkpoint's shapes used to mean reading the checkpoint.

    `list_snapshots` reported sizes and steps, `load` gave back everything,
    and there was nothing in between — so a caller that needed to know how
    long a shard was loaded the shard to find out. A reshard from N ranks to M
    does that once per old rank before it can plan a single slice: N complete
    reads of data it discards.
    """

    def _trained(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(64, 32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.randn(8, 64)).sum().backward()
        optimizer.step()
        return model, optimizer

    def test_describe_reports_every_tensor_with_its_shape(self, tmp_path):
        model, optimizer = self._trained()
        mgr = CheckpointManager(storage_root=str(tmp_path), async_save=False)
        snap = mgr.save(step=3, model=model, optimizer=optimizer, metadata={"lr": "1e-3"})
        mgr.flush()

        described = mgr.describe(snap)

        assert described["step"] == 3
        assert described["id"] == snap
        assert described["metadata"]["lr"] == "1e-3"

        by_name = {t["name"]: t for t in described["tensors"]}
        for name, tensor in model.state_dict().items():
            entry = by_name["model/" + name]
            assert tuple(entry["shape"]) == tuple(tensor.shape), name
            assert entry["dtype"] == str(tensor.dtype).replace("torch.", ""), name

    def test_describe_reports_each_tensors_raw_hash(self, tmp_path):
        """`hash_raw` identifies what a snapshot holds without opening
        `manifest.json`, which on Windows is not safe while the writer may be
        renaming a new manifest over it (GPU-93, GPU-128).

        It hashes what a tensor holds, not how it was stored: an unchanged
        tensor keeps its hash when the step becomes a delta, and a tensor whose
        values moved gets a new one.
        """
        model, optimizer = self._trained()
        mgr = CheckpointManager(storage_root=str(tmp_path), async_save=False)
        first = mgr.save(step=1, model=model, optimizer=optimizer)
        with torch.no_grad():
            model.weight.add_(0.01)
        second = mgr.save(step=2, model=model, optimizer=optimizer)
        mgr.flush()

        before = {t["name"]: t["hash_raw"] for t in mgr.describe(first)["tensors"]}
        after = {t["name"]: t["hash_raw"] for t in mgr.describe(second)["tensors"]}

        assert all(len(value) == 32 for value in after.values())
        assert after["model/bias"] == before["model/bias"], (
            "an unchanged tensor's hash moved when it was stored as a delta"
        )
        assert after["model/weight"] != before["model/weight"]

    def test_a_cast_tensor_reports_the_dtype_a_load_would_give_back(self, tmp_path):
        """`dtype` is what comes back, `stored_dtype` is what is on disk.

        They differ exactly under `save_dtype`, and conflating them would have
        a caller size an fp32 buffer for what it reads as bf16.
        """
        model, _ = self._trained()
        mgr = CheckpointManager(
            storage_root=str(tmp_path), async_save=False, save_dtype="bf16"
        )
        snap = mgr.save(step=1, model=model)
        mgr.flush()

        weight = next(
            t for t in mgr.describe(snap)["tensors"] if t["name"] == "model/weight"
        )
        assert weight["stored_dtype"] == "bfloat16"
        assert weight["dtype"] == "float32"

    def test_the_shapes_survive_the_step_becoming_a_delta(self, tmp_path):
        """The case a real run is in from step two onwards.

        An unchanged tensor is stored `skipped` — no bytes at all — and still
        has to report its shape, or this API answers only the first checkpoint
        of a run.
        """
        model, optimizer = self._trained()
        mgr = CheckpointManager(storage_root=str(tmp_path), async_save=False)
        mgr.save(step=1, model=model, optimizer=optimizer)
        with torch.no_grad():
            model.weight.add_(0.01)
        snap = mgr.save(step=2, model=model, optimizer=optimizer)
        mgr.flush()

        described = mgr.describe(snap)
        assert described["is_delta"], "step 2 was written as a full snapshot"
        by_name = {t["name"]: t for t in described["tensors"]}
        for name, tensor in model.state_dict().items():
            assert tuple(by_name["model/" + name]["shape"]) == tuple(tensor.shape)

    def test_describe_state_dict_rebuilds_the_tree_out_of_the_metadata_alone(
        self, tmp_path
    ):
        """The whole point, end to end: the structure and every shape, from
        one small entry, with no tensor materialized and no pack read whole."""
        from moonclip import MoonclipManager, TensorStub, describe_state_dict

        model, _ = self._trained()
        mgr = CheckpointManager(storage_root=str(tmp_path), async_save=False)
        snap = mgr.save(step=1, model=model)
        mgr.flush()

        low = MoonclipManager(storage_root=str(tmp_path))
        raw = low.load_tensors(snap, ["model._metadata"])
        assert set(raw) == {"model._metadata"}, "more than the template was read"

        described = describe_state_dict(raw)["model"]
        for name, tensor in model.state_dict().items():
            stub = described[name]
            assert isinstance(stub, TensorStub)
            assert stub.shape == tuple(tensor.shape), name
            assert stub.name == "model/" + name, name

        # And the names are usable: having measured, fetch exactly one.
        wanted = described["weight"].name
        got = low.load_tensors(snap, [wanted])
        assert set(got) == {wanted}
        back = torch.frombuffer(bytes(got[wanted]), dtype=model.weight.dtype)
        assert torch.equal(back.reshape(model.weight.shape), model.weight)

    def test_describe_state_dict_keeps_what_was_never_a_tensor(self, tmp_path):
        """Scalars, flags and whatever structure the caller wrapped around its
        tensors come back untouched — they were in the template all along.

        This is what lets a caller read placements, layout tags and the like
        without loading anything: Moonclip does not model them, and does not
        need to."""
        from moonclip import MoonclipManager, TensorStub, describe_state_dict

        state = {
            "group": {
                "sharded": True,
                "placements": [{"kind": "shard", "dim": 0}],
                "local": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            }
        }
        mgr = CheckpointManager(storage_root=str(tmp_path), async_save=False)
        snap = mgr.save(step=1, extra=state)
        mgr.flush()

        low = MoonclipManager(storage_root=str(tmp_path))
        # One prefix per `extra` entry, so this is `extra/group._metadata`.
        raw = low.load_tensors(snap, ["extra/group._metadata"])
        node = describe_state_dict(raw)["extra/group"]

        assert node["sharded"] is True
        assert node["placements"] == [{"kind": "shard", "dim": 0}]
        assert isinstance(node["local"], TensorStub)
        assert node["local"].shape == (3, 4)
        assert node["local"].dtype == "float32"

    def test_it_is_reachable_from_the_package(self):
        import moonclip

        assert callable(moonclip.describe_state_dict)
        assert "describe_state_dict" in moonclip.__all__
        assert "TensorStub" in moonclip.__all__

    def test_load_tensors_refuses_a_name_that_is_not_there(self, tmp_path):
        model, _ = self._trained()
        mgr = CheckpointManager(storage_root=str(tmp_path), async_save=False)
        snap = mgr.save(step=1, model=model)
        mgr.flush()

        with pytest.raises(RuntimeError, match="model/nope"):
            mgr.load_tensors(snap, ["model/nope"])
