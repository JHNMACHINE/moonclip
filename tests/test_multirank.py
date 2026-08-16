"""Multi-rank save and load, actually run.

The distributed path was covered only by ``manual_torchrun.py``, which pytest
never collects because of its name - so the headline distributed feature was
verified whenever someone remembered to type the command. ``test_distributed.py``
looks like it fills the gap but only checks that ``RANK`` and ``WORLD_SIZE`` are
*parsed*; nothing there ever writes a shard.

This runs the real thing: several processes, a gloo process group, and the
``create_snapshot`` / ``save_rank`` / ``finalize_snapshot`` flow underneath
``CheckpointManager.save()``.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="the PyTorch adapter needs torch")

WORKER = Path(__file__).parent / "multirank_worker.py"
TIMEOUT = 300


def free_port() -> int:
    """Ask the OS for a port, so parallel test runs do not collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_ranks(world_size: int, checkpoint_dir: Path):
    """Launch one process per rank with the environment torchrun would set."""
    base = {
        **os.environ,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(free_port()),
        "WORLD_SIZE": str(world_size),
        # gloo on a single host: keep it off any real network interface.
        "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "lo"),
    }
    if sys.platform == "win32":
        base.pop("GLOO_SOCKET_IFNAME", None)

    processes = []
    for rank in range(world_size):
        environment = {**base, "RANK": str(rank), "LOCAL_RANK": str(rank)}
        processes.append(
            subprocess.Popen(
                [sys.executable, str(WORKER), "--dir", str(checkpoint_dir)],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    results = []
    for rank, process in enumerate(processes):
        try:
            output, _ = process.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            results.append((rank, -1, f"timed out after {TIMEOUT}s\n{output}"))
            continue
        results.append((rank, process.returncode, output))
    return results


@pytest.mark.parametrize("world_size", [2, 4])
def test_every_rank_saves_and_reloads_its_shard(tmp_path, world_size):
    checkpoint_dir = tmp_path / "checkpoints"
    results = run_ranks(world_size, checkpoint_dir)

    for rank, code, output in results:
        assert code == 0, f"rank {rank} failed (exit {code}):\n{output}"
        assert f"rank {rank}/{world_size} ok" in output, output

    # Rank 0 finalises, so the snapshots are visible from a plain reader too.
    assert checkpoint_dir.is_dir()
    assert any(checkpoint_dir.iterdir()), "nothing was written"


def test_a_single_rank_run_still_uses_the_simple_path(tmp_path):
    """world_size=1 must not need the multi-rank flow.

    Guards the boundary the two paths meet at: the manager auto-detects the
    topology, and a run launched with one rank has to behave like a plain
    single-process save.
    """
    results = run_ranks(1, tmp_path / "checkpoints")
    rank, code, output = results[0]
    assert code == 0, output
    assert "rank 0/1 ok" in output, output
