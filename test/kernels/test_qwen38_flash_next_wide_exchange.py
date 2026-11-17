# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8-Flash-Next wide decode's TP exchange between two processes on two
gfx950 GPUs, through ``common.peer.PeerBuffer`` (``qwen38_flash_next/check_wide_exchange.py``):
both codecs, every decode width SGLang captures up to 192 rows and 129, 1,000
launches a width, in vLLM's build setting and SGLang's (stop mode, ``grid``). Every
launch's new state equals the host reference bit for bit on both ranks. Three
controls break one launch on rank 1 in vLLM's setting and check that the matching
check fires: a lost publish (rank 0's wait gives up at ``abi.WAIT_TICKS`` and the
launch ends), a corrupt input (the reference check) and different injection scales
(the cross-rank digest). In SGLang's setting the fault injection stalls one rank's
wait: its deadline faults it, and its flag faults the other rank long before that
rank's own deadline.

The harness spawns and waits for each rank itself, so a rank that never stops is
seen and killed at the hang limit instead of hanging the test."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

WORKER = Path(__file__).with_name("qwen38_flash_next") / "check_wide_exchange.py"
# SGLang's decode CUDA graph capture sizes up to 192 rows without speculative
# decoding (generate_decode_cuda_graph_batch_sizes), and 129: a partial line
WIDTHS = sorted({1, 2, 4, 8, 12, *range(16, 193, 8), 129})
ITERS = 1000
CONTROL_ROWS, CONTROL_ITER = 129, 10
WAIT_S = 10  # abi.WAIT_SECONDS
SITE_PEER = 2  # abi.SITE_PEER
# the fault test: the stalled rank's deadline (SGLang's least: half of the 10 s
# `--watchdog-timeout` it accepts) and the other rank's (SGLang's default), and
# the worker kernel's first wait (the attention lines in `hc.combine`) stalled
# once the epoch reaches 2: launch LAYERS, after a whole step
FAULT_WAIT_S, OTHER_WAIT_S = 5.0, 60.0
FAULT_ROWS, FAULT_SITE, FAULT_AFTER, LAYERS = 129, 1, 2, 48
# a stop's fault, at the null page (`guard.stop_launch`), as ROCr reports it
# ("... on address %p%s. Reason: %s.", glibc's %p of 0 is "(nil)")
NULL_FAULT = re.compile(
    r"Memory access fault by GPU .* on address (\(nil\)|0x0+)(?![0-9a-f])"
)


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


pytestmark = pytest.mark.skipif(_gfx950_gpus() < 2, reason="needs 2 gfx950 GPUs")


def _start(tmp_path, args, rank_args=((), ()), env=None):
    """Start both ranks (`rank_args`: each rank's own arguments)."""
    procs = []
    for r in range(2):
        renv = {**os.environ, **(env or {})}
        renv["FLYDSL_RUNTIME_CACHE_DIR"] = str(tmp_path / f"flydsl{r}")
        with open(tmp_path / f"rank{r}.log", "w") as log:
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(WORKER), "--rank", str(r),
                     "--init-file", str(tmp_path / "init"),
                     "--out", str(tmp_path / f"rank{r}.json"), *args, *rank_args[r]],
                    env=renv, stdout=log, stderr=subprocess.STDOUT,
                )
            )  # fmt: skip
    return procs


def _wait(procs, hang_s):
    """Each rank's exit time (monotonic s, to 50 ms); a rank still running at
    `hang_s` is killed and fails the test."""
    t0, ends = time.monotonic(), [None] * len(procs)
    while None in ends and time.monotonic() - t0 < hang_s:
        for r, p in enumerate(procs):
            if ends[r] is None and p.poll() is not None:
                ends[r] = time.monotonic()
        time.sleep(0.05)
    hung = [r for r, e in enumerate(ends) if e is None]
    for r in hung:
        procs[r].kill()
        procs[r].wait()
    assert not hung, f"rank(s) {hung} still running after {hang_s} s"
    return ends


def _spawn(tmp_path, args, hang_s):
    """Run both ranks; return each rank's results."""
    procs = _start(tmp_path, args)
    _wait(procs, hang_s)
    for r, p in enumerate(procs):
        log = (tmp_path / f"rank{r}.log").read_text()[-3000:]
        assert p.returncode == 0, f"rank {r} exited {p.returncode}:\n{log}"
    return [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(2)]


def _control(tmp_path, control, codec="bf16"):
    out = _spawn(
        tmp_path,
        ["--codecs", codec, "--widths", str(CONTROL_ROWS),
         "--iters", str(CONTROL_ITER + 1), "--control", control,
         "--control-iter", str(CONTROL_ITER)],
        hang_s=1200,
    )  # fmt: skip
    return [o["records"][0] for o in out]


@pytest.mark.parametrize("setting", ["vllm", "sglang"])
def test_exchange_every_width_bit_equal(tmp_path, setting):
    out = _spawn(
        tmp_path,
        ["--widths", ",".join(map(str, WIDTHS)), "--iters", str(ITERS),
         "--setting", setting],
        hang_s=3600,
    )  # fmt: skip
    for o in out:
        assert [(r["codec"], r["rows"]) for r in o["records"]] == [
            (c, w) for c in ("bf16", "int4") for w in WIDTHS
        ]
        bad = [r for r in o["records"] if not r["pass"] or r["iters"] != ITERS]
        assert not bad, f"rank {o['rank']}: {json.dumps(bad, indent=1)}"


def test_lost_publish_gives_up_at_the_deadline(tmp_path):
    """Rank 1 publishes its lines to its own mailbox only: rank 0's waits on them
    give up after WAIT_SECONDS (its grid barrier's waits may too), record the peer
    site, raise its alarm and end the launch, whose state then differs from the
    reference."""
    r0, _ = _control(tmp_path, "lost_publish")
    assert r0["failed"] == 1
    assert str(SITE_PEER) in r0["failed_record"]["sites"]
    assert r0["alarm_words"][0] == 1
    assert r0["launch_ms_control"] >= 0.95 * WAIT_S * 1000
    assert r0["mismatch_at_control"] > 0


@pytest.mark.parametrize("codec", ["bf16", "int4"])
def test_corrupt_input_fails_the_reference(tmp_path, codec):
    """Rank 1's attention partial is not the reference's: both ranks combine the
    same lines, so their states agree with each other and not with the reference."""
    for r in _control(tmp_path, "corrupt_input", codec):
        assert r["failed"] == 0
        assert r["mismatch_iters"] == 1 and r["mismatch_at_control"] > 0
        assert r["digest_differ_iters"] == 0


def test_rank_mismatch_fails_the_digest(tmp_path):
    """Rank 1's injection scales differ from rank 0's: the ranks' states differ, which
    the cross-rank digest sees; rank 0 still matches the reference."""
    r0, r1 = _control(tmp_path, "rank_mismatch")
    for r in (r0, r1):
        assert r["failed"] == 0
        assert r["digest_differ_iters"] == 1 and r["digest_differs_at_control"]
    assert r0["mismatch_iters"] == 0
    assert r1["mismatch_iters"] == 1 and r1["mismatch_at_control"] > 0


@pytest.mark.parametrize("stalled", [1, 0])
def test_a_stalled_rank_faults_both_ranks_by_its_deadline(tmp_path, stalled):
    """SGLang's setting, rank `stalled`'s first wait of launch LAYERS stalled
    (SGLANG_AMD_MONO_FAULT_INJECT): after the first step on both ranks, its
    FAULT_WAIT_S deadline faults it at the null page, and the flag it raised in
    the other rank's peer buffer faults the other rank (waiting on its MLP
    lines) at the null page too, long before that rank's own OTHER_WAIT_S
    deadline."""
    other = 1 - stalled
    waits = [OTHER_WAIT_S, OTHER_WAIT_S]
    waits[stalled] = FAULT_WAIT_S
    procs = _start(
        tmp_path,
        ["--setting", "sglang", "--codecs", "bf16", "--widths", str(FAULT_ROWS),
         "--iters", str(2 * LAYERS)],
        rank_args=[("--wait-s", str(w), "--progress", str(tmp_path / f"progress{r}"))
                   for r, w in enumerate(waits)],
        env={"SGLANG_AMD_MONO_FAULT_INJECT": f"{stalled}:{FAULT_SITE}:{FAULT_AFTER}"},
    )  # fmt: skip
    ends = _wait(procs, hang_s=600)
    last = []
    for r, p in enumerate(procs):
        log = (tmp_path / f"rank{r}.log").read_text()
        assert p.returncode != 0 and NULL_FAULT.search(log), (
            f"rank {r} exited {p.returncode}:\n{log[-3000:]}"
        )
        # every launch of the first step returned, and none after it
        rows = (tmp_path / f"progress{r}").read_text().split()
        assert [int(i) for i in rows[::2]] == list(range(LAYERS)), rows[::2]
        last.append(float(rows[-1]))
    # the stalled rank: its deadline, from its stalled launch's start, then the
    # runtime's teardown
    took = ends[stalled] - last[stalled]
    assert 0.95 * FAULT_WAIT_S <= took <= FAULT_WAIT_S + 15, took
    # the other rank: the stalled rank's flag, not its own deadline
    assert ends[other] - last[other] < OTHER_WAIT_S / 2, ends[other] - last[other]
    assert abs(ends[other] - ends[stalled]) < 15, ends
