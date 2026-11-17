# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8-Flash-Next wide decode's TP exchange between two processes on two
gfx950 GPUs, through ``common.peer.PeerBuffer`` (``qwen38_flash_next/check_wide_exchange.py``):
both codecs, every decode width SGLang captures up to 192 rows and 129, 1,000
launches a width. Every launch's new state equals the host reference bit for bit
on both ranks. Three controls break one launch on rank 1 and check that the matching
check fires: a lost publish (rank 0's wait gives up at ``abi.WAIT_TICKS`` and the
launch ends), a corrupt input (the reference check) and different injection scales
(the cross-rank digest).

The harness spawns and waits for each rank itself, so a rank that never stops is
seen and killed at the hang limit instead of hanging the test."""

import json
import os
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


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


pytestmark = pytest.mark.skipif(_gfx950_gpus() < 2, reason="needs 2 gfx950 GPUs")


def _spawn(tmp_path, args, hang_s):
    """Run both ranks; return each rank's results. A rank still running at
    `hang_s` is killed and fails the test."""
    procs = []
    for r in range(2):
        env = {**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / f"flydsl{r}")}
        with open(tmp_path / f"rank{r}.log", "w") as log:
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(WORKER), "--rank", str(r),
                     "--init-file", str(tmp_path / "init"),
                     "--out", str(tmp_path / f"rank{r}.json"), *args],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
            )  # fmt: skip
    t0, hung = time.monotonic(), []
    for r, p in enumerate(procs):
        try:
            p.wait(timeout=max(1.0, hang_s - (time.monotonic() - t0)))
        except subprocess.TimeoutExpired:
            hung.append(r)
            p.kill()
            p.wait()
    assert not hung, f"rank(s) {hung} still running after {hang_s} s"
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


def test_exchange_every_width_bit_equal(tmp_path):
    out = _spawn(
        tmp_path,
        ["--widths", ",".join(map(str, WIDTHS)), "--iters", str(ITERS)],
        hang_s=3600,
    )
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
