"""Qwen3.8-Flash-Next (the MXFP4 checkpoint) with --monokernel-backend amd on two
MI355X, TP2 decode of 1 to 192 rows: the server runs at most 192 requests, so every
decode step is one the wide launches take. The startup line shows every captured
decode graph runs them. GSM8K and MMMU against the normal path on the same server
arguments, with requests joining and leaving batches of up to 192 rows; the tokens of covered steps scored again by the normal prefill path (KL
consistency), the second turn of a conversation decoding from the GDN states the
radix cache saved during the first; each layer's launches against SGLang's own layer
under SGLANG_AMD_MONO_CHECK, at 1 to 192 rows and at contexts past QSA's 2,048-token
selection budget (about 6,000 and 108,000 tokens); and a rank stalled by
SGLANG_AMD_MONO_FAULT_INJECT stopping the server without tokens."""

import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from qwen38_flash_next_decode_server import ARGS, ENV, MODEL
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper as assert_decode_hits_match,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.run_eval import run_eval
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_amd_mono import BACKEND
from sgl_amd_mono.qwen38_flash_next.spec import SPEC

# a cold FlyDSL cache compiles every width's launches at startup
LAUNCH_TIMEOUT = 3600
# The decode graphs: the spec's widths (both paths') up to 192, the largest capture
# size; SGLang's default sizes up to 192 are among them
WIDTHS = sorted(w for w in SPEC.devices["gfx950"].widths if w <= 192)
# Score drops against the normal path that fail the run: bounds for gross failures
GSM8K_DROP = 0.03
MMMU_DROP = 0.05
# Decode against prefill, one batch of 1 to 192 LongBench-v2 prompts at a time (every
# decode step covered), mean KL a batch: 0.022 to 0.085 on the normal path (3 runs,
# one server), 0.026 to 0.097 with the monokernels (6 runs, 4 servers)
KL_BOUND = 0.12
# 200 requests at a time, more than the server runs: its batches fill to the
# largest width, 192 rows
THREADS = 200
# Check mode: batches of (rows, paragraphs of 12 tokens): one a width from 1 row to
# the largest at about 55 tokens, then contexts QSA's indexer selects from (past its
# budget of 2,048 tokens): about 6,000 tokens at 1, 16 and 64 rows, and about 108,000
# at 1 and 4
CHECK_BATCHES = (
    *((rows, 4) for rows in (1, 4, 16, 64, 128, 192)),
    *((rows, 500) for rows in (1, 16, 64)),
    *((rows, 9000) for rows in (1, 4)),
)
PARAGRAPH = "The river valley archive lists the flood height of every year. "
CHECK_METRICS = (
    "gdn.streams.cosine",
    "gdn.delta.cosine",
    "gdn.conv.cosine",
    "gdn.state.cosine",
    "gdn.update.cosine",
    "qsa.streams.cosine",
    "qsa.delta.cosine",
    "qsa.mix.cosine",
    "qsa.qkv.cosine",
    "qsa.ix.cosine",
)
# The fault test: a stopped launch faults at its wait kind's page (a grid barrier,
# a TP exchange line, a tagged hand-off between CTAs), or at the flag's for a peer's
# flag
WAIT_PAGES = {0x1000, 0x2000, 0x3000}
FLAG_PAGE = 0x4000


def _launch(mono=True, extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    mono_args = ["--monokernel-backend", BACKEND] if mono else []
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=ARGS + mono_args + list(extra_args),
        env=env,
        return_stdout_stderr=log,
    )


def _logs(files) -> str:
    return "".join(Path(f.name).read_text() for f in files)


def _scores() -> dict:
    gsm8k = run_sgl_eval(
        SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            model=MODEL,
            eval_name="gsm8k",
            num_examples=400,
            num_threads=THREADS,
            temperature=0.0,
            seed=42,
        )
    )
    mmmu = run_eval(
        SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            model=MODEL,
            eval_name="mmmu",
            num_examples=100,
            num_threads=THREADS,
            max_tokens=30,
        )
    )
    return {"gsm8k": gsm8k["score"], "mmmu": mmmu["score"]}


class TestMonokernelAccuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores = {}
        for mono in (False, True):
            process = _launch(mono=mono)
            try:
                cls.scores[mono] = _scores()
            finally:
                terminate_and_kill_process_tree(process)
        print(f"normal path {cls.scores[False]}, monokernels {cls.scores[True]}")

    def test_gsm8k_against_the_normal_path(self):
        drop = self.scores[False]["gsm8k"] - self.scores[True]["gsm8k"]
        self.assertLess(drop, GSM8K_DROP, self.scores)

    def test_mmmu_against_the_normal_path(self):
        drop = self.scores[False]["mmmu"] - self.scores[True]["mmmu"]
        self.assertLess(drop, MMMU_DROP, self.scores)


class TestMonokernelDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(extra_args=["--watchdog-timeout", "600"], log=cls.logs)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in getattr(cls, "logs", ()):
            log.close()

    def test_a_every_decode_graph_runs_the_monokernels(self):
        # routing is fixed at startup; without this, the other tests could pass on
        # the normal path alone
        self.assertIn(
            "monokernel qwen38_flash_next_hip on gfx950 runs decode steps of batch "
            f"size {WIDTHS} ({len(WIDTHS)} of {len(WIDTHS)} captured decode graphs)",
            _logs(self.logs),
        )

    def test_b_covered_steps_match_prefill(self):
        # one graph per batch (ignore_eos keeps its rows); 3, 13, 17 and 129 rows one
        # past a graph's, so their steps carry pad rows; the helper flushes the
        # cache first, so a
        # graph replayed first on another's reads none of this batch's pages. Up to
        # 48 rows the batches share one LongBench-v2 download, 129 and 192 one each.
        for rows in (1, 2, 3, 12, 13, 16, 17, 48, 129, 192):
            with self.subTest(rows=rows):
                assert_logprobs_match(
                    DEFAULT_URL_FOR_TEST,
                    {MODEL: {"kl_div": KL_BOUND}},
                    MODEL,
                    max_samples=rows,
                    max_new_tokens=512,
                )

    def test_c_a_second_turn_decodes_from_the_saved_gdn_states(self):
        # the second turn hits the first turn's decoded tokens in the radix cache:
        # its GDN states are the ones the launches' steps saved through SGLang's
        # track-save; a state saved before its layer's launch, or none, fails the KL
        assert_decode_hits_match(
            DEFAULT_URL_FOR_TEST,
            {MODEL: {"kl_div": KL_BOUND}},
            MODEL,
            max_samples=8,
            max_new_tokens=512,
        )


class TestMonokernelCheck(CustomTestCase):
    def test_each_covered_layer_within_its_numerics(self):
        # SGLANG_AMD_MONO_CHECK (eager) runs SGLang's layer next to each layer's
        # launches; with the MXFP4 reference the GDN projections are the launches'
        # arithmetic, so the bounds hold the kernels' own difference (streams and
        # delta: with the two MoE implementations'). One batch at a time
        # (ignore_eos: a request that stops at its first token never decodes); the
        # radix cache on, so the GDN state saves are compared where a step tracks
        # one (gdn.track)
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={
                    "SGLANG_AMD_MONO_CHECK": "1",
                    "SGLANG_AMD_MONO_CHECK_REF": "mxfp4",
                },
                extra_args=["--disable-cuda-graph"],
                log=(out, err),
            )
            try:
                for rows, paragraphs in CHECK_BATCHES:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": [
                                f"Record {i}. " + PARAGRAPH * paragraphs
                                for i in range(rows)
                            ],
                            "sampling_params": {
                                "max_new_tokens": 16,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=1800,
                    )
                    r.raise_for_status()
            finally:
                terminate_and_kill_process_tree(process)
            log = _logs((out, err))
        steps = [
            (int(rows), int(first), int(last), fields)
            for rows, first, last, fields in re.findall(
                r"monokernel check step \((\d+) rows, positions (\d+)\.\.(\d+)\): (.*)",
                log,
            )
        ]
        values = {}
        for rows, first, _, fields in steps:
            context = "long" if first >= 2048 else "short"
            for name, v in re.findall(r"(\S+)=(\S+)@\d+", fields):
                values.setdefault(name, []).append(float(v))
                values.setdefault(f"{name} ({context})", []).append(float(v))
        for name, vs in sorted(values.items()):
            print(f"{name}: min {min(vs):.6g} max {max(vs):.6g} ({len(vs)} steps)")
        # each batch's steps ran at its rows and its context: a prompt's decode
        # positions start past its paragraphs, by the record number's few tokens
        for rows, paragraphs in CHECK_BATCHES:
            low, high = 12 * paragraphs, 12 * paragraphs + 64
            self.assertTrue(
                any(
                    r == rows and low <= first <= last <= high
                    for r, first, last, _ in steps
                ),
                (rows, paragraphs, sorted({(r, f) for r, f, _, _ in steps})),
            )
        for metric in CHECK_METRICS:
            self.assertIn(metric, values)
        violations = re.findall(r"monokernel check violation: .*", log)
        self.assertEqual(violations, [], "\n".join(violations[:20]))


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # once the step epoch reaches 1000 (startup's graph warmups run a few steps
        # of each width), rank 1 stalls at the first wait of its launches; at the
        # deadline (watchdog / 2) it raises its flag and both ranks stop: rank 1 at
        # its wait kind's page, rank 0 at the flag's
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={
                    "SGLANG_AMD_MONO_FAULT_INJECT": "1:1:1000",
                    # the stop faults every GPU on purpose: no crash dumps, which
                    # would hold the ranks up for minutes
                    "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
                    "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "0",
                    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
                },
                extra_args=["--watchdog-timeout", "20"],
                log=(out, err),
            )
            try:
                try:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": "Write a long essay about lighthouses.",
                            "sampling_params": {
                                "max_new_tokens": 2048,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=600,
                    )
                    ok = r.status_code == 200 and r.json().get("text")
                except requests.RequestException:
                    ok = False
                self.assertFalse(ok, "a failed step's tokens reached the client")
                deadline = time.time() + 300
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                log = _logs((out, err))
                # a launch stops with a store to its wait kind's page (CLR's
                # "faulting addr: 0x...", ROCr's "on address 0x..."): the only
                # faults are there, rank 0's at the flag's page
                faults = {
                    0 if a == "(nil)" else int(a, 16)
                    for a in re.findall(
                        r"(?:faulting addr: |on address )(0x[0-9a-fA-F]+|\(nil\))",
                        log,
                    )
                }
                self.assertTrue(faults, log[-2000:])
                self.assertLessEqual(faults, WAIT_PAGES | {FLAG_PAGE}, faults)
                self.assertIn(FLAG_PAGE, faults)
                self.assertTrue(faults & WAIT_PAGES, faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
