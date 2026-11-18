# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next decode steps of 1 to 192 rows at TP2 on the wide kernels of
vllm-project/vllm#65188 (``wide``): one persistent launch a GDN layer, and two a QSA
layer around SGLang's QSA attention. The TP2 path of
``qwen38_flash_next.spec.SPEC``: its refusals, build and forward in ``spec``, the
launch state in ``launcher``, the layers' operands in ``weights`` and check mode
in ``check`` (README.md)."""

import os

from sgl_amd_mono.qwen38_flash_next.decode.wide import layout as L


def decode_widths(capture_sizes) -> tuple[int, ...]:
    """The widths that get a build: the decode CUDA graph capture sizes from
    ``MIN_TOKENS`` to ``MAX_TOKENS`` (SGLang pads a step to the smallest captured
    width that holds it)."""
    return tuple(
        sorted(s for s in set(capture_sizes) if L.MIN_TOKENS <= s <= L.MAX_TOKENS)
    )


def allreduce_codec(tp: int) -> str:
    """The TP exchange's line codec: exact bf16 lines, or QuickReduce's INT4 codec
    when ``ROCM_QUICK_REDUCE_QUANTIZATION`` is INT4. SGLang's own all-reduce of a
    decode step stays exact at these sizes (below QuickReduce's minimum), so INT4
    is lossy here where SGLang is not. At TP1 the exchange is a round trip through
    the rank's own mailbox and SGLang runs no all-reduce: bf16 whatever the
    setting."""
    level = os.environ.get("ROCM_QUICK_REDUCE_QUANTIZATION", "NONE")
    return "int4" if tp > 1 and level == "INT4" else "bf16"
