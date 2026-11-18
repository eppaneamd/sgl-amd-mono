# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65188 at 294644b81e (Apache-2.0):
# tests/models/qwen4_exp/amd_specialized_decode/test_stages.py
"""The Qwen3.8-Flash-Next wide decode's build choices on CPU: the widths that get
a build and the TP exchange's codec."""

import pytest

from sgl_amd_mono.qwen38_flash_next.decode import allreduce_codec, decode_widths


def test_decode_widths_take_every_capture_size():
    """Every CUDA graph capture size in 1-192 gets a build (SGLang pads a step
    to the smallest captured width that holds it), and a build of any width
    in that range is valid at TP2, 1-64 at TP1; a TP the kernels are not built
    for has no build."""
    from sgl_amd_mono.qwen38_flash_next.decode.wide.layer import LayerBuild

    widths = (1, 2, 3, 4, 8, 12, 16, 24, 32, 64, 128, 129, 136, 160, 184, 192)
    assert decode_widths([*widths, 193, 200, 256]) == widths
    for tp, cap in ((2, 192), (1, 64)):
        for rows in range(1, cap + 1):
            LayerBuild(kind="gdn", tokens=rows, tp=tp)
        for rows in (0, cap + 1):
            with pytest.raises(ValueError):
                LayerBuild(kind="gdn", tokens=rows, tp=tp)
    with pytest.raises(ValueError, match="TP 3"):
        LayerBuild(kind="gdn", tokens=1, tp=3)


@pytest.mark.parametrize(
    "tp, level, codec",
    [
        *(
            pytest.param(2, level, codec, id=f"{level}-{codec}")
            for level, codec in (
                ("INT4", "int4"),
                ("INT6", "bf16"),
                ("INT8", "bf16"),
                ("FP", "bf16"),
                ("NONE", "bf16"),
                (None, "bf16"),
            )
        ),
        pytest.param(1, "INT4", "bf16", id="tp1-INT4-bf16"),
    ],
)
def test_allreduce_codec_follows_quickreduce(monkeypatch, tp, level, codec):
    """The TP exchange quantizes as the deployment's QuickReduce setting asks
    (INT4 lines for INT4, SGLang's only 4-bit level), and is exact otherwise;
    at TP1 nothing is reduced, so it is exact whatever the setting."""
    if level is None:
        monkeypatch.delenv("ROCM_QUICK_REDUCE_QUANTIZATION", raising=False)
    else:
        monkeypatch.setenv("ROCM_QUICK_REDUCE_QUANTIZATION", level)
    assert allreduce_codec(tp) == codec
