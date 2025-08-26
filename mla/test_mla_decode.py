# Adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import math
import random
import time

import torch
from vllm.attention.ops.flashmla import (flash_mla_with_kvcache,
                                         get_mla_metadata,
                                         is_flashmla_supported)
from vllm.triton_utils import triton

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    assert cos_diff < 1e-5


FLASH_MLA_UNSUPPORTED_REASON = (
    is_flashmla_supported()[1]
    if not is_flashmla_supported()[0]
    else "FlashMLA is supported"
)


def flash_mla(
    q,
    blocked_k,
    block_table,
    cache_seqlens,
    dv,
    tile_scheduler_metadata,
    num_splits,
    causal,
):
    return flash_mla_with_kvcache(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        dv,
        tile_scheduler_metadata,
        num_splits,
        causal=causal,
    )


def test_flash_mla(
    b, s_q, mean_sk, h_q, h_kv, d, dv, block_size, causal, varlen, device
):
    # TODO: parametrize using pytest
    dtype = torch.bfloat16
    # dtype = torch.float8_e4m3fn
    # device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

    q = torch.randn(b, s_q, h_q, d)
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item() :] = (
            float("nan")
        )
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv
    )

    return (
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        dv,
        tile_scheduler_metadata,
        num_splits,
        causal,
    )


def get_mla_params():
    params = {
        "b": 128,
        "s_q": 1,
        "mean_sk": 1024,
        "h_q": 128,
        "h_kv": 1,
        "d": 576,
        "dv": 512,
        "block_size": 64,
        "causal": True,
        "varlen": False,
    }
    return params


if __name__ == "__main__":
    params = get_mla_params()
    device = torch.device("cuda:7")

    p = test_flash_mla(**params, device=device)

    # profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        out_flash, lse_flash = flash_mla(*p)

    events = prof.key_averages()
    max_cuda_time = max(
        (
            event.device_time
            for event in events
            if event.device_type != torch.autograd.DeviceType.CPU
        ),
        default=0.0,
    )

    if __name__ == "__main__":
        for evt in events:
            logger.info(
                f"event.device_type: {evt.device_type}, device_time: {evt.device_time}"
            )
        print(
            events.table(
                sort_by="cuda_time_total",
                row_limit=10,
            )
        )
        print(f"max_cuda_time is {max_cuda_time}")
