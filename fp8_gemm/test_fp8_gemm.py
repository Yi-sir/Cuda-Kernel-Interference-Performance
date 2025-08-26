# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import itertools

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_col_major_tma_aligned_tensor, per_token_group_quant_fp8,
    w8a8_block_fp8_matmul)
from vllm.platforms import current_platform
from vllm.utils import has_deep_gemm
from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype):
    return w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)


def test_w8a8_block_fp8_matmul(M, N, K, block_size, out_dtype, seed, device):
    # torch.set_default_device(device)
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

    A_fp8 = A_fp8.to(device)
    B_fp8 = B_fp8.to(device)
    As = As.to(device)
    Bs = Bs.to(device)

    return (A_fp8, B_fp8, As, Bs, block_size, out_dtype)


def get_gemm_params():
    params = {
        "M": 4096,
        "N": 13824,
        "K": 16384,
        "block_size": [128, 128],
        "out_dtype": torch.bfloat16,
        "seed": 0,
    }
    return params


if __name__ == "__main__":
    params = get_gemm_params()
    device = torch.device("cuda:7")
    gemm_input = test_w8a8_block_fp8_matmul(**params, device=device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        _ = fp8_matmul(*gemm_input)

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
