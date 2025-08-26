# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    _valid_deep_gemm_shape, deep_gemm_moe_fp8)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, modular_triton_fused_moe)
from vllm.platforms import current_platform

from moe.utils import (make_test_weights, native_per_token_group_quant_fp8,
                       native_w8a8_block_matmul)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if current_platform.get_device_capability() < (9, 0):
    pytest.skip("FP8 Triton requires CUDA 9.0 or higher", allow_module_level=True)

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

# Test configurations
DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
# Deepseek-V3's intermediate size 18432, so N is 18432*2/8=4608 at TP8
# and its hidden size is 7168.
MNK_FACTORS = [
    (1, 128, 128),
    (1, 512, 512),
    (1, 128, 7168),
    (1, 1024, 7168),
    (1, 4608, 128),
    (1, 4608, 512),
    (1, 4608, 7168),
    (83, 128, 128),
    (83, 512, 512),
    (83, 1024, 7168),
    (83, 4608, 512),
    (83, 4608, 7168),
    (128, 128, 128),
    (128, 512, 512),
    (128, 1024, 7168),
    (128, 4608, 512),
    (128, 4608, 7168),
    (2048, 128, 128),
    (2048, 1024, 7168),
    (2048, 4608, 512),
    (2048, 4608, 7168),
    (8192, 128, 128),
    (8192, 512, 512),
    (8192, 128, 7168),
    (8192, 1024, 7168),
    (8192, 4608, 512),
    (8192, 4608, 7168),
]

BLOCK_SIZE = [[128, 128]]
E = [2, 8, 16]  # [128, 256]
TOP_KS = [1, 2, 6]
SEEDS = [0]


def prepare_data(M, N, K, E, topk, block_size, dtype, seed, device):
    if topk > E:
        pytest.skip(f"Skipping test; topk={topk} > E={E}")

    torch.set_default_device(device)
    torch.manual_seed(seed)

    os.environ["VLLM_FUSED_MOE_CHUNK_SIZE"] = "2048"

    a = torch.randn((M, K), dtype=dtype) / 10
    score = torch.randn((M, E), dtype=dtype)

    _, w1, w1_s, _, w2, w2_s = make_test_weights(
        E,
        N,
        K,
        dtype,
        torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=block_size,
    )

    topk_weights, topk_ids, _ = fused_topk(a, score.float(), topk, False)

    return {
        "a": a,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1_s": w1_s,
        "w2_s": w2_s,
    }


@torch.inference_mode()
def test_w8a8_block_fp8_fused_moe(
    M,
    N,
    K,
    E,
    topk,
    block_size,
    dtype,
    seed,
    a,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_s,
    w2_s,
    device,
):

    torch.set_default_device(device)
    # Set the context to avoid lots of warning spam.
    with set_current_vllm_config(vllm_config):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            out = fused_experts(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                use_fp8_w8a8=True,
                w1_scale=w1_s,
                w2_scale=w2_s,
                block_shape=block_size,
            )

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

    return max_cuda_time


def get_moe_params():
    return {
        "M": 128,  # batch size
        "N": 4608,  # for TP8(?)
        "K": 7168,  # hidden size
        "E": 32,
        "topk": 4,
        "block_size": [128, 128],
        "dtype": torch.bfloat16,
        "seed": 0,
    }


if __name__ == "__main__":
    device = torch.device("cuda")

    params = get_moe_params()

    inputs = prepare_data(**params, device=device)

    t = test_w8a8_block_fp8_fused_moe(**params, **inputs, device=device)
