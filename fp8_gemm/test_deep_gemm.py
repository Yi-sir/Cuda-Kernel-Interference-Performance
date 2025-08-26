# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import itertools

import pytest
import torch
# from tests.kernels.quant_utils import (native_per_token_group_quant_fp8,
#                                        native_w8a8_block_matmul)
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_col_major_tma_aligned_tensor, per_token_group_quant_fp8,
    w8a8_block_fp8_matmul)
from vllm.platforms import current_platform
from vllm.utils import has_deep_gemm
from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8

if current_platform.get_device_capability() < (9, 0):
    pytest.skip("FP8 Triton requires CUDA 9.0 or higher", allow_module_level=True)

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

# Test configurations
DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
NUM_TOKENS = [7, 2050]
D = [512, 4096, 5120, 13824]
GROUP_SIZE = [64, 128, 512]
M = [1, 7, 8, 83, 84, 4096]
N = [128, 512, 7168, 7748, 13824]
K = [256, 3884, 4096, 13824, 16384]
# Deepseek-V3's intermediate size 18432, so N is 18432*2/8=4608 at TP8
# and its hidden size is 7168.
BLOCK_SIZE = [[128, 128]]
OUT_DTYPES = [torch.bfloat16]  # [torch.float32, torch.half, torch.bfloat16]
SEEDS = [0]

# Skip all tests if CUDA is not available


# cannot run
@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGemm kernels not available.")
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_matmul(M, N, K, block_size, out_dtype, seed):
    # only aligned sizes
    if M % 4 != 0 or K % 128 != 0 or N % 64 != 0:
        pytest.skip(f"Skipping test; invalid size {M}, {N}, {K}")

    # torch.set_default_device(torch.device("cuda:7"))

    torch.manual_seed(seed)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max

    A_fp8, As_fp8 = per_token_group_quant_fp8(A_fp32, block_size[1])
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_fp32)

    # Transpose earlier so that the testing will not trigger transposing kernels
    As_fp8 = get_col_major_tma_aligned_tensor(As_fp8)

    out = torch.zeros((M, N), device="cuda", dtype=out_dtype)

    assert As_fp8.shape == (
        M,
        (K + 127) // 128,
    ), f"{As_fp8.shape} != {(M, (K + 127) // 128)}"

    fp8_gemm_nt((A_fp8, As_fp8), (B_fp8, Bs_fp8), out)
