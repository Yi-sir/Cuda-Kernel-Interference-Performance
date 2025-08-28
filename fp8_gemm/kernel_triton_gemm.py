import logging
import math
import os
import random
import sys
import time

import torch
from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_block_fp8_matmul

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from kernel_base.kernel_base import KernelBase
from kernel_base.registry import register_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_kernel
class TritonGemm(KernelBase):

    _default_params = {
        "M": 4096,
        "N": 13824,
        "K": 16384,
        "block_size": [128, 128],
        "out_dtype": torch.bfloat16,
        "seed": 0,
    }

    _kernel_name = "triton_gemm"
    _key = "_w8a8_block_fp8_matmul"

    def __init__(self, device: torch.device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(M, N, K, block_size, out_dtype, seed, device):
            torch.set_default_device(device)
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

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype):
            return w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

        return fp8_matmul(*self.inputs)
