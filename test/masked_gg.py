import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch
from masked_group_gemm.kernel_mgg import MaskedGroupGemm
from green_ctx_test.const import G, T, us, CALC_H200

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

params_list = [
    # sglang DeepSeek moe-gemm0
    {
        "G": 32,
        "M": 1024,
        "N": 4096,
        "K": 7168,
        "expected_m": 1,
        "seed": 0,
    },
    # sglang DeepSeek-V3 moe-gemm1
    {
        "G": 32,
        "M": 1024,
        "N": 7168,
        "K": 2048,
        "expected_m": 1,
        "seed": 0,
    },
    # DeepGEMM test case
    {
        "G": 4,
        "M": 4096,
        "N": 7168,
        "K": 2048,
        "expected_m": 256,
        "seed": 0
    }
]

if __name__ == "__main__":
    device = torch.device("cuda:0")

    mgg = MaskedGroupGemm(device)

    for param in params_list:
        mgg.set_params(param)
        t = mgg.profile_kernel_us()

        t_s = t / us

        mask_m = mgg.inputs[3]
        valid_m = sum(mask_m).item()

        FLOPs = 2 * param["N"] * param["K"] * valid_m
        actual_FLOPs = FLOPs / t_s

        A_bytes = valid_m * param["K"]
        B_bytes = param["G"] * param["N"] * param["K"]
        D_bytes = 2 * valid_m * param["N"]

        bindwidth = (A_bytes + B_bytes + D_bytes) / G / t_s

        logger.info(f"params {param} | {t:.2f}us | {(actual_FLOPs / T):.2f} TFLOPS | {bindwidth:.2f} GB/s")
