import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch
from test_utils import profile_a_kernel

from fp8_gemm.test_fp8_gemm import (fp8_matmul, get_gemm_params,
                                    test_w8a8_block_fp8_matmul)
from green_context.green_context import get_all_streams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    device = torch.device("cuda:0")

    streams = get_all_streams(device)

    params = get_gemm_params()

    all_times = {}

    matmul_input = test_w8a8_block_fp8_matmul(**params, device=device)

    for stream, num_sm in streams:
        t = profile_a_kernel(fp8_matmul, matmul_input, stream)
        all_times[num_sm] = t

    for k, v in all_times.items():
        logger.info(f"With SMs {k}, MLA cost {v:.3f}us")
