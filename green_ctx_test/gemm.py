import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch
from test_utils import profile_a_kernel

from fp8_gemm.kernel_triton_gemm import TritonGemm
from green_context.green_context import get_all_streams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    device = torch.device("cuda:0")
    streams = get_all_streams(device)

    triton_gemm = TritonGemm(device)

    all_times = {}

    triton_gemm.prepare_input()

    for stream, num_sm in streams:
        t = triton_gemm.profile_kernel_us(stream)
        all_times[num_sm] = t

    for k, v in all_times.items():
        logger.info(f"With SMs {k}, MLA cost {v:.3f}us")
