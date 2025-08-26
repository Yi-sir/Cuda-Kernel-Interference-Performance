import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from green_context.green_context import get_all_streams
from moe.test_block_fp8 import (get_moe_params, prepare_data,
                                test_w8a8_block_fp8_fused_moe)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    device = torch.device("cuda:0")

    streams = get_all_streams(device)

    moe_params = get_moe_params()

    all_times = {}

    inputs = prepare_data(**moe_params, device=device)

    for stream, num_sm in streams:
        with torch.cuda.stream(stream):
            t = test_w8a8_block_fp8_fused_moe(**moe_params, **inputs, device=device)
            all_times[num_sm] = t

    for k, v in all_times.items():
        logger.info(f"With SMs {k}, MoE cost {v:.3f}us")
