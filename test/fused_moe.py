import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from green_context.green_context import get_all_streams
from moe.kernel_fused_moe import FusedMoE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    device = torch.device("cuda:0")
    streams = get_all_streams(device)

    fused_moe = FusedMoE(device)

    fused_moe.prepare_input()

    all_times = {}

    for stream, num_sm in streams:
        with torch.cuda.stream(stream):
            t = fused_moe.profile_kernel_us(stream)
            all_times[num_sm] = t

    for k, v in all_times.items():
        logger.info(f"With SMs {k}, MoE cost {v:.3f}us")
