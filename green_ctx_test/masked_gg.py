import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from green_context.green_context import get_all_streams
from masked_group_gemm.kernel_mgg import MaskedGroupGemm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    streams = get_all_streams(device)

    mgg = MaskedGroupGemm(device)
    mgg.prepare_input()

    mgg_times = {}

    for stream, num_sm in streams:
        with torch.cuda.stream(stream):
            t = mgg.profile_kernel_us(stream)
            mgg_times[num_sm] = t

    for k, v in mgg_times.items():
        logger.info(f"With SMs {k}, MGG cost {v:.3f}us")