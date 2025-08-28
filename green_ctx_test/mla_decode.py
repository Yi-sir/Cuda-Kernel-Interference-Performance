import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch
from test_utils import profile_a_kernel

from green_context.green_context import get_all_streams
from mla.kernel_mla_decode import MLADecode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    device = torch.device("cuda:0")
    streams = get_all_streams(device)

    mla_decode = MLADecode(device)

    mla_times = {}

    mla_decode.prepare_input()

    for stream, num_sm in streams:
        with torch.cuda.stream(stream):
            mla_time = mla_decode.profile_kernel_us(stream)
            mla_times[num_sm] = mla_time

    for k, v in mla_times.items():
        logger.info(f"With SMs {k}, MLA cost {v:.3f}us")
