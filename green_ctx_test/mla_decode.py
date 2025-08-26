import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch
from test_utils import profile_a_kernel

from green_context.green_context import get_all_streams
from mla.test_mla_decode import flash_mla, get_mla_params, test_flash_mla

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    device = torch.device("cuda:7")

    streams = get_all_streams(device)

    mla_params = get_mla_params()

    mla_times = {}

    mla_input = test_flash_mla(**mla_params, device=device)

    for stream, num_sm in streams:
        with torch.cuda.stream(stream):
            mla_time = profile_a_kernel(flash_mla, mla_input, stream)
            mla_times[num_sm] = mla_time

    for k, v in mla_times.items():
        logger.info(f"With SMs {k}, MLA cost {v:.3f}us")
