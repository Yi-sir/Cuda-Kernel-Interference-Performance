import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from green_context.green_context import get_all_streams
from mla.test_flash_attn import (get_prefill_attn_params,
                                 test_varlen_with_paged_kv)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    device = torch.device("cuda:7")
    streams = get_all_streams(device)
    params = get_prefill_attn_params()

    attn_times = {}

    for stream, num_sm in streams:
        with torch.cuda.stream(stream):
            attn_time = test_varlen_with_paged_kv(**params, device=device)
            attn_times[num_sm] = attn_time

    for k, v in attn_times.items():
        logger.info(f"With SMs {k}, MLA cost {v:.3f}us")
