import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from mha.kernel_triton_mha import TritonMHAPrefill
from test_utils import generate_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

all_params = {
    "batch_size": [8],
    "prompt_len": [1024],
    "cached_len": [256],
    "q_head_num": [128],      # ds-v3.1-terminus
    "kv_head_num": [1, 128],  # ds-v3.1-terminus, 1 for mla, 128 for mha
    "qk_head_dim": [576],     # ds-v3.1-terminus
    "v_head_dim": [512],      # ds-v3.1-terminus
    "page_size": [1],
    "max_num_pages": [128],
    "tp": [1, 8]
}

if __name__ == "__main__":

    device = torch.device("cuda:2")

    k = TritonMHAPrefill(device)

    params_list = generate_params(all_params)

    for param in params_list:

        k.set_params(param)
        t = k.profile_kernel_us()

        t_ms = t / 1000

        logger.info(f"Triton MHA Prefill with {param} costs {t:.3f}ms")
