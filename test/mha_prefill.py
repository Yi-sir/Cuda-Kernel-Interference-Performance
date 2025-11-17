import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from mha.kernel_flashinfer_mha import FlashinferMHAPrefill
from test_utils import generate_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

all_params = {
    "batch_size": [8],
    "q_head_num": [64],
    "kv_head_num": [64],
    "head_dim": [128],
    "page_size": [16],
    "prompt_len": [1024],
    "cached_len": [256],
    "tp": [1, 8],
    "backend": ["fa2", "fa3"]
}

if __name__ == "__main__":

    device = torch.device("cuda:0")

    flashinfer_mha = FlashinferMHAPrefill(device)

    params_list = generate_params(all_params)

    for param in params_list:
        backend = param.get("backend", None)

        flashinfer_mha.set_params(param)
        t = flashinfer_mha.profile_kernel_us()

        t_ms = t / 1000

        logger.info(f"Flashinfer MHA Prefill with param: {param} costs {t:.3f}ms")
