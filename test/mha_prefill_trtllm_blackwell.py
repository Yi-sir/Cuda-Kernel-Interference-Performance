import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from mha.kernel_trtllm_mha import TRTLLMMHAPrefill
from test_utils import generate_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

all_params = {
    "batch_size": [7],
    "num_qo_heads": [128],
    "num_kv_heads": [128],
    "head_dim": [128],
    "max_num_pages": [128],
    "page_size": [16],
    "prompt_len": [1024],
    "max_context_len": [4096],
    "tp": [1, 8]
}

if __name__ == "__main__":

    device = torch.device("cuda:0")

    flashinfer_mha = TRTLLMMHAPrefill(device)

    params_list = generate_params(all_params)

    for param in params_list:

        flashinfer_mha.set_params(param)
        t = flashinfer_mha.profile_kernel_us()

        t_ms = t / 1000

        logger.info(f"Flashinfer MHA Prefill with {param} costs {t:.3f}ms")
