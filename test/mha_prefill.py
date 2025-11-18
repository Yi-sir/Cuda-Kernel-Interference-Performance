import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from mha.kernel_triton_mha import TritonMHAPrefill
from mha.kernel_fa3_mha_mla import FA3Prefill
from test_utils import generate_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

all_params = {
    "batch_size": [8],
    "prompt_len": [1024],
    "cached_len": [256],
    "q_head_num": [128],
    "kv_head_num": [128],
    "qk_nope_head_dim": [128],
    "qk_rope_head_dim": [64],
    "kv_lora_rank": [512],
    "v_head_dim": [128],
    "page_size": [1],
    "tp": [1, 8],
    "attn_type": ["mha", "mla"],
    "attn_backend": [TritonMHAPrefill, FA3Prefill]
}

# ATTN_BACKENDS = [TritonMHAPrefill, FA3Prefill]

if __name__ == "__main__":

    device = torch.device("cuda:2")

    params_list = generate_params(all_params)

    for param in params_list:

        attn = param["attn_backend"](device)
        param.pop("attn_backend")

        attn.set_params(param)
        t = attn.profile_kernel_us()

        t_ms = t / 1000
        logger.info(f"{attn._kernel_name} with {param} costs {t_ms:.3f}ms")
