import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from mha.kernel_flashinfer_mha import FlashinferMHAPrefill

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

params_list = [
    {
        "batch_size": 7,
        "num_layers": 1,
        "num_qo_heads": 64,
        "num_kv_heads": 16,
        "head_dim": 128,
        "max_num_pages": 128,
        "page_size": 16,
        "prompt_len": 1024,
        "backend": "fa2"
    },
    {
        "batch_size": 7,
        "num_layers": 1,
        "num_qo_heads": 64,
        "num_kv_heads": 16,
        "head_dim": 128,
        "max_num_pages": 128,
        "page_size": 16,
        "prompt_len": 1024,
        "backend": "fa3"
    }
]

if __name__ == "__main__":

    device = torch.device("cuda:0")

    flashinfer_mha = FlashinferMHAPrefill(device)

    for param in params_list:
        backend = param.get("backend", None)

        flashinfer_mha.set_params(param)
        t = flashinfer_mha.profile_kernel_us()

        t_ms = t / 1000

        logger.info(f"Flashinfer MHA Prefill with {backend} backend costs {t:.3f}ms")
