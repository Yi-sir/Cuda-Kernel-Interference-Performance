import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

from tabulate import tabulate
import torch

from mla.kernel_flashmla_mla import FlashMLADecode
from mha.kernel_fa3_mha_mla import FA3Decode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

all_params = [
    {
        "batch_size": 8,
        "prompt_len": 1024,
        "q_head_num": 128,
        "kv_head_num": 1,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "page_size": 1,
        "tp": 8
    },
    {
        "batch_size": 8,
        "prompt_len": 1024,
        "q_head_num": 16,
        "kv_head_num": 1,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "page_size": 1,
        "tp": 1
    }
]

ATTN_BACKENDS = [FlashMLADecode, FA3Decode]

def test(device):
    attns = [backend(device) for backend in ATTN_BACKENDS]

    for param in all_params:
        for attn in attns:
            attn.set_params(param)
            t = attn.profile_kernel_us()

            # t_ms = t / 1000
            logger.info(f"param: {param}\n\tattn: {attn}\n\tt_us: {t:.2f}")

if __name__ == "__main__":
    device = torch.device("cuda:2")
    test(device)
