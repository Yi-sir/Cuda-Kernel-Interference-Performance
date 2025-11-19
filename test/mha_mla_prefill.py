import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

from tabulate import tabulate
import torch

from mha.kernel_fa3_mha_mla import FA3Prefill
# from mha.kernel_flashinfer_mha import FlashinferMHAPrefill
from mha.kernel_triton_mha import TritonMHAPrefill
from test_utils import generate_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

all_params = {
    "batch_size": [8],
    "prompt_len": [512, 1024, 2048, 4096, 8192, 16384],
    "cached_len": [1],
    "q_head_num": [128],
    "kv_head_num": [128],
    "qk_nope_head_dim": [128],
    "qk_rope_head_dim": [64],
    "kv_lora_rank": [512],
    "v_head_dim": [128],
    "page_size": [1],
    "tp": [1, 8],
    "attn_type": ["mha", "mla"],
}

ATTN_BACKENDS = [TritonMHAPrefill, FA3Prefill]

def get_varying_params(params_list):
    if not params_list:
        return []

    all_keys = set(params_list[0].keys())

    varying_params = []
    for key in all_keys:
        values = set(param[key] for param in params_list)
        if len(values) > 1:
            varying_params.append(key)

    return sorted(varying_params)

if __name__ == "__main__":

    device = torch.device("cuda:2")

    attns = [backend(device) for backend in ATTN_BACKENDS]

    params_list = generate_params(all_params)
    varying_params = get_varying_params(params_list)
    results = []

    for param in params_list:
        config_parts = [f"{key}={param[key]}" for key in varying_params]
        row = [", ".join(config_parts)]
        for attn in attns:
            attn.set_params(param)
            t = attn.profile_kernel_us()

            t_ms = t / 1000
            row.append(f"{t_ms:.3f}")
        results.append(row)

    headers = ["Config"] + [attn._kernel_name + "(ms)" for attn in attns]
    print(tabulate(results, headers=headers, tablefmt="grid"))
