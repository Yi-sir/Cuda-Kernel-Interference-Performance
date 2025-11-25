import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

from tabulate import tabulate
import torch

from mha.kernel_fa3_mha_mla import FA3Prefill

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREFILL_CHUNKED_CAPACITY = 8 * 1024
SEQ_LENS = [256, 512, 1024, 1024 * 2, 1024 * 4, 1024 * 8, 1024 * 16, 1024 * 32]

all_params = {
    "batch_size": 8,
    "prompt_len": 16384 * 2,
    "cached_len": 1,
    "q_head_num": 128,
    "kv_head_num": 128,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
    "v_head_dim": 128,
    "page_size": 1,
    "tp": 8,
    "attn_type": "mha",
}

def get_params(bs, seqlen):
    iterations = bs
    params = []
    for i in range(iterations):
        param = all_params.copy()
        # param["batch_size"] = bs
        param["batch_size"] = 1
        param["prompt_len"] = seqlen * (i + 1)
        param["cached_len"] = seqlen * i

        params.append(param)
    return params

def run(device: torch.device):
    bss = [PREFILL_CHUNKED_CAPACITY // seqlen for seqlen in SEQ_LENS]
    attn = FA3Prefill(device)

    results = []

    for bs, seqlen in zip(bss, SEQ_LENS):
        if seqlen > PREFILL_CHUNKED_CAPACITY:
            continue
        params = get_params(bs, seqlen)
        logger.info(f"profiling attn {attn._kernel_name} with {bs, seqlen}")
        config_parts = [f"{key}={value}" for key, value in zip(["batch_size", "prompt_len"], [bs, seqlen])]
        row = [", ".join(config_parts)]
        t_ms = 0
        for param in params:
            attn.set_params(param)
            t = attn.profile_kernel_us()
            t_ms += t / 1000
        row.append(f"{t_ms:.3f}")
        results.append(row)

    headers = ["Config", attn._kernel_name + "(ms)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    run(torch.device("cuda:3"))
