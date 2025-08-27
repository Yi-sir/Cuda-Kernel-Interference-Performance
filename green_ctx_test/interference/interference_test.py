from itertools import combinations
import logging
import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from green_context.green_context import get_stream_pairs
from green_ctx_test.test_utils import (KERNEL_DICT, benchmark_parallel_ops,
                                       get_best_performance, plt_draw2)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KERNELS = ["mla", "gemm"]

def get_next_pair(lst):
    yield from combinations(lst, 2)


def check_kernel(k):
    assert k in KERNEL_DICT, f"Unknown kernel: {k}"
    methods = KERNEL_DICT[k]
    assert "kernel" in methods
    assert "params" in methods
    assert "prepare" in methods
    assert "key" in methods


def interference_test():
    logger.info(">>>> START >>>>")

    device = torch.device("cuda:0")
    stream_pairs = get_stream_pairs(device=device)

    for (k0, k1) in get_next_pair(KERNELS):
        check_kernel(k0)
        check_kernel(k1)
        logger.info(f"==== KERNEL {k0, k1} START ====")

        k0_times = []
        k1_times = []
        sms_labels = []

        k0_params = KERNEL_DICT[k0]["params"]()
        k0_input = KERNEL_DICT[k0]["prepare"](**k0_params, device=device)

        k1_params = KERNEL_DICT[k1]["params"]()
        k1_input = KERNEL_DICT[k1]["prepare"](**k1_params, device=device)

        for stream0, stream1, sms0, sms1 in stream_pairs:
            logger.info(f"==== SMs: {sms0, sms1} ====")
            t0, t1 = benchmark_parallel_ops(
                stream0,
                stream1,
                k0_input,
                k1_input,
                k0,
                k1
            )

            k0_times.append(t0)
            k1_times.append(t1)
            sms_labels.append(f"({sms0},{sms1})")

        k0_best_time = get_best_performance(k0, k0_input)
        k1_best_time = get_best_performance(k1, k1_input)

        k0_times.insert(0, float("inf"))
        k0_times.append(k0_best_time)

        k1_times.insert(0, k1_best_time)
        k1_times.append(float("inf"))

        sms_labels.append((132, 0))
        sms_labels.insert(0, (0, 132))

        k0_performance = [k0_best_time / t for t in k0_times]
        k1_performance = [k1_best_time / t for t in k1_times]

        plt_draw2(
            k0_times,
            k0_performance,
            k0,
            k1_times,
            k1_performance,
            k1,
            sms_labels,
            f"{k0}_{k1}.png"
        )

        logger.info(f"==== KERNEL {k0, k1} COMPLETE ====")

    logger.info(f">>>> COMPLETE >>>>")

if __name__ == "__main__":
    interference_test()
