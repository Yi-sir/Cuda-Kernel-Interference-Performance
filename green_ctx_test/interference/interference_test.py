import logging
import os
import sys
from itertools import combinations

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from green_context.green_context import get_stream_pairs
from green_ctx_test.test_utils import (benchmark_parallel_ops, plt_draw2)

from mla.kernel_mla_decode import MLADecode
from fp8_gemm.kernel_triton_gemm import TritonGemm
from kernel_base.registry import get_kernel_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KERNELS = ["mla_decode", "triton_gemm"]


def get_next_pair(lst):
    yield from combinations(lst, 2)


def interference_test():
    logger.info(">>>> START >>>>")

    device = torch.device("cuda:0")
    stream_pairs = get_stream_pairs(device=device)

    for k0, k1 in get_next_pair(KERNELS):
        obj0 = get_kernel_class(k0)(device)
        obj1 = get_kernel_class(k1)(device)

        logger.info(f"==== KERNEL {k0, k1} START ====")

        k0_times = []
        k1_times = []
        sms_labels = []

        obj0.prepare_input()
        obj1.prepare_input()

        for stream0, stream1, sms0, sms1 in stream_pairs:
            logger.info(f"==== SMs: {sms0, sms1} ====")
            t0, t1 = benchmark_parallel_ops(
                stream0, stream1, obj0, obj1
            )

            k0_times.append(t0)
            k1_times.append(t1)
            sms_labels.append(f"({sms0},{sms1})")

        k0_best_time = obj0.get_best_performance_us() / 1000
        k1_best_time = obj1.get_best_performance_us() / 1000

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
            f"{k0}_{k1}.png",
        )

        logger.info(f"==== KERNEL {k0, k1} COMPLETE ====")

    logger.info(f">>>> COMPLETE >>>>")


if __name__ == "__main__":
    interference_test()
