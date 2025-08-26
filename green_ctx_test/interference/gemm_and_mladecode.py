import logging
import os
import sys
from pathlib import Path

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from fp8_gemm.test_fp8_gemm import (fp8_matmul, get_gemm_params,
                                    test_w8a8_block_fp8_matmul)
from green_context.green_context import get_stream_pairs
from green_ctx_test.test_utils import (EVENT_KEY_DICT, benchmark_parallel_ops,
                                       get_best_performance, plt_draw,
                                       plt_draw2)
from mla.test_mla_decode import flash_mla, get_mla_params, test_flash_mla

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    device = torch.device("cuda:0")

    stream_pairs = get_stream_pairs(device)

    mla_params = get_mla_params()
    gemm_params = get_gemm_params()

    mla_times = []
    gemm_times = []
    sms_labels = []

    mla_input = test_flash_mla(**mla_params, device=device)
    gemm_input = test_w8a8_block_fp8_matmul(**gemm_params, device=device)

    for stream0, stream1, sms0, sms1 in stream_pairs:
        logger.info(f"===== Kernel: MLA, GEMM, SMs: {sms0, sms1} =====")
        t0, t1 = benchmark_parallel_ops(
            stream0,
            stream1,
            flash_mla,
            fp8_matmul,
            mla_input,
            gemm_input,
            "mla",
            "gemm",
        )

        mla_times.append(t0)
        gemm_times.append(t1)
        sms_labels.append(f"({sms0},{sms1})")

    mla_best_time = get_best_performance(flash_mla, mla_input)
    gemm_best_time = get_best_performance(fp8_matmul, gemm_input)

    mla_times.insert(0, float("inf"))
    mla_times.append(mla_best_time)

    gemm_times.insert(0, gemm_best_time)
    gemm_times.append(float("inf"))

    sms_labels.append((132, 0))
    sms_labels.insert(0, (0, 132))

    mla_performance = [mla_best_time / t for t in mla_times]
    gemm_performance = [gemm_best_time / t for t in gemm_times]

    # plt_draw(mla_times, gemm_times, "MLA", "GEMM", sms_labels, "mla_gemm_time_cost.png", "bar")
    # plt_draw(mla_performance, gemm_performance, "MLA", "GEMM", sms_labels, "mla_gemm_performance.png", "line")

    plt_draw2(
        mla_times,
        mla_performance,
        "MLA",
        gemm_times,
        gemm_performance,
        "GEMM",
        sms_labels,
        "mla_gemm.png",
    )
