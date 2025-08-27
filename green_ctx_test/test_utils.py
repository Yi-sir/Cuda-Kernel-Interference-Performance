import logging
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch

from fp8_gemm.test_fp8_gemm import (fp8_matmul, get_gemm_params,
                                    test_w8a8_block_fp8_matmul)
from mla.test_mla_decode import flash_mla, get_mla_params, test_flash_mla

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KERNEL_DICT = {
    "mla": {
        "kernel": flash_mla,
        "params": get_mla_params,
        "prepare": test_flash_mla,
        "key": "void flash::flash_fwd_splitkv_mla_kernel"
    },
    "gemm": {
        "kernel": fp8_matmul,
        "params": get_gemm_params,
        "prepare": test_w8a8_block_fp8_matmul,
        "key": "_w8a8_block_fp8_matmul"
    }
}


def profile_a_kernel(key, kernel_inputs, stream, repeat=10):

    kernel_function = KERNEL_DICT[key]["kernel"]
    # warmup
    with torch.cuda.stream(stream):
        kernel_function(*kernel_inputs)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.cuda.stream(stream):
            for _ in range(repeat):
                kernel_function(*kernel_inputs)

    events = prof.key_averages()
    max_cuda_time = max(
        (
            event.device_time
            for event in events
            if event.device_type != torch.autograd.DeviceType.CPU
        ),
        default=0.0,
    )
    return max_cuda_time  # us


def get_best_performance(key, kernel_inputs):
    stream = torch.cuda.Stream()
    return profile_a_kernel(key, kernel_inputs, stream) / 1000  # ms


def benchmark_parallel_ops(
    stream0,
    stream1,
    input0,
    input1,
    key0,
    key1,
    target_duration_ms=1000,
):
    kernel0 = KERNEL_DICT[key0]["kernel"]
    kernel1 = KERNEL_DICT[key1]["kernel"]

    # Warmup
    with torch.cuda.stream(stream0):
        _ = kernel0(*input0)
    with torch.cuda.stream(stream1):
        _ = kernel1(*input1)
    torch.cuda.synchronize()

    # Estimate individual op durations
    duration_0 = profile_a_kernel(key0, input0, stream0) / 1000  # ms
    duration_1 = profile_a_kernel(key1, input1, stream1) / 1000  # ms

    # Calculate loop counts based on target duration
    kernel0_loops = max(1, int(target_duration_ms / duration_0))
    kernel1_loops = max(1, int(target_duration_ms / duration_1))

    logger.info(
        f"Estimated durations - {key0}: {duration_0:.3f}ms, {key1}: {duration_1:.3f}ms"
    )
    logger.info(f"Using loop counts - {key0}: {kernel0_loops}, {key1}: {kernel1_loops}")

    # Profile parallel execution
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    )

    prof.start()
    with torch.cuda.stream(stream0):
        for _ in range(kernel0_loops):
            _ = kernel0(*input0)
    with torch.cuda.stream(stream1):
        for _ in range(kernel1_loops):
            _ = kernel1(*input1)
    torch.cuda.synchronize()
    prof.stop()

    # Process results
    v0 = KERNEL_DICT[key0]["key"]
    v1 = KERNEL_DICT[key1]["key"]

    events = prof.events()

    kernel0_times = [
        e.device_time
        for e in events
        if e.device_type != torch.autograd.DeviceType.CPU and e.key.startswith(v0)
    ]
    kernel1_times = [
        e.device_time
        for e in events
        if e.device_type != torch.autograd.DeviceType.CPU and e.key.startswith(v1)
    ]

    avg_k0 = sum(kernel0_times) / len(kernel0_times) / 1000
    avg_k1 = sum(kernel1_times) / len(kernel1_times) / 1000

    logger.info(f"Parallel execution results:")
    logger.info(f"  {key0} - count: {len(kernel0_times)}, avg: {avg_k0:.3f}ms")
    logger.info(f"  {key1} - count: {len(kernel1_times)}, avg: {avg_k1:.3f}ms")

    return avg_k0, avg_k1


def plt_draw(
    kernel0_times,
    kernel1_times,
    kernel0_name,
    kernel1_name,
    sms_labels,
    filename,
    plot_type="bar",
):
    """
    绘制比较两种kernel执行时间的图表

    参数:
        kernel0_times: 第一个kernel的执行时间列表
        kernel1_times: 第二个kernel的执行时间列表
        kernel0_name: 第一个kernel的名称
        kernel1_name: 第二个kernel的名称
        sms_labels: x轴的标签列表
        filename: 保存图片的文件名
        plot_type: 图表类型，'bar'表示直方图，'line'表示折线图
    """
    plt.figure(figsize=(12, 6))
    x = np.arange(len(sms_labels))
    width = 0.35

    if plot_type == "bar":
        plt.bar(
            x - width / 2, kernel0_times, width, label=kernel0_name, color="royalblue"
        )
        plt.bar(x + width / 2, kernel1_times, width, label=kernel1_name, color="orange")
    elif plot_type == "line":
        plt.plot(
            x,
            kernel0_times,
            "o-",
            label=kernel0_name,
            color="royalblue",
            linewidth=2,
            markersize=8,
        )
        plt.plot(
            x,
            kernel1_times,
            "s-",
            label=kernel1_name,
            color="orange",
            linewidth=2,
            markersize=8,
        )
    else:
        raise ValueError("plot_type必须是'bar'或'line'")

    plt.xlabel(f"SM Combinations ({kernel0_name}, {kernel1_name})")
    plt.ylabel("Execution Time (ms)")
    plt.title("Operator Execution Time under Different SM Configurations")
    plt.xticks(x, sms_labels)
    plt.legend()

    for i in range(len(sms_labels)):
        if plot_type == "bar":
            plt.text(
                i - width / 2,
                kernel0_times[i] + 0.1,
                f"{kernel0_times[i]:.3f}",
                ha="center",
            )
            plt.text(
                i + width / 2,
                kernel1_times[i] + 0.1,
                f"{kernel1_times[i]:.3f}",
                ha="center",
            )
        else:
            plt.text(
                x[i], kernel0_times[i] + 0.1, f"{kernel0_times[i]:.3f}", ha="center"
            )
            plt.text(
                x[i], kernel1_times[i] + 0.1, f"{kernel1_times[i]:.3f}", ha="center"
            )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def plt_draw2(
    k0_times, k0_perf, k0_name, k1_times, k1_perf, k1_name, sms_labels, filename
):
    """
    在同一图表中使用双Y轴绘制执行时间和性能的比较图

    参数:
        k0_times: 第一个kernel的执行时间列表
        k0_perf: 第一个kernel的性能列表
        k0_name: 第一个kernel的名称
        k1_times: 第二个kernel的执行时间列表
        k1_perf: 第二个kernel的性能列表
        k1_name: 第二个kernel的名称
        sms_labels: x轴的标签列表
        filename: 保存图片的文件名
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(sms_labels))
    width = 0.35

    ax1.bar(
        x - width / 2,
        k0_times,
        width,
        label=f"{k0_name} Time",
        color="royalblue",
        alpha=0.7,
    )
    ax1.bar(
        x + width / 2,
        k1_times,
        width,
        label=f"{k1_name} Time",
        color="orange",
        alpha=0.7,
    )
    ax1.set_xlabel(f"SM Combinations ({k0_name}, {k1_name})")
    ax1.set_ylabel("Execution Time (ms)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    for i in range(len(sms_labels)):
        ax1.text(
            x[i] - width / 2,
            k0_times[i] + 0.1,
            f"{k0_times[i]:.3f}",
            ha="center",
            color="black",
        )
        ax1.text(
            x[i] + width / 2,
            k1_times[i] + 0.1,
            f"{k1_times[i]:.3f}",
            ha="center",
            color="black",
        )

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        k0_perf,
        "o-",
        label=f"{k0_name} Perf",
        color="royalblue",
        linewidth=2,
        markersize=8,
    )
    ax2.plot(
        x,
        k1_perf,
        "s-",
        label=f"{k1_name} Perf",
        color="orange",
        linewidth=2,
        markersize=8,
    )
    ax2.set_ylabel("Performance", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    for i in range(len(sms_labels)):
        ax2.text(
            x[i], k0_perf[i] + 0.1, f"{k0_perf[i]:.3f}", ha="center", color="royalblue"
        )
        ax2.text(
            x[i], k1_perf[i] + 0.1, f"{k1_perf[i]:.3f}", ha="center", color="orange"
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(sms_labels)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(f"{k0_name} vs {k1_name} - Time and Performance Comparison")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
