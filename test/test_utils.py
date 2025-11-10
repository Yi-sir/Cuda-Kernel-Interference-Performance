import logging
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_parallel_ops(
    stream0,
    stream1,
    obj0,
    obj1,
    target_duration_ms=1000,
):
    key0 = obj0._kernel_name
    key1 = obj1._kernel_name

    # Warmup
    with torch.cuda.stream(stream0):
        _ = obj0.launch_kernel()
    with torch.cuda.stream(stream1):
        _ = obj1.launch_kernel()
    torch.cuda.synchronize()

    # Estimate individual op durations
    duration_0 = obj0.profile_kernel_us(stream0) / 1000
    duration_1 = obj1.profile_kernel_us(stream1) / 1000


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
            _ = obj0.launch_kernel()
    with torch.cuda.stream(stream1):
        for _ in range(kernel1_loops):
            _ = obj1.launch_kernel()
    torch.cuda.synchronize()
    prof.stop()

    # Process results
    v0 = obj0._key
    v1 = obj1._key

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
