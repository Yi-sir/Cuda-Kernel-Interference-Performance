import copy
import logging
import random
import time

import deep_gemm
import torch
from deep_gemm.testing import bench, bench_kineto, calc_diff, count_bytes
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
from generators import (KernelType, MajorTypeAB,
                        enumerate_k_grouped_contiguous,
                        enumerate_m_grouped_contiguous,
                        enumerate_m_grouped_masked, enumerate_normal,
                        generate_k_grouped_contiguous,
                        generate_m_grouped_contiguous,
                        generate_m_grouped_masked, generate_normal,
                        get_ue8m0_usage)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 请放在DeepGEMM/tests目录下使用


def get_all_streams(device: torch.device):
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    logger.info(f"Current device: {device}, sm count is {sm_count}")

    num_sms_start = 8
    num_sms_step = 8
    num_sms_end = (sm_count // num_sms_step) * num_sms_step
    num_sms = [i for i in range(num_sms_start, num_sms_end, num_sms_step)]
    num_sm_counts = len(num_sms)

    streams = []

    for sms in num_sms:
        logger.info(f"Creating green ctx with SM count: {sms}")
        (stream, _), _ = split_device_green_ctx_by_sm_count(device, [sms])

        streams.append((stream, sms))

    return streams


def test_gemm() -> None:
    print("Testing GEMM:")
    for (
        kernel_type,
        m,
        n,
        k,
        major_a,
        major_b,
        accumulate,
        out_dtype,
    ) in enumerate_normal():
        major_opt = "N" if major_a.is_k_major() else "T"
        major_opt += "T" if major_b.is_k_major() else "N"
        out_opt = "FP32" if out_dtype == torch.float else "BF16"
        acc_opt = f"acc={int(accumulate)}"
        kernel_opt = f"1D1D" if kernel_type.is_1d1d() else "1D2D"
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(
                m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0
            )
            func_name = f'fp8_gemm_{major_opt.lower() if test_alias else "nt"}'
            if test_alias:
                a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(
                a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast
            )
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, (
                f"{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, "
                f"{diff:.5f}, alias={test_alias}"
            )
        a, b, c, d, ref_d = generate_normal(
            m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0
        )

        # Test launch overhead
        launch_start_t = time.time_ns()
        deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
        launch_end_t = time.time_ns()
        torch.cuda.synchronize()

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)

        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
        print(
            f" > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): "
            f"launch {(launch_end_t - launch_start_t) / 1e3:4.0f} us | {t * 1e6:4.0f} us | "
            f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
            f"{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s"
        )
    print()


def xyz_test_gemm():
    logger.info("xyz test gemm")
    m = 4096
    n = 129280
    k = 7168
    major_a = MajorTypeAB.KMajor
    major_b = MajorTypeAB.KMajor
    kernel_opt = "1D2D"
    major_opt = "NT"
    out_opt = "BF16"
    acc_opt = 0
    accumulate = False
    out_dtype = torch.bfloat16
    use_ue8m0 = False
    test_alias = False
    disable_ue8m0_cast = not use_ue8m0

    a, b, c, d, ref_d = generate_normal(
        m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0
    )

    func_name = f'fp8_gemm_{major_opt.lower() if test_alias else "nt"}'

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        getattr(deep_gemm, func_name)(
            a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast
        )

    events = prof.key_averages()
    max_cuda_time = max(
        (
            event.device_time
            for event in events
            if event.device_type != torch.autograd.DeviceType.CPU
        ),
        default=0.0,
    )

    diff = calc_diff(d, ref_d)
    assert diff < 0.001, (
        f"{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, "
        f"{diff:.5f}, alias={test_alias}"
    )

    return max_cuda_time


def green_ctx_gemm(device):
    streams = get_all_streams(device)
    all_times = {}
    for stream, num_sm in streams:
        with torch.cuda.stream(stream):
            t = xyz_test_gemm()
            all_times[num_sm] = t
    for k, v in all_times.items():
        logger.info(f"With SMs {k}, GEMM cost {v:.3f}us")


def test_m_grouped_gemm_contiguous() -> None:
    print("Testing m-grouped contiguous GEMM:")

    for (
        kernel_type,
        num_groups,
        expected_m_per_group,
        n,
        k,
        major_a,
        major_b,
    ) in enumerate_m_grouped_contiguous():
        major_opt = "N" if major_a.is_k_major() else "T"
        major_opt += "T" if major_b.is_k_major() else "N"
        kernel_opt = f"1D1D" if kernel_type.is_1d1d() else "1D2D"
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        for test_alias in (False, True):
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                num_groups,
                expected_m_per_group,
                n,
                k,
                major_a,
                major_b,
                use_ue8m0=use_ue8m0,
            )
            func_name = f"m_grouped_fp8_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(
                a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast
            )
            d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
            diff = calc_diff(d, ref_d)
            assert (
                diff < 0.001
            ), f"{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}"
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
            num_groups,
            expected_m_per_group,
            n,
            k,
            major_a,
            major_b,
            use_ue8m0=use_ue8m0,
        )

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast
            )

        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
        print(
            f" > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, {kernel_opt}, layout={major_opt}): "
            f"{t * 1e6:4.0f} us | "
            f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
            f"{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s"
        )
    print()


def test_m_grouped_gemm_masked() -> None:
    print("Testing m-grouped masked GEMM:")

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for (
        kernel_type,
        num_groups,
        max_m,
        expected_m_per_group,
        n,
        k,
    ) in enumerate_m_grouped_masked():
        kernel_opt = f"1D1D" if kernel_type.is_1d1d() else "1D2D"
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(
                num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0
            )
            deep_gemm.fp8_m_grouped_gemm_nt_masked(
                a,
                b,
                d,
                masked_m,
                expected_m_per_group,
                disable_ue8m0_cast=disable_ue8m0_cast,
            )
            for j in range(num_groups):
                diff = calc_diff(
                    d[j, : masked_m[j].item()], ref_d[j, : masked_m[j].item()]
                )
                assert (
                    diff < 0.001
                ), f"{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}"

        # Construct full cases
        a, b, masked_m, d, ref_d = generate_m_grouped_masked(
            num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0
        )

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.fp8_m_grouped_gemm_nt_masked(
                a,
                b,
                d,
                masked_m,
                expected_m_per_group,
                disable_ue8m0_cast=disable_ue8m0_cast,
            )

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
        print(
            f" > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): "
            f"{t * 1e6:4.0f} us | "
            f"{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | "
            f"{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s"
        )
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print("Testing k-grouped contiguous GEMM:")

    for num_groups, m, n, ks, expected_k_per_group in enumerate_k_grouped_contiguous():
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

        for test_empty_groups in (False, True):
            new_ks = copy.deepcopy(ks)
            if test_empty_groups:
                new_ks[random.randint(0, num_groups - 1)] = 0
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous(
                num_groups, m, n, new_ks, use_ue8m0=use_ue8m0
            )
            new_ks_tensor = torch.tensor(new_ks, dtype=torch.int, device="cuda")
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                a, b, d, new_ks, new_ks_tensor, c=c
            )
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f"{m=}, {n=}, {k=}, {i=}, {diff:.5f}"

        # Test performance
        k, a, b, c, d, ref_d = generate_k_grouped_contiguous(
            num_groups, m, n, ks, use_ue8m0=use_ue8m0
        )
        ks_tensor = torch.tensor(ks, dtype=torch.int, device="cuda")

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, ks_tensor, c=c)

        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
        print(
            f" > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={k:5}): "
            f"{t * 1e6:4.0f} us | "
            f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
            f"{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s"
        )
    print()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda:0")
    torch.set_default_device(device)

    torch.manual_seed(0)
    random.seed(0)

    print("Library path:")
    print(f" > {deep_gemm.__path__}\n")

    # test_gemm()
    # test_m_grouped_gemm_contiguous()
    # test_m_grouped_gemm_masked()
    # test_k_grouped_gemm_contiguous()

    # xyz_test_gemm()
    green_ctx_gemm(device)
