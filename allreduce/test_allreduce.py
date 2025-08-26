from __future__ import annotations

import logging
import os
import socket
from pathlib import Path
from typing import Any, Callable

import pytest
import ray
import torch
import vllm.envs as envs
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              tensor_model_parallel_all_reduce)

WORK_PATH = Path(__file__).parent.parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_open_port() -> int:
    port = envs.VLLM_PORT
    if port is not None:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def get_open_port() -> int:
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by
    the data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    if "VLLM_DP_MASTER_PORT" in os.environ:
        dp_master_port = envs.VLLM_DP_MASTER_PORT
        reserved_port_range = range(dp_master_port, dp_master_port + 10)
        while True:
            candidate_port = _get_open_port()
            if candidate_port not in reserved_port_range:
                return candidate_port
    return _get_open_port()


def init_test_distributed_environment(
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
    local_rank: int = -1,
) -> None:
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    init_distributed_environment(
        world_size=pp_size * tp_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
        local_rank=local_rank,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_test_worker(
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    # monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # logger.info(f"Current rank: {rank}")
    # pid = os.getpid()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    num_elements = 8
    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="cuda") * (r + 1)
        for r in range(tp_size)
    ]
    expected = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
    t = all_tensors[rank % tp_size]

    # warmup
    # seems useless
    _ = tensor_model_parallel_all_reduce(t)

    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="cuda") * (r + 1)
        for r in range(tp_size)
    ]
    t = all_tensors[rank % tp_size]

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        t = tensor_model_parallel_all_reduce(t)

    events = prof.key_averages()
    max_cuda_time = max(
        (
            event.device_time
            for event in events
            if event.device_type != torch.autograd.DeviceType.CPU
        ),
        default=0.0,
    )

    if __name__ == "__main__":
        if rank == 0:
            for evt in events:
                print(
                    f"event.device_type: {evt.device_type}, device_time: {evt.device_time}"
                )
            # print(events.table(sort_by="cuda_time_total", row_limit=10,))
        print(f"rank is {rank}, max_cuda_time is {max_cuda_time}")

    # torch.testing.assert_close(t, expected)

    return max_cuda_time


def multi_process_parallel(
    tp_size: int,
    pp_size: int,
) -> None:
    import ray

    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    # NOTE: We need to set working_dir for distributed tests,
    # otherwise we may get import errors on ray workers
    # NOTE: Force ray not to use gitignore file as excluding, otherwise
    # it will not move .so files to working dir.
    # So we have to manually add some of large directories
    os.environ["RAY_RUNTIME_ENV_IGNORE_GITIGNORE"] = "1"
    ray.init(runtime_env={"working_dir": WORK_PATH})

    distributed_init_port = get_open_port()
    refs = []
    for rank in range(tp_size * pp_size):
        refs.append(
            all_reduce_test_worker.remote(
                tp_size,
                pp_size,
                rank,
                distributed_init_port,
            ),
        )
    ret = ray.get(refs)

    print(ret)

    ray.shutdown()


if __name__ == "__main__":
    tp_size = 8
    pp_size = 1

    multi_process_parallel(tp_size, pp_size)
