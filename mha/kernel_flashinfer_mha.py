import logging
import os
# import random
import sys
import time

import torch
import flashinfer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from kernel_base.kernel_base import KernelBase
from kernel_base.registry import register_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _generate_random_partition(max_num_pages, batch_size):
    random_points = torch.randperm(max_num_pages - 1)[:batch_size - 1] + 1
    points = torch.cat([
        torch.tensor([0]),
        torch.sort(random_points)[0],
        torch.tensor([max_num_pages])
    ])
    return points.to(dtype=torch.int32)

def _generate_random_page_len(page_size, batch_size):
    return torch.randint(1, page_size+1, (batch_size, )).to(dtype=torch.int32)

WORKSPACE_BUFFER_SIZE = 384 * 1024 * 1024


@register_kernel
class FlashinferMHADecode(KernelBase):

    _default_params = {
        "batch_size": 7,
        "num_layers": 1,
        "num_qo_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "max_num_pages": 128,
        "page_size": 16,
    }

    _kernel_name = "flashinfer_mha_decode"
    _key = "void flashinfer::BatchDecodeWithPagedKVCacheKernel"

    def __init__(self, device: torch.device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(
            batch_size, num_layers, num_qo_heads, num_kv_heads, head_dim, max_num_pages, page_size, device
        ):

            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            self.workspace_buffer = torch.zeros(WORKSPACE_BUFFER_SIZE, dtype=torch.uint8)
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
            kv_page_indices = torch.arange(max_num_pages).int()
            kv_page_indptr = _generate_random_partition(max_num_pages, batch_size)

            kv_last_page_len = _generate_random_page_len(page_size, batch_size)

            kv_cache_at_layer = [
                torch.randn(
                    max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
                ) for _ in range(num_layers)
            ]

            self.decode_wrapper.plan(
                kv_page_indptr,
                kv_page_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                pos_encoding_mode="NONE",
                data_type=torch.float16
            )

            qs = [torch.randn(batch_size, num_qo_heads, head_dim).half() for _ in range(num_layers)]

            return (
                qs,
                kv_cache_at_layer,
            )

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def flashinfer_mha(
            qs,
            kv_cache_at_layer,
        ):
            num_layers = self.params["num_layers"]
            for i in range(num_layers):
                q = qs[i]
                kv_cache = kv_cache_at_layer[i]
                o = self.decode_wrapper.run(q, kv_cache)
                logger.debug(f"decode_wrapper output' s shape is {o.shape}")

        return flashinfer_mha(*self.inputs)


@register_kernel
class FlashinferMHAPrefill(KernelBase):

    _default_params = {
        "batch_size": 7,
        "num_layers": 1,
        "num_qo_heads": 64,
        "num_kv_heads": 16,
        "head_dim": 128,
        "max_num_pages": 128,
        "page_size": 16,
        "prompt_len": 1024,
        "backend": "fa2"
    }

    _kernel_name = "flashinfer_mha_prefill"
    _key = "void flashinfer::PrefillWithKVCacheKernel"

    _backend_key_dict = {
        "fa2": "void flashinfer::BatchPrefillWithPagedKVCacheKernel",
        "fa3": "void flashinfer::PrefillWithKVCacheKernel",
        # "cudnn": ""
    }

    def __init__(self, device: torch.device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(
          batch_size, num_layers, num_qo_heads, num_kv_heads, head_dim, max_num_pages, page_size, prompt_len, backend, device
        ):
            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            self._key = self._backend_key_dict.get(backend, "")
            logger.debug(f"backend is {backend}, key is {self._key}")

            workspace_buffer = torch.zeros(WORKSPACE_BUFFER_SIZE, dtype=torch.uint8)
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD", backend=backend
            )

            # query长度累计
            qo_indptr = torch.tensor([prompt_len * i for i in range(batch_size + 1)], dtype=torch.int32)

            # query总长度
            nnz_qo = prompt_len * batch_size

            # kv page长度累计
            paged_kv_indptr = _generate_random_partition(max_num_pages, batch_size)
            # 足够大的buffer，只有启用cudagraph时生效
            paged_kv_indices = torch.arange(max_num_pages).to(torch.int32)
            # 最后一页的token数
            paged_kv_last_page_len = _generate_random_page_len(page_size, batch_size)
            q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half()

            kv_cache_at_layer = torch.randn(
                num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
            )

            # max_token_per_sequance = prompt_len
            # max_sequence_kv = max_num_pages * page_size
            # seq_lens = torch.tensor([prompt_len] * batch_size).to(torch.int32)

            self.prefill_wrapper.plan(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                causal=True,
            )

            return (q_at_layer, kv_cache_at_layer)

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def flashinfer_mha_prefill(
            q_at_layer,
            kv_cache_at_layer
        ):
            num_layers = self.params["num_layers"]
            for i in range(num_layers):
                q = q_at_layer[i]
                kv_cache = kv_cache_at_layer[i]
                o = self.prefill_wrapper.run(q, kv_cache)
                logger.debug(f"prefil_wrapper output' s shape is {o.shape}")

        return flashinfer_mha_prefill(*self.inputs)

def test_flashinfer_mha_decode():
    device = torch.device("cuda:0")
    logger.setLevel(logging.DEBUG)
    k = FlashinferMHADecode(device)
    k.prepare_input()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        k.launch_kernel()

    events = prof.key_averages()
    for evt in events:
        print(
            f"event.device_type: {evt.device_type}, device_time: {evt.device_time}"
        )
    print(events.table(sort_by="cuda_time_total", row_limit=10,))

def test_flashinfer_mha_prefill():
    device = torch.device("cuda:0")
    logger.setLevel(logging.DEBUG)
    k = FlashinferMHAPrefill(device)
    k.prepare_input()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        k.launch_kernel()

    events = prof.key_averages()
    for evt in events:
        print(
            f"event.device_type: {evt.device_type}, device_time: {evt.device_time}"
        )
    print(events.table(sort_by="cuda_time_total", row_limit=10,))

if __name__ == "__main__":
    # test_flashinfer_mha_decode()
    test_flashinfer_mha_prefill()
