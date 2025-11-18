import logging
import os
import sys

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
        "q_head_num": 64,
        "kv_head_num": 8,
        "head_dim": 128,
        "max_num_pages": 128,
        "page_size": 16,
        "tp": 1
    }

    _kernel_name = "flashinfer_mha_decode"
    _key = "void flashinfer::BatchDecodeWithPagedKVCacheKernel"

    def __init__(self, device: torch.device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(
            batch_size, num_layers, q_head_num, kv_head_num, head_dim, max_num_pages, page_size, tp, device
        ):
            assert q_head_num % tp == 0 and kv_head_num % tp == 0

            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            self.workspace_buffer = torch.zeros(WORKSPACE_BUFFER_SIZE, dtype=torch.uint8)
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )

            tp_q_head_num = q_head_num // tp
            tp_kv_head_num = kv_head_num // tp

            kv_page_indices = torch.arange(max_num_pages).int()
            kv_page_indptr = _generate_random_partition(max_num_pages, batch_size)

            kv_last_page_len = _generate_random_page_len(page_size, batch_size)

            kv_cache_at_layer = [
                torch.randn(
                    max_num_pages, 2, page_size, tp_kv_head_num, head_dim, dtype=torch.float16
                ) for _ in range(num_layers)
            ]

            self.decode_wrapper.plan(
                kv_page_indptr,
                kv_page_indices,
                kv_last_page_len,
                tp_q_head_num,
                tp_kv_head_num,
                head_dim,
                page_size,
                pos_encoding_mode="NONE",
                data_type=torch.float16
            )

            qs = [torch.randn(batch_size, tp_q_head_num, head_dim).half() for _ in range(num_layers)]

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
        "batch_size": 8,
        "prompt_len": 1024,
        "cached_len": 256,
        "q_head_num": 128,
        "kv_head_num": 128,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "page_size": 16,
        "tp": 1,
        "backend": "fa3"
    }

    _kernel_name = "flashinfer_mha_prefill"
    _key = "void flashinfer::PrefillWithKVCacheKernel"

    _backend_key_dict = {
        "fa2": "void flashinfer::BatchPrefillWithPagedKVCacheKernel",
        "fa3": "void flashinfer::PrefillWithKVCacheKernel",
    }

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.num_layers = 1

    def prepare_input(self):
        def prepare_paged(
          batch_size, prompt_len, cached_len, q_head_num, kv_head_num, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, page_size, v_head_dim, tp, backend, device
        ):
            assert q_head_num % tp == 0 or q_head_num == 1
            assert kv_head_num % tp == 0 or kv_head_num == 1
            assert cached_len < prompt_len
            assert cached_len >= 0

            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            self._key = self._backend_key_dict.get(backend, "")
            logger.debug(f"backend is {backend}, key is {self._key}")

            workspace_buffer = torch.zeros(WORKSPACE_BUFFER_SIZE, dtype=torch.uint8)
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD", backend=backend
            )

            new_token_len = prompt_len - cached_len

            total_tokens = prompt_len * batch_size
            total_new_tokens = new_token_len * batch_size
            total_cached_tokens = cached_len * batch_size

            qk_head_dim = qk_rope_head_dim + qk_nope_head_dim

            qo_indptr = torch.tensor([new_token_len * i for i in range(batch_size + 1)], dtype=torch.int32)

            tp_q_head_num = q_head_num // tp
            tp_kv_head_num = kv_head_num // tp

            max_num_pages = batch_size * prompt_len * 2 // page_size

            paged_kv_indptr = torch.tensor([prompt_len * i for i in range(batch_size + 1)]).int()
            paged_kv_indices = torch.randint(low=0, high=max_num_pages, size=(total_tokens,)).int()
            paged_kv_last_page_len = _generate_random_page_len(page_size, batch_size)

            q_at_layer = torch.randn(self.num_layers, total_new_tokens, tp_q_head_num, qk_head_dim, dtype=torch.float16)

            k_cache_at_layer = torch.randn(
                self.num_layers, max_num_pages, page_size, tp_kv_head_num, qk_head_dim, dtype=torch.float16
            )
            v_cache_at_layer = torch.randn(
                self.num_layers, max_num_pages, page_size, tp_kv_head_num, v_head_dim, dtype=torch.float16
            )
            kv_cache_at_layer = (k_cache_at_layer, v_cache_at_layer)

            self.prefill_wrapper.plan(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                tp_q_head_num,
                tp_kv_head_num,
                v_head_dim,
                page_size,
                causal=True,
            )

            return (q_at_layer, kv_cache_at_layer)

        def prepare_ragged(
            batch_size, num_layers, q_head_num, kv_head_num, head_dim, max_num_pages, page_size, prompt_len, backend, device
        ):
            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            self._key = self._backend_key_dict.get(backend, "")
            logger.debug(f"backend is {backend}, key is {self._key}")


            workspace_buffer = torch.zeros(WORKSPACE_BUFFER_SIZE, dtype=torch.uint8)
            self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer, "NHD", backend=backend
            )

            qo_indptr = torch.tensor([prompt_len * i for i in range(batch_size + 1)], dtype=torch.int32)
            nnz_qo = prompt_len * batch_size
            nnz_kv = nnz_qo

            kv_indptr = qo_indptr.clone()

            q_at_layer = torch.randn(num_layers, nnz_qo, q_head_num, head_dim).half()
            k_at_layer = torch.randn(num_layers, nnz_kv, kv_head_num, head_dim).half()
            v_at_layer = torch.randn(num_layers, nnz_kv, kv_head_num, head_dim).half()

            self.prefill_wrapper.plan(
                qo_indptr,
                kv_indptr,
                q_head_num,
                kv_head_num,
                head_dim,
                causal=True,
            )

            return (q_at_layer, k_at_layer, v_at_layer)

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        if self.params.get("backend", "") != "trtllm-gen":
            self.inputs = prepare_paged(**self.params, device=self.device)
        else:
            self.inputs = prepare_ragged(**self.params, device=self.device)

    def launch_kernel(self):
        def flashinfer_mha_prefill_paged(
            q_at_layer,
            kv_cache_at_layer
        ):
            for i in range(self.num_layers):
                q = q_at_layer[i]
                kv_cache = (kv_cache_at_layer[0][i], kv_cache_at_layer[1][i])
                o = self.prefill_wrapper.run(q, kv_cache)
                logger.debug(f"prefil_wrapper output' s shape is {o.shape}")

        def flashinfer_mha_prefill_ragged(
            q_at_layer,
            k_at_layer,
            v_at_layer
        ):
            num_layers = self.params["num_layers"]
            for i in range(num_layers):
                q = q_at_layer[i]
                k = k_at_layer[i]
                v = v_at_layer[i]
                o = self.prefill_wrapper.run(q, k, v)
                logger.debug(f"prefil_wrapper output' s shape is {o.shape}")

        if self.params.get("backend", "") != "trtllm-gen":
            return flashinfer_mha_prefill_paged(*self.inputs)
        else:
            return flashinfer_mha_prefill_ragged(*self.inputs)

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
    # for evt in events:
    #     print(
    #         f"event.device_type: {evt.device_type}, device_time: {evt.device_time}"
    #     )
    print(events.table(sort_by="cuda_time_total", row_limit=10,))

if __name__ == "__main__":
    # test_flashinfer_mha_decode()
    test_flashinfer_mha_prefill()
