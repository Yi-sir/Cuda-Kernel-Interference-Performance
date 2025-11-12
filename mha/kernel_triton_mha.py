import logging
import os
import sys

import torch
import flashinfer

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from kernel_base.kernel_base import KernelBase
from kernel_base.registry import register_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_kernel
class TritonMHAPrefill(KernelBase):

    _default_params = {
        "batch_size": 8,
        "prompt_len": 1024,
        "cached_len": 256,
        "q_head_num": 64,
        "kv_head_num": 1,
        "qk_head_dim": 576,
        "v_head_dim": 512,
        "page_size": 1,
        "max_num_pages": 128,
        "tp": 1
    }

    _kernel_name = "triton_mha_prefill"
    _key = "_fwd_kernel"

    def __init__(self, device):
        super().__init__(device)
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

    def prepare_input(self):
        def prepare(
            batch_size, prompt_len, cached_len, q_head_num, kv_head_num, qk_head_dim, v_head_dim, page_size, max_num_pages, tp, device
        ):
            assert q_head_num % tp == 0 and kv_head_num % tp == 0
            assert cached_len < prompt_len
            assert cached_len >= 0

            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            tp_q_head_num = q_head_num // tp
            tp_kv_head_num = kv_head_num // tp

            new_token_len = prompt_len - cached_len

            total_tokens = max_num_pages * page_size
            total_new_tokens = batch_size * new_token_len
            total_cached_tokens = batch_size * cached_len

            k_cache = torch.randn(total_tokens + page_size, tp_kv_head_num, qk_head_dim, dtype=torch.float16)
            v_cache = torch.randn(total_tokens + page_size, tp_kv_head_num, v_head_dim, dtype=torch.float16)

            q = torch.randn(total_new_tokens, tp_q_head_num, qk_head_dim, dtype=torch.float16)
            k = torch.randn(total_new_tokens, tp_kv_head_num, qk_head_dim, dtype=torch.float16)
            v = torch.randn(total_new_tokens, tp_kv_head_num, v_head_dim, dtype=torch.float16)

            o = torch.randn(total_new_tokens, tp_q_head_num * v_head_dim, dtype=torch.float16)

            qo_indptr = torch.tensor([new_token_len * i for i in range(batch_size + 1)])
            kv_indptr = torch.tensor([cached_len * i for i in range(batch_size + 1)])
            kv_indices = torch.randint(low=0, high=total_tokens+page_size, size=(batch_size, prompt_len))

            return (
                q.view(-1, tp_q_head_num, qk_head_dim),
                k,
                v,
                o.view(-1, tp_q_head_num, v_head_dim),
                k_cache,
                v_cache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                None,
                True,
                None,
                new_token_len,
            )

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def triton_mha_prefill(
            q, k, v, o, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, mask, causal, mask_indptr, new_token_len
        ):
            self.extend_attention_fwd(
                q,
                k,
                v,
                o,
                k_cache,
                v_cache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                mask,
                causal,
                mask_indptr,
                new_token_len
            )

            logger.debug(f"triton prefill output' s shape is {o.shape}")

        return triton_mha_prefill(*self.inputs)

def test_trtllm_mha_prefill():
    device = torch.device("cuda:0")

    logger.setLevel(logging.DEBUG)
    k = TritonMHAPrefill(device)
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
    test_trtllm_mha_prefill()
