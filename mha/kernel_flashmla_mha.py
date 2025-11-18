import logging
import os
import sys

import torch
from flash_mla import flash_attn_varlen_func

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from kernel_base.kernel_base import KernelBase
from kernel_base.registry import register_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_sm100_supported(device: torch.device) -> bool:
    return (torch.cuda.get_device_capability(device)[0] == 10) and (
        torch.version.cuda >= "12.8"
    )

@register_kernel
class FlashMLAMHAPrefill(KernelBase):

    _default_params = {
        "batch_size": 8,
        "q_head_num": 128,
        "kv_head_num": 128,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "page_size": 16,
        "prompt_len": 1024,
        "cached_len": 256,
        "tp": 1,
    }

    _kernel_name = "flashmla_mha_prefill_varlen"
    _key = ""

    def __init__(self, device: torch.device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(
            batch_size, prompt_len, cached_len, q_head_num, kv_head_num, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, page_size, tp, device
        ):
            assert q_head_num % tp == 0 or q_head_num == 1
            assert kv_head_num % tp == 0 or kv_head_num == 1
            assert cached_len < prompt_len
            assert cached_len >= 0

            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            tp_q_head_num = max(q_head_num // tp, 1)
            tp_kv_head_num = max(kv_head_num // tp, 1)

            new_token_len = prompt_len - cached_len

            total_tokens = batch_size * prompt_len * 2
            total_new_tokens = batch_size * new_token_len
            total_cached_tokens = batch_size * cached_len

            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

            q = torch.randn(total_new_tokens, tp_q_head_num, qk_head_dim, dtype=torch.bfloat16)
            k = torch.randn(total_tokens, tp_kv_head_num, qk_head_dim, dtype=torch.bfloat16)
            v = torch.randn(total_tokens, tp_kv_head_num, v_head_dim, dtype=torch.bfloat16)

            cu_seqlens_q = torch.tensor([new_token_len * i for i in range(batch_size + 1)])
            cu_seqlens_k = torch.tensor([prompt_len * i for i in range(batch_size + 1)])
            max_seqlen_q = new_token_len
            max_seqlen_k = prompt_len
            softmax_scale = (qk_head_dim + 100) ** (-0.5)

            is_varlen = True
            causal = True

            return (
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                is_varlen,
                causal
            )

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def flashmla_mha_prefill_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, is_varlen, causal
        ):
            o, lse = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                is_varlen=is_varlen
            )
            logger.debug(f"flashmla mha prefill output' s shape is {o.shape}")
            return o

        return flashmla_mha_prefill_varlen(*self.inputs)

def test_flashmla_mha_prefill():
    device = torch.device("cuda:0")

    assert is_sm100_supported(device)

    logger.setLevel(logging.DEBUG)
    k = FlashMLAMHAPrefill(device)
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
    test_flashmla_mha_prefill()
