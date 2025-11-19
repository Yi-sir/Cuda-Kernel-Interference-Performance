import logging
import os
import sys

import torch

from sgl_kernel.flash_attn import flash_attn_with_kvcache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from kernel_base.kernel_base import KernelBase
from kernel_base.registry import register_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_kernel
class FA3Prefill(KernelBase):

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
        "attn_type": "mha"
    }

    _kernel_name = "fa3_mha_prefill"
    _key = "void cutlass::device_kernel"

    def __init__(self, device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(
            batch_size, prompt_len, cached_len, q_head_num, kv_head_num, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, page_size, tp, attn_type, device
        ):
            assert q_head_num % tp == 0 or q_head_num == 1
            assert kv_head_num % tp == 0 or kv_head_num == 1
            assert cached_len < prompt_len
            assert cached_len >= 0

            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            kv_head_num = 1 if attn_type == "mla" else q_head_num

            tp_q_head_num = max(q_head_num // tp, 1)
            tp_kv_head_num = max(kv_head_num // tp, 1)

            new_token_len = prompt_len - cached_len

            total_tokens = batch_size * prompt_len * 2
            total_new_tokens = batch_size * new_token_len
            total_cached_tokens = batch_size * cached_len

            if attn_type == "mla":
                qk_head_dim = kv_lora_rank + qk_rope_head_dim
                v_head_dim = kv_lora_rank
                q = torch.randn(total_new_tokens, tp_q_head_num, qk_head_dim - v_head_dim, dtype=torch.bfloat16)
                k = torch.randn(total_tokens, 1, tp_kv_head_num, qk_head_dim - v_head_dim, dtype=torch.bfloat16)
                v = torch.randn(total_tokens, 1, tp_kv_head_num, v_head_dim, dtype=torch.bfloat16)
                qv = torch.randn(total_new_tokens, tp_q_head_num, v_head_dim, dtype=torch.bfloat16)
            else:
                qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
                q = torch.randn(total_new_tokens, tp_q_head_num, qk_head_dim, dtype=torch.bfloat16)
                k = torch.randn(total_tokens, 1, tp_kv_head_num, qk_head_dim, dtype=torch.bfloat16)
                v = torch.randn(total_tokens, 1, tp_kv_head_num, v_head_dim, dtype=torch.bfloat16)
                qv = None

            page_table = torch.randint(low=0, high=total_tokens, size=(batch_size, new_token_len + cached_len,), dtype=torch.int32)
            if page_size > 1:
                strided_indices = torch.arange(0, page_table.shape[1], page_size)
                page_table = page_table[:, strided_indices] // page_size

            cache_seqlens = torch.tensor([prompt_len for _ in range(batch_size)], dtype=torch.int32)

            cu_seqlens_q = torch.tensor([new_token_len * i for i in range(batch_size + 1)], dtype=torch.int32)
            cu_seqlens_k = torch.tensor([prompt_len * i for i in range(batch_size + 1)], dtype=torch.int32)
            max_seqlen_q = new_token_len

            layer_scaling = 0.1147213867929261
            causal = True
            layer_logit_cap = 0.0
            k_descale, v_descale = None, None
            num_splits = 0

            return (
                q,
                k,
                v,
                qv,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                layer_scaling,
                causal,
                layer_logit_cap,
                k_descale,
                v_descale,
                num_splits
            )

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def fa3_mha_prefill(
            q,
            k,
            v,
            qv,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            layer_scaling,
            causal,
            layer_logit_cap,
            k_descale,
            v_descale,
            num_splits
        ):
            result = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k,
                    v_cache=v,
                    qv=qv,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer_scaling,
                    causal=causal,
                    softcap=layer_logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=False,
                    num_splits=num_splits,
                )
            logger.debug(f"fa3 prefill output' s shape is {result.shape}")
            return result

        return fa3_mha_prefill(*self.inputs)

def test_fa3_mha_prefill():
    device = torch.device("cuda:3")

    logger.setLevel(logging.DEBUG)
    k = FA3Prefill(device)
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
    test_fa3_mha_prefill()
