import logging
import os
import sys

import torch

from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from kernel_base.kernel_base import KernelBase
from kernel_base.registry import register_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_kernel
class FlashMLADecode(KernelBase):

    _default_params = {
        "batch_size": 8,
        "prompt_len": 1024,
        "q_head_num": 128,
        "kv_head_num": 1,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "tp": 1
    }

    _kernel_name = "flashmla_decode"
    _key = "void sm90::flash_fwd_splitkv_mla_kernel"

    def __init__(self, device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(batch_size, prompt_len, q_head_num, kv_head_num, qk_nope_head_dim, qk_rope_head_dim,
                    kv_lora_rank, v_head_dim, tp, device):
            assert q_head_num % tp == 0 or q_head_num == 1
            assert kv_head_num == 1

            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            tp_q_head_num = max(q_head_num // tp, 1)
            tp_kv_head_num = 1

            qk_head_dim = kv_lora_rank + qk_rope_head_dim
            v_head_dim = kv_lora_rank

            max_token_slots = 3064640
            page_size = 64
            max_pages = max_token_slots // page_size

            reshape_q = torch.randn(batch_size, 1, tp_q_head_num, qk_head_dim, dtype=torch.bfloat16)
            k_cache = torch.randn(max_token_slots, tp_kv_head_num, qk_head_dim, dtype=torch.bfloat16).view(-1, page_size, 1, qk_head_dim)

            cache_seqlens = torch.tensor([prompt_len] * batch_size, dtype=torch.int32)

            num_pages_per_req = (prompt_len + page_size - 1) // page_size
            block_table = torch.randint(low=0, high=max_pages, size=(batch_size, num_pages_per_req), dtype=torch.int32)

            # num_splits = torch.tensor([num_pages_per_req * i for i in range(batch_size + 1)], dtype=torch.int32)

            mla_metadata, num_splits = get_mla_metadata(
                cache_seqlens,
                tp_q_head_num,
                1
            )
            scaling = qk_head_dim**-0.5
            causal = True

            return (
                reshape_q,
                k_cache,
                block_table,
                cache_seqlens,
                kv_lora_rank,
                mla_metadata,
                num_splits,
                scaling,
                causal
            )

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def flashmla_decode(
            reshape_q,
            k_cache,
            block_table,
            cache_seqlens,
            kv_lora_rank,
            tile_scheduler_metadata,
            num_splits,
            scaling,
            causal
        ):
            result, _ = flash_mla_with_kvcache(
                q=reshape_q,
                k_cache=k_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                head_dim_v=kv_lora_rank,
                tile_scheduler_metadata=tile_scheduler_metadata,
                num_splits=num_splits,
                softmax_scale=scaling,
                causal=causal
            )
            logger.debug(f"flashmla decode output' s shape is {result.shape}")
            return result

        return flashmla_decode(*self.inputs)

def test_flashmla_decode():
    device = torch.device("cuda:0")

    logger.setLevel(logging.DEBUG)
    k = FlashMLADecode(device)
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
    test_flashmla_decode()
