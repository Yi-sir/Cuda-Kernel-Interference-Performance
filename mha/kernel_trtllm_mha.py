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

WORKSPACE_BUFFER_SIZE = 384 * 1024 * 1024

def is_sm100_supported(device: torch.device) -> bool:
    return (torch.cuda.get_device_capability(device)[0] == 10) and (
        torch.version.cuda >= "12.8"
    )

@register_kernel
class TRTLLMMHAPrefill(KernelBase):

    _default_params = {
        "batch_size": 7,
        "num_qo_heads": 64,
        "num_kv_heads": 64,
        "head_dim": 128,
        "max_num_pages": 128,
        "page_size": 16,
        "prompt_len": 1024,
        "max_context_len": 4096,
        "tp": 1
    }

    _kernel_name = "trtllm_mha_prefill"
    _key = "fmhaSm100fKernel_QkvFp16OFp16H128PagedKvCausalP16"

    def __init__(self, device: torch.device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(
            batch_size, num_qo_heads, num_kv_heads, head_dim, max_num_pages, page_size, prompt_len, max_context_len, tp, device
        ):
            assert num_qo_heads % tp == 0 and num_kv_heads % tp == 0

            torch.set_default_device(device)
            torch.cuda.set_device(device)

            torch.manual_seed(0)

            workspace_buffer = torch.zeros(WORKSPACE_BUFFER_SIZE, dtype=torch.uint8)

            num_tokens = prompt_len * batch_size
            seq_lens = torch.tensor([prompt_len] * batch_size)

            cum_seq_lens_q = torch.tensor([prompt_len * i for i in range(batch_size + 1)])
            cum_seq_lens_kv = cum_seq_lens_q.clone()

            tp_q_head_num = num_qo_heads // tp
            tp_kv_head_num = num_kv_heads // tp

            query = torch.randn(num_tokens, tp_q_head_num, head_dim, dtype=torch.float16)

            total_tokens = max_num_pages * page_size
            k_cache = torch.randn(total_tokens + page_size, num_kv_heads, head_dim, dtype=torch.float16)
            v_cache = torch.randn(total_tokens + page_size, num_kv_heads, head_dim, dtype=torch.float16)

            # [num_pages, head_num, page_size, head_dim]
            k_cache = k_cache.view(-1, page_size, tp_kv_head_num, head_dim).permute(0, 2, 1, 3)
            v_cache = v_cache.view(-1, page_size, tp_kv_head_num, head_dim).permute(0, 2, 1, 3)

            kv_cache = (k_cache, v_cache)

            # sglang的调用里，block_tables取自req_to_token_pool[indices, :max_seq_len_k]
            # 值域 [0, total_tokens]
            # 这里需要调成page_size个递增的吗，只跑性能好像无所谓
            block_tables = torch.randint(low=0, high=total_tokens+page_size, size=(batch_size, prompt_len))

            if page_size > 1:
                strided_indices = torch.arange(0, block_tables.shape[1], page_size)
                block_tables = block_tables[:, strided_indices] // page_size

            return (query, kv_cache, workspace_buffer, block_tables, seq_lens, cum_seq_lens_q, cum_seq_lens_kv)

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def trtllm_mha_prefill(
          query, kv_cache, workspace_buffer, block_tables, seq_lens, cum_seq_lens_q, cum_seq_lens_kv
        ):
            max_q_len = self.params.get("prompt_len")
            max_kv_len = self.params.get("max_context_len")

            batch_size = self.params.get("batch_size")

            bmm1_scale = 1.0
            bmm2_scale = 1.0

            output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=query,
                kv_cache=kv_cache,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_q_len=max_q_len,
                max_kv_len=max_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=batch_size,
                cum_seq_lens_q=cum_seq_lens_q,
                cum_seq_lens_kv=cum_seq_lens_kv
            )

            logger.debug(f"trtllm prefill output' s shape is {output.shape}")

        return trtllm_mha_prefill(*self.inputs)

def test_trtllm_mha_prefill():
    device = torch.device("cuda:0")

    assert is_sm100_supported(device)

    logger.setLevel(logging.DEBUG)
    k = TRTLLMMHAPrefill(device)
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
