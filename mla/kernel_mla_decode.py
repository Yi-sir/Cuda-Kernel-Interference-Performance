import logging
import os
import random
import sys
import time

import torch
from vllm.attention.ops.flashmla import (flash_mla_with_kvcache,
                                         get_mla_metadata)
from vllm.triton_utils import triton

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from kernel_base.kernel_base import KernelBase
from kernel_base.registry import register_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_kernel
class MLADecode(KernelBase):

    _default_params = {
        "b": 128,
        "s_q": 1,
        "mean_sk": 1024,
        "h_q": 128,
        "h_kv": 1,
        "d": 576,
        "dv": 512,
        "block_size": 64,
        "causal": True,
        "varlen": False,
    }

    _kernel_name = "mla_decode"
    _key = "void flash::flash_fwd_splitkv_mla_kernel"

    def __init__(self, device: torch.device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(
            b, s_q, mean_sk, h_q, h_kv, d, dv, block_size, causal, varlen, device
        ):
            dtype = torch.bfloat16
            torch.set_default_dtype(dtype)
            torch.set_default_device(device)
            torch.cuda.set_device(device)
            torch.manual_seed(0)
            random.seed(0)

            cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
            if varlen:
                for i in range(b):
                    cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
            total_seqlens = cache_seqlens.sum().item()
            max_seqlen = cache_seqlens.max().item()
            max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

            q = torch.randn(b, s_q, h_q, d)
            block_table = torch.arange(
                b * max_seqlen_pad // block_size, dtype=torch.int32
            ).view(b, max_seqlen_pad // block_size)
            blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
            for i in range(b):
                blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item() :] = (
                    float("nan")
                )
            blocked_v = blocked_k[..., :dv]

            tile_scheduler_metadata, num_splits = get_mla_metadata(
                cache_seqlens, s_q * h_q // h_kv, h_kv
            )

            return (
                q,
                blocked_k,
                block_table,
                cache_seqlens,
                dv,
                tile_scheduler_metadata,
                num_splits,
                causal,
            )
        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def flash_mla(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal,
        ):
            return flash_mla_with_kvcache(
                q,
                blocked_k,
                block_table,
                cache_seqlens,
                dv,
                tile_scheduler_metadata,
                num_splits,
                causal=causal,
            )

        return flash_mla(*self.inputs)
