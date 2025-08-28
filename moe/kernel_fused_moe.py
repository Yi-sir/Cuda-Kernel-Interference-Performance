import logging
import os
import random
import sys
import time

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.triton_utils import triton

from .utils import (make_test_weights, native_per_token_group_quant_fp8,
                       native_w8a8_block_matmul)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from kernel_base.kernel_base import KernelBase
from kernel_base.registry import register_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_kernel
class FusedMoE(KernelBase):

    _default_params = {
        "M": 128,  # batch size
        "N": 4608,  # for TP8(?)
        "K": 7168,  # hidden size
        "E": 32,
        "topk": 4,
        "block_size": [128, 128],
        "dtype": torch.bfloat16,
        "seed": 0,
    }

    _kernel_name = "fused_moe"
    _key = "fused_moe_kernel"

    def __init__(self, device: torch.device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(M, N, K, E, topk, block_size, dtype, seed, device):
            torch.set_default_device(device)
            torch.manual_seed(seed)

            os.environ["VLLM_FUSED_MOE_CHUNK_SIZE"] = "2048"

            a = torch.randn((M, K), dtype=dtype) / 10
            score = torch.randn((M, E), dtype=dtype)

            _, w1, w1_s, _, w2, w2_s = make_test_weights(
                E,
                N,
                K,
                dtype,
                torch.float8_e4m3fn,
                per_act_token_quant=False,
                block_shape=block_size,
            )

            topk_weights, topk_ids, _ = fused_topk(a, score.float(), topk, False)

            return (
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                w1_s,
                w2_s,
            )
        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        block_size = self.params["block_size"]
        def test_w8a8_block_fp8_fused_moe(
            a,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_s,
            w2_s,
            block_size,
        ):
            out = fused_experts(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                use_fp8_w8a8=True,
                w1_scale=w1_s,
                w2_scale=w2_s,
                block_shape=block_size,
            )

        return test_w8a8_block_fp8_fused_moe(*self.inputs, block_size=block_size)
