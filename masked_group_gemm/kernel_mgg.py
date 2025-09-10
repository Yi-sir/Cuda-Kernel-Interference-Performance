import logging
import os
import sys

os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGL_DG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm")
)
import deep_gemm
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from kernel_base.kernel_base import KernelBase
from kernel_base.registry import register_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_kernel
class MaskedGroupGemm(KernelBase):

    _default_params = {
        "G": 32,
        "M": 1024,
        "N": 4096,
        "K": 7168,
        "active_experts_per_rank": 2,
        "expected_m": 1,
        "seed": 0
    }

    _kernel_name = "masked_group_gemm"
    _key = "void deep_gemm::sm90_fp8_gemm_1d2d_impl"

    def __init__(self, device: torch.device):
        super().__init__(device)

    def prepare_input(self):
        def prepare(G, M, N, K, active_experts_per_rank, expected_m, seed, device):
            input_dtype = torch.float8_e4m3fn
            output_dtype = torch.bfloat16
            factor_for_scale = 1e-2
            block_k = 128
            block_n = 128

            torch.set_default_device(device)
            torch.manual_seed(seed)

            fp8_info = torch.finfo(input_dtype)
            fp8_max, fp8_min = fp8_info.max, fp8_info.min

            A_fp32 = (torch.rand(G, M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
            A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(input_dtype)
            k_tiles = (K + block_k - 1) // block_k
            As = torch.rand(G, M, k_tiles, dtype=torch.float32) * factor_for_scale

            hidden_states_fp8 = (A_fp8, As)

            B_fp32 = (torch.rand(G, N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
            B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
            n_tiles = (N + block_n - 1) // block_n
            Bs = torch.rand(G, n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

            weight_fp8 = (B_fp8, Bs)

            output = torch.empty((G, M, N), dtype=torch.bfloat16)

            mask_m = torch.zeros(G, dtype=torch.int32)
            indices = torch.randperm(G)[:active_experts_per_rank]
            mask_m[indices] = expected_m

            return (hidden_states_fp8, weight_fp8, output, mask_m, expected_m)

        logger.debug(f"Prepare input for {self.__class__._kernel_name} >>>>>")
        logger.debug(f"Params are {self.params}")
        self.inputs = prepare(**self.params, device=self.device)

    def launch_kernel(self):
        def masked_group_gemm(hidden_states_fp8, weight_fp8, output, mask_m, expected_m):
            return deep_gemm.fp8_m_grouped_gemm_nt_masked(
                hidden_states_fp8,
                weight_fp8,
                output,
                mask_m,
                expected_m
            )
        return masked_group_gemm(*self.inputs)