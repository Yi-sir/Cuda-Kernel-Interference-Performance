from typing import Optional, Union

import torch
import vllm._custom_ops as ops
from vllm.utils import round_up

DEFAULT_BLOCK_SHAPE = [128, 128]


def per_block_cast_to_int8(
    x: torch.Tensor,
    block_shape: list[int] = DEFAULT_BLOCK_SHAPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_m, block_n = block_shape
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (round_up(m, block_m), round_up(n, block_n)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (256.0 / x_amax)).to(torch.int8)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 256.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def per_block_cast_to_fp8(
    x: torch.Tensor,
    block_shape: list[int] = DEFAULT_BLOCK_SHAPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_m, block_n = block_shape
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (round_up(m, block_m), round_up(n, block_n)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def moe_quantize_weights(
    w: torch.Tensor,
    w_s: Optional[torch.Tensor],
    quant_dtype: Optional[torch.dtype],
    per_token_quant: bool,
    block_shape: Optional[list[int]],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert (
        quant_dtype == torch.float8_e4m3fn or quant_dtype == torch.int8
    ), "only fp8/int8 supported"

    if block_shape is not None:
        assert not per_token_quant
        if quant_dtype == torch.int8:
            w, w_s = per_block_cast_to_int8(w, block_shape)
        else:
            w, w_s = per_block_cast_to_fp8(w, block_shape)
    else:
        if quant_dtype == torch.int8:
            w, w_s = ops.scaled_int8_quant(
                w, w_s, use_per_token_if_dynamic=per_token_quant
            )
        else:
            w, w_s = ops.scaled_fp8_quant(
                w, w_s, use_per_token_if_dynamic=per_token_quant
            )

    return w, w_s


def make_test_weight(
    e: int,
    rows: int,
    cols: int,
    in_dtype: torch.dtype = torch.bfloat16,
    quant_dtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
    per_act_token_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    w_16 = torch.randn((e, rows, cols), device="cuda", dtype=in_dtype) / 15

    if quant_dtype is not None:
        w_l = [None] * e
        w_s_l = [None] * e
        for idx in range(e):
            w_l[idx], w_s_l[idx] = moe_quantize_weights(
                w_16[idx], None, quant_dtype, per_act_token_quant, block_shape
            )

        w = torch.stack(w_l)
        w_s = torch.stack(w_s_l)
        if w_s.ndim == 2:
            assert w_s.shape[-1] == 1
            w_s = w_s.view(-1, 1, 1)

        if block_shape is not None:
            block_n, block_k = block_shape
            n_tiles = (rows + block_n - 1) // block_n
            k_tiles = (cols + block_k - 1) // block_k
            assert w_s.shape == (e, n_tiles, k_tiles)
    else:
        w = w_16
        w_s = None

    return w_16, w, w_s


def make_test_weights(
    e: int,
    n: int,
    k: int,
    in_dtype: torch.dtype = torch.bfloat16,
    quant_dtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
    per_act_token_quant: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    return (
        *make_test_weight(
            e, 2 * n, k, in_dtype, quant_dtype, block_shape, per_act_token_quant
        ),
        *make_test_weight(
            e, k, n, in_dtype, quant_dtype, block_shape, per_act_token_quant
        ),
    )


def native_per_token_group_quant_fp8(
    x, group_size, eps=1e-10, dtype=torch.float8_e4m3fn
):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch."""
    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` must " "be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


def native_w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
    compute_type: torch.dtype = torch.float32,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization using native torch.
    It is agnostic to the input data type and can be used for both int8 and
    fp8 data types.

    It takes two input tensors `A` and `B` (int8) with scales `As` and
    `Bs` (float32).
    The output is returned in the specified `output_dtype`.
    """
    A = A.to(compute_type)
    B = B.to(compute_type)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N,)
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0], f"{n_tiles} == {Bs.shape[0]}"
    assert k_tiles == Bs.shape[1], f"{k_tiles} == {Bs.shape[1]}"

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=compute_type, device=A.device)

    A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [
        [
            B[
                j * block_n : min((j + 1) * block_n, N),
                i * block_k : min((i + 1) * block_k, K),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C
