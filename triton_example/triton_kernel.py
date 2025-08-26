import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(
    a_ptr, b_ptr, c_ptr, M, stride_a, stride_b, stride_c, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    a_ptrs = a_ptr + offsets * stride_a
    b_ptrs = b_ptr + offsets * stride_b

    a = tl.load(a_ptrs, mask=offsets < M, other=0.0)
    b = tl.load(b_ptrs, mask=offsets < M, other=0.0)

    c = a + b

    c_ptrs = c_ptr + offsets * stride_c
    tl.store(c_ptrs, c, mask=offsets < M)


def vector_add(input1: torch.Tensor, input2: torch.Tensor, block_size: int = 1024):
    assert input1.is_cuda and input2.is_cuda
    assert input1.ndim == 1 and input2.ndim == 1 and input1.size(0) == input2.size(0)

    vector_size = input1.size(0)
    output = torch.zeros_like(input1)

    block_size = triton.next_power_of_2(block_size)
    num_programs = triton.cdiv(vector_size, block_size)
    programs_shape = (num_programs,)

    compiled_kernel = _vector_add_kernel[(programs_shape)](
        input1,
        input2,
        output,
        vector_size,
        input1.stride(0),
        input2.stride(0),
        output.stride(0),
        block_size,
    )

    return output


if __name__ == "__main__":
    torch.set_default_device("cuda")
    a = torch.rand((512,))
    b = torch.rand((512,))

    c = vector_add(a, b)

    # print(a[:5], b[:5], c[:5])
