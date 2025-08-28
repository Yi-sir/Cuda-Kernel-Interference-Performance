# Cuda Kernel Interference Performance

## Example

```shell
cd green_ctx_test/interference
# if meet errors about cuda sync, try CUDA_LAUNCH_BLOCKING=1
python3 ./interference_test.py
```

![1](green_ctx_test/interference/mla_decode_triton_gemm.png)
![2](green_ctx_test/interference/mla_decode_fused_moe.png)
![3](green_ctx_test/interference/triton_gemm_fused_moe.png)