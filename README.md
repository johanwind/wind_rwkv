# Wind RWKV
A repository with optimized kernels for [RWKV](https://github.com/BlinkDL/RWKV-LM/) language models. Currently focused on RWKV-7.

## Kernel benchmarks
The kernels were timed using [tests/speed_test.py](tests/speed_test.py) with modeldim 4096 and varying (batch size, head size, seqlence length) as labeled in the table.
| Kernel  | (8,64,4096) | (8,128,4096) | (8,256,4096) | (1,256,32768) | Peak VRAM[^1] | Typical error |
|:----------------------------|------:|-------:|-------:|--------:|--------:|-----:|
| Chunked bf16                |  8 ms |  11 ms |  54 ms |  224 ms |  5 - 8 GB | 5e-3 |
| Backstepping fp32 smallhead | 17 ms | 101 ms | 862 ms | 1802 ms | 7 - 13 GB | 9e-5 |
| Backstepping fp32 longhead  | 23 ms |  46 ms |  80 ms |  124 ms | 8 - 14 GB | 9e-5 |
| Triton bighead fp32         | 66 ms |  87 ms | 168 ms | 1175 ms | 6 - 12 GB | 5e-5 |
| Triton bighead bf16         |   [^2]|  29 ms |  59 ms |  358 ms | 6 - 12 GB | 5e-3 |
| FLA chunk_rwkv7             | 64 ms |  62 ms |  89 ms |   93 ms |12 - 13 GB | 4e-3 |
[^1]: Smallest peak VRAM was typically for (8,64,4096) and largest for (8,256,4096).
[^2]: Triton fails to compile the kernel, only seen on H100. 


## Kernel descriptions
### [Chunked bf16](wind_rwkv/rwkv7/chunked_cuda/chunked_cuda.cu)
This is the fastest kernel when applicable. It processes the sequence in chunks of length 16 (chunked formulation) and uses Ampere (SM80, i.e., A100 and later) instructions for fast bfloat16 matmuls.
### [Backstepping fp32 smallhead](wind_rwkv/rwkv7/backstepping_smallhead/backstepping_smallhead.cu)
This is essentially the [official](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/cuda/wkv7_cuda.cu) kernel which was used to train the [RWKV-7 World models](https://huggingface.co/BlinkDL/rwkv-7-world). Calculates gradients by iterating the state backwards in time (max 15 steps). This makes the code simple, but requires 32-bit floats and limits the decay to ca. 0.5.
### [Backstepping fp32 longhead](wind_rwkv/rwkv7/backstepping_longhead/backstepping_longhead.cu)
Backstepping fp32 smallhead becomes very slow for large head sizes, since the full state is kept in registers, which overflow into global memory. To fix this, backstepping fp32 longhead uses the observation that the columns of the state are essentially updated independently. So it processes blocks of 64 or 32 columns indepdently. This increasing parallelization, and keeps less state in shared memory at a time, while keeping most of the simplicity of backstepping fp32 smallhead.
### [Triton bighead](wind_rwkv/rwkv7/triton_bighead.py)
A simple chunked kernel written in triton. The kernel stores intermediate states in global memory instead of shared memory, so it handles large head sizes (like 1024) without crashing. It takes a flag to choose fp32 or bf16 precision[^3] which affects all matmuls inside the triton kernel.
### [FLA chunk_rwkv7](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/chunk.py)
RWKV-7 triton kernel from [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention). Chunked implementation with partial sequence length parallelization.

[^3]: The kernel also supports tf32 precision for matmuls, but tf32 seems to run into bugs in the triton language, so I didn't expose it. 
