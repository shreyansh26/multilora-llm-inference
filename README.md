## Multi-LoRA Inference

This repository contains several implementations and test/benchmark scripts for high-throughput multi-LoRA inference on Llama-like decoder models. It includes reference baselines, Triton kernels, and a CUDA extension, plus end-to-end benchmarks over chat-style LLM inference workloads.

### Implementations
- **Baselines (decode and prefill microbenchmarks)** in `multilora_impl_prefill.py` (for prefill) and `multilora_impl.py` (for decode):
  - `lora_loop`: plain per-sample loop applying $y \mathrel{+}= x A_i B_i$.
  - `lora_cheat_bmm`: idealized batched variant assuming pre-grouped A/B per sample.
  - `lora_gbmm` (GBMM): groups adapters by index then uses batched matmuls.
  - `lora_bgmv_triton`: Implementing Batched Gather Matrix-Vector Multiplication (BGMV) in Triton. [[Punica Paper](https://arxiv.org/pdf/2310.18547)]
  - `lora_bgmv_cuda`: BGMV implemneted in CUDA and exposed as a PyTorch extension.
  - `lora_sgmv_triton`: Segmented Gather Matric-Vector Multiplication (SGMV) in Triton. [[Punica Paper](https://arxiv.org/pdf/2310.18547)]

- **BGMV**
  - Triton: `bgmv_triton/bgmv_triton.py` (+ experimental split-k variant and helpers).
  - CUDA: `bgmv_cuda/` contains the CUDA kernel (`bgmv_kernel.cuh`) and extension glue (`bgmv_ext.cu`, `bgmv.cu`).
  - Test files: `bgmv_triton/run_bgmv.py` for validation and benchmarking.

- **SGMV — Triton**
  - Shrink kernel: `sgmv_triton/sgmv_shrink_triton.py`
  - Expand kernel: `sgmv_triton/sgmv_expand_triton.py`
  - Torch reference: `sgmv_triton/sgmv_torch_sequential.py`
  - Test files: `sgmv_triton/run_sgmv_shrink.py` and `sgmv_triton/run_sgmv_expand.py` for validation and benchmarking.

### Microbenchmarks (decode vs prefill)
- `multilora_impl_prefill.py` (prefill path: sequence length n > 1)
  - Same suite as above but for prefill; includes an extra token dimension in the kernels.
  - Produces plot: `plots/prefill_lora_loop_vs_cheat_bmm_vs_gbmm_vs_bgmv_cuda_triton_vs_sgmv_triton.png`.

- `multilora_impl.py` (decode path: sequence length n = 1)
  - Benchmarks: `loop`, `cheat_bmm`, `gbmm`, `bgmv_cuda`, `bgmv_triton`, `sgmv_triton` over different batch sizes.
  - Produces plot: `plots/decode_lora_loop_vs_cheat_bmm_vs_gbmm_vs_bgmv_cuda_triton_vs_sgmv_triton.png`.

Run either directly, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 python multilora_impl.py
CUDA_VISIBLE_DEVICES=0 python multilora_impl_prefill.py
```

### End-to-end inference and benchmarks
- `run_inference_multilora.py`
  - Loads a base Llama model and injects multiple LoRA adapters (A/B) and scales.
  - Supports modes via `--lora-inference-mode`: `gbmm`, `bgmv_triton`, `bgmv_cuda`, `sgmv_triton`.

- `benchmark_inference_multilora.py`
  - Runs full text generation over a mixed batch with random per-sample LoRA indices.
  - Measures tokens/sec for each mode with warmup and consistent decoding parameters.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_inference_multilora.py \
  --modes gbmm bgmv_triton bgmv_cuda sgmv_triton \
  --max_new_tokens 512 --batch-size 64
```

### Benchmark results

Configuration excerpt:

```text
Number of LoRA adapters: 7
Batch size: 64
Max new tokens: 512
Temperature: 0.7
Top-p: 0.9
Modes: [gbmm, bgmv_triton, bgmv_cuda, sgmv_triton]
```

Summary:

| Mode          | Throughput (tok/s) | Time (s) | Total Tokens |
|---------------|--------------------|----------|--------------|
| gbmm          | 414.30             | 79.09    | 32768        |
| sgmv_triton   | 387.90             | 84.48    | 32768        |
| bgmv_cuda     | 362.13             | 90.49    | 32768        |
| bgmv_triton   | 353.44             | 92.71    | 32768        |

Speedup vs slowest (higher is better):

```text
gbmm         1.17x
sgmv_triton  1.10x
bgmv_cuda    1.02x
bgmv_triton  1.00x
```