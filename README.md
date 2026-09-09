# Diffusion Transformer (DiT) Sampling Efficiency & Acceleration Analysis

This repository contains the official implementation and benchmark suite for evaluating sampling strategies, acceleration techniques, and guidance methods on Diffusion Transformers (DiT-XL/2). The project systematically compares stochastic **DDPM** against deterministic **DDIM** solvers while evaluating the runtime overhead and perceptual quality of advanced guidance packages (Full Cosine Guidance + FreeU).

---

## 📌 Key Findings

* **$40\times$ Inference Acceleration:** DDIM with 25 steps cuts per-image generation latency from **61.68s down to 1.53s** compared to standard DDPM-1000.
* **Zero Runtime Overhead:** Applying **Package A** (Full Cosine CFG guidance + FreeU feature re-weighting with $b_1=1.05, b_2=0.95$) incurs $0\%$ latency or VRAM penalty across all sampling modes.
* **Metric Realignment:** Lower pixel-wise SSIM ($\sim0.28$) observed under DDIM is caused by deterministic spatial trajectory shifts ($\eta = 0$) rather than visual degradation; structural fidelity and edge definition remain preserved.

---

## 📊 Benchmark Summary Results

### Latency & Throughput Comparison

| Configuration Name | Sampling Steps | Deterministic (DDIM) | Package A Applied | Latency / Img (s) | Speedup Factor | Peak VRAM (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DDPM1000_baseline** | 1000 | ❌ | ❌ | 61.68s | $1.00\times$ | 4.36 GB |
| **DDPM1000_packageA** | 1000 | ❌ | ⚠️ | 61.26s | $1.01\times$ | 4.36 GB |
| **DDPM250_baseline** | 250 | ❌ | ❌ | 15.40s | $4.01\times$ | 4.36 GB |
| **DDPM250_packageA** | 250 | ❌ | ⚠️ | 15.28s | $4.04\times$ | 4.36 GB |
| **DDIM50_baseline** | 50 | ⚠️ | ❌ | 3.08s | $20.01\times$ | 4.36 GB |
| **DDIM50_packageA** | 50 | ⚠️ | ⚠️ | 3.06s | $20.14\times$ | 4.36 GB |
| **DDIM25_baseline** | 25 | ⚠️ | ❌ | 1.53s | $40.25\times$ | 4.36 GB |
| **DDIM25_packageA** | 25 | ⚠️ | ⚠️ | 1.53s | $40.25\times$ | 4.36 GB |

### Perceptual & Structural Fidelity Matrix (SSIM vs DDPM-1000)

| Configuration | DDPM-1000 | DDPM-250 | DDIM-50 | DDIM-25 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | 1.0000 | 0.2496 | 0.2267 | 0.2324 |
| **Package A** | 0.6831 | 0.2555 | 0.2509 | 0.2542 |

---

## 📁 Repository Structure

```text
.
├── DiT/
│   ├── exp4_ddim_benchmark.py    # Main DDPM vs DDIM sampling benchmark script
│   └── merge pics.py             # Utility script to compile comparison grids
├── outputs/
│   └── exp4_ddim_benchmark/
│       ├── benchmark_results.csv           # Raw performance numbers & SSIM/PSNR metrics
│       ├── latency_benchmark_chart.png     # Latency & Speedup comparison plots
│       └── grid_comparison_matrix.png      # Qualitative visual outputs across models
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation
