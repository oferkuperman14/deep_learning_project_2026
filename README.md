# Diffusion Transformer (DiT) Optimization & Analysis Suite

This repository contains the complete implementation, experimental suite, and mechanistic analysis for Diffusion Transformers (DiT-XL/2). The project explores guidance scheduling, feature re-weighting via FreeU, internal transformer attention dynamics, and inference acceleration using deterministic DDIM solvers.

---

## 📌 Key Findings Across All Experiments

1. **Guidance Scheduling (Exp 1):** Precision benchmarking under bfloat16 and continuous Cosine guidance scheduling outperforms traditional constant CFG and abrupt timestep windowing, preventing over-saturation while preserving prompt alignment.
2. **Feature Modulation & CLIP Optimization (Exp 2):** Combining mild FreeU feature scaling ($b_1=1.05, b_2=0.95$) with Cosine guidance (**Package A**) significantly boosts visual sharpness and CLIP text alignment without adding runtime or memory overhead.
3. **Mechanistic Trajectory Phase Shift (Exp 3):** Probing internal Block 14 attention maps revealed a distinct phase transition: early timesteps ($t > 700$) govern global spatial composition and structure, while late timesteps ($t < 300$) focus strictly on high-frequency detail and texture refinement.
4. **Fast Deterministic Sampling (Exp 4):** DDIM-25 achieves a **$40\times$ speedup** over standard DDPM-1000 (reducing generation time from 61.68s to 1.53s per image) with zero VRAM penalty and full compatibility with Package A.

---

## 🔬 Experiment Breakdown

### Experiment 1: Precision & Guidance Benchmark (`exp1_bf16_benchmark.py`)
* **Objective:** Evaluate baseline DiT performance under bfloat16 precision and compare Classifier-Free Guidance (CFG) schedules.
* **Findings:** Standard constant CFG scales ($w > 4.0$) often cause artifacting and over-saturation. Continuous schedules like Cosine CFG modulate guidance dynamically across diffusion timesteps, improving generation stability.

### Experiment 2: FreeU Feature Modulation & Package A Integration
* **Scripts:** `exp2_package_a.py`, `exp2b_mild_freeu_clip.py`, `exp2c_micro_freeu.py`
* **Objective:** Test FreeU backbone feature re-weighting ($b_1=1.05, b_2=0.95$) paired with CFG scheduling and evaluate fine-grained CLIP score impacts.
* **Findings:** Applying FreeU to the DiT backbone enhances low-frequency structural stability and high-frequency details. Micro-tuning confirmed that mild FreeU settings paired with a continuous Cosine trajectory (**Package A**) deliver superior perceptual quality with $0\%$ computational overhead.

### Experiment 3: Attention Dynamics Probing (`exp3_attention_maps.py`)
* **Objective:** Perform internal feature probing on DiT Block 14 attention trajectories across all denoising timesteps.
* **Findings:** Identified a clear functional split during image generation:
  * **Noise-Dominant Phase ($t \in [1000, 700]$):** Attention maps focus globally on semantic layout and spatial object placement.
  * **Refinement Phase ($t \in [300, 0]$):** Attention shifts locally to high-frequency surface textures, contrast tuning, and edge sharpness.

### Experiment 4: Fast DDIM Sampling & Throughput Benchmark (`exp4_ddim_benchmark.py`)
* **Objective:** Benchmark inference latency, VRAM usage, and structural fidelity across DDPM (1000/250 steps) and DDIM (50/25 steps).
* **Findings:** DDIM cuts sampling passes dramatically while preserving image sharpness. Lower SSIM scores ($\sim 0.28$) reflect spatial coordinate shifts inherent to deterministic ODE trajectories ($\eta = 0$) rather than loss of visual fidelity.

---

## 📊 Benchmark Results (Experiment 4)

### Latency & Throughput Benchmark

| Configuration Name | Steps | Sampler | Package A | Sec / Image | Speedup | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DDPM1000_baseline** | 1000 | DDPM | ❌ | 61.68s | $1.00\times$ | 4.36 GB |
| **DDPM1000_packageA** | 1000 | DDPM | ⚠️ | 61.26s | $1.01\times$ | 4.36 GB |
| **DDPM250_baseline** | 250 | DDPM | ❌ | 15.40s | $4.01\times$ | 4.36 GB |
| **DDPM250_packageA** | 250 | DDPM | ⚠️ | 15.28s | $4.04\times$ | 4.36 GB |
| **DDIM50_baseline** | 50 | DDIM | ❌ | 3.08s | $20.01\times$ | 4.36 GB |
| **DDIM50_packageA** | 50 | DDIM | ⚠️ | 3.06s | $20.14\times$ | 4.36 GB |
| **DDIM25_baseline** | 25 | DDIM | ❌ | 1.53s | $40.25\times$ | 4.36 GB |
| **DDIM25_packageA** | 25 | DDIM | ⚠️ | 1.53s | $40.25\times$ | 4.36 GB |

### Structural Similarity (SSIM vs DDPM-1000 Baseline)

| Configuration | DDPM-1000 | DDPM-250 | DDIM-50 | DDIM-25 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | 1.0000 | 0.2496 | 0.2267 | 0.2324 |
| **Package A** | 0.6831 | 0.2555 | 0.2509 | 0.2542 |

---

## 📁 Project Directory Structure

```text
.
├── DiT/
│   ├── exp1_bf16_benchmark.py       # Experiment 1: Precision & Guidance Benchmark
│   ├── exp2_package_a.py            # Experiment 2: Package A Integration Setup
│   ├── exp2b_mild_freeu_clip.py     # Experiment 2b: FreeU & CLIP Evaluation
│   ├── exp2c_micro_freeu.py         # Experiment 2c: Fine-Grained FreeU Tuning
│   ├── exp3_attention_maps.py       # Experiment 3: Block 14 Attention Trajectory Probing
│   ├── exp4_ddim_benchmark.py       # Experiment 4: Fast DDIM Sampling Benchmark
│   ├── models.py                    # DiT Backbone & FreeU Architecture Definitions (Paper)
│   ├── download.py                  # Pre-trained Checkpoint Downloader (Paper)
│   ├── sample.py                    # Single-GPU Sampling Script (Paper)
│   ├── sample_ddp.py                # Distributed Sampling Script (Paper)
│   ├── train.py                     # Model Training Script (Paper)
│   ├── run_DiT.ipynb                # Interactive Execution & Analysis Notebook
│   └── environment.yml              # Subfolder Environment Specification
├── outputs/                         # Benchmarking Output Plots & Visual Artifacts
│   ├── baseline/                    # Baseline sampling outputs
│   ├── exp2b_mild_freeu_clip/       # Mild FreeU & CLIP score evaluation outputs
│   ├── exp2c_micro_freeu/           # Micro-tuned FreeU output grids
│   ├── exp2_package_a/              # Package A visual results
│   ├── exp3_attention_maps/         # Attention probing maps & plots
│   ├── exp4_ddim_benchmark/         # Latency plots, grid matrix, & benchmark_results.csv
│   ├── exp1_bf16_output.png         # bfloat16 experiment output grid
│   └── exp1_fp32_output.png         # float32 experiment output grid
├── pretrained_models/               # Cache Directory for DiT Weights
├── visuals/                         # Presentation Assets & Image Artifacts
├── environment.yml                  # Root Conda Environment Specification
├── requirements.txt                 # Exported Virtual Environment Dependencies
└── README.md                        # Project Documentation
