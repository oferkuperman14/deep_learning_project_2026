#!/usr/bin/env python3
"""
Experiment 4: Fast DDIM Sampling Benchmark (DDPM-1000 vs DDPM-250 vs DDIM-50 vs DDIM-25)
Compares Baseline vs Package A (Full Cosine + FreeU 1.05/0.95, No Interval Windowing)
Optimized with GPU Warmup, Single Model Lifetime, Perceptual Metrics, and Visual Analytics.
"""

import os
import csv
import gc
import time
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image

# Optional perceptual quality metrics import with safe fallback
try:
    from skimage.metrics import structural_similarity as ssim_func
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from diffusers import AutoencoderKL
from models import DiT_models
from download import find_model
from diffusion import create_diffusion

OUTPUT_DIR = os.path.expanduser("~/deep_learning_project/outputs/exp4_ddim_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def schedule_cosine(t_float, s_max=4.0, s_min=1.0, t_max=1000.0):
    ratio = t_float / t_max
    w = 0.5 * (1.0 + math.cos(math.pi * (1.0 - ratio)))
    return s_min + (s_max - s_min) * w


def effective_cfg_scale(t_tensor, base_s=4.0, use_package_a=False):
    if not use_package_a:
        return base_s
    t_float = float(t_tensor[0].item()) if torch.is_tensor(t_tensor) else float(t_tensor)
    # Full Cosine guidance schedule across the entire timestep range [0, 1000] (no interval windowing)
    s = schedule_cosine(t_float, s_max=base_s)
    return float(s)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FreeUController:
    def __init__(self, b1=1.05, b2=0.95):
        self.b1 = float(b1)
        self.b2 = float(b2)
        self._orig = []

    def apply(self, model):
        blocks = model.blocks if hasattr(model, "blocks") else model.module.blocks
        for block in blocks:
            orig_forward = block.forward

            def make_patched(blk, b1=self.b1, b2=self.b2):
                def patched_forward(x, c):
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        blk.adaLN_modulation(c).chunk(6, dim=1)
                    )
                    attn_out = blk.attn(modulate(blk.norm1(x), shift_msa, scale_msa))
                    x = x + b1 * gate_msa.unsqueeze(1) * attn_out
                    mlp_out = blk.mlp(modulate(blk.norm2(x), shift_mlp, scale_mlp))
                    x = x + b2 * gate_mlp.unsqueeze(1) * mlp_out
                    return x
                return patched_forward

            block.forward = make_patched(block)
            self._orig.append((block, orig_forward))

    def restore(self):
        for block, orig_forward in self._orig:
            block.forward = orig_forward
        self._orig.clear()


class GuidedModel(nn.Module):
    def __init__(self, model, base_s=4.0, use_package_a=False):
        super().__init__()
        self.model = model
        self.base_s = base_s
        self.use_package_a = use_package_a

    def forward(self, x, t, y):
        return self.model.forward(x, t, y)

    def forward_with_cfg(self, x, t, y, cfg_scale=None):
        s = effective_cfg_scale(t_tensor=t, base_s=self.base_s, use_package_a=self.use_package_a)

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.model.forward(combined, t, y)

        C = 4
        eps, rest = model_out[:, :C], model_out[:, C:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + s * (cond_eps - uncond_eps)
        eps_out = torch.cat([guided_eps, guided_eps], dim=0)
        return torch.cat([eps_out, rest], dim=1) if rest.numel() > 0 else eps_out


@torch.no_grad()
def run_benchmark_config(model, vae, class_labels, timestep_respacing, use_ddim, use_package_a, seed=42):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    diffusion = create_diffusion(timestep_respacing)
    guided = GuidedModel(model, base_s=4.0, use_package_a=use_package_a).to(DEVICE)

    z = torch.randn(len(class_labels), 4, 32, 32, device=DEVICE, dtype=DTYPE)
    y = torch.tensor(class_labels, device=DEVICE)
    z = torch.cat([z, z], dim=0)
    y_null = torch.tensor([1000] * len(class_labels), device=DEVICE)
    y_input = torch.cat([y, y_null], dim=0)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        if use_ddim:
            samples = diffusion.ddim_sample_loop(
                guided.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=dict(y=y_input, cfg_scale=4.0),
                progress=False,
                device=DEVICE,
            )
        else:
            samples = diffusion.p_sample_loop(
                guided.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=dict(y=y_input, cfg_scale=4.0),
                progress=False,
                device=DEVICE,
            )

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3) if DEVICE == "cuda" else 0.0

    samples, _ = samples.chunk(2, dim=0)
    
    vae_dtype = next(vae.parameters()).dtype
    decoded = vae.decode(samples.to(dtype=vae_dtype) / 0.18215).sample

    return decoded, dt, vram


# Dynamic Latency & Acceleration Chart Plotter
def generate_benchmark_plots(results, output_dir):
    df = pd.DataFrame(results)
    # Speedup relative to 1000-step baseline if present, otherwise 250-step
    ref_name = "DDPM1000_baseline" if "DDPM1000_baseline" in df["name"].values else "DDPM250_baseline"
    baseline_time = df.loc[df["name"] == ref_name, "sampling_time_sec"].values[0]
    df["speedup"] = baseline_time / df["sampling_time_sec"]

    num_bars = len(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, num_bars * 2), 5))
    
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_bars)]
    labels = [r["name"].replace("_", "\n") for r in results]

    # Subplot 1: Per-Image Latency
    bars1 = ax1.bar(labels, df["sec_per_image"], color=colors, edgecolor="black", width=0.55)
    ax1.set_ylabel("Latency per Image (s)", fontsize=11, fontweight="bold")
    ax1.set_title("Sampling Latency per Image", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Subplot 2: Acceleration Factor vs Baseline
    bars2 = ax2.bar(labels, df["speedup"], color=colors, edgecolor="black", width=0.55)
    ax2.set_ylabel(f"Speedup Factor vs {ref_name}", fontsize=11, fontweight="bold")
    ax2.set_title(f"Acceleration Relative to {ref_name}", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.2, f"{yval:.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "latency_benchmark_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[Exp4] Saved benchmark chart to: {chart_path}")


# Dynamic Matrix Grid Aggregator
def generate_matrix_grid(benchmarks, class_names, output_dir):
    col_titles = [b["name"].replace("_", " ") for b in benchmarks]
    fig, axes = plt.subplots(len(class_names), len(benchmarks), figsize=(3 * len(benchmarks), 12))

    for col_idx, bench in enumerate(benchmarks):
        img_path = os.path.join(output_dir, f"{bench['name']}.png")
        grid_img = Image.open(img_path)
        W, H = grid_img.size
        pad = 2
        w_sub = (W - 3 * pad) // 2
        h_sub = (H - 3 * pad) // 2

        # Slicing 2x2 grid image into 4 individual class crops
        crops = [
            grid_img.crop((pad, pad, pad + w_sub, pad + h_sub)),
            grid_img.crop((2 * pad + w_sub, pad, 2 * pad + 2 * w_sub, pad + h_sub)),
            grid_img.crop((pad, 2 * pad + h_sub, pad + w_sub, 2 * pad + 2 * h_sub)),
            grid_img.crop((2 * pad + w_sub, 2 * pad + h_sub, 2 * pad + 2 * w_sub, 2 * pad + 2 * h_sub))
        ]

        for row_idx, crop in enumerate(crops):
            ax = axes[row_idx, col_idx] if len(benchmarks) > 1 else axes[row_idx]
            ax.imshow(crop)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(class_names[row_idx], fontsize=11, fontweight="bold", labelpad=10)

    plt.tight_layout()
    matrix_path = os.path.join(output_dir, "grid_comparison_matrix.png")
    plt.savefig(matrix_path, dpi=300)
    plt.close()
    print(f"[Exp4] Saved matrix visual comparison to: {matrix_path}")


# Quantitative SSIM / PSNR Metrics Evaluator
def compute_quality_metrics(results, output_dir):
    if not HAS_SKIMAGE:
        print("[Exp4] Warning: scikit-image not found. Skipping SSIM/PSNR calculation.")
        return results

    # Determine reference ground truth (DDPM1000 baseline if available, else DDPM250 baseline)
    ref_name = "DDPM1000_baseline" if any(r["name"] == "DDPM1000_baseline" for r in results) else "DDPM250_baseline"
    ref_path = os.path.join(output_dir, f"{ref_name}.png")
    ref_arr = np.array(Image.open(ref_path))

    for row in results:
        test_path = os.path.join(output_dir, f"{row['name']}.png")
        test_arr = np.array(Image.open(test_path))

        val_ssim = ssim_func(ref_arr, test_arr, channel_axis=2)
        val_psnr = psnr_func(ref_arr, test_arr)

        row["ssim_vs_baseline"] = round(float(val_ssim), 4)
        row["psnr_db"] = round(float(val_psnr), 2)

    return results


def main():
    print("=" * 64)
    print(" Exp 4: Fast DDIM Sampling Benchmark ")
    print("=" * 64)

    image_size = 256
    latent_size = image_size // 8
    class_labels = [207, 360, 387, 974]
    class_names = ["Golden Retriever", "Otter", "Red Panda", "Geyser"]

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")

    print("[Exp4] Loading DiT-XL/2 backbone into memory...")
    model = DiT_models["DiT-XL/2"](input_size=latent_size).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    if DEVICE == "cuda":
        print("[Exp4] Executing CUDA Warmup Pass...")
        warmup_diffusion = create_diffusion("ddim10")
        dummy_z = torch.randn(2, 4, 32, 32, device=DEVICE, dtype=DTYPE)
        dummy_y = torch.tensor([100, 100], device=DEVICE)
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE), torch.no_grad():
            _ = warmup_diffusion.ddim_sample_loop(
                model.forward,
                dummy_z.shape,
                dummy_z,
                model_kwargs=dict(y=dummy_y),
                progress=False,
                device=DEVICE,
            )
        torch.cuda.synchronize()

    benchmarks = [
        # Full 1000-Step Ground Truth Reference
        dict(name="DDPM1000_baseline", spacing="1000", ddim=False, package_a=False),
        dict(name="DDPM1000_packageA", spacing="1000", ddim=False, package_a=True),
        # 250-Step Subsampled DDPM
        dict(name="DDPM250_baseline", spacing="250", ddim=False, package_a=False),
        dict(name="DDPM250_packageA", spacing="250", ddim=False, package_a=True),
        # Deterministic Fast DDIM
        dict(name="DDIM50_baseline", spacing="ddim50", ddim=True, package_a=False),
        dict(name="DDIM50_packageA", spacing="ddim50", ddim=True, package_a=True),
        dict(name="DDIM25_baseline", spacing="ddim25", ddim=True, package_a=False),
        dict(name="DDIM25_packageA", spacing="ddim25", ddim=True, package_a=True),
    ]

    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    results = []

    for bench in benchmarks:
        print(f"\n[Exp4] Running {bench['name']}...")

        freeu = None
        if bench["package_a"]:
            freeu = FreeUController(b1=1.05, b2=0.95)
            freeu.apply(model)

        images, dt, vram = run_benchmark_config(
            model=model,
            vae=vae,
            class_labels=class_labels,
            timestep_respacing=bench["spacing"],
            use_ddim=bench["ddim"],
            use_package_a=bench["package_a"],
            seed=42,
        )

        if freeu:
            freeu.restore()

        sec_per_img = dt / len(class_labels)
        img_path = os.path.join(OUTPUT_DIR, f"{bench['name']}.png")
        save_image(images, img_path, nrow=2, normalize=True, value_range=(-1, 1))

        row = {
            "name": bench["name"],
            "spacing": bench["spacing"],
            "ddim": bench["ddim"],
            "package_a": bench["package_a"],
            "sampling_time_sec": round(dt, 2),
            "sec_per_image": round(sec_per_img, 2),
            "peak_vram_gb": round(vram, 3),
        }
        results.append(row)
        print(f"  Sampling time: {dt:.2f}s ({sec_per_img:.2f}s/img) | Peak VRAM: {vram:.2f}GB")

        del images
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Post-Processing & Analytics Execution
    ref_time = results[0]["sampling_time_sec"]
    for r in results:
        r["speedup_factor"] = round(ref_time / r["sampling_time_sec"], 2)

    results = compute_quality_metrics(results, OUTPUT_DIR)
    generate_benchmark_plots(results, OUTPUT_DIR)
    generate_matrix_grid(benchmarks, class_names, OUTPUT_DIR)

    # Save full updated results to CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 80)
    print(" SUMMARY BENCHMARK RESULTS ")
    print("=" * 80)
    for r in results:
        ssim_str = f"{r.get('ssim_vs_baseline', 'N/A'):>6}"
        psnr_str = f"{r.get('psnr_db', 'N/A'):>5}"
        print(
            f"{r['name']:<20} | Time: {r['sampling_time_sec']:>6.2f}s | "
            f"Per-Img: {r['sec_per_image']:>5.2f}s | Speedup: {r['speedup_factor']:>5.2fx} | "
            f"VRAM: {r['peak_vram_gb']:>4.2f}GB | SSIM: {ssim_str} | PSNR: {psnr_str}dB"
        )

    print(f"\n[Exp4] Complete. Detailed outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()