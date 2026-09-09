import os
import time
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from models import DiT_models
from download import find_model

# Ensure output directory exists
OUTPUT_DIR = os.path.expanduser("~/deep_learning_project/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def measure_memory_and_time(
    model, vae, latent_size, class_labels, cfg_scale, num_steps, precision="fp32"
):
    """Runs DDPM sampling pass and measures latency + peak VRAM usage."""
    device = "cuda"

    # Reset max memory tracker
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    start_time = time.time()

    # Fixed seed for fair comparison
    torch.manual_seed(42)

    # Sample random noise
    z = torch.randn(
        len(class_labels), 4, latent_size, latent_size, device=device
    )
    y = torch.tensor(class_labels, device=device)

    # Setup Classifier-Free Guidance (CFG) inputs
    z = torch.cat([z, z], dim=0)
    y_null = torch.tensor([1000] * len(class_labels), device=device)
    y_input = torch.cat([y, y_null], dim=0)

    # DDPM Timesteps setup
    from diffusion import create_diffusion

    diffusion = create_diffusion(str(num_steps))

    # Inference execution block
    if precision == "bf16":
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=dict(y=y_input, cfg_scale=cfg_scale),
                    progress=True,
                    device=device,
                )
    else:
        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=dict(y=y_input, cfg_scale=cfg_scale),
                progress=True,
                device=device,
            )

    # Split CFG conditional samples
    samples, _ = samples.chunk(2, dim=0)

    # Decode latents with VAE
    with torch.no_grad():
        samples = vae.decode(samples / 0.18215).sample

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    return samples, elapsed_time, peak_vram_gb


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"  DiT-XL/2 Precision Benchmark (FP32 vs BF16)     ")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    # Load DiT-XL/2
    image_size = 256
    latent_size = image_size // 8
    model_name = "DiT-XL/2"

    print("\n[1/3] Loading DiT-XL/2 checkpoint...")
    model = DiT_models[model_name](input_size=latent_size).to(device)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval()

    print("[2/3] Loading VAE checkpoint...")
    vae = (
        AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        .to(device)
        .eval()
    )

    # Test ImageNet classes: Golden Retriever (207), Otter (360), Red Panda (387), Geyser (974)
    class_labels = [207, 360, 387, 974]
    num_steps = 250
    cfg_scale = 4.0

    print("\n[3/3] Running Precision Benchmark Sweeps...")

    # Run FP32 Benchmark
    print("\n---> Running FP32 (Standard Full Precision)...")
    samples_fp32, time_fp32, vram_fp32 = measure_memory_and_time(
        model, vae, latent_size, class_labels, cfg_scale, num_steps, "fp32"
    )
    save_image(
        samples_fp32,
        os.path.join(OUTPUT_DIR, "exp1_fp32_output.png"),
        nrow=2,
        normalize=True,
        value_range=(-1, 1),
    )

    # Run BF16 Benchmark
    print("\n---> Running BF16 (bfloat16 Mixed Precision)...")
    samples_bf16, time_bf16, vram_bf16 = measure_memory_and_time(
        model, vae, latent_size, class_labels, cfg_scale, num_steps, "bf16"
    )
    save_image(
        samples_bf16,
        os.path.join(OUTPUT_DIR, "exp1_bf16_output.png"),
        nrow=2,
        normalize=True,
        value_range=(-1, 1),
    )

    # Benchmark Results Summary
    speedup = (
        ((time_fp32 - time_bf16) / time_fp32) * 100 if time_fp32 > 0 else 0
    )
    vram_saved = (
        ((vram_fp32 - vram_bf16) / vram_fp32) * 100 if vram_fp32 > 0 else 0
    )

    print("\n==================================================")
    print("               EXPERIMENT 1 RESULTS               ")
    print("==================================================")
    print(
        f" FP32 | Time: {time_fp32:6.2f}s | Peak VRAM: {vram_fp32:5.2f} GB"
    )
    print(
        f" BF16 | Time: {time_bf16:6.2f}s | Peak VRAM: {vram_bf16:5.2f} GB"
    )
    print("--------------------------------------------------")
    print(f" VRAM Reduction : {vram_saved:.2f}%")
    print(f" Sampling Speedup: {speedup:.2f}%")
    print(f" Output images saved in: {OUTPUT_DIR}")
    print("==================================================")


if __name__ == "__main__":
    main()