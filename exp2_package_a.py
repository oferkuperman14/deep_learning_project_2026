#!/usr/bin/env python3
"""
Experiment 2 / Package A (Fully Corrected)
------------------------------------------
Fixes:
1. Python closure bug in FreeU patching (explicit blk=block binding).
2. Correct 4-channel latent CFG splitting (:4 eps, 4: variance).
"""

import os
import csv
import time
import math
import torch
import torch.nn as nn
from torchvision.utils import save_image

from diffusers import AutoencoderKL
from models import DiT_models
from download import find_model
from diffusion import create_diffusion


OUTPUT_DIR = os.path.expanduser("~/deep_learning_project/outputs/exp2_package_a")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1) s(t) Schedules
# =========================
def schedule_constant(t_float, s_max=4.0, **kwargs):
    return s_max

def schedule_linear(t_float, s_max=4.0, s_min=1.0, t_max=1000.0, **kwargs):
    ratio = t_float / t_max
    return s_min + (s_max - s_min) * ratio

def schedule_cosine(t_float, s_max=4.0, s_min=1.0, t_max=1000.0, **kwargs):
    ratio = t_float / t_max
    w = 0.5 * (1.0 + math.cos(math.pi * (1.0 - ratio)))
    return s_min + (s_max - s_min) * w

def schedule_late_ramp(t_float, s_max=4.0, s_min=1.0, t_max=1000.0, **kwargs):
    ratio = 1.0 - (t_float / t_max)
    return s_min + (s_max - s_min) * ratio

SCHEDULES = {
    "constant": schedule_constant,
    "linear": schedule_linear,
    "cosine": schedule_cosine,
    "late_ramp": schedule_late_ramp,
}

def effective_cfg_scale(t_tensor, base_s=4.0, schedule_name="constant", interval=None, t_max=1000.0):
    t_float = float(t_tensor[0].item()) if torch.is_tensor(t_tensor) else float(t_tensor)
    sched_fn = SCHEDULES[schedule_name]
    s = sched_fn(t_float, s_max=base_s, s_min=1.0, t_max=t_max)
    if interval is not None:
        t_low, t_high = interval
        if not (t_low <= t_float <= t_high):
            s = 1.0
    return float(s)


# =========================
# 2) FreeU Controller (Fixed Closure)
# =========================
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FreeUController:
    def __init__(self, b1=1.0, b2=1.0):
        self.b1 = float(b1)
        self.b2 = float(b2)
        self._orig = []

    def apply(self, model):
        blocks = model.blocks if hasattr(model, "blocks") else model.module.blocks
        for block in blocks:
            orig_forward = block.forward

            def make_patched(orig_fn, blk):
                def patched_forward(x, c):
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        blk.adaLN_modulation(c).chunk(6, dim=1)
                    )
                    attn_out = blk.attn(modulate(blk.norm1(x), shift_msa, scale_msa))
                    x = x + self.b1 * gate_msa.unsqueeze(1) * attn_out

                    mlp_out = blk.mlp(modulate(blk.norm2(x), shift_mlp, scale_mlp))
                    x = x + self.b2 * gate_mlp.unsqueeze(1) * mlp_out
                    return x
                return patched_forward

            block.forward = make_patched(orig_forward, block)
            self._orig.append((block, orig_forward))

    def restore(self):
        for block, orig_forward in self._orig:
            block.forward = orig_forward
        self._orig.clear()


# =========================
# 3) CFG Wrapper (Correct 4-Channel Latent Split)
# =========================
class GuidedModel(nn.Module):
    def __init__(self, model, base_s=4.0, schedule_name="constant", interval=None, t_max=1000.0):
        super().__init__()
        self.model = model
        self.base_s = base_s
        self.schedule_name = schedule_name
        self.interval = interval
        self.t_max = t_max

    def forward(self, x, t, y):
        return self.model.forward(x, t, y)

    def forward_with_cfg(self, x, t, y, cfg_scale=None):
        s = effective_cfg_scale(
            t_tensor=t,
            base_s=self.base_s,
            schedule_name=self.schedule_name,
            interval=self.interval,
            t_max=self.t_max,
        )

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.model.forward(combined, t, y)

        # DiT latent space is 4 channels. Output has 8 channels (4 eps + 4 var).
        C = 4
        eps, rest = model_out[:, :C], model_out[:, C:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + s * (cond_eps - uncond_eps)
        eps_out = torch.cat([guided_eps, guided_eps], dim=0)

        return torch.cat([eps_out, rest], dim=1) if rest.numel() > 0 else eps_out


# =========================
# 4) Sampling & Matrix
# =========================
@torch.no_grad()
def run_sample(guided_model, vae, diffusion, class_labels, latent_size=32, seed=42):
    torch.manual_seed(seed)
    z = torch.randn(len(class_labels), 4, latent_size, latent_size, device=DEVICE)
    y = torch.tensor(class_labels, device=DEVICE)

    z = torch.cat([z, z], dim=0)
    y_null = torch.tensor([1000] * len(class_labels), device=DEVICE)
    y_input = torch.cat([y, y_null], dim=0)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        samples = diffusion.p_sample_loop(
            guided_model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=dict(y=y_input, cfg_scale=4.0),
            progress=True,
            device=DEVICE,
        )

    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    torch.cuda.synchronize()
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
    return samples, dt, vram


def get_experiment_matrix():
    exps = [
        dict(name="B0_constant_cfg", schedule="constant", interval=None, b1=1.0, b2=1.0, base_s=4.0),
        dict(name="B1_interval_only", schedule="constant", interval=(200, 800), b1=1.0, b2=1.0, base_s=4.0),
    ]
    for sched in ["linear", "cosine", "late_ramp"]:
        exps.append(dict(name=f"B2_schedule_{sched}", schedule=sched, interval=None, b1=1.0, b2=1.0, base_s=4.0))
    for b1, b2 in [(1.1, 0.9), (1.2, 0.9), (1.1, 1.0)]:
        exps.append(dict(name=f"B3_freeu_b1_{b1}_b2_{b2}", schedule="constant", interval=None, b1=b1, b2=b2, base_s=4.0))
    exps.append(dict(name="B12_interval_plus_cosine", schedule="cosine", interval=(200, 800), b1=1.0, b2=1.0, base_s=4.0))
    for b1, b2 in [(1.1, 0.9), (1.2, 0.9)]:
        exps.append(dict(name=f"B123_full_interval_cosine_freeu_{b1}_{b2}", schedule="cosine", interval=(200, 800), b1=b1, b2=b2, base_s=4.0))
    return exps


def main():
    print("=" * 64)
    print(" Package A: Corrected DiT-XL/2 (BF16) ")
    print(f" Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'}")
    print("=" * 64)

    image_size = 256
    latent_size = image_size // 8
    class_labels = [207, 360, 387, 974]
    num_steps = 250
    seed = 42

    base_model = DiT_models["DiT-XL/2"](input_size=latent_size).to(DEVICE)
    state = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    base_model.load_state_dict(state)
    base_model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
    diffusion = create_diffusion(str(num_steps))

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "schedule", "interval", "b1", "b2", "base_s", "time_sec", "peak_vram_gb", "image_path"]
        )
        writer.writeheader()

        exps = get_experiment_matrix()
        for i, cfg in enumerate(exps, 1):
            print(f"[{i}/{len(exps)}] {cfg['name']}...")
            freeu = FreeUController(b1=cfg["b1"], b2=cfg["b2"])
            freeu.apply(base_model)

            guided = GuidedModel(
                model=base_model, base_s=cfg["base_s"], schedule_name=cfg["schedule"], interval=cfg["interval"]
            ).to(DEVICE)

            samples, dt, vram = run_sample(guided, vae, diffusion, class_labels, latent_size, seed)
            freeu.restore()

            img_path = os.path.join(OUTPUT_DIR, f"{cfg['name']}.png")
            save_image(samples, img_path, nrow=2, normalize=True, value_range=(-1, 1))

            writer.writerow({
                "name": cfg["name"], "schedule": cfg["schedule"], "interval": str(cfg["interval"]),
                "b1": cfg["b1"], "b2": cfg["b2"], "base_s": cfg["base_s"],
                "time_sec": round(dt, 2), "peak_vram_gb": round(vram, 3), "image_path": img_path
            })
            f.flush()
            print(f"  Saved {img_path} ({dt:.2f}s, {vram:.2f} GB)")


if __name__ == "__main__":
    main()