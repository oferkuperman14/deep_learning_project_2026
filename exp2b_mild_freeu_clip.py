#!/usr/bin/env python3
"""
Exp2b: Mild FreeU sweep + CLIP score measurement (training-free)
----------------------------------------------------------------
- Keeps corrected 4-channel CFG split and FreeU closure binding
- Sweeps small FreeU multipliers (e.g. 1.02/0.98)
- Reports mean CLIP score over (classes x seeds)
"""

import os
import csv
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torchvision.transforms import Normalize

from diffusers import AutoencoderKL
from models import DiT_models
from download import find_model
from diffusion import create_diffusion

# -----------------------------
# Paths / device
# -----------------------------
OUTPUT_DIR = os.path.expanduser("~/deep_learning_project/outputs/exp2b_mild_freeu_clip")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Class prompts for CLIP
# -----------------------------
# ImageNet class ids used in the project
CLASS_PROMPTS = {
    207: "a photo of a golden retriever",
    360: "a photo of an otter",
    387: "a photo of a red panda",
    974: "a photo of a geyser",
}

# -----------------------------
# Schedules / CFG
# -----------------------------
def schedule_constant(t_float, s_max=4.0, **kwargs):
    return s_max

def schedule_cosine(t_float, s_max=4.0, s_min=1.0, t_max=1000.0, **kwargs):
    ratio = t_float / t_max
    w = 0.5 * (1.0 + math.cos(math.pi * (1.0 - ratio)))
    return s_min + (s_max - s_min) * w

SCHEDULES = {
    "constant": schedule_constant,
    "cosine": schedule_cosine,
}

def effective_cfg_scale(t_tensor, base_s=4.0, schedule_name="constant", interval=None, t_max=1000.0):
    t_float = float(t_tensor[0].item()) if torch.is_tensor(t_tensor) else float(t_tensor)
    s = SCHEDULES[schedule_name](t_float, s_max=base_s, s_min=1.0, t_max=t_max)
    if interval is not None:
        t_low, t_high = interval
        if not (t_low <= t_float <= t_high):
            s = 1.0
    return float(s)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# -----------------------------
# FreeU controller (closure-safe)
# -----------------------------
class FreeUController:
    def __init__(self, b1=1.0, b2=1.0):
        self.b1 = float(b1)
        self.b2 = float(b2)
        self._orig = []

    def apply(self, model):
        blocks = model.blocks if hasattr(model, "blocks") else model.module.blocks
        for block in blocks:
            orig_forward = block.forward

            def make_patched(orig_fn, blk, b1=self.b1, b2=self.b2):
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

            block.forward = make_patched(orig_forward, block)
            self._orig.append((block, orig_forward))

    def restore(self):
        for block, orig_forward in self._orig:
            block.forward = orig_forward
        self._orig.clear()

# -----------------------------
# Guided model (4-channel split)
# -----------------------------
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

        C = 4  # DiT latent channels
        eps, rest = model_out[:, :C], model_out[:, C:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + s * (cond_eps - uncond_eps)
        eps_out = torch.cat([guided_eps, guided_eps], dim=0)
        return torch.cat([eps_out, rest], dim=1) if rest.numel() > 0 else eps_out

# -----------------------------
# CLIP metric
# -----------------------------
class CLIPScorer:
    """
    Uses OpenAI CLIP via open_clip if available, else torchvision CLIP-ish fallback through open_clip only.
    Install once if needed:
        pip install open_clip_torch
    """
    def __init__(self, device=DEVICE):
        import open_clip
        self.device = device
        # ViT-B-32 is light and standard for CLIP score reporting
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(device).eval()
        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def score_batch(self, images_minus1_1: torch.Tensor, class_ids):
        """
        images_minus1_1: (B,3,H,W) in [-1,1] as produced by VAE decode path before save_image normalize
        returns list of float CLIP scores (cosine * 100)
        """
        # to [0,1]
        imgs = ((images_minus1_1.clamp(-1, 1) + 1.0) * 0.5)
        # resize to 224
        imgs = F.interpolate(imgs, size=(224, 224), mode="bicubic", align_corners=False)
        imgs = (imgs - self.image_mean) / self.image_std

        text = [CLASS_PROMPTS[int(c)] for c in class_ids]
        text_tokens = self.tokenizer(text).to(self.device)

        image_features = self.model.encode_image(imgs)
        text_features = self.model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # cosine similarity * 100 (common CLIP score scaling)
        sims = (image_features * text_features).sum(dim=-1) * 100.0
        return [float(x) for x in sims.cpu()]

# -----------------------------
# Sampling
# -----------------------------
@torch.no_grad()
def run_sample(guided_model, vae, diffusion, class_labels, latent_size=32, seed=42):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    z = torch.randn(len(class_labels), 4, latent_size, latent_size, device=DEVICE)
    y = torch.tensor(class_labels, device=DEVICE)
    z = torch.cat([z, z], dim=0)
    y_null = torch.tensor([1000] * len(class_labels), device=DEVICE)
    y_input = torch.cat([y, y_null], dim=0)

    if DEVICE == "cuda":
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
            progress=False,
            device=DEVICE,
        )

    samples, _ = samples.chunk(2, dim=0)
    decoded = vae.decode(samples / 0.18215).sample  # ~[-1,1]

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3) if DEVICE == "cuda" else 0.0
    return decoded, dt, vram

# -----------------------------
# Experiment matrix
# -----------------------------
def get_experiment_matrix():
    return [
        dict(name="B0_constant", schedule="constant", interval=None, b1=1.0, b2=1.0),
        dict(name="B12_interval_cosine", schedule="cosine", interval=(200, 800), b1=1.0, b2=1.0),
        dict(name="P_cosine_full_1.05_0.95", schedule="cosine", interval=None, b1=1.05, b2=0.95),
        # mild FreeU alone
        dict(name="F_1.02_0.98", schedule="constant", interval=None, b1=1.02, b2=0.98),
        dict(name="F_1.05_0.95", schedule="constant", interval=None, b1=1.05, b2=0.95),
        dict(name="F_1.08_0.92", schedule="constant", interval=None, b1=1.08, b2=0.92),
        dict(name="F_1.10_0.90", schedule="constant", interval=None, b1=1.10, b2=0.90),
        # mild FreeU + interval + cosine  (Package A candidates)
        dict(name="P_intcos_1.02_0.98", schedule="cosine", interval=(200, 800), b1=1.02, b2=0.98),
        dict(name="P_intcos_1.05_0.95", schedule="cosine", interval=(200, 800), b1=1.05, b2=0.95),
        dict(name="P_intcos_1.08_0.92", schedule="cosine", interval=(200, 800), b1=1.08, b2=0.92),
    ]

def main():
    print("=" * 64)
    print(" Exp2b: Mild FreeU + CLIP score")
    print("=" * 64)

    # ---- knobs ----
    image_size = 256
    latent_size = image_size // 8
    class_labels = [207, 360, 387, 974]
    seeds = [42, 43, 44, 45]   # use [42, 43] for a faster first pass
    num_steps = 250
    base_s = 4.0

    # optional fast pass via env:
    #   FAST=1 python exp2b_mild_freeu_clip.py
    if os.environ.get("FAST", "0") == "1":
        seeds = [42, 43]
        print("[Info] FAST=1 -> using seeds", seeds)

    # model / vae / diffusion
    base_model = DiT_models["DiT-XL/2"](input_size=latent_size).to(DEVICE)
    state = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    base_model.load_state_dict(state)
    base_model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
    diffusion = create_diffusion(str(num_steps))

    print("[Info] Loading CLIP...")
    clip_scorer = CLIPScorer(device=DEVICE)

    # CSV outputs
    per_image_csv = os.path.join(OUTPUT_DIR, "per_image_scores.csv")
    summary_csv = os.path.join(OUTPUT_DIR, "summary.csv")

    per_image_rows = []
    summary_rows = []

    exps = get_experiment_matrix()
    for i, cfg in enumerate(exps, 1):
        print(f"\n[{i}/{len(exps)}] {cfg['name']}  b1={cfg['b1']} b2={cfg['b2']}  schedule={cfg['schedule']} interval={cfg['interval']}")
        scores = []
        times = []
        vrams = []

        # collect images across seeds for a contact sheet
        all_imgs = []

        for seed in seeds:
            freeu = FreeUController(b1=cfg["b1"], b2=cfg["b2"])
            freeu.apply(base_model)

            guided = GuidedModel(
                model=base_model,
                base_s=base_s,
                schedule_name=cfg["schedule"],
                interval=cfg["interval"],
            ).to(DEVICE)

            samples, dt, vram = run_sample(
                guided, vae, diffusion, class_labels, latent_size=latent_size, seed=seed
            )
            freeu.restore()

            times.append(dt)
            vrams.append(vram)

            # CLIP scores
            clip_scores = clip_scorer.score_batch(samples, class_labels)
            scores.extend(clip_scores)

            for cls, sc in zip(class_labels, clip_scores):
                per_image_rows.append({
                    "name": cfg["name"],
                    "seed": seed,
                    "class_id": cls,
                    "prompt": CLASS_PROMPTS[cls],
                    "clip_score": round(sc, 4),
                    "b1": cfg["b1"],
                    "b2": cfg["b2"],
                    "schedule": cfg["schedule"],
                    "interval": str(cfg["interval"]),
                    "time_sec": round(dt, 2),
                })
                print(f"  seed={seed} class={cls} CLIP={sc:.2f}")

            # save per-seed grid
            grid_path = os.path.join(OUTPUT_DIR, f"{cfg['name']}_seed{seed}.png")
            save_image(samples, grid_path, nrow=2, normalize=True, value_range=(-1, 1))
            all_imgs.append(samples.cpu())

        # summary stats
        import statistics as stats
        mean_clip = sum(scores) / len(scores)
        std_clip = stats.stdev(scores) if len(scores) > 1 else 0.0
        mean_t = sum(times) / len(times)
        mean_v = sum(vrams) / len(vrams)

        summary_rows.append({
            "name": cfg["name"],
            "b1": cfg["b1"],
            "b2": cfg["b2"],
            "schedule": cfg["schedule"],
            "interval": str(cfg["interval"]),
            "n_images": len(scores),
            "clip_mean": round(mean_clip, 4),
            "clip_std": round(std_clip, 4),
            "time_sec_mean": round(mean_t, 2),
            "peak_vram_gb_mean": round(mean_v, 3),
        })
        print(f"  >> CLIP mean±std = {mean_clip:.3f} ± {std_clip:.3f} | time≈{mean_t:.1f}s | VRAM≈{mean_v:.2f}GB")

        # contact sheet over seeds (first seed already saved; write multi-seed strip if desired)
        # save a single summary grid from seed 42 only is enough; keep files tidy

    # write CSVs
    with open(per_image_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # pretty print ranking
    ranking = sorted(summary_rows, key=lambda r: r["clip_mean"], reverse=True)
    print("\n" + "=" * 64)
    print(" RANKING by mean CLIP score")
    print("=" * 64)
    for r in ranking:
        print(f"{r['clip_mean']:.3f} ± {r['clip_std']:.3f}  |  {r['name']}")

    with open(os.path.join(OUTPUT_DIR, "ranking.json"), "w") as f:
        json.dump(ranking, f, indent=2)

    print(f"\nSaved:\n  {per_image_csv}\n  {summary_csv}\n  grids in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()