#!/usr/bin/env python3
"""
Exp3: DiT Block 14 Self-Attention Map Evolution Across Timesteps
Demonstrates Global Semantic Layout (high t) -> Local Texture (low t).

Critical implementation notes:
- create_diffusion("250") is SpacedDiffusion: loop indices are 0..249
- DiT model must receive ORIGINAL timesteps via diffusion.timestep_map
- diffusion.p_sample must receive SPACED indices 0..249
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from diffusers import AutoencoderKL
from models import DiT_models
from download import find_model
from diffusion import create_diffusion

OUTPUT_DIR = os.path.expanduser("~/deep_learning_project/outputs/exp3_attention_maps")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

captured_attentions = {}  # keys = ideal labels {900,700,...}


def make_attn_capture_forward(attn_module):
    """Explicit attention forward so we can record full attn maps (not fused SDPA-only)."""
    def forward(x):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B, N, 3, attn_module.num_heads, C // attn_module.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = (q.shape[-1]) ** -0.5
        attn_scores = (q @ k.transpose(-2, -1)) * scale
        attn_probs = attn_scores.softmax(dim=-1)

        cap = getattr(attn_module, "_capture_t", None)
        if cap is not None:
            # avg over heads -> (B, N, N)
            captured_attentions[cap] = attn_probs.mean(dim=1).detach().float().cpu()

        out = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        out = attn_module.proj(out)
        out = attn_module.proj_drop(out)
        return out

    return forward


def main():
    print("=" * 64)
    print(" Exp 3: Attention Map Evolution Across Timesteps ")
    print("=" * 64)

    image_size = 256
    latent_size = image_size // 8

    model = DiT_models["DiT-XL/2"](input_size=latent_size).to(DEVICE)
    state = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state)
    model.eval()
    model.to(dtype=DTYPE)

    # Hook Block 14
    block_idx = 14
    target_attn = model.blocks[block_idx].attn
    target_attn.forward = make_attn_capture_forward(target_attn)
    target_attn._capture_t = None

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE, dtype=DTYPE).eval()
    diffusion = create_diffusion("250")

    # SpacedDiffusion: timestep_map[i] = original t in [0, 999]
    if not hasattr(diffusion, "timestep_map"):
        raise RuntimeError(
            "Expected SpacedDiffusion with timestep_map. "
            "Check create_diffusion('250')."
        )
    timestep_map = list(diffusion.timestep_map)
    num_steps = diffusion.num_timesteps  # 250
    assert len(timestep_map) == num_steps, (
        f"timestep_map length {len(timestep_map)} != num_timesteps {num_steps}"
    )

    # Spaced indices high -> low
    spaced_indices = list(range(num_steps))[::-1]

    # Ideal display labels
    target_labels = [900, 700, 500, 300, 100]

    # Map each label -> best spaced index (match on ORIGINAL t)
    capture_at_idx = {}
    used_idx = set()
    for label in target_labels:
        closest_idx = min(
            range(len(timestep_map)),
            key=lambda i: abs(int(timestep_map[i]) - label),
        )
        if closest_idx in used_idx:
            ranked = sorted(
                range(len(timestep_map)),
                key=lambda i: abs(int(timestep_map[i]) - label),
            )
            for cand in ranked:
                if cand not in used_idx:
                    closest_idx = cand
                    break
        used_idx.add(closest_idx)
        capture_at_idx[closest_idx] = label

    print("[Exp3] Capture plan (spaced_idx -> original_t -> label):")
    for idx in sorted(capture_at_idx.keys(), reverse=True):
        print(
            f"  idx={idx:3d}  original_t={int(timestep_map[idx]):3d}  "
            f"label={capture_at_idx[idx]}"
        )

    class_id = 387  # red panda
    seed = 42
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    z = torch.randn(1, 4, latent_size, latent_size, device=DEVICE, dtype=DTYPE)
    y = torch.tensor([class_id], device=DEVICE)
    y_null = torch.tensor([1000], device=DEVICE)
    z_combined = torch.cat([z, z], dim=0)
    y_combined = torch.cat([y, y_null], dim=0)
    cfg_scale = 4.0

    print("[Exp3] Running denoising trajectory + capturing attention...")
    with torch.no_grad():
        for t_idx in spaced_indices:
            t_orig = int(timestep_map[t_idx])

            # Capture under ideal LABEL for clean plot titles
            target_attn._capture_t = capture_at_idx.get(t_idx, None)

            # Model needs ORIGINAL timestep
            t_model = torch.full((2,), t_orig, device=DEVICE, dtype=torch.long)

            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                model_out = model(z_combined, t_model, y_combined)

            # Official DiT 4-channel CFG split
            C = 4
            eps, rest = model_out[:, :C], model_out[:, C:]
            cond_eps, uncond_eps = eps.chunk(2, dim=0)
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            model_out_guided = torch.cat(
                [guided_eps, rest.chunk(2, dim=0)[0]], dim=1
            )

            # Diffusion step needs SPACED index
            t_diff = torch.full((1,), t_idx, device=DEVICE, dtype=torch.long)
            out = diffusion.p_sample(
                lambda *args, **kwargs: model_out_guided,
                z,
                t_diff,
                clip_denoised=False,
            )
            z = out["sample"]
            z_combined = torch.cat([z, z], dim=0)

        # Explicitly cast back to DTYPE
        decoded = vae.decode((z / 0.18215).to(dtype=DTYPE)).sample
        decoded = (decoded.clamp(-1, 1) + 1.0) / 2.0
        sample_path = os.path.join(OUTPUT_DIR, "final_red_panda_sample.png")
        save_image(decoded, sample_path)

    print(f"[Exp3] Sample saved: {sample_path}")
    print(f"[Exp3] Captured labels: {sorted(captured_attentions.keys())}")

    if len(captured_attentions) < len(target_labels):
        print(
            f"[Exp3][WARN] Only captured {len(captured_attentions)}/"
            f"{len(target_labels)} maps. Plot may have empty panels."
        )

    # -------- Plot --------
    # Patch (8, 8) center token index = 8 * 16 + 8
    center_token_idx = 8 * 16 + 8
    # Exact center pixel position for patch (8, 8) in 256x256 image space
    patch_center_px = 136

    fig, axes = plt.subplots(1, len(target_labels) + 1, figsize=(18, 3.5))

    img_np = decoded[0].cpu().float().permute(1, 2, 0).numpy()
    axes[0].imshow(np.clip(img_np, 0, 1))
    axes[0].plot([patch_center_px], [patch_center_px], "r+", markersize=12, markeredgewidth=2)
    axes[0].set_title("Generated Image\n(Query Patch Marked +)", fontsize=10, pad=8)
    axes[0].axis("off")

    for col_idx, label in enumerate(target_labels, start=1):
        ax = axes[col_idx]
        if label not in captured_attentions:
            ax.set_title(f"t ≈ {label}\n(missing)", fontsize=10, pad=8)
            ax.axis("off")
            continue

        attn_matrix = captured_attentions[label][0]  # cond batch
        query_attn = attn_matrix[center_token_idx].reshape(16, 16).numpy()
        qmin, qmax = query_attn.min(), query_attn.max()
        query_attn = (query_attn - qmin) / (qmax - qmin + 1e-8)

        im = ax.imshow(query_attn, cmap="viridis", interpolation="bilinear")
        stage = "Global Layout" if label >= 500 else "Local Texture"
        ax.set_title(f"t ≈ {label}\n({stage})", fontsize=10, pad=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "DiT-XL/2 Block 14 Self-Attention Evolution Across Timesteps",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "attention_evolution_block14.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Exp3] Heatmap figure saved: {plot_path}")
    print("[Exp3] Done.")


if __name__ == "__main__":
    main()