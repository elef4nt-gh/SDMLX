"""LUA -- Latent Upscale Adapter for diffusion models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from pathlib import Path

from .swinir_arch import SwinIRMultiHead

__all__ = ["SwinIRMultiHead", "load_model", "upscale_latent"]

# Default HuggingFace repository for LUA weights
HF_REPO_ID = "vaskers5/LUA-FLUX"
HF_FILENAME = "lua_flux.pth"

# Model configuration for the released FLUX checkpoint
_FLUX_MODEL_CONFIG = dict(
    in_chans=16,
    img_size=32,
    window_size=16,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=360,
    num_heads=[12, 12, 12, 12, 12, 12],
    mlp_ratio=2,
    resi_connection="1conv",
    primary_head="x4",
    head_num_feat=256,
    heads=[
        {"name": "x2", "scale": 2, "out_chans": 16},
        {"name": "x4", "scale": 4, "out_chans": 16, "primary": True},
    ],
)


def load_model(
    weights_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SwinIRMultiHead:
    """Load the LUA model with pretrained weights.

    Args:
        weights_path: Path to local checkpoint. If ``None``, downloads from
            HuggingFace (``vaskers5/LUA-FLUX``).
        device: Device to load the model onto.
        dtype: Data type for inference (``torch.float32`` recommended).

    Returns:
        Loaded ``SwinIRMultiHead`` model in eval mode.
    """
    if weights_path is None:
        weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    weights_path = str(weights_path)

    model = SwinIRMultiHead(**_FLUX_MODEL_CONFIG)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    key = "params_ema" if "params_ema" in ckpt else "params"
    model.load_state_dict(ckpt[key], strict=True)
    model.eval().to(device=device, dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[LUA] Loaded ({n_params:.1f}M params, heads: {list(model.heads.keys())})")
    return model


@torch.inference_mode()
def upscale_latent(
    model: SwinIRMultiHead,
    latent: torch.Tensor,
    head: str = "x2",
) -> torch.Tensor:
    """Upscale a VAE latent tensor using LUA.

    Args:
        model: Loaded LUA model.
        latent: VAE-space latent of shape ``(B, 16, H, W)``.
        head: Which upscaling head to use (``"x2"`` or ``"x4"``).

    Returns:
        Upscaled latent of shape ``(B, 16, H*scale, W*scale)``.
    """
    x = latent.to(dtype=torch.float32, device=next(model.parameters()).device)
    ws = model.window_size
    _, _, h, w = x.shape
    ph = (ws - h % ws) % ws
    pw = (ws - w % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), "reflect")
    sc = int(model.head_scales[head])
    out = model.forward_single_head(x, head)
    _, _, oh, ow = out.shape
    return out[:, :, : oh - ph * sc, : ow - pw * sc]
