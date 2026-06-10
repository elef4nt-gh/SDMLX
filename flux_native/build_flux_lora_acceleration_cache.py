#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open


NATIVE_ROOT = Path(__file__).resolve().parent
if str(NATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(NATIVE_ROOT))

import native_flux_core  # noqa: E402


HIDDEN_DIM = native_flux_core.HIDDEN_DIM
MLP_DIM = native_flux_core.MLP_DIM


def flux_key_from_underscores(value: str) -> str:
    key = value.replace("_", ".")
    fixes = (
        ("single.transformer.blocks", "single_transformer_blocks"),
        ("transformer.blocks", "transformer_blocks"),
        ("double.blocks", "double_blocks"),
        ("single.blocks", "single_blocks"),
        ("img.attn", "img_attn"),
        ("txt.attn", "txt_attn"),
        ("img.mlp", "img_mlp"),
        ("txt.mlp", "txt_mlp"),
        ("to.q", "to_q"),
        ("to.k", "to_k"),
        ("to.v", "to_v"),
        ("to.out.0", "to_out.0"),
        ("to.add.out", "to_add_out"),
        ("add.q.proj", "add_q_proj"),
        ("add.k.proj", "add_k_proj"),
        ("add.v.proj", "add_v_proj"),
        ("proj.mlp", "proj_mlp"),
        ("norm1.context", "norm1_context"),
        ("norm.query.norm", "norm.query_norm"),
        ("norm.key.norm", "norm.key_norm"),
        ("query.norm", "query_norm"),
        ("key.norm", "key_norm"),
        ("x.embedder", "x_embedder"),
        ("context.embedder", "context_embedder"),
    )
    for source, target in fixes:
        key = key.replace(source, target)
    return key


def direct_flux_weight_key(base: str) -> str | None:
    key = base
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    if not key.endswith(".weight"):
        key = f"{key}.weight"
    key = native_flux_core.normalize_flux_weight_key(key)
    if key is None:
        return None
    native_prefixes = (
        "img_in.",
        "txt_in.",
        "time_in.",
        "vector_in.",
        "guidance_in.",
        "final_layer.",
        "double_blocks.",
        "single_blocks.",
    )
    if key.startswith(native_prefixes):
        return key
    return None


def diffusers_flux_target(base: str) -> tuple[str, int | None, int | None] | None:
    parts = base.split(".")
    if len(parts) < 2:
        return None

    if parts[0] == "transformer_blocks" and len(parts) >= 4:
        try:
            index = int(parts[1])
        except ValueError:
            return None
        tail = ".".join(parts[2:])
        qkv = {
            "attn.to_q": ("img_attn.qkv", 0, HIDDEN_DIM),
            "attn.to_k": ("img_attn.qkv", HIDDEN_DIM, HIDDEN_DIM),
            "attn.to_v": ("img_attn.qkv", HIDDEN_DIM * 2, HIDDEN_DIM),
            "attn.add_q_proj": ("txt_attn.qkv", 0, HIDDEN_DIM),
            "attn.add_k_proj": ("txt_attn.qkv", HIDDEN_DIM, HIDDEN_DIM),
            "attn.add_v_proj": ("txt_attn.qkv", HIDDEN_DIM * 2, HIDDEN_DIM),
        }
        if tail in qkv:
            target, start, length = qkv[tail]
            return f"double_blocks.{index}.{target}.weight", start, length
        block_map = {
            "attn.to_out.0": "img_attn.proj",
            "norm1.linear": "img_mod.lin",
            "norm1_context.linear": "txt_mod.lin",
            "attn.to_add_out": "txt_attn.proj",
            "ff.net.0.proj": "img_mlp.0",
            "ff.linear_in": "img_mlp.0",
            "ff.net.2": "img_mlp.2",
            "ff.linear_out": "img_mlp.2",
            "ff_context.net.0.proj": "txt_mlp.0",
            "ff_context.linear_in": "txt_mlp.0",
            "ff_context.net.2": "txt_mlp.2",
            "ff_context.linear_out": "txt_mlp.2",
        }
        if tail in block_map:
            return f"double_blocks.{index}.{block_map[tail]}.weight", None, None

    if parts[0] == "single_transformer_blocks" and len(parts) >= 3:
        try:
            index = int(parts[1])
        except ValueError:
            return None
        tail = ".".join(parts[2:])
        qkv_mlp = {
            "attn.to_q": (0, HIDDEN_DIM),
            "attn.to_k": (HIDDEN_DIM, HIDDEN_DIM),
            "attn.to_v": (HIDDEN_DIM * 2, HIDDEN_DIM),
            "proj_mlp": (HIDDEN_DIM * 3, MLP_DIM),
        }
        if tail in qkv_mlp:
            start, length = qkv_mlp[tail]
            return f"single_blocks.{index}.linear1.weight", start, length
        block_map = {
            "norm.linear": "modulation.lin",
            "proj_out": "linear2",
            "attn.to_qkv_mlp_proj": "linear1",
            "attn.to_out": "linear2",
        }
        if tail in block_map:
            return f"single_blocks.{index}.{block_map[tail]}.weight", None, None

    basic_map = {
        "x_embedder": "img_in",
        "context_embedder": "txt_in",
        "time_text_embed.timestep_embedder.linear_1": "time_in.in_layer",
        "time_text_embed.timestep_embedder.linear_2": "time_in.out_layer",
        "time_text_embed.text_embedder.linear_1": "vector_in.in_layer",
        "time_text_embed.text_embedder.linear_2": "vector_in.out_layer",
        "time_text_embed.guidance_embedder.linear_1": "guidance_in.in_layer",
        "time_text_embed.guidance_embedder.linear_2": "guidance_in.out_layer",
        "proj_out": "final_layer.linear",
    }
    if base in basic_map:
        return f"{basic_map[base]}.weight", None, None
    return None


def flux_lora_targets(prefix: str) -> list[tuple[str, int | None, int | None]]:
    bases = [prefix]
    if prefix.startswith("lora_unet_"):
        bases.append(flux_key_from_underscores(prefix[len("lora_unet_") :]))
    if prefix.startswith("lora_transformer_"):
        bases.append(flux_key_from_underscores(prefix[len("lora_transformer_") :]))
    if prefix.startswith("lycoris_"):
        bases.append(flux_key_from_underscores(prefix[len("lycoris_") :]))

    expanded: list[str] = []
    for base in bases:
        for strip in ("base_model.model.", "unet.", "transformer."):
            if base.startswith(strip):
                expanded.append(base[len(strip) :])
        expanded.append(base)

    targets: list[tuple[str, int | None, int | None]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for base in expanded:
        direct = direct_flux_weight_key(base)
        if direct is not None:
            item = (direct, None, None)
            if item not in seen:
                targets.append(item)
                seen.add(item)
        diffusers = diffusers_flux_target(base)
        if diffusers is not None and diffusers not in seen:
            targets.append(diffusers)
            seen.add(diffusers)
    return targets


def factor_array(handle: safe_open, key: str) -> np.ndarray | None:
    try:
        tensor = handle.get_tensor(key)
    except Exception:
        return None
    if not tensor.is_floating_point():
        return None
    return tensor.to(torch.float16).cpu().contiguous().numpy()


def alpha_value(handle: safe_open, prefix: str, rank: int) -> float:
    for suffix in (".alpha", ".lora_alpha", ".scale"):
        key = f"{prefix}{suffix}"
        if key in handle.keys():
            try:
                return float(handle.get_tensor(key).float().reshape(()).item())
            except Exception:
                return float(rank)
    return float(rank)


def iter_lora_pairs(lora_path: Path):
    with safe_open(str(lora_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        key_set = set(keys)
        down_suffixes = (".lora_down.weight", ".lora_A.weight")
        for key in keys:
            suffix = next((item for item in down_suffixes if key.endswith(item)), None)
            if suffix is None:
                continue
            prefix = key[: -len(suffix)]
            up_candidates = (
                f"{prefix}.lora_up.weight",
                f"{prefix}.lora_B.weight",
            )
            up_key = next((candidate for candidate in up_candidates if candidate in key_set), None)
            if up_key is None:
                yield prefix, None, None, 0.0
                continue
            down = factor_array(handle, key)
            up = factor_array(handle, up_key)
            if down is None or up is None:
                yield prefix, None, None, 0.0
                continue
            rank = int(down.shape[0])
            yield prefix, up, down, alpha_value(handle, prefix, rank)


def apply_lora_to_weight(
    weights: dict[str, mx.array],
    target_key: str,
    start: int | None,
    length: int | None,
    up: np.ndarray,
    down: np.ndarray,
    alpha: float,
    strength: float,
) -> bool:
    base = weights.get(target_key)
    if base is None or len(base.shape) != 2:
        return False
    rank = max(1, int(down.shape[0]))
    out_slice = int(up.shape[0])
    out_dim, in_dim = (int(base.shape[0]), int(base.shape[1]))
    if int(down.shape[1]) != in_dim or int(up.shape[1]) != rank:
        return False
    if start is not None:
        end = start + (length or out_slice)
        if end > out_dim or out_slice != end - start:
            return False
    elif out_slice != out_dim:
        return False

    scale = float(strength) * float(alpha) / float(rank)
    delta = (mx.array(up).astype(base.dtype) @ mx.array(down).astype(base.dtype)) * scale
    if start is None:
        patched = base + delta.astype(base.dtype)
    else:
        end = start + (length or out_slice)
        patched = base.at[start:end, :].add(delta.astype(base.dtype))
    weights[target_key] = patched
    mx.eval(patched)
    return True


def build_cache(
    base_model: Path,
    lora: Path,
    output: Path,
    strength: float,
    metadata: dict[str, str],
    *,
    quiet: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f"{output.stem}.tmp-{os.getpid()}.safetensors")
    if tmp_path.exists():
        tmp_path.unlink()

    t0 = time.perf_counter()
    weights, load_mode, fp8_weight_keys, fp8_mxfp8, gguf_affine = native_flux_core.load_flux_weights(
        base_model,
        mx.float16,
        fp8_mode="dequant",
    )
    if fp8_mxfp8 or fp8_weight_keys:
        raise RuntimeError("packed/native FP8 cache input was not dequantized as expected.")
    if gguf_affine:
        raise RuntimeError("GGUF affine cache input is not supported for acceleration-patch baking yet.")
    for key, value in list(weights.items()):
        if value.dtype in (mx.float16, mx.bfloat16, mx.float32):
            weights[key] = value.astype(mx.float16)
    mx.eval(weights)

    matched = 0
    skipped = 0
    unsupported = 0
    for prefix, up, down, alpha in iter_lora_pairs(lora):
        if up is None or down is None:
            unsupported += 1
            continue
        applied = False
        for target_key, start, length in flux_lora_targets(prefix):
            if apply_lora_to_weight(weights, target_key, start, length, up, down, alpha, strength):
                matched += 1
                applied = True
                break
        if not applied:
            skipped += 1

    save_metadata = {
        "format": "sdmlx-flux-acceleration-cache-v1",
        "base_model": str(base_model),
        "lora": str(lora),
        "strength": str(float(strength)),
        "base_load_mode": load_mode,
        "matched": str(matched),
        "skipped": str(skipped),
        "unsupported": str(unsupported),
    }
    save_metadata.update(metadata)
    mx.save_safetensors(str(tmp_path), weights, metadata=save_metadata)
    os.replace(tmp_path, output)
    if not quiet:
        print(
            "SDMLX FLUX Acceleration Cache: "
            f"built {output.name}, matched={matched}, skipped={skipped}, unsupported={unsupported}, "
            f"time={time.perf_counter() - t0:.2f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a merged SDMLX FLUX acceleration cache from a FLUX LoRA.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    build_cache(
        Path(args.base_model),
        Path(args.lora),
        Path(args.output),
        args.strength,
        json.loads(args.metadata_json),
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
