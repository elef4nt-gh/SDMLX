import hashlib
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

SDMLX_IMPORT_ERROR = None

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    import numpy as np
    from transformers import CLIPTokenizer
    import torch
    from PIL import Image
except ModuleNotFoundError as exc:
    SDMLX_IMPORT_ERROR = exc

    class _MissingModule:
        float16 = "float16"
        float32 = "float32"

        class Module:
            pass

        def __getattr__(self, name):
            raise RuntimeError(
                "SDMLX runtime dependencies are unavailable. "
                "Install SDMLX on Apple Silicon with its package requirements."
            )

    mx = _MissingModule()
    np = _MissingModule()
    torch = _MissingModule()
    Image = _MissingModule()

    class _MissingNN(_MissingModule):
        Module = _MissingModule.Module

    nn = _MissingNN()

    class CLIPTokenizer:
        pass

    def tree_unflatten(*args, **kwargs):
        raise RuntimeError(
            "SDMLX runtime dependencies are unavailable. "
            "Install SDMLX on Apple Silicon with its package requirements."
        )

SDMLX_VERSION = "0.1.16"
SDMLX_CACHE_VERSION = "adapter-v7"

if SDMLX_IMPORT_ERROR is None:
    from .sdxl_adapter import (
        apply_mapped_weights,
        ldm_unet_key_to_diffusers,
        map_clip_g_weights,
        map_clip_l_weights,
        map_vae_weights_for_apple,
        split_sdxl_checkpoint,
        validate_sdxl_checkpoint_keys,
    )
    from .mlx_sd.model_io import map_unet_weights, map_vae_weights
    from .mlx_sd.controlnet_union import (
        ControlNetUnionModel,
        UNION_CONTROL_TYPES,
        map_controlnet_union_weights,
    )
else:
    UNION_CONTROL_TYPES = {
        "pose": 0,
        "depth": 1,
        "soft edge to scribble": 2,
        "line to canny": 3,
        "normal": 4,
        "segment": 5,
        "tile": 6,
        "repaint": 7,
    }

    def _missing_runtime(*args, **kwargs):
        raise RuntimeError(
            "SDMLX runtime dependencies are unavailable. "
            "Install SDMLX on Apple Silicon with its package requirements."
        )

    apply_mapped_weights = _missing_runtime
    ldm_unet_key_to_diffusers = _missing_runtime
    map_clip_g_weights = _missing_runtime
    map_clip_l_weights = _missing_runtime
    map_vae_weights_for_apple = _missing_runtime
    split_sdxl_checkpoint = _missing_runtime
    validate_sdxl_checkpoint_keys = _missing_runtime
    map_unet_weights = _missing_runtime
    map_vae_weights = _missing_runtime
    map_controlnet_union_weights = _missing_runtime

    class ControlNetUnionModel:
        pass

MODEL_CACHE = {}
MODEL_CACHE_META = {}
TOKENIZER_CACHE = {}
CONDITIONING_CACHE = {}
CONDITIONING_GUARD_CACHE = {}
COMPILED_STEP_DENOISERS = {}
COMPILED_VAE_DECODERS = {}
TAESD_PREVIEWER_CACHE = {}
SPEED_PATCH_FACTORS_CACHE = {}
LORA_MODULES_CACHE = {}
CONTROLNET_MODEL_CACHE = {}
IPADAPTER_MODEL_CACHE = {}
CLIP_VISION_MODEL_CACHE = {}
INSIGHTFACE_MODEL_CACHE = {}
def sdmlx_env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def sdmlx_env_value(name):
    return os.environ.get(name, "").strip().lower()


SDMLX_VERBOSE_LOGS = sdmlx_env_flag("SDMLX_VERBOSE")
SDMLX_CONDITIONING_DIAGNOSTICS_MODE = sdmlx_env_value("SDMLX_CONDITIONING_DIAGNOSTICS")
SDMLX_CONDITIONING_DIAGNOSTICS = SDMLX_CONDITIONING_DIAGNOSTICS_MODE in {
    "1",
    "true",
    "yes",
    "on",
    "basic",
    "full",
    "all",
}
SDMLX_SAFE_MODE = sdmlx_env_flag("SDMLX_SAFE_MODE")
SDMLX_DISABLE_STEP_COMPILE = SDMLX_SAFE_MODE or sdmlx_env_flag("SDMLX_DISABLE_STEP_COMPILE")
SDMLX_DISABLE_FAST_ATTENTION = SDMLX_SAFE_MODE or sdmlx_env_flag("SDMLX_DISABLE_FAST_ATTENTION")
SDMLX_CONDITIONING_GUARD = not sdmlx_env_flag("SDMLX_DISABLE_CONDITIONING_GUARD")
SDMLX_CONDITIONING_DIAGNOSTICS_HEADER_PRINTED = False
TIMING_LOGS_ENABLED = SDMLX_VERBOSE_LOGS
CONDITIONING_CACHE_VERSION = "shared-tokenizer-v2"
MEMORY_CACHE_POLICY = {
    "mode": "balanced",
    "reserve_gb": None,
}
MLX_MEMORY_LIMIT_STATE = {
    "cache_key": None,
    "original_cache_limit": None,
}
MEMORY_ASSIST_OPTIONS = ["auto", "max_performance", "low_memory", "off"]
SDMLX_SELECT_CHECKPOINT = ""

SDXL_SIZE_PRESETS = [
    "1024x1024",
    "1152x896",
    "896x1152",
    "1216x832",
    "832x1216",
    "1344x768",
    "768x1344",
    "1536x640",
    "640x1536",
]
SIZE_PRESETS = ["Custom"] + SDXL_SIZE_PRESETS
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "lcm"]
TILED_UPSCALE_SCALE_OPTIONS = ["1.5x", "2x", "3x", "4x", "custom"]
MASK_DETAILER_SCALE_OPTIONS = ["1.5x", "2x", "3x", "4x"]
HIRES_RESIZE_METHODS = ["lanczos", "bicubic", "bilinear"]
HIRES_MAX_PIXELS = 4_500_000
TILED_UPSCALE_MAX_PIXELS = 32_000_000
VAE_DECODE_MODES = ["auto", "full", "tiled"]
MASK_FEATHER_MODES = ["gaussian", "box"]
SPEED_PATCH_NONE = "None"
SPEED_PATCH_REPO_ID = "elef4nt/sdmlx-acceleration-patches"
KNOWN_SPEED_PATCHES = [
    "dmd2_sdxl_4step_lora_fp16.sdmlxpatch",
    "dmd2-lighting8step_cfg1.5.sdmlxpatch",
    "sdxl_lightning_4step_lora.sdmlxpatch",
    "sdxl_lightning_8step_lora.sdmlxpatch",
    "lcm-lora-sdxl.sdmlxpatch",
    "Hyper-SDXL-8steps-CFG-lora.sdmlxpatch",
    "Hyper-SDXL-12steps-CFG-lora.sdmlxpatch",
]
SPEED_PATCH_LABELS = {
    "dmd2_sdxl_4step_lora_fp16.sdmlxpatch": "DMD2 / 4-step fp16",
    "dmd2-lighting8step_cfg1.5.sdmlxpatch": "DMD2 / Lightning 8-step CFG 1.5",
    "sdxl_lightning_4step_lora.sdmlxpatch": "Lightning / 4-step",
    "sdxl_lightning_8step_lora.sdmlxpatch": "Lightning / 8-step",
    "lcm-lora-sdxl.sdmlxpatch": "LCM / SDXL",
    "Hyper-SDXL-8steps-CFG-lora.sdmlxpatch": "Hyper-SD / 8-step CFG",
    "Hyper-SDXL-12steps-CFG-lora.sdmlxpatch": "Hyper-SD / 12-step CFG",
}
SPEED_PATCH_BY_LABEL = {label: name for name, label in SPEED_PATCH_LABELS.items()}
SUPPORTED_SPEED_LORA_PLACEHOLDER = "Put a supported speed LoRA in models/loras"
SUPPORTED_SPEED_LORA_PATCHES = {
    "dmd2_sdxl_4step_lora_fp16.safetensors": {
        "package": "dmd2_sdxl_4step_lora_fp16.sdmlxpatch",
        "source_repo": "tianweiy/DMD2",
        "source_file": "dmd2_sdxl_4step_lora_fp16.safetensors",
        "license": "cc-by-nc-4.0 / model card mentions CC-BY-NC-SA-4.0",
        "recommendations": {"steps": 4, "cfg": 1.0, "sampler": "lcm", "scheduler": "normal", "force_no_cfg": True},
    },
    "dmd2-lighting8step_cfg1.5.safetensors": {
        "package": "dmd2-lighting8step_cfg1.5.sdmlxpatch",
        "source_repo": "tianweiy/DMD2",
        "source_file": "dmd2-lighting8step_cfg1.5.safetensors",
        "license": "cc-by-nc-4.0 / model card mentions CC-BY-NC-SA-4.0",
        "recommendations": {"steps": 8, "cfg": 1.5, "sampler": "lcm", "scheduler": "normal", "force_no_cfg": False},
    },
    "sdxl_lightning_4step_lora.safetensors": {
        "package": "sdxl_lightning_4step_lora.sdmlxpatch",
        "source_repo": "ByteDance/SDXL-Lightning",
        "source_file": "sdxl_lightning_4step_lora.safetensors",
        "license": "openrail++",
        "recommendations": {"steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "sgm_uniform", "force_no_cfg": True},
    },
    "sdxl_lightning_8step_lora.safetensors": {
        "package": "sdxl_lightning_8step_lora.sdmlxpatch",
        "source_repo": "ByteDance/SDXL-Lightning",
        "source_file": "sdxl_lightning_8step_lora.safetensors",
        "license": "openrail++",
        "recommendations": {"steps": 8, "cfg": 1.0, "sampler": "lcm", "scheduler": "normal", "force_no_cfg": True},
    },
    "lcm-lora-sdxl.safetensors": {
        "package": "lcm-lora-sdxl.sdmlxpatch",
        "source_repo": "latent-consistency/lcm-lora-sdxl",
        "source_file": "pytorch_lora_weights.safetensors",
        "license": "openrail++",
        "recommendations": {"steps": 8, "cfg": 1.0, "sampler": "lcm", "scheduler": "normal", "force_no_cfg": True},
    },
    "Hyper-SDXL-8steps-CFG-lora.safetensors": {
        "package": "Hyper-SDXL-8steps-CFG-lora.sdmlxpatch",
        "source_repo": "ByteDance/Hyper-SD",
        "source_file": "Hyper-SDXL-8steps-CFG-lora.safetensors",
        "license": "CreativeML Open RAIL++-M",
        "recommendations": {"steps": 8, "cfg": 5.0, "sampler": "lcm", "scheduler": "normal", "force_no_cfg": False},
    },
    "Hyper-SDXL-12steps-CFG-lora.safetensors": {
        "package": "Hyper-SDXL-12steps-CFG-lora.sdmlxpatch",
        "source_repo": "ByteDance/Hyper-SD",
        "source_file": "Hyper-SDXL-12steps-CFG-lora.safetensors",
        "license": "CreativeML Open RAIL++-M",
        "recommendations": {"steps": 12, "cfg": 7.5, "sampler": "lcm", "scheduler": "normal", "force_no_cfg": False},
    },
}
CONTROL_STRENGTH_EPSILON = 1e-6
CONTROLNET_PROMAX_AUTO = "Auto download Xinsir Union ProMax"
CONTROLNET_PROMAX_REPO_ID = "xinsir/controlnet-union-sdxl-1.0"
CONTROLNET_PROMAX_FILENAME = "diffusion_pytorch_model_promax.safetensors"
IPADAPTER_PLACEHOLDER = "Put IPAdapter models in models/ipadapter"
CLIP_VISION_PLACEHOLDER = "Put CLIP Vision models in models/clip_vision"
AUTO_CLIP_VISION_OPTION = "auto"
IPADAPTER_AUTO_PLUS_SDXL_VITH = "Auto download IP-Adapter Plus SDXL ViT-H"
IPADAPTER_AUTO_SDXL_VITH = "Auto download IP-Adapter SDXL ViT-H"
IPADAPTER_AUTO_PLUS_FACE_SDXL_VITH = "Auto download IP-Adapter Plus Face SDXL ViT-H"
IPADAPTER_AUTO_FACEID_SDXL = "Auto download FaceID SDXL"
IPADAPTER_AUTO_FACEID_PLUSV2_SDXL = "Auto download FaceID PlusV2 SDXL"
CLIP_VISION_AUTO_VITH = "Auto download CLIP-ViT-H-14"
IPADAPTER_DOWNLOAD_SPECS = {
    IPADAPTER_AUTO_PLUS_SDXL_VITH: {
        "repo_id": "h94/IP-Adapter",
        "filename": "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        "local_name": "ip-adapter-plus_sdxl_vit-h.safetensors",
    },
    IPADAPTER_AUTO_SDXL_VITH: {
        "repo_id": "h94/IP-Adapter",
        "filename": "sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
        "local_name": "ip-adapter_sdxl_vit-h.safetensors",
    },
    IPADAPTER_AUTO_PLUS_FACE_SDXL_VITH: {
        "repo_id": "h94/IP-Adapter",
        "filename": "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "local_name": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
    },
    IPADAPTER_AUTO_FACEID_SDXL: {
        "repo_id": "h94/IP-Adapter-FaceID",
        "filename": "ip-adapter-faceid_sdxl.bin",
        "local_name": "ip-adapter-faceid_sdxl.bin",
        "lora": {
            "repo_id": "h94/IP-Adapter-FaceID",
            "filename": "ip-adapter-faceid_sdxl_lora.safetensors",
            "local_name": "ip-adapter-faceid_sdxl_lora.safetensors",
        },
    },
    IPADAPTER_AUTO_FACEID_PLUSV2_SDXL: {
        "repo_id": "h94/IP-Adapter-FaceID",
        "filename": "ip-adapter-faceid-plusv2_sdxl.bin",
        "local_name": "ip-adapter-faceid-plusv2_sdxl.bin",
        "lora": {
            "repo_id": "h94/IP-Adapter-FaceID",
            "filename": "ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
            "local_name": "ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
        },
    },
}
CLIP_VISION_DOWNLOAD_SPECS = {
    CLIP_VISION_AUTO_VITH: {
        "repo_id": "h94/IP-Adapter",
        "filename": "models/image_encoder/model.safetensors",
        "local_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    },
}
LORA_MERGE_VERSION = 2
SEED_MAX = 0xffffffffffffffff
IPADAPTER_WEIGHT_TYPES = [
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "strong style and composition",
]
LORA_SCHEDULE_MODES = ["blend in", "blend out", "bell"]
LORA_SCHEDULE_CURVES = [
    "linear",
    "progressive",
    "progressive fast",
    "degressive",
    "degressive fast",
    "s-curve",
    "positive",
    "negative",
]
IPADAPTER_EMBEDS_SCALING = [
    "V only",
    "K+V",
    "K+V w/ C penalty",
    "K+mean(V) w/ C penalty",
]
SDMLX_IPADAPTER_CONTEXT = {
    "adapters": [],
    "step_percent": 0.0,
    "use_cfg": False,
}
SDMLX_LORA_CONTEXT = {
    "step_percent": 0.0,
}


def sdmlx_gelu_mode():
    env_value = os.environ.get("SDMLX_GELU_MODE", "").strip().lower()
    if env_value in {"exact", "approx", "fast"}:
        return env_value
    return "fast"


SDMLX_GELU_MODE = sdmlx_gelu_mode()


def sdmlx_gelu(x):
    if SDMLX_GELU_MODE == "fast":
        return nn.gelu_fast_approx(x)
    if SDMLX_GELU_MODE == "approx":
        return nn.gelu_approx(x)
    return nn.gelu(x)


class FusedGEGLUFFN(nn.Module):
    def __init__(self, linear1, linear2, linear3):
        super().__init__()
        hidden_dims, input_dims = linear1.weight.shape
        self.hidden_dims = hidden_dims
        self.linear12 = nn.Linear(input_dims, hidden_dims * 2)
        self.linear12.weight = mx.concatenate([linear1.weight, linear2.weight], axis=0)
        if "bias" in linear1 and "bias" in linear2:
            self.linear12.bias = mx.concatenate([linear1.bias, linear2.bias], axis=0)
        self.linear3 = linear3

    def __call__(self, x):
        y = self.linear12(x)
        y_a, y_b = mx.split(y, 2, axis=-1)
        return self.linear3(y_a * sdmlx_gelu(y_b))


class FastFFNTransformerBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1 = source.norm1
        self.attn1 = source.attn1
        self.norm2 = source.norm2
        self.attn2 = source.attn2
        self.norm3 = source.norm3
        self.ffn = FusedGEGLUFFN(source.linear1, source.linear2, source.linear3)

    def __call__(self, x, memory, attn_mask=None, memory_mask=None):
        y = self.norm1(x)
        x = x + self.attn1(y, y, y, attn_mask)

        y = self.norm2(x)
        x = x + self.attn2(y, memory, memory, memory_mask)

        y = self.norm3(x)
        return x + self.ffn(y)


class FusedSelfAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        dims, input_dims = source.query_proj.weight.shape
        self.num_heads = source.num_heads
        self.qkv_proj = nn.Linear(input_dims, dims * 3, bias="bias" in source.query_proj)
        self.qkv_proj.weight = mx.concatenate(
            [source.query_proj.weight, source.key_proj.weight, source.value_proj.weight],
            axis=0,
        )
        if "bias" in source.query_proj:
            self.qkv_proj.bias = mx.concatenate(
                [source.query_proj.bias, source.key_proj.bias, source.value_proj.bias],
                axis=0,
            )
        self.out_proj = source.out_proj

    def __call__(self, queries, keys, values, mask=None):
        if queries is not keys or keys is not values:
            raise ValueError("FusedSelfAttention expects shared query/key/value input.")
        qkv = self.qkv_proj(queries)
        queries, keys, values = mx.split(qkv, 3, axis=-1)
        num_heads = self.num_heads
        queries = mx.unflatten(queries, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        keys = mx.unflatten(keys, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        values = mx.unflatten(values, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        scale = math.sqrt(1 / queries.shape[-1])
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).flatten(-2, -1)
        return self.out_proj(output)


class FusedCrossAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        key_dims, key_input_dims = source.key_proj.weight.shape
        value_dims, value_input_dims = source.value_proj.weight.shape
        if key_input_dims != value_input_dims:
            raise ValueError("FusedCrossAttention requires matching key/value input dimensions.")
        self.num_heads = source.num_heads
        self.query_proj = source.query_proj
        self.kv_proj = nn.Linear(key_input_dims, key_dims + value_dims, bias="bias" in source.key_proj)
        self.kv_proj.weight = mx.concatenate([source.key_proj.weight, source.value_proj.weight], axis=0)
        if "bias" in source.key_proj:
            self.kv_proj.bias = mx.concatenate([source.key_proj.bias, source.value_proj.bias], axis=0)
        self.key_dims = key_dims
        self.out_proj = source.out_proj

    def __call__(self, queries, keys, values, mask=None):
        if keys is not values:
            raise ValueError("FusedCrossAttention expects shared key/value memory input.")
        queries = self.query_proj(queries)
        kv = self.kv_proj(keys)
        keys = kv[..., : self.key_dims]
        values = kv[..., self.key_dims :]
        num_heads = self.num_heads
        queries = mx.unflatten(queries, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        keys = mx.unflatten(keys, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        values = mx.unflatten(values, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        scale = math.sqrt(1 / queries.shape[-1])
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).flatten(-2, -1)
        return self.out_proj(output)


def projected_attention(queries, keys, values, num_heads, mask=None):
    queries = mx.unflatten(queries, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
    keys = mx.unflatten(keys, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
    values = mx.unflatten(values, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
    scale = math.sqrt(1 / queries.shape[-1])
    output = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
    )
    return output.transpose(0, 2, 1, 3).flatten(-2, -1)


def linear_no_bias(x, weight):
    return mx.matmul(x, weight.T)


def linear_with_bias(x, weight, bias=None):
    y = mx.matmul(x, weight.T)
    if bias is not None:
        y = y + bias
    return y


def batch_to_size(source, dest_size):
    dest_size = int(dest_size)
    if source.shape[0] == dest_size:
        return source
    if source.shape[0] > dest_size:
        return source[:dest_size]
    repeat_shape = (dest_size - source.shape[0],) + tuple(source.shape[1:])
    return mx.concatenate([source, mx.broadcast_to(source[-1:], repeat_shape)], axis=0)


def ipadapter_layer_weight(adapter, transformer_index):
    weight = adapter.get("weight", 1.0)
    schedule = adapter.get("schedule")
    schedule_factor = 1.0
    if isinstance(schedule, dict):
        schedule_factor = schedule_effective_strength(
            schedule,
            SDMLX_IPADAPTER_CONTEXT.get("step_percent", 0.0),
        )
    if isinstance(weight, dict):
        if transformer_index not in weight:
            return None
        return float(weight[transformer_index]) * schedule_factor

    weight = float(weight)
    weight_type = adapter.get("weight_type", "linear")
    layers = int(adapter.get("layers", 11))
    denom = max(layers - 1, 1)
    t = min(max(float(transformer_index) / denom, 0.0), 1.0)

    if weight_type == "ease in":
        return weight * (0.05 + 0.95 * (1.0 - t)) * schedule_factor
    if weight_type == "ease out":
        return weight * (0.05 + 0.95 * t) * schedule_factor
    if weight_type == "ease in-out":
        return weight * (0.05 + 0.95 * (1.0 - abs(t - 0.5) / 0.5)) * schedule_factor
    if weight_type == "reverse in-out":
        return weight * (0.05 + 0.95 * (abs(t - 0.5) / 0.5)) * schedule_factor
    return weight * schedule_factor


def ipadapter_tokens_for_batch(adapter, batch_size, dtype, use_cfg=None):
    cond = adapter["cond"].astype(dtype)
    uncond = adapter["uncond"].astype(dtype)
    if use_cfg is None:
        use_cfg = bool(SDMLX_IPADAPTER_CONTEXT.get("use_cfg", False))
    else:
        use_cfg = bool(use_cfg)
    if use_cfg and batch_size % 2 == 0:
        prompt_batch = batch_size // 2
        return mx.concatenate(
            [
                batch_to_size(cond, prompt_batch),
                batch_to_size(uncond, prompt_batch),
            ],
            axis=0,
        )
    return batch_to_size(cond, batch_size)


def prepare_ipadapter_kv_cache(ip_adapters, batch_size, dtype, use_cfg):
    if not ip_adapters:
        return [], 0

    start = time.perf_counter()
    prepared_adapters = []
    eval_tensors = []
    prepared_layers = 0
    for adapter in ip_adapters:
        prepared = dict(adapter)
        prepared["debug_stats"] = []
        tokens = ipadapter_tokens_for_batch(prepared, batch_size, dtype, use_cfg=use_cfg)
        prepared_kv = {}
        for key, k_weight in prepared.get("ip_adapter", {}).items():
            if not key.endswith(".to_k_ip.weight"):
                continue
            module_key = key[:-len(".to_k_ip.weight")]
            v_weight = prepared["ip_adapter"].get(f"{module_key}.to_v_ip.weight")
            if v_weight is None:
                continue
            ip_k = linear_no_bias(tokens, k_weight.astype(dtype))
            ip_v = linear_no_bias(tokens, v_weight.astype(dtype))
            prepared_kv[str(module_key)] = (ip_k, ip_v)
            eval_tensors.extend([ip_k, ip_v])
        prepared["prepared_kv"] = prepared_kv
        prepared_adapters.append(prepared)
        prepared_layers += len(prepared_kv)

    if eval_tensors:
        mx.eval(*eval_tensors)
    elapsed = time.perf_counter() - start
    print(
        "SDMLX: IP-Adapter K/V Cache vorbereitet "
        f"({len(prepared_adapters)} Adapter, {prepared_layers} Layer, "
        f"batch={batch_size}, dtype={dtype}, {elapsed:.3f}s)."
    )
    return prepared_adapters, prepared_layers


def ipadapter_attention_delta(queries, adapter, module_key, transformer_index, num_heads):
    step_percent = float(SDMLX_IPADAPTER_CONTEXT.get("step_percent", 0.0))
    if step_percent < float(adapter.get("start_at", 0.0)) or step_percent > float(adapter.get("end_at", 1.0)):
        return None

    weight = ipadapter_layer_weight(adapter, transformer_index)
    if weight is None or abs(weight) <= 1e-6:
        return None

    prepared = adapter.get("prepared_kv", {}).get(str(module_key))
    if prepared is not None and prepared[0].shape[0] == queries.shape[0]:
        ip_k, ip_v = prepared
        ip_k = ip_k.astype(queries.dtype)
        ip_v = ip_v.astype(queries.dtype)
    else:
        k_weight = adapter["ip_adapter"].get(f"{module_key}.to_k_ip.weight")
        v_weight = adapter["ip_adapter"].get(f"{module_key}.to_v_ip.weight")
        if k_weight is None or v_weight is None:
            return None

        tokens = ipadapter_tokens_for_batch(adapter, queries.shape[0], queries.dtype)
        ip_k = linear_no_bias(tokens, k_weight.astype(queries.dtype))
        ip_v = linear_no_bias(tokens, v_weight.astype(queries.dtype))

    embeds_scaling = adapter.get("embeds_scaling", "V only")
    if embeds_scaling == "K+mean(V) w/ C penalty":
        scaling = float(ip_k.shape[-1]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v_mean = mx.mean(ip_v, axis=1, keepdims=True)
        ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
        delta = projected_attention(queries, ip_k, ip_v, num_heads)
    elif embeds_scaling == "K+V w/ C penalty":
        scaling = float(ip_k.shape[-1]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        delta = projected_attention(queries, ip_k, ip_v, num_heads)
    elif embeds_scaling == "K+V":
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        delta = projected_attention(queries, ip_k, ip_v, num_heads)
    else:
        delta = projected_attention(queries, ip_k, ip_v, num_heads) * weight

    return delta


def record_ipadapter_debug_stat(adapter, module_key, base_output, delta):
    if not adapter.get("is_faceid"):
        return
    stats = adapter.setdefault("debug_stats", [])
    if len(stats) >= 8:
        return
    base_f = base_output.astype(mx.float32)
    delta_f = delta.astype(mx.float32)
    base_rms = float(mx.sqrt(mx.mean(mx.square(base_f))).item())
    delta_rms = float(mx.sqrt(mx.mean(mx.square(delta_f))).item())
    stats.append(
        {
            "module": str(module_key),
            "base_rms": base_rms,
            "delta_rms": delta_rms,
            "ratio": delta_rms / max(base_rms, 1e-8),
        }
    )


class IPAdapterCrossAttention(nn.Module):
    def __init__(self, source, module_key, transformer_index):
        super().__init__()
        self.source = source
        self.module_key = str(module_key)
        self.transformer_index = int(transformer_index)
        self.num_heads = source.num_heads

    def __call__(self, queries, keys, values, mask=None):
        q = self.source.query_proj(queries)
        if "kv_proj" in self.source:
            kv = self.source.kv_proj(keys)
            k = kv[..., : self.source.key_dims]
            v = kv[..., self.source.key_dims :]
        else:
            k = self.source.key_proj(keys)
            v = self.source.value_proj(values)

        output = projected_attention(q, k, v, self.num_heads, mask)
        for adapter in SDMLX_IPADAPTER_CONTEXT.get("adapters", []):
            delta = ipadapter_attention_delta(q, adapter, self.module_key, self.transformer_index, self.num_heads)
            if delta is not None:
                record_ipadapter_debug_stat(adapter, self.module_key, output, delta)
                output = output + delta.astype(output.dtype)
        return self.source.out_proj(output)


class FastTransformerBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1 = source.norm1
        self.attn1 = source.attn1
        self.norm2 = source.norm2
        self.attn2 = source.attn2
        self.norm3 = source.norm3
        self.linear1 = source.linear1
        self.linear2 = source.linear2
        self.linear3 = source.linear3

    def __call__(self, x, memory, attn_mask=None, memory_mask=None):
        y = self.norm1(x)
        x = x + self.attn1(y, y, y, attn_mask)

        y = self.norm2(x)
        x = x + self.attn2(y, memory, memory, memory_mask)

        y = self.norm3(x)
        y = self.linear1(y) * nn.gelu(self.linear2(y))
        return x + self.linear3(y)


class Conv2dFloat16(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight.astype(mx.float16)
        if "bias" in source:
            self.bias = source.bias.astype(mx.float16)
        self.padding = source.padding
        self.dilation = source.dilation
        self.stride = source.stride
        self.groups = source.groups

    def __call__(self, x):
        output_dtype = x.dtype
        y = mx.conv2d(
            x.astype(mx.float16),
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if "bias" in self:
            y = y + self.bias
        return y.astype(output_dtype)


class ScheduledLoRALinear(nn.Module):
    def __init__(self, source, target_base):
        super().__init__()
        self.source = source
        self.target_base = str(target_base)
        self.scheduled_loras = []

    def clear_scheduled_loras(self):
        self.scheduled_loras = []

    def add_scheduled_lora(self, item, module, dtype):
        self.scheduled_loras.append(
            {
                "item": item,
                "up": mx.array(module["up"]).astype(dtype),
                "down": mx.array(module["down"]).astype(dtype),
                "scale_base": float(module["alpha"]) / float(module["rank"]),
            }
        )

    def __call__(self, x):
        y = self.source(x)
        if not self.scheduled_loras:
            return y
        step_percent = float(SDMLX_LORA_CONTEXT.get("step_percent", 0.0))
        for module in self.scheduled_loras:
            strength = lora_effective_strength(module["item"], step_percent)
            if abs(strength) <= 1e-6:
                continue
            down = module["down"].astype(x.dtype)
            up = module["up"].astype(x.dtype)
            delta = mx.matmul(mx.matmul(x, down.T), up.T)
            y = y + delta.astype(y.dtype) * (strength * module["scale_base"])
        return y


class ScheduledLoRAConv2d(nn.Module):
    def __init__(self, source, target_base):
        super().__init__()
        self.source = source
        self.target_base = str(target_base)
        self.scheduled_loras = []

    def clear_scheduled_loras(self):
        self.scheduled_loras = []

    def add_scheduled_lora(self, item, module, dtype):
        try:
            delta = lora_delta(module["up"], module["down"], self.source.weight.shape, mx.float32)
        except Exception:
            return False
        self.scheduled_loras.append(
            {
                "item": item,
                "delta": delta.astype(dtype),
                "scale_base": float(module["alpha"]) / float(module["rank"]),
            }
        )
        return True

    def __call__(self, x):
        y = self.source(x)
        if not self.scheduled_loras:
            return y
        step_percent = float(SDMLX_LORA_CONTEXT.get("step_percent", 0.0))
        for module in self.scheduled_loras:
            strength = lora_effective_strength(module["item"], step_percent)
            if abs(strength) <= 1e-6:
                continue
            source = self.source
            delta = mx.conv2d(
                x,
                module["delta"].astype(x.dtype),
                source.stride,
                source.padding,
                source.dilation,
                source.groups,
            )
            y = y + delta.astype(y.dtype) * (strength * module["scale_base"])
        return y


def get_child_module(parent, name):
    if isinstance(parent, list):
        return parent[int(name)]
    if isinstance(parent, dict):
        if name not in parent and "source" in parent:
            return get_child_module(parent["source"], name)
        return parent[name]
    return getattr(parent, name)


def set_child_module(parent, name, value):
    if isinstance(parent, list):
        parent[int(name)] = value
    elif isinstance(parent, dict):
        if name not in parent and "source" in parent:
            set_child_module(parent["source"], name, value)
        else:
            parent[name] = value
    else:
        setattr(parent, name, value)


def get_module_by_path(root, path):
    current = root
    if not path:
        return current
    for part in str(path).split("."):
        current = get_child_module(current, part)
    return current


def set_module_by_path(root, path, value):
    parts = str(path).split(".")
    parent = root
    for part in parts[:-1]:
        parent = get_child_module(parent, part)
    set_child_module(parent, parts[-1], value)


def clear_scheduled_lora_wrappers(unet):
    for _, module in unet.named_modules():
        if isinstance(module, (ScheduledLoRALinear, ScheduledLoRAConv2d)):
            module.clear_scheduled_loras()


def ensure_scheduled_lora_wrapper(unet, target_base):
    try:
        module = get_module_by_path(unet, target_base)
    except Exception:
        return None
    if isinstance(module, (ScheduledLoRALinear, ScheduledLoRAConv2d)):
        return module
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    if len(weight.shape) == 2:
        wrapper = ScheduledLoRALinear(module, target_base)
    elif len(weight.shape) == 4:
        wrapper = ScheduledLoRAConv2d(module, target_base)
    else:
        return None
    set_module_by_path(unet, target_base, wrapper)
    return wrapper


def prepare_scheduled_loras_for_unet(unet, scheduled_loras, dtype):
    clear_scheduled_lora_wrappers(unet)
    if not scheduled_loras:
        return {"loras": 0, "modules": 0, "skipped": 0}

    patched = 0
    skipped = 0
    for item in scheduled_loras:
        internal_modules = item.get("internal_lora_modules")
        if internal_modules is not None:
            lora_info = {
                "modules": internal_modules,
                "skipped_prefixes": [],
                "text_prefixes": 0,
                "state_prefix_count": len(internal_modules),
            }
        else:
            lora_info = load_lora_modules(item["path"])
        item_modules = 0
        for module in lora_info["modules"]:
            target_base = module["target_base"]
            wrapper = ensure_scheduled_lora_wrapper(unet, target_base)
            if wrapper is None:
                skipped += 1
                continue
            if isinstance(wrapper, ScheduledLoRALinear):
                if len(np.shape(module["up"])) != 2 or len(np.shape(module["down"])) != 2:
                    skipped += 1
                    continue
                wrapper.add_scheduled_lora(item, module, dtype)
                item_modules += 1
            elif isinstance(wrapper, ScheduledLoRAConv2d):
                if wrapper.add_scheduled_lora(item, module, dtype):
                    item_modules += 1
                else:
                    skipped += 1
        patched += item_modules
        log_timing(
            "SDMLX: Scheduled LoRA prepared "
            f"({item.get('name', os.path.basename(item.get('path', 'LoRA')))}, "
            f"Module={item_modules}, strength={float(item.get('strength_model', 1.0)):g}, "
            f"curve={(item.get('schedule') or {}).get('curve', 'constant')})."
        )
        if lora_info["text_prefixes"]:
            log_timing(f"SDMLX: Scheduled LoRA contains {lora_info['text_prefixes']} text-encoder prefixes; currently only UNet is patched dynamically.")
    if skipped:
        log_timing(f"SDMLX: Scheduled LoRA: {skipped} target modules skipped.")
    return {"loras": len(scheduled_loras), "modules": patched, "skipped": skipped}


class FastTransformer2D(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm = source.norm
        self.proj_in = source.proj_in
        self.transformer_blocks = [FastTransformerBlock(block) for block in source.transformer_blocks]
        self.proj_out = source.proj_out

    def __call__(self, x, encoder_x, attn_mask=None, encoder_attn_mask=None):
        input_x = x
        B, H, W, C = x.shape
        x = self.norm(x).reshape(B, -1, C)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, encoder_x, attn_mask, encoder_attn_mask)
        x = self.proj_out(x)
        return x.reshape(B, H, W, C) + input_x


def new_profile():
    return {
        "conv_in": 0.0,
        "down_blocks": {},
        "mid_blocks": {},
        "up_blocks": {},
        "details": {},
        "out": 0.0,
        "calls": 0,
    }


def add_profile_time(profile, section, key, elapsed):
    if key is None:
        profile[section] += elapsed
    else:
        profile[section][key] = profile[section].get(key, 0.0) + elapsed


def timed_eval(label, fn, profile, section, key=None):
    start = time.perf_counter()
    result = fn()
    if isinstance(result, tuple):
        mx.eval(*result)
    else:
        mx.eval(result)
    add_profile_time(profile, section, key, time.perf_counter() - start)
    return result


def timed_detail(profile, label, fn):
    start = time.perf_counter()
    result = fn()
    if isinstance(result, tuple):
        mx.eval(*result)
    else:
        mx.eval(result)
    profile["details"][label] = profile["details"].get(label, 0.0) + time.perf_counter() - start
    return result


def print_unet_profile(profile):
    if not profile or profile["calls"] == 0:
        return
    total = profile["conv_in"] + profile["out"]
    total += sum(profile["down_blocks"].values())
    total += sum(profile["mid_blocks"].values())
    total += sum(profile["up_blocks"].values())

    print("SDMLX UNet Profiling:")
    print(f"  calls: {profile['calls']}")
    print(f"  total profiled: {total:.2f}s")
    print(f"  conv_in: {profile['conv_in']:.2f}s")
    for index, elapsed in sorted(profile["down_blocks"].items()):
        print(f"  down_blocks.{index}: {elapsed:.2f}s")
    for index, elapsed in sorted(profile["mid_blocks"].items()):
        print(f"  mid_blocks.{index}: {elapsed:.2f}s")
    for index, elapsed in sorted(profile["up_blocks"].items()):
        print(f"  up_blocks.{index}: {elapsed:.2f}s")
    print(f"  out: {profile['out']:.2f}s")
    if profile["details"]:
        buckets = {
            "transformer_attn1": 0.0,
            "transformer_attn2": 0.0,
            "transformer_ffn": 0.0,
            "transformer_norm": 0.0,
            "transformer_proj": 0.0,
            "resnets": 0.0,
            "up_downsample": 0.0,
        }
        for key, elapsed in profile["details"].items():
            if ".transformer_blocks." in key and key.endswith(".attn1"):
                buckets["transformer_attn1"] += elapsed
            elif ".transformer_blocks." in key and key.endswith(".attn2"):
                buckets["transformer_attn2"] += elapsed
            elif ".transformer_blocks." in key and key.endswith(".ffn"):
                buckets["transformer_ffn"] += elapsed
            elif ".transformer_blocks." in key and (key.endswith(".norm1") or key.endswith(".norm2") or key.endswith(".norm3")):
                buckets["transformer_norm"] += elapsed
            elif key.endswith(".norm_proj_in") or key.endswith(".proj_out"):
                buckets["transformer_proj"] += elapsed
            elif ".resnets." in key:
                buckets["resnets"] += elapsed
            elif key.endswith(".upsample") or key.endswith(".downsample"):
                buckets["up_downsample"] += elapsed

        print("  aggregate details:")
        for key, elapsed in buckets.items():
            percent = (elapsed / total * 100.0) if total > 0 else 0.0
            print(f"    {key}: {elapsed:.2f}s ({percent:.1f}%)")

        print("  details:")
        for key, elapsed in sorted(profile["details"].items()):
            print(f"    {key}: {elapsed:.2f}s")


def patch_unet_transformers(unet):
    patched = 0
    for block_group in (unet.down_blocks, unet.up_blocks):
        for block in block_group:
            if "attentions" in block:
                block.attentions = [FastTransformer2D(attention) for attention in block.attentions]
                patched += len(block.attentions)
    unet.mid_blocks[1] = FastTransformer2D(unet.mid_blocks[1])
    patched += 1
    return patched


def patch_transformer_blocks_fast_ffn(transformer):
    patched = 0
    new_blocks = []
    for block in transformer.transformer_blocks:
        if "ffn" in block:
            new_blocks.append(block)
        else:
            new_blocks.append(FastFFNTransformerBlock(block))
            patched += 1
    transformer.transformer_blocks = new_blocks
    return patched


def patch_unet_fast_ffn(unet):
    patched = 0
    for block_group in (unet.down_blocks, unet.up_blocks):
        for block in block_group:
            if "attentions" in block:
                for attention in block.attentions:
                    patched += patch_transformer_blocks_fast_ffn(attention)
    patched += patch_transformer_blocks_fast_ffn(unet.mid_blocks[1])
    return patched


def patch_transformer_blocks_fast_attention(transformer):
    patched = 0
    for block in transformer.transformer_blocks:
        if not isinstance(block.attn1, FusedSelfAttention):
            block.attn1 = FusedSelfAttention(block.attn1)
            patched += 1
        if not isinstance(block.attn2, FusedCrossAttention):
            block.attn2 = FusedCrossAttention(block.attn2)
            patched += 1
    return patched


def patch_unet_fast_attention(unet):
    patched = 0
    for block_group in (unet.down_blocks, unet.up_blocks):
        for block in block_group:
            if "attentions" in block:
                for attention in block.attentions:
                    patched += patch_transformer_blocks_fast_attention(attention)
    patched += patch_transformer_blocks_fast_attention(unet.mid_blocks[1])
    return patched


def patch_ipadapter_transformer(transformer, module_index):
    patched = 0
    for transformer_index, block in enumerate(transformer.transformer_blocks):
        if not isinstance(block.attn2, IPAdapterCrossAttention):
            block.attn2 = IPAdapterCrossAttention(block.attn2, module_index * 2 + 1, transformer_index)
            patched += 1
        module_index += 1
    return module_index, patched


def ensure_unet_ipadapter_wrapped(unet):
    if getattr(unet, "_sdmlx_ipadapter_wrapped", False):
        return 0

    patched = 0
    module_index = 0
    for block in unet.down_blocks:
        if "attentions" in block:
            for attention in block.attentions:
                module_index, count = patch_ipadapter_transformer(attention, module_index)
                patched += count

    for block in unet.up_blocks:
        if "attentions" in block:
            for attention in block.attentions:
                module_index, count = patch_ipadapter_transformer(attention, module_index)
                patched += count

    mid_start = module_index
    module_index, count = patch_ipadapter_transformer(unet.mid_blocks[1], module_index)
    patched += count
    unet._sdmlx_ipadapter_wrapped = True
    unet._sdmlx_ipadapter_modules = module_index
    unet._sdmlx_ipadapter_mid_start = mid_start
    return patched


def patch_down_mid_transformers(model):
    patched = 0
    for block in model.down_blocks:
        if "attentions" in block:
            block.attentions = [FastTransformer2D(attention) for attention in block.attentions]
            patched += len(block.attentions)
    model.mid_blocks[1] = FastTransformer2D(model.mid_blocks[1])
    patched += 1
    return patched


def patch_down_mid_fast_ffn(model):
    patched = 0
    for block in model.down_blocks:
        if "attentions" in block:
            for attention in block.attentions:
                patched += patch_transformer_blocks_fast_ffn(attention)
    patched += patch_transformer_blocks_fast_ffn(model.mid_blocks[1])
    return patched


def patch_down_mid_fast_attention(model):
    patched = 0
    for block in model.down_blocks:
        if "attentions" in block:
            for attention in block.attentions:
                patched += patch_transformer_blocks_fast_attention(attention)
    patched += patch_transformer_blocks_fast_attention(model.mid_blocks[1])
    return patched


def apply_controlnet_fast_stack(model, fast_transformer=False, fast_ffn=False, fast_attention=False):
    if fast_transformer and not getattr(model, "_sdmlx_fast_transformer", False):
        patched = patch_down_mid_transformers(model)
        model._sdmlx_fast_transformer = True
        log_timing(f"SDMLX: ControlNet FastTransformer2D enabled ({patched} Module).")
    if fast_attention and not getattr(model, "_sdmlx_fast_attention", False):
        patched = patch_down_mid_fast_attention(model)
        model._sdmlx_fast_attention = True
        log_timing(f"SDMLX: ControlNet Fast Attention enabled ({patched} Attention-Module).")
    if fast_ffn and not getattr(model, "_sdmlx_fast_ffn", False):
        patched = patch_down_mid_fast_ffn(model)
        model._sdmlx_fast_ffn = True
        log_timing(f"SDMLX: ControlNet Fast FFN enabled ({patched} Transformer-Blocks).")


def patch_unet_resnet_convs_float16(unet):
    patched = 0
    for block_group in (unet.down_blocks, unet.up_blocks):
        for block in block_group:
            for resnet in block.resnets:
                resnet.conv1 = Conv2dFloat16(resnet.conv1)
                resnet.conv2 = Conv2dFloat16(resnet.conv2)
                patched += 2
    for resnet in (unet.mid_blocks[0], unet.mid_blocks[2]):
        resnet.conv1 = Conv2dFloat16(resnet.conv1)
        resnet.conv2 = Conv2dFloat16(resnet.conv2)
        patched += 2
    return patched

# --- PFAD-FIX ---
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.insert(0, current_path)

def is_mlx_array(value):
    value_type = type(value)
    return value_type.__name__ == "array" and value_type.__module__.startswith("mlx")


def get_mlx_array(tensor):
    if is_mlx_array(tensor):
        return tensor
    if hasattr(tensor, "detach"):
        return mx.array(tensor.detach().cpu().numpy())
    return mx.array(tensor)


def get_numpy_array(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if is_mlx_array(tensor):
        return np.array(tensor)
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def format_bytes(num_bytes):
    value = float(num_bytes or 0)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0


def short_dtype_name(dtype):
    name = str(dtype).replace("torch.", "").replace("numpy.", "")
    aliases = {
        "float16": "fp16",
        "float32": "fp32",
        "bfloat16": "bf16",
    }
    return aliases.get(name, name)


def summarize_source_dtypes(state_dict):
    totals = {}
    for value in (state_dict or {}).values():
        if not hasattr(value, "dtype") or not hasattr(value, "numel"):
            continue
        try:
            size = int(value.numel()) * int(value.element_size())
        except Exception:
            size = 0
        dtype = short_dtype_name(value.dtype)
        totals[dtype] = totals.get(dtype, 0) + size
    if not totals:
        return "unknown"
    ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    total = sum(totals.values()) or 1
    if len(ordered) == 1:
        return ordered[0][0]
    main_name, main_size = ordered[0]
    main_share = main_size / total
    if main_share >= 0.95:
        return f"{main_name}~{main_share * 100:.0f}%"
    return "mixed:" + ",".join(f"{name}:{size / total * 100:.0f}%" for name, size in ordered[:3])


def cache_dir():
    try:
        import folder_paths
        models_dir = folder_paths.models_dir
    except Exception:
        comfy_dir = os.path.dirname(os.path.dirname(current_path))
        models_dir = os.path.join(comfy_dir, "models")
    path = os.path.join(models_dir, "SDMLX")
    os.makedirs(path, exist_ok=True)
    return path


def speed_patch_dir():
    path = os.path.join(cache_dir(), "SpeedPatches")
    os.makedirs(path, exist_ok=True)
    return path


def normalized_speed_patch_name(speed_patch):
    if not speed_patch or speed_patch == SPEED_PATCH_NONE:
        return None
    value = str(speed_patch)
    name = SPEED_PATCH_BY_LABEL.get(value, value)
    name = os.path.basename(name)
    if not name.endswith(".sdmlxpatch"):
        name += ".sdmlxpatch"
    return name


def speed_patch_override(widget_value, speed_patch_input=None):
    if speed_patch_input and speed_patch_input != SPEED_PATCH_NONE:
        return speed_patch_input
    return widget_value


def speed_patch_label(package_name):
    return SPEED_PATCH_LABELS.get(package_name, package_name.removesuffix(".sdmlxpatch"))


SPEED_PATCH_OPTIONS_CACHE = None
SPEED_PATCH_MARKED_PACKAGES = set()


def invalidate_speed_patch_options_cache():
    global SPEED_PATCH_OPTIONS_CACHE
    SPEED_PATCH_OPTIONS_CACHE = None


def speed_patch_options():
    global SPEED_PATCH_OPTIONS_CACHE
    if SPEED_PATCH_OPTIONS_CACHE is not None:
        return list(SPEED_PATCH_OPTIONS_CACHE)

    names = set(KNOWN_SPEED_PATCHES)
    try:
        root = speed_patch_dir()
        for entry in os.listdir(root):
            if entry.endswith(".sdmlxpatch"):
                names.add(entry)
                package_path = os.path.join(root, entry)
                if os.path.isdir(package_path) and package_path not in SPEED_PATCH_MARKED_PACKAGES:
                    mark_macos_package(package_path)
                    SPEED_PATCH_MARKED_PACKAGES.add(package_path)
    except Exception:
        pass
    options = [SPEED_PATCH_NONE] + sorted(
        (speed_patch_label(name) for name in names),
        key=str.lower,
    )
    SPEED_PATCH_OPTIONS_CACHE = tuple(options)
    return list(options)


def speed_patch_package_path(package_name):
    return os.path.join(speed_patch_dir(), package_name)


def speed_lora_lookup_key(lora_name):
    return os.path.basename(str(lora_name)).lower()


SUPPORTED_SPEED_LORA_PATCHES_BY_KEY = {
    speed_lora_lookup_key(name): info
    for name, info in SUPPORTED_SPEED_LORA_PATCHES.items()
}


def supported_speed_lora_info(lora_name):
    return SUPPORTED_SPEED_LORA_PATCHES_BY_KEY.get(speed_lora_lookup_key(lora_name))


def supported_speed_lora_options():
    try:
        import folder_paths
        names = []
        for name in folder_paths.get_filename_list("loras"):
            if supported_speed_lora_info(name):
                names.append(name)
        return sorted(names, key=str.lower) or [SUPPORTED_SPEED_LORA_PLACEHOLDER]
    except Exception:
        return [SUPPORTED_SPEED_LORA_PLACEHOLDER]


def model_folder_paths(folder_name, default_subdir):
    import folder_paths
    paths = []
    if hasattr(folder_paths, "get_folder_paths"):
        try:
            paths = list(folder_paths.get_folder_paths(folder_name))
        except Exception:
            paths = []
    if not paths:
        entry = getattr(folder_paths, "folder_names_and_paths", {}).get(folder_name)
        if entry:
            paths = list(entry[0])
    if not paths:
        paths = [os.path.join(folder_paths.models_dir, default_subdir)]
    return paths


def model_file_exists(folder_name, default_subdir, local_name):
    local_name = str(local_name or "")
    if not local_name:
        return False
    try:
        import folder_paths
        names = folder_paths.get_filename_list(folder_name)
        local_lower = local_name.lower()
        if any(name.lower() == local_lower or os.path.basename(name).lower() == local_lower for name in names):
            return True
    except Exception:
        pass
    try:
        return any(os.path.exists(os.path.join(path, local_name)) for path in model_folder_paths(folder_name, default_subdir))
    except Exception:
        return False


def refresh_folder_path_cache(folder_name):
    try:
        import folder_paths
        mapped_name = folder_paths.map_legacy(folder_name) if hasattr(folder_paths, "map_legacy") else folder_name
        cache_helper = getattr(folder_paths, "cache_helper", None)
        if cache_helper is not None:
            if hasattr(cache_helper, "cache"):
                cache_helper.cache.pop(mapped_name, None)
            elif hasattr(cache_helper, "clear"):
                cache_helper.clear()
        filename_cache = getattr(folder_paths, "filename_list_cache", None)
        if isinstance(filename_cache, dict):
            filename_cache.pop(mapped_name, None)
    except Exception:
        pass


def notify_model_downloaded(folder_name, local_name):
    try:
        from server import PromptServer
        server = getattr(PromptServer, "instance", None)
        if server is not None:
            server.send_sync("sdmlx_model_downloaded", {
                "folder_name": folder_name,
                "local_name": local_name,
            })
    except Exception:
        pass


def download_spec_exists(spec, folder_name, default_subdir):
    local_name = spec.get("local_name") or os.path.basename(spec["filename"])
    if not model_file_exists(folder_name, default_subdir, local_name):
        return False
    lora_spec = spec.get("lora")
    if lora_spec:
        lora_name = lora_spec.get("local_name") or os.path.basename(lora_spec["filename"])
        if not model_file_exists("loras", "loras", lora_name):
            return False
    return True


def download_hf_model_file(spec, target_dir, folder_name=None):
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "SDMLX: huggingface_hub is not available; "
            f"{spec.get('local_name') or spec.get('filename')} cannot be downloaded automatically."
        ) from exc

    os.makedirs(target_dir, exist_ok=True)
    local_name = spec.get("local_name") or os.path.basename(spec["filename"])
    final_path = os.path.join(target_dir, local_name)
    if os.path.exists(final_path):
        if folder_name:
            refresh_folder_path_cache(folder_name)
            notify_model_downloaded(folder_name, local_name)
        return final_path

    print(
        "SDMLX: Downloading model from Hugging Face "
        f"({spec['repo_id']}/{spec['filename']} -> {local_name})..."
    )
    downloaded = hf_hub_download(
        repo_id=spec["repo_id"],
        repo_type="model",
        filename=spec["filename"],
        local_dir=target_dir,
    )
    if os.path.abspath(downloaded) != os.path.abspath(final_path):
        os.replace(downloaded, final_path)
        parent = os.path.dirname(downloaded)
        while os.path.abspath(parent).startswith(os.path.abspath(target_dir)) and parent != target_dir:
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = os.path.dirname(parent)
    if folder_name:
        refresh_folder_path_cache(folder_name)
        notify_model_downloaded(folder_name, local_name)
    return final_path


def read_speed_patch_manifest(package_path):
    try:
        with open(os.path.join(package_path, "manifest.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def ensure_speed_patch_package(speed_patch):
    package_name = normalized_speed_patch_name(speed_patch)
    if package_name is None:
        return None

    package_path = speed_patch_package_path(package_name)
    manifest = read_speed_patch_manifest(package_path)
    factors_path = os.path.join(package_path, "patch.safetensors")
    if manifest and os.path.exists(factors_path):
        if os.path.isdir(package_path) and package_path not in SPEED_PATCH_MARKED_PACKAGES:
            mark_macos_package(package_path)
            SPEED_PATCH_MARKED_PACKAGES.add(package_path)
        return {
            "name": package_name,
            "path": package_path,
            "manifest": manifest,
            "factors_path": factors_path,
        }

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "SDMLX: huggingface_hub is not available; "
            f"Speed Patch {package_name} cannot be downloaded."
        ) from exc

    os.makedirs(package_path, exist_ok=True)
    print(f"SDMLX: Downloading Speed Patch {package_name} from Hugging Face...")
    for filename in ("manifest.json", "patch.safetensors", "source_metadata.json"):
        try:
            hf_hub_download(
                repo_id=SPEED_PATCH_REPO_ID,
                repo_type="model",
                filename=f"{package_name}/{filename}",
                local_dir=speed_patch_dir(),
            )
        except Exception:
            if filename != "source_metadata.json":
                raise

    manifest = read_speed_patch_manifest(package_path)
    if not manifest or not os.path.exists(factors_path):
        raise RuntimeError(f"SDMLX: Speed Patch {package_name} could not be downloaded completely.")
    mark_macos_package(package_path)
    SPEED_PATCH_MARKED_PACKAGES.add(package_path)
    invalidate_speed_patch_options_cache()
    return {
        "name": package_name,
        "path": package_path,
        "manifest": manifest,
        "factors_path": factors_path,
    }


def load_speed_patch_factors(patch_info):
    factors_path = patch_info["factors_path"]
    try:
        stat = os.stat(factors_path)
    except OSError as exc:
        raise RuntimeError(f"SDMLX: Speed Patch missing: {factors_path}") from exc

    cache_key = (factors_path, stat.st_size, stat.st_mtime_ns)
    if cache_key not in SPEED_PATCH_FACTORS_CACHE:
        from safetensors.numpy import load_file
        SPEED_PATCH_FACTORS_CACHE[cache_key] = load_file(factors_path)
    return SPEED_PATCH_FACTORS_CACHE[cache_key]


def speed_patch_modules(manifest, factors):
    modules = manifest.get("modules") or []
    if modules:
        return modules
    result = []
    for key in factors:
        if key.endswith(".lora_up"):
            base = key[: -len(".lora_up")]
            result.append({"target": f"{base}.weight"})
    return result


def normalize_speed_patch_target_base(target_base):
    replacements = {
        ".to.k": ".key_proj",
        ".to.q": ".query_proj",
        ".to.v": ".value_proj",
        ".to.out.0": ".out_proj",
    }
    for source, target in replacements.items():
        target_base = target_base.replace(source, target)
    return target_base


def lora_delta(up, down, target_shape, dtype):
    rank = int(down.shape[0])
    up = mx.array(up).astype(dtype)
    down = mx.array(down).astype(dtype)
    if len(down.shape) == 2:
        delta = up @ down
    elif len(down.shape) == 4:
        delta = (up @ down.reshape(rank, -1)).reshape((up.shape[0],) + tuple(down.shape[1:]))
    else:
        raise ValueError(f"unsupported lora_down rank {len(down.shape)}")
    if tuple(delta.shape) != tuple(target_shape):
        raise ValueError(f"delta shape {tuple(delta.shape)} != target shape {tuple(target_shape)}")
    return delta


def cast_mapped_weights(mapped_weights, dtype):
    casted = []
    for key, value in mapped_weights:
        try:
            casted.append((key, value.astype(dtype)))
        except Exception:
            casted.append((key, value))
    return casted


def apply_speed_patch_to_mapped_weights(mapped_weights, speed_patch, strength):
    package_name = normalized_speed_patch_name(speed_patch)
    strength = float(strength)
    if package_name is None or strength == 0.0:
        return mapped_weights, None

    patch_info = ensure_speed_patch_package(package_name)
    manifest = patch_info["manifest"]
    if manifest.get("format") != "sdmlx-acceleration-patch-v1":
        raise RuntimeError(f"SDMLX: Unknown Speed Patch format in {package_name}.")

    factors = load_speed_patch_factors(patch_info)
    weight_map = dict(mapped_weights)
    applied = 0
    skipped = []
    pending_eval = []

    for module in speed_patch_modules(manifest, factors):
        target_key = module.get("target")
        if not target_key:
            continue
        factor_base = target_key[: -len(".weight")] if target_key.endswith(".weight") else target_key
        weight_base = normalize_speed_patch_target_base(factor_base)
        weight_key = f"{weight_base}.weight"
        up_key = f"{factor_base}.lora_up"
        down_key = f"{factor_base}.lora_down"
        alpha_key = f"{factor_base}.alpha"
        if up_key not in factors and f"{weight_base}.lora_up" in factors:
            up_key = f"{weight_base}.lora_up"
            down_key = f"{weight_base}.lora_down"
            alpha_key = f"{weight_base}.alpha"

        if weight_key not in weight_map or up_key not in factors or down_key not in factors:
            skipped.append(weight_key)
            continue

        base_weight = weight_map[weight_key]
        down = factors[down_key]
        alpha = float(np.asarray(factors.get(alpha_key, np.array(down.shape[0], dtype=np.float32))).reshape(()))
        scale = strength * alpha / float(down.shape[0])
        try:
            delta = lora_delta(factors[up_key], down, base_weight.shape, base_weight.dtype)
        except Exception:
            skipped.append(weight_key)
            continue

        updated = (base_weight + delta * scale).astype(base_weight.dtype)
        weight_map[weight_key] = updated
        pending_eval.append(updated)
        applied += 1
        if len(pending_eval) >= 24:
            mx.eval(*pending_eval)
            pending_eval.clear()

    if pending_eval:
        mx.eval(*pending_eval)

    log_timing(
        f"SDMLX: Speed Patch {package_name} applied "
        f"({applied} Module, strength={strength:g})."
    )
    if skipped:
        log_timing(f"SDMLX: Speed Patch: {len(skipped)} target weights skipped.")
    return list(weight_map.items()), {
        "name": package_name,
        "path": patch_info["path"],
        "strength": strength,
        "applied": applied,
        "skipped": len(skipped),
        "recommendations": manifest.get("recommendations", {}),
    }


def lora_file_identity(path):
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def lora_stack_key(loras):
    result = [("merge_version", LORA_MERGE_VERSION)]
    for item in loras or []:
        try:
            if item.get("schedule"):
                continue
            identity = item.get("identity") or lora_file_identity(item["path"])
            result.append(
                (
                    identity["path"],
                    identity["size"],
                    identity["mtime_ns"],
                    round(float(item.get("strength_model", 1.0)), 6),
                )
            )
        except Exception:
            continue
    return tuple(result)


def lora_schedule_key(loras):
    result = [("schedule_version", 4)]
    for item in loras or []:
        schedule = item.get("schedule")
        if not schedule:
            continue
        try:
            identity = item.get("identity") or lora_file_identity(item["path"])
            result.append(
                (
                    identity["path"],
                    identity["size"],
                    identity["mtime_ns"],
                    round(float(item.get("strength_model", 1.0)), 6),
                    round(float(schedule.get("start_percent", 0.0)), 6),
                    round(float(schedule.get("end_percent", 1.0)), 6),
                    round(float(schedule.get("minimum_strength", 0.0)), 6),
                    round(float(schedule.get("maximum_strength", 1.0)), 6),
                    str(schedule.get("mode", "blend in")),
                    str(schedule.get("curve", "linear")),
                )
            )
        except Exception:
            continue
    return tuple(result)


def split_loras_by_schedule(loras):
    static_loras = []
    scheduled_loras = []
    for item in loras or []:
        if item.get("schedule"):
            scheduled_loras.append(item)
        else:
            static_loras.append(item)
    return static_loras, scheduled_loras


def interpolation_curve_value(curve, progress):
    curve = str(curve)
    progress = min(max(float(progress), 0.0), 1.0)
    if curve == "constant":
        return 1.0
    if curve == "linear":
        return progress
    if curve in ("progressive", "exp_in"):
        k = 4.0
        return (math.exp(k * progress) - 1.0) / (math.exp(k) - 1.0)
    if curve in ("degressive", "exp_out"):
        k = 4.0
        return 1.0 - ((math.exp(k * (1.0 - progress)) - 1.0) / (math.exp(k) - 1.0))
    if curve == "fast_degressive":
        k = 8.0
        return (1.0 - math.exp(-k * progress)) / (1.0 - math.exp(-k))
    if curve in ("s_curve", "exp_in_out"):
        if progress < 0.5:
            return 0.5 * interpolation_curve_value("progressive", progress * 2.0)
        return 0.5 + 0.5 * interpolation_curve_value("degressive", (progress - 0.5) * 2.0)
    if curve == "bell":
        return math.sin(math.pi * progress)
    return progress


def normalize_lora_schedule_mode(mode):
    mode = str(mode).strip().lower()
    if mode in LORA_SCHEDULE_MODES:
        return mode
    return "blend in"


def normalize_lora_schedule_curve(mode, curve):
    mode = normalize_lora_schedule_mode(mode)
    curve = str(curve).strip().lower()
    allowed = {
        "blend in": ("linear", "progressive", "progressive fast", "s-curve"),
        "blend out": ("linear", "degressive", "degressive fast", "s-curve"),
        "bell": ("positive", "negative"),
    }[mode]
    if curve in allowed:
        return curve
    return allowed[0]


def lora_schedule_curve_value(mode, curve, progress):
    mode = normalize_lora_schedule_mode(mode)
    curve = normalize_lora_schedule_curve(mode, curve)
    progress = min(max(float(progress), 0.0), 1.0)
    if mode == "bell":
        value = math.sin(math.pi * progress)
        if curve == "negative":
            value = 1.0 - value
        return value

    if curve == "linear":
        value = progress
    elif curve == "progressive":
        value = interpolation_curve_value("degressive", progress)
    elif curve == "progressive fast":
        value = interpolation_curve_value("fast_degressive", progress)
    elif curve == "degressive":
        value = interpolation_curve_value("degressive", progress)
    elif curve == "degressive fast":
        value = interpolation_curve_value("fast_degressive", progress)
    elif curve == "s-curve":
        value = interpolation_curve_value("s_curve", progress)
    else:
        value = progress

    if mode == "blend out":
        value = 1.0 - value
    return value


def schedule_effective_strength(schedule, step_percent):
    schedule = schedule or {}
    start = float(schedule.get("start_percent", 0.0))
    end = float(schedule.get("end_percent", 1.0))
    if end <= start:
        return 0.0
    if step_percent < start or step_percent > end:
        return 0.0
    local = (step_percent - start) / (end - start)
    minimum_strength = float(schedule.get("minimum_strength", 0.0))
    maximum_strength = float(schedule.get("maximum_strength", 1.0))
    if minimum_strength > maximum_strength:
        minimum_strength, maximum_strength = maximum_strength, minimum_strength
    mode = schedule.get("mode", "blend in")
    curve = schedule.get("curve", "linear")
    curve_value = lora_schedule_curve_value(mode, curve, local)
    return minimum_strength + (maximum_strength - minimum_strength) * curve_value


def lora_effective_strength(item, step_percent):
    schedule = item.get("schedule") or {}
    strength_factor = schedule_effective_strength(schedule, step_percent)
    return float(item.get("strength_model", 1.0)) * strength_factor


def lora_strength_for_item(strength_model, schedule=None):
    if isinstance(schedule, dict):
        return 1.0
    return float(strength_model)


def make_lora_schedule(
    mode="blend in",
    minimum_strength=0.0,
    maximum_strength=1.0,
    curve="linear",
    advanced=False,
    start_percent=0.0,
    end_percent=1.0,
):
    mode = normalize_lora_schedule_mode(mode)
    curve = normalize_lora_schedule_curve(mode, curve)
    start_percent = float(start_percent)
    end_percent = float(end_percent)
    minimum_strength = float(minimum_strength)
    maximum_strength = float(maximum_strength)
    if minimum_strength > maximum_strength:
        print(
            "SDMLX: Scheduler received minimum_strength > maximum_strength; "
            "swapping the values automatically."
        )
        minimum_strength, maximum_strength = maximum_strength, minimum_strength
    if not bool(advanced):
        start_percent = 0.0
        end_percent = 1.0
    if end_percent <= start_percent:
        raise ValueError("SDMLX: Scheduler end_percent must be greater than start_percent.")
    return {
        "start_percent": start_percent,
        "end_percent": end_percent,
        "minimum_strength": minimum_strength,
        "maximum_strength": maximum_strength,
        "mode": mode,
        "curve": curve,
    }


def is_probable_union_promax_name(name):
    lower = str(name).lower()
    return "promax" in lower


def controlnet_file_options():
    try:
        import folder_paths
        models = [
            name
            for name in folder_paths.get_filename_list("controlnet")
            if is_probable_union_promax_name(name)
        ]
        options = sorted(models, key=str.lower)
        if not model_file_exists("controlnet", "controlnet", CONTROLNET_PROMAX_FILENAME):
            options.insert(0, CONTROLNET_PROMAX_AUTO)
        return options or [CONTROLNET_PROMAX_AUTO]
    except Exception:
        return [CONTROLNET_PROMAX_AUTO]


def controlnet_model_dirs():
    return model_folder_paths("controlnet", "controlnet")


def ensure_xinsir_promax_controlnet():
    try:
        import folder_paths
        existing = [
            name
            for name in folder_paths.get_filename_list("controlnet")
            if is_probable_union_promax_name(name)
        ]
        if existing:
            path = folder_paths.get_full_path("controlnet", sorted(existing, key=str.lower)[0])
            if path and os.path.exists(path):
                return path, sorted(existing, key=str.lower)[0]
    except Exception:
        pass

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "SDMLX: huggingface_hub is not available; "
            "Xinsir Union ProMax ControlNet cannot be downloaded automatically."
        ) from exc

    target_dir = controlnet_model_dirs()[0]
    os.makedirs(target_dir, exist_ok=True)
    print(
        "SDMLX: Downloading Xinsir Union ProMax ControlNet from Hugging Face "
        f"({CONTROLNET_PROMAX_REPO_ID}/{CONTROLNET_PROMAX_FILENAME})..."
    )
    path = download_hf_model_file(
        {
            "repo_id": CONTROLNET_PROMAX_REPO_ID,
            "filename": CONTROLNET_PROMAX_FILENAME,
            "local_name": CONTROLNET_PROMAX_FILENAME,
        },
        target_dir,
        folder_name="controlnet",
    )
    return path, os.path.basename(path)


def resolve_controlnet_path(control_net_name):
    import folder_paths
    if not control_net_name:
        control_net_name = CONTROLNET_PROMAX_AUTO
    if control_net_name == CONTROLNET_PROMAX_AUTO:
        return ensure_xinsir_promax_controlnet()
    path = folder_paths.get_full_path("controlnet", control_net_name)
    return path, control_net_name


def validate_controlnet_union_promax_weights(weights, control_net_name):
    required = {
        "task_embedding": (8, 320),
        "control_add_embedding.linear_1.weight": (1280, 2048),
        "controlnet_cond_embedding.conv_in.weight": (16, 3, 3, 3),
        "controlnet_down_blocks.8.weight": (1280, 1280, 1, 1),
        "controlnet_mid_block.weight": (1280, 1280, 1, 1),
        "transformer_layes.0.attn.in_proj_weight": (960, 320),
    }
    missing = [key for key in required if key not in weights]
    mismatched = [
        f"{key}: expected {shape}, got {tuple(weights[key].shape)}"
        for key, shape in required.items()
        if key in weights and tuple(weights[key].shape) != shape
    ]
    if missing or mismatched:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:4]))
        if mismatched:
            details.append("shape_mismatch=" + "; ".join(mismatched[:3]))
        raise ValueError(
            "SDMLX: This loader currently only supports Xinsir SDXL ControlNet Union ProMax. "
            f"`{control_net_name}` does not look like this model ({' | '.join(details)})."
        )


def controlnet_cache_key(mlx_controlnet):
    identity = mlx_controlnet.get("identity", {})
    return (
        identity.get("path", mlx_controlnet.get("cache_key", "")),
        identity.get("size", 0),
        identity.get("mtime_ns", 0),
        mlx_controlnet.get("dtype", "float16"),
    )


def get_controlnet_union_model(mlx_controlnet, fast_transformer=False, fast_ffn=False, fast_attention=False):
    key = controlnet_cache_key(mlx_controlnet)
    if key in CONTROLNET_MODEL_CACHE:
        model = CONTROLNET_MODEL_CACHE[key]
    else:
        model = ControlNetUnionModel()
        mapped_weights = []
        for weight_key, value in mlx_controlnet["weights"].items():
            mapped_weights.extend(map_controlnet_union_weights(weight_key, value.astype(mx.float16)))
        model.update(tree_unflatten(mapped_weights))
        model.set_dtype(mx.float16)
        mx.eval(model.parameters())
        CONTROLNET_MODEL_CACHE[key] = model
        log_timing(f"SDMLX: ControlNet Union loaded ({len(mapped_weights)} mapped tensors).")
    apply_controlnet_fast_stack(model, fast_transformer, fast_ffn, fast_attention)
    return model


def control_effective_strength(control, step_percent):
    schedule = control.get("schedule")
    if isinstance(schedule, dict):
        return schedule_effective_strength(schedule, step_percent)
    start = float(control.get("start_percent", 0.0))
    end = float(control.get("end_percent", 1.0))
    if end <= start:
        return 0.0
    if step_percent < start or step_percent > end:
        return 0.0
    return float(control.get("strength", 1.0))


def controlnets_active_at_percent(controlnets, step_percent):
    return any(abs(control_effective_strength(control, step_percent)) > CONTROL_STRENGTH_EPSILON for control in controlnets)


def add_controlnet_to_conditioning(conditioning, control):
    result = dict(conditioning)
    controls = list(result.get("controlnets", []))
    controls.append(control)
    result["controlnets"] = controls
    return result


def collect_conditioning_controlnets(mlx_model, positive, negative):
    controls = list(mlx_model.get("controlnets", []))
    positive_controls = list(positive.get("controlnets", [])) if isinstance(positive, dict) else []
    negative_controls = list(negative.get("controlnets", [])) if isinstance(negative, dict) else []
    if positive_controls:
        controls.extend(positive_controls)
    elif negative_controls:
        controls.extend(negative_controls)
    return controls


def resize_control_pil(pil, width, height, resize_mode):
    mode_aliases = {
        "crop": "crop center",
        "center crop": "crop center",
        "fit": "fit pad",
    }
    resize_mode = mode_aliases.get(str(resize_mode), str(resize_mode))
    resize_mode = resize_mode if resize_mode in ("crop center", "crop top", "fit pad", "stretch") else "crop center"
    source_w, source_h = pil.size
    if source_w == width and source_h == height:
        return pil
    if resize_mode == "stretch":
        return pil.resize((width, height), Image.Resampling.BICUBIC)

    source_aspect = source_w / max(source_h, 1)
    target_aspect = width / max(height, 1)
    if resize_mode in ("crop center", "crop top"):
        if source_aspect > target_aspect:
            resized_h = height
            resized_w = max(width, int(round(height * source_aspect)))
        else:
            resized_w = width
            resized_h = max(height, int(round(width / max(source_aspect, 1e-6))))
        pil = pil.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        left = max(0, (resized_w - width) // 2)
        top = 0 if resize_mode == "crop top" else max(0, (resized_h - height) // 2)
        return pil.crop((left, top, left + width, top + height))

    if source_aspect > target_aspect:
        resized_w = width
        resized_h = max(1, int(round(width / max(source_aspect, 1e-6))))
    else:
        resized_h = height
        resized_w = max(1, int(round(height * source_aspect)))
    resized = pil.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - resized_w) // 2, (height - resized_h) // 2))
    return canvas


def control_image_to_mlx(image, width, height, batch, dtype=mx.float16, resize_mode="crop center"):
    array = get_numpy_array(image).astype(np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    array = np.clip(array, 0.0, 1.0)
    resized = []
    for item in array:
        if item.shape[0] != height or item.shape[1] != width:
            pil = Image.fromarray((item[:, :, :3] * 255.0).astype(np.uint8), mode="RGB")
            pil = resize_control_pil(pil, width, height, resize_mode)
            item = np.asarray(pil).astype(np.float32) / 255.0
        else:
            item = item[:, :, :3]
        resized.append(item)
    while len(resized) < batch:
        resized.append(resized[-1])
    result = mx.array(np.stack(resized[:batch], axis=0)).astype(dtype)
    return result


def prepare_controlnets_for_sampling(controlnets, width, height, batch, dtype):
    prepared = []
    for control in controlnets:
        item = dict(control)
        item["prepared_image"] = control_image_to_mlx(
            item["image"],
            width,
            height,
            batch,
            dtype=dtype,
            resize_mode=item.get("resize_mode", "crop center"),
        )
        prepared.append(item)
    if prepared:
        mx.eval(*[item["prepared_image"] for item in prepared])
    return prepared


def ensure_ipadapter_folder_paths():
    try:
        import folder_paths

        default_path = os.path.join(folder_paths.models_dir, "ipadapter")
        os.makedirs(default_path, exist_ok=True)
        if "ipadapter" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["ipadapter"] = (
                [default_path],
                folder_paths.supported_pt_extensions,
            )
        else:
            paths, extensions = folder_paths.folder_names_and_paths["ipadapter"]
            if not extensions:
                folder_paths.folder_names_and_paths["ipadapter"] = (
                    paths,
                    folder_paths.supported_pt_extensions,
                )
        return folder_paths
    except Exception:
        return None


def ipadapter_model_options():
    folder_paths = ensure_ipadapter_folder_paths()
    if folder_paths is None:
        return list(IPADAPTER_DOWNLOAD_SPECS.keys())
    names = folder_paths.get_filename_list("ipadapter")
    options = [
        auto_name
        for auto_name, spec in IPADAPTER_DOWNLOAD_SPECS.items()
        if not download_spec_exists(spec, "ipadapter", "ipadapter")
    ]
    for name in names:
        if name not in options:
            options.append(name)
    return options


def faceid_ipadapter_model_options():
    names = ipadapter_model_options()
    faceid_tokens = ("faceid", "face-id", "face_id", "portrait")
    faceid_auto_names = {IPADAPTER_AUTO_FACEID_PLUSV2_SDXL, IPADAPTER_AUTO_FACEID_SDXL}
    options = [name for name in names if name in faceid_auto_names]
    faceid_names = [
        name for name in names
        if any(token in os.path.basename(name).lower() for token in faceid_tokens)
        and name not in options
    ]
    options.extend(faceid_names)
    return options


def ipadapter_model_dirs():
    return model_folder_paths("ipadapter", "ipadapter")


def lora_model_dirs():
    return model_folder_paths("loras", "loras")


def ensure_ipadapter_download(ipadapter_name):
    if not ipadapter_name:
        ipadapter_name = IPADAPTER_AUTO_PLUS_SDXL_VITH
    spec = IPADAPTER_DOWNLOAD_SPECS.get(ipadapter_name)
    if spec is None:
        return None, ipadapter_name
    target_dir = ipadapter_model_dirs()[0]
    path = download_hf_model_file(spec, target_dir, folder_name="ipadapter")
    lora_spec = spec.get("lora")
    if lora_spec:
        download_hf_model_file(lora_spec, lora_model_dirs()[0], folder_name="loras")
    return path, spec.get("local_name") or os.path.basename(path)


def resolve_ipadapter_path(ipadapter_name):
    downloaded_path, resolved_name = ensure_ipadapter_download(ipadapter_name)
    if downloaded_path is not None:
        return downloaded_path, resolved_name
    folder_paths = ensure_ipadapter_folder_paths()
    if folder_paths is None:
        raise ValueError("SDMLX: folder_paths is not available; IP-Adapter cannot be loaded.")
    path = folder_paths.get_full_path("ipadapter", ipadapter_name)
    return path, ipadapter_name


def clip_vision_dim_hint(dim, source="image_embeds"):
    dim = int(dim) if dim is not None else 0
    if source == "insightface_embedding":
        return "InsightFace buffalo_l Face-Embedding"
    if source == "penultimate_hidden_states":
        if dim == 1280:
            return "CLIP-ViT-H-14-laion2B-s32B-b79K"
        if dim == 1664:
            return "CLIP-ViT-bigG-14-laion2B-39B-b160k"
    else:
        if dim == 1024:
            return "CLIP-ViT-H-14-laion2B-s32B-b79K"
        if dim == 1280:
            return "CLIP-ViT-bigG-14-laion2B-39B-b160k"
    return f"CLIP-Vision with {dim} dimensions"


def raw_ipadapter_image_proj_dim(raw_image_proj):
    proj_weight = raw_image_proj.get("proj.weight")
    if proj_weight is not None and len(proj_weight.shape) == 2:
        return int(proj_weight.shape[1])
    for key in ("proj_in.weight", "perceiver_resampler.proj_in.weight"):
        weight = raw_image_proj.get(key)
        if weight is not None and len(weight.shape) == 2:
            return int(weight.shape[1])
    proj0_weight = raw_image_proj.get("proj.0.weight")
    if proj0_weight is not None and len(proj0_weight.shape) == 2:
        return int(proj0_weight.shape[1])
    return 0


def load_sdmlx_ipadapter_model(ipadapter_name):
    if ipadapter_name == IPADAPTER_PLACEHOLDER:
        raise ValueError("SDMLX: Please place an IP-Adapter model in ComfyUI/models/ipadapter.")
    path, resolved_name = resolve_ipadapter_path(ipadapter_name)
    if path is None:
        raise FileNotFoundError(f"SDMLX: IP-Adapter model not found: {ipadapter_name}")

    stat = os.stat(path)
    cache_key = (path, stat.st_size, stat.st_mtime_ns)
    if cache_key in IPADAPTER_MODEL_CACHE:
        return IPADAPTER_MODEL_CACHE[cache_key]

    import comfy.utils

    raw = comfy.utils.load_torch_file(path, safe_load=True)
    if path.lower().endswith(".safetensors"):
        split = {"image_proj": {}, "ip_adapter": {}}
        for key, value in raw.items():
            if key.startswith("image_proj."):
                split["image_proj"][key[len("image_proj."):]] = value
            elif key.startswith("ip_adapter."):
                split["ip_adapter"][key[len("ip_adapter."):]] = value
        raw = split

    if "image_proj" not in raw or "ip_adapter" not in raw or not raw["ip_adapter"]:
        raise ValueError(f"SDMLX: Ungueltiges IP-Adapter-Modell: {resolved_name}")

    image_proj = {key: mx.array(get_numpy_array(value).astype(np.float32)) for key, value in raw["image_proj"].items()}
    ip_adapter = {key: mx.array(get_numpy_array(value).astype(np.float16)) for key, value in raw["ip_adapter"].items()}
    clip_vision_dim = raw_ipadapter_image_proj_dim(raw["image_proj"])
    output_cross_attention_dim = int(raw["ip_adapter"]["1.to_k_ip.weight"].shape[1])
    is_sdxl = output_cross_attention_dim == 2048
    is_full = "proj.3.weight" in raw["image_proj"]
    has_lora_ip_weights = "0.to_q_lora.down.weight" in raw["ip_adapter"]
    is_portrait_unnorm = "portrait" in os.path.basename(path).lower() and "unnorm" in os.path.basename(path).lower()
    basename_lower = os.path.basename(path).lower()
    is_faceid_plusv2 = "faceidplusv2" in basename_lower or "faceid-plusv2" in basename_lower
    is_portrait = (
        "proj.2.weight" in raw["image_proj"]
        and not is_full
        and not has_lora_ip_weights
        and not is_portrait_unnorm
    )
    is_plus = (
        "latents" in raw["image_proj"]
        or "perceiver_resampler.proj_in.weight" in raw["image_proj"]
    ) and not is_portrait_unnorm
    is_faceid = (
        has_lora_ip_weights
        or is_portrait
        or is_portrait_unnorm
        or "faceid" in basename_lower
    )
    has_key_101 = "101.to_k_ip.weight" in raw["ip_adapter"]
    if is_faceid_plusv2:
        clip_vision_source = "penultimate_hidden_states"
    elif is_faceid:
        clip_vision_source = "insightface_embedding"
    else:
        clip_vision_source = "penultimate_hidden_states" if is_plus else "image_embeds"

    model = {
        "name": resolved_name,
        "path": path,
        "cache_key": cache_key,
        "image_proj": image_proj,
        "ip_adapter": ip_adapter,
        "is_sdxl": is_sdxl,
        "is_plus": is_plus,
        "is_full": is_full,
        "is_faceid": is_faceid,
        "is_faceid_plusv2": is_faceid_plusv2,
        "is_portrait": is_portrait,
        "is_portrait_unnorm": is_portrait_unnorm,
        "has_lora_ip_weights": has_lora_ip_weights,
        "clip_vision_dim": clip_vision_dim,
        "clip_vision_source": clip_vision_source,
        "clip_vision_hint": clip_vision_dim_hint(clip_vision_dim, clip_vision_source),
        "output_cross_attention_dim": output_cross_attention_dim,
        "layers": 11 if has_key_101 else 16,
    }
    IPADAPTER_MODEL_CACHE[cache_key] = model
    print(
        "SDMLX: IP-Adapter loaded "
        f"({resolved_name}, sdxl={is_sdxl}, plus={is_plus}, full={is_full}, faceid={is_faceid}, "
        f"faceid_plusv2={is_faceid_plusv2}, portrait={is_portrait}, "
        f"portrait_unnorm={is_portrait_unnorm}, "
        f"clip_vision={model['clip_vision_hint']})."
    )
    return model


def clip_vision_tensor(output, key):
    if output is None:
        return None
    if isinstance(output, dict):
        value = output.get(key)
    else:
        value = getattr(output, key, None)
    if value is None:
        return None
    if is_mlx_array(value):
        return value.astype(mx.float32)
    return mx.array(get_numpy_array(value).astype(np.float32))


def clip_vision_fit_pad_preprocess(
    image,
    size=224,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
):
    image = image[:, :, :, :3] if image.shape[3] > 3 else image
    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        height, width = image.shape[2], image.shape[3]
        scale = size / max(height, width)
        scaled_height = max(1, round(height * scale))
        scaled_width = max(1, round(width * scale))
        image = torch.nn.functional.interpolate(
            image,
            size=(scaled_height, scaled_width),
            mode="bicubic",
            antialias=True,
        )
        pad_top = (size - scaled_height) // 2
        pad_bottom = size - scaled_height - pad_top
        pad_left = (size - scaled_width) // 2
        pad_right = size - scaled_width - pad_left
        image = torch.nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    image = torch.clip((255.0 * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])


def clip_vision_crop_preprocess(
    image,
    size=224,
    anchor="center",
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
):
    image = image[:, :, :, :3] if image.shape[3] > 3 else image
    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        scale = size / min(image.shape[2], image.shape[3])
        scale_size = (round(scale * image.shape[2]), round(scale * image.shape[3]))
        image = torch.nn.functional.interpolate(image, size=scale_size, mode="bicubic", antialias=True)
        if anchor == "top":
            top = 0
        elif anchor == "bottom":
            top = image.shape[2] - size
        else:
            top = (image.shape[2] - size) // 2
        left = (image.shape[3] - size) // 2
        image = image[:, :, top:top + size, left:left + size]
    image = torch.clip((255.0 * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])


def clip_vision_stretch_preprocess(
    image,
    size=224,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
):
    image = image[:, :, :, :3] if image.shape[3] > 3 else image
    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        image = torch.nn.functional.interpolate(
            image,
            size=(size, size),
            mode="bicubic",
            antialias=True,
        )
    image = torch.clip((255.0 * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])


def ensure_clip_vision_folder_paths():
    try:
        import folder_paths

        default_path = os.path.join(folder_paths.models_dir, "clip_vision")
        os.makedirs(default_path, exist_ok=True)
        if "clip_vision" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["clip_vision"] = (
                [default_path],
                folder_paths.supported_pt_extensions,
            )
        return folder_paths
    except Exception:
        return None


def clip_vision_model_options():
    folder_paths = ensure_clip_vision_folder_paths()
    if folder_paths is None:
        return list(CLIP_VISION_DOWNLOAD_SPECS.keys())
    names = folder_paths.get_filename_list("clip_vision")
    options = [
        auto_name
        for auto_name, spec in CLIP_VISION_DOWNLOAD_SPECS.items()
        if not download_spec_exists(spec, "clip_vision", "clip_vision")
    ]
    for name in names:
        if name not in options:
            options.append(name)
    return options


def faceid_clip_vision_model_options():
    options = [AUTO_CLIP_VISION_OPTION]
    for name in clip_vision_model_options():
        if name not in options:
            options.append(name)
    return options


def resolve_faceid_clip_vision_name(clip_vision_name):
    if clip_vision_name not in (None, "", AUTO_CLIP_VISION_OPTION, CLIP_VISION_PLACEHOLDER):
        return clip_vision_name
    names = [name for name in clip_vision_model_options() if name != CLIP_VISION_PLACEHOLDER]
    if not names:
        return CLIP_VISION_AUTO_VITH
    if names == [CLIP_VISION_AUTO_VITH]:
        return CLIP_VISION_AUTO_VITH
    preferred_tokens = ("vit-h", "vit_h", "h-14", "clip-vit-h", "laion2b-s32b-b79k")
    for token in preferred_tokens:
        for name in names:
            lowered = os.path.basename(name).lower()
            if token in lowered and "bigg" not in lowered:
                return name
    for name in names:
        if "bigg" not in os.path.basename(name).lower():
            return name
    return names[0]


def clip_vision_model_dirs():
    return model_folder_paths("clip_vision", "clip_vision")


def ensure_clip_vision_download(clip_vision_name):
    if not clip_vision_name:
        clip_vision_name = CLIP_VISION_AUTO_VITH
    spec = CLIP_VISION_DOWNLOAD_SPECS.get(clip_vision_name)
    if spec is None:
        return None, clip_vision_name
    target_dir = clip_vision_model_dirs()[0]
    path = download_hf_model_file(spec, target_dir, folder_name="clip_vision")
    return path, spec.get("local_name") or os.path.basename(path)


def resolve_clip_vision_path(clip_vision_name):
    downloaded_path, resolved_name = ensure_clip_vision_download(clip_vision_name)
    if downloaded_path is not None:
        return downloaded_path, resolved_name
    folder_paths = ensure_clip_vision_folder_paths()
    if folder_paths is None:
        raise ValueError("SDMLX: folder_paths is not available; CLIP Vision cannot be loaded.")
    path = folder_paths.get_full_path("clip_vision", clip_vision_name)
    return path, clip_vision_name


def sdmlx_clip_vision_layer_indices(weights):
    layers = []
    pattern = re.compile(r"^vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.weight$")
    for key in weights:
        match = pattern.match(key)
        if match:
            layers.append(int(match.group(1)))
    return sorted(layers)


def sdmlx_clip_vision_config(weights):
    patch_weight = weights["vision_model.embeddings.patch_embedding.weight"]
    position_weight = weights["vision_model.embeddings.position_embedding.weight"]
    layer_indices = sdmlx_clip_vision_layer_indices(weights)
    hidden_size = int(position_weight.shape[1])
    patch_size = int(patch_weight.shape[-1])
    image_size = int(round((int(position_weight.shape[0]) - 1) ** 0.5) * patch_size)
    first_layer = layer_indices[0] if layer_indices else 0
    fc1 = weights[f"vision_model.encoder.layers.{first_layer}.mlp.fc1.weight"]
    visual_projection = weights.get("visual_projection.weight")
    return {
        "hidden_size": hidden_size,
        "intermediate_size": int(fc1.shape[0]),
        "num_hidden_layers": len(layer_indices),
        "layer_indices": layer_indices,
        "num_attention_heads": 16,
        "patch_size": patch_size,
        "image_size": image_size,
        "projection_dim": int(visual_projection.shape[0]) if visual_projection is not None else hidden_size,
        "image_mean": (0.48145466, 0.4578275, 0.40821073),
        "image_std": (0.26862954, 0.26130258, 0.27577711),
    }


def load_sdmlx_clip_vision_model(clip_vision_name, compute_dtype="float16"):
    if clip_vision_name == CLIP_VISION_PLACEHOLDER:
        raise ValueError("SDMLX: Please place a CLIP Vision model in ComfyUI/models/clip_vision.")
    path, resolved_name = resolve_clip_vision_path(clip_vision_name)
    if path is None:
        raise FileNotFoundError(f"SDMLX: CLIP Vision model not found: {clip_vision_name}")

    stat = os.stat(path)
    dtype_name = "float16" if compute_dtype == "float16" else "float32"
    cache_key = (path, stat.st_size, stat.st_mtime_ns, dtype_name)
    if cache_key in CLIP_VISION_MODEL_CACHE:
        return CLIP_VISION_MODEL_CACHE[cache_key]

    start = time.perf_counter()
    dtype = mx.float16 if dtype_name == "float16" else mx.float32
    weights = {}
    for key, value in mx.load(path).items():
        if key.endswith("position_ids"):
            continue
        weights[key] = value.astype(dtype) if value.dtype == mx.float32 else value
    config = sdmlx_clip_vision_config(weights)
    result = {
        "name": resolved_name,
        "path": path,
        "cache_key": cache_key,
        "weights": weights,
        "config": config,
        "dtype": dtype,
    }
    CLIP_VISION_MODEL_CACHE[cache_key] = result
    print(
        "SDMLX: CLIP Vision MLX loaded "
        f"({resolved_name}, layers={config['num_hidden_layers']}, hidden={config['hidden_size']}, "
        f"dtype={dtype_name}, {time.perf_counter() - start:.2f}s)."
    )
    return result


def sdmlx_clip_weight(model, key):
    try:
        return model["weights"][key]
    except KeyError as exc:
        raise KeyError(f"SDMLX: CLIP Vision weight missing: {key}") from exc


def sdmlx_clip_layer_norm(x, weight, bias=None, eps=1e-5):
    target_dtype = x.dtype
    x_f = x.astype(mx.float32)
    mean = mx.mean(x_f, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x_f - mean), axis=-1, keepdims=True)
    y = (x_f - mean) * mx.rsqrt(variance + eps)
    y = y * weight.astype(mx.float32)
    if bias is not None:
        y = y + bias.astype(mx.float32)
    return y.astype(target_dtype)


def sdmlx_clip_linear(x, weight, bias=None):
    y = mx.matmul(x, weight.T)
    if bias is not None:
        y = y + bias
    return y


def sdmlx_clip_attention(x, model, layer_index):
    prefix = f"vision_model.encoder.layers.{layer_index}.self_attn."
    heads = int(model["config"]["num_attention_heads"])
    q = sdmlx_clip_linear(
        x,
        sdmlx_clip_weight(model, prefix + "q_proj.weight"),
        sdmlx_clip_weight(model, prefix + "q_proj.bias"),
    )
    k = sdmlx_clip_linear(
        x,
        sdmlx_clip_weight(model, prefix + "k_proj.weight"),
        sdmlx_clip_weight(model, prefix + "k_proj.bias"),
    )
    v = sdmlx_clip_linear(
        x,
        sdmlx_clip_weight(model, prefix + "v_proj.weight"),
        sdmlx_clip_weight(model, prefix + "v_proj.bias"),
    )
    out = projected_attention(q, k, v, heads)
    return sdmlx_clip_linear(
        out,
        sdmlx_clip_weight(model, prefix + "out_proj.weight"),
        sdmlx_clip_weight(model, prefix + "out_proj.bias"),
    )


def sdmlx_clip_mlp(x, model, layer_index):
    prefix = f"vision_model.encoder.layers.{layer_index}.mlp."
    x = sdmlx_clip_linear(
        x,
        sdmlx_clip_weight(model, prefix + "fc1.weight"),
        sdmlx_clip_weight(model, prefix + "fc1.bias"),
    )
    x = nn.gelu(x)
    return sdmlx_clip_linear(
        x,
        sdmlx_clip_weight(model, prefix + "fc2.weight"),
        sdmlx_clip_weight(model, prefix + "fc2.bias"),
    )


def sdmlx_clip_vision_forward(model, pixel_values):
    weights = model["weights"]
    config = model["config"]
    dtype = model["dtype"]
    pixel_values = pixel_values.astype(dtype)
    patch_weight = sdmlx_clip_weight(model, "vision_model.embeddings.patch_embedding.weight")
    patch_size = int(config["patch_size"])
    batch, channels, height, width = pixel_values.shape
    patches = pixel_values.reshape(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size)
    patches = patches.transpose(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * patch_size * patch_size)
    x = mx.matmul(patches, patch_weight.reshape(patch_weight.shape[0], -1).T)
    class_embedding = sdmlx_clip_weight(model, "vision_model.embeddings.class_embedding")
    class_tokens = mx.broadcast_to(class_embedding[None, None, :], (batch, 1, class_embedding.shape[0]))
    x = mx.concatenate([class_tokens, x], axis=1)
    x = x + sdmlx_clip_weight(model, "vision_model.embeddings.position_embedding.weight")
    x = sdmlx_clip_layer_norm(
        x,
        sdmlx_clip_weight(model, "vision_model.pre_layrnorm.weight"),
        sdmlx_clip_weight(model, "vision_model.pre_layrnorm.bias"),
    )

    penultimate = None
    penultimate_index = len(config["layer_indices"]) - 2
    for ordinal, layer_index in enumerate(config["layer_indices"]):
        ln1 = sdmlx_clip_layer_norm(
            x,
            sdmlx_clip_weight(model, f"vision_model.encoder.layers.{layer_index}.layer_norm1.weight"),
            sdmlx_clip_weight(model, f"vision_model.encoder.layers.{layer_index}.layer_norm1.bias"),
        )
        x = x + sdmlx_clip_attention(ln1, model, layer_index)
        ln2 = sdmlx_clip_layer_norm(
            x,
            sdmlx_clip_weight(model, f"vision_model.encoder.layers.{layer_index}.layer_norm2.weight"),
            sdmlx_clip_weight(model, f"vision_model.encoder.layers.{layer_index}.layer_norm2.bias"),
        )
        x = x + sdmlx_clip_mlp(ln2, model, layer_index)
        if ordinal == penultimate_index:
            penultimate = x

    pooled = sdmlx_clip_layer_norm(
        x[:, 0, :],
        sdmlx_clip_weight(model, "vision_model.post_layernorm.weight"),
        sdmlx_clip_weight(model, "vision_model.post_layernorm.bias"),
    )
    visual_projection = weights.get("visual_projection.weight")
    image_embeds = sdmlx_clip_linear(pooled, visual_projection) if visual_projection is not None else pooled
    if penultimate is None:
        penultimate = x
    mx.eval(x, penultimate, image_embeds)
    return x, penultimate, image_embeds


def encode_sdmlx_clip_vision_for_ipadapter(sdmlx_clip_vision, image, resize_mode="crop center"):
    config = sdmlx_clip_vision["config"]
    image_t = image.detach().cpu().float() if hasattr(image, "detach") else torch.from_numpy(get_numpy_array(image).astype(np.float32))
    if image_t.ndim == 3:
        image_t = image_t[None, ...]
    if resize_mode == "stretch":
        pixel_values = clip_vision_stretch_preprocess(
            image_t,
            size=config["image_size"],
            mean=config["image_mean"],
            std=config["image_std"],
        )
    elif resize_mode == "crop top":
        pixel_values = clip_vision_crop_preprocess(
            image_t,
            size=config["image_size"],
            anchor="top",
            mean=config["image_mean"],
            std=config["image_std"],
        )
    elif resize_mode == "fit pad":
        pixel_values = clip_vision_fit_pad_preprocess(
            image_t,
            size=config["image_size"],
            mean=config["image_mean"],
            std=config["image_std"],
        )
    else:
        pixel_values = clip_vision_crop_preprocess(
            image_t,
            size=config["image_size"],
            anchor="center",
            mean=config["image_mean"],
            std=config["image_std"],
        )
    pixel_mx = mx.array(pixel_values.detach().cpu().numpy().astype(np.float32))
    start = time.perf_counter()
    last_hidden, penultimate, image_embeds = sdmlx_clip_vision_forward(sdmlx_clip_vision, pixel_mx)
    print(
        "SDMLX: CLIP Vision MLX encode finished "
        f"({sdmlx_clip_vision['name']}, batch={pixel_mx.shape[0]}, mode={resize_mode}, "
        f"{time.perf_counter() - start:.3f}s)."
    )
    return {
        "last_hidden_state": last_hidden.astype(mx.float32),
        "penultimate_hidden_states": penultimate.astype(mx.float32),
        "image_embeds": image_embeds.astype(mx.float32),
        "image_sizes": [tuple(pixel_values.shape[1:])] * int(pixel_values.shape[0]),
        "mm_projected": None,
    }


def ensure_insightface_folder_paths():
    try:
        import folder_paths

        default_path = os.path.join(folder_paths.models_dir, "insightface")
        os.makedirs(default_path, exist_ok=True)
        if "insightface" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["insightface"] = ([default_path], set())
        return folder_paths
    except Exception:
        return None


def insightface_model_root():
    folder_paths = ensure_insightface_folder_paths()
    candidates = []
    if folder_paths is not None:
        local_default = os.path.join(folder_paths.models_dir, "insightface")
        candidates.append(local_default)
        try:
            candidates.extend(folder_paths.get_folder_paths("insightface"))
        except Exception:
            pass
    seen = set()
    candidates = [path for path in candidates if not (path in seen or seen.add(path))]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "models", "buffalo_l")):
            return path
    return candidates[0] if candidates else None


def force_buffalo_l_w600k_recognition(face_app, provider_name, root):
    model_dir = os.path.join(root, "models", "buffalo_l")
    preferred = os.path.join(model_dir, "w600k_r50.onnx")
    if not os.path.exists(preferred):
        return

    current = face_app.models.get("recognition")
    current_file = os.path.basename(str(getattr(current, "model_file", ""))) if current is not None else ""
    if current_file == "w600k_r50.onnx":
        return

    try:
        from insightface import model_zoo

        replacement = model_zoo.get_model(preferred, providers=[provider_name])
        if replacement is None or getattr(replacement, "taskname", None) != "recognition":
            raise RuntimeError("w600k_r50.onnx was not recognized as a recognition model")
        replacement.prepare(ctx_id=0)
        face_app.models["recognition"] = replacement
        print(
            "SDMLX: InsightFace recognition set to w600k_r50.onnx "
            f"(instead of {current_file or 'unknown'})."
        )
    except Exception as exc:
        print(f"SDMLX: InsightFace w600k_r50 override failed ({exc}).")


def load_sdmlx_insightface(provider="CoreML"):
    provider = str(provider or "CoreML")
    cache_key = provider
    if cache_key in INSIGHTFACE_MODEL_CACHE:
        return INSIGHTFACE_MODEL_CACHE[cache_key]

    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:
        available = set()

    provider_name = f"{provider}ExecutionProvider"
    if available and provider_name not in available:
        print(f"SDMLX: InsightFace provider {provider_name} not available, using CPUExecutionProvider.")
        provider_name = "CPUExecutionProvider"

    root = insightface_model_root()
    if root is None:
        raise FileNotFoundError("SDMLX: InsightFace model root not found.")

    from insightface.app import FaceAnalysis

    start = time.perf_counter()
    model = FaceAnalysis(name="buffalo_l", root=root, providers=[provider_name])
    model.prepare(ctx_id=0, det_size=(640, 640))
    force_buffalo_l_w600k_recognition(model, provider_name, root)
    result = {"model": model, "provider": provider_name, "root": root}
    INSIGHTFACE_MODEL_CACHE[cache_key] = result
    print(
        "SDMLX: InsightFace loaded "
        f"(buffalo_l, provider={provider_name}, root={root}, {time.perf_counter() - start:.2f}s)."
    )
    return result


def fallback_to_cpu_insightface(insightface, reason):
    provider = insightface.get("provider") if isinstance(insightface, dict) else ""
    if "CoreML" not in str(provider):
        raise reason
    print(f"SDMLX: InsightFace CoreML error, falling back to CPU ({reason}).")
    cpu_info = load_sdmlx_insightface("CPU")
    if isinstance(insightface, dict):
        insightface.update(cpu_info)
    return cpu_info.get("model")


def image_tensor_to_bgr_uint8_batch(image):
    image_t = image.detach().cpu().float() if hasattr(image, "detach") else torch.from_numpy(get_numpy_array(image).astype(np.float32))
    if image_t.ndim == 3:
        image_t = image_t[None, ...]
    if image_t.ndim != 4 or image_t.shape[-1] < 3:
        raise ValueError("SDMLX: FaceID image muss ein IMAGE Tensor with RGB channels sein.")
    image_t = torch.clamp(image_t[..., :3], 0.0, 1.0)
    return (image_t[..., [2, 1, 0]] * 255.0).round().byte().numpy()


def extract_insightface_face_data(insightface, image, unnorm=False, aligned_crop=False, aligned_size=256):
    face_app = insightface.get("model") if isinstance(insightface, dict) else insightface
    if face_app is None:
        raise ValueError("SDMLX: InsightFace Loader missing.")
    face_align = None
    if aligned_crop:
        from insightface.utils import face_align

    images = image_tensor_to_bgr_uint8_batch(image)
    embeddings = []
    aligned_faces = []
    for batch_index, image_bgr in enumerate(images):
        face = None
        for size in range(640, 256, -64):
            try:
                face_app.det_model.input_size = (size, size)
            except Exception:
                pass
            try:
                faces = face_app.get(image_bgr)
            except Exception as exc:
                face_app = fallback_to_cpu_insightface(insightface, exc)
                faces = face_app.get(image_bgr)
            if faces:
                face = faces[0]
                if size != 640:
                    print(f"SDMLX: InsightFace detection for image {batch_index} with {size}px.")
                break
        if face is None:
            raise ValueError(f"SDMLX: InsightFace detected no face in image {batch_index}.")
        embedding = face.embedding if unnorm else face.normed_embedding
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        if aligned_crop:
            crop_bgr = face_align.norm_crop(image_bgr, landmark=face.kps, image_size=int(aligned_size))
            crop_rgb = torch.clamp(torch.from_numpy(crop_bgr).float() / 255.0, 0.0, 1.0)[..., [2, 1, 0]]
            aligned_faces.append(crop_rgb)

    result = mx.array(np.stack(embeddings, axis=0)).astype(mx.float32)
    mx.eval(result)
    if not aligned_crop:
        return result, None
    return result, torch.stack(aligned_faces, dim=0)


def extract_insightface_embeddings(insightface, image, unnorm=False):
    embeddings, _ = extract_insightface_face_data(insightface, image, unnorm=unnorm, aligned_crop=False)
    return embeddings


def mlx_layer_norm(x, weight, bias=None, eps=1e-5):
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * mx.rsqrt(variance + eps)
    y = y * weight
    if bias is not None:
        y = y + bias
    return y


def image_proj_get(image_proj, key, default=None):
    if key in image_proj:
        return image_proj[key]
    prefixed = f"perceiver_resampler.{key}"
    if prefixed in image_proj:
        return image_proj[prefixed]
    return default


def image_proj_require(image_proj, key):
    value = image_proj_get(image_proj, key)
    if value is None:
        raise ValueError(f"SDMLX: IP-Adapter Plus Projection-Key missing: {key}")
    return value


def image_proj_resampler_depth(image_proj):
    depth = 0
    pattern = re.compile(r"^(?:perceiver_resampler\.)?layers\.(\d+)\.0\.to_q\.weight$")
    for key in image_proj:
        match = pattern.match(key)
        if match:
            depth = max(depth, int(match.group(1)) + 1)
    return depth


def validate_ipadapter_embed_dim(embeds, expected_dim, label, source="image_embeds"):
    actual_dim = int(embeds.shape[-1])
    if actual_dim != expected_dim:
        raise ValueError(
            f"SDMLX: {label} does not match the IP-Adapter. "
            f"Der Adapter erwartet {expected_dim} dimensions "
            f"({clip_vision_dim_hint(expected_dim, source)}), bekommen hat er {actual_dim}."
        )


def ipadapter_plus_perceiver_attention(x, latents, image_proj, layer_index, heads):
    prefix = f"layers.{layer_index}.0."
    x_norm = mlx_layer_norm(
        x,
        image_proj_require(image_proj, prefix + "norm1.weight"),
        image_proj_get(image_proj, prefix + "norm1.bias"),
    )
    latents_norm = mlx_layer_norm(
        latents,
        image_proj_require(image_proj, prefix + "norm2.weight"),
        image_proj_get(image_proj, prefix + "norm2.bias"),
    )
    q = linear_with_bias(latents_norm, image_proj_require(image_proj, prefix + "to_q.weight"))
    kv_input = mx.concatenate([x_norm, latents_norm], axis=1)
    kv = linear_with_bias(kv_input, image_proj_require(image_proj, prefix + "to_kv.weight"))
    inner_dim = kv.shape[-1] // 2
    k = kv[..., :inner_dim]
    v = kv[..., inner_dim:]
    output = projected_attention(q, k, v, heads)
    return linear_with_bias(output, image_proj_require(image_proj, prefix + "to_out.weight"))


def ipadapter_plus_feed_forward(latents, image_proj, layer_index):
    prefix = f"layers.{layer_index}.1."
    y = mlx_layer_norm(
        latents,
        image_proj_require(image_proj, prefix + "0.weight"),
        image_proj_get(image_proj, prefix + "0.bias"),
    )
    y = linear_with_bias(y, image_proj_require(image_proj, prefix + "1.weight"))
    y = nn.gelu(y)
    return linear_with_bias(y, image_proj_require(image_proj, prefix + "3.weight"))


def project_plus_ipadapter_tokens(ipadapter, embeds):
    image_proj = ipadapter["image_proj"]
    proj_in_weight = image_proj_require(image_proj, "proj_in.weight")
    expected_dim = int(proj_in_weight.shape[1])
    validate_ipadapter_embed_dim(
        embeds,
        expected_dim,
        "CLIP-Vision Plus-Embeds",
        source="penultimate_hidden_states",
    )

    x = embeds.astype(mx.float32)
    pos_emb = image_proj_get(image_proj, "pos_emb.weight")
    if pos_emb is not None:
        x = x + pos_emb[: x.shape[1]]

    latents = image_proj_require(image_proj, "latents").astype(mx.float32)
    if len(latents.shape) == 2:
        latents = latents[None, ...]
    latents = mx.broadcast_to(latents, (x.shape[0],) + tuple(latents.shape[1:]))
    x = linear_with_bias(
        x,
        proj_in_weight,
        image_proj_get(image_proj, "proj_in.bias"),
    )

    depth = image_proj_resampler_depth(image_proj)
    if depth <= 0:
        raise ValueError("SDMLX: IP-Adapter Plus resampler layer not recognized.")
    heads = int(ipadapter.get("plus_heads", 20 if ipadapter.get("is_sdxl") else 12))
    for layer_index in range(depth):
        latents = latents + ipadapter_plus_perceiver_attention(x, latents, image_proj, layer_index, heads)
        latents = latents + ipadapter_plus_feed_forward(latents, image_proj, layer_index)

    latents = linear_with_bias(
        latents,
        image_proj_require(image_proj, "proj_out.weight"),
        image_proj_get(image_proj, "proj_out.bias"),
    )
    return mlx_layer_norm(
        latents,
        image_proj_require(image_proj, "norm_out.weight"),
        image_proj_get(image_proj, "norm_out.bias"),
    ).astype(mx.float16)


def project_plus_ipadapter_embeds(ipadapter, clip_vision_output, negative_clip_vision_output=None):
    embeds = clip_vision_tensor(clip_vision_output, "penultimate_hidden_states")
    if embeds is None:
        raise ValueError(
            "SDMLX: IP-Adapter Plus needs CLIP_VISION_OUTPUT.penultimate_hidden_states. "
            "Please use the SDMLX CLIP Vision Encode node."
        )
    negative_embeds = clip_vision_tensor(negative_clip_vision_output, "penultimate_hidden_states")
    if negative_embeds is None:
        negative_embeds = mx.zeros_like(embeds)

    cond = project_plus_ipadapter_tokens(ipadapter, embeds)
    uncond = project_plus_ipadapter_tokens(ipadapter, negative_embeds)
    mx.eval(cond, uncond)
    return cond, uncond


def project_faceid_mlp_tokens(ipadapter, face_embeds):
    image_proj = ipadapter["image_proj"]
    if len(face_embeds.shape) == 3 and face_embeds.shape[1] == 1:
        face_embeds = face_embeds[:, 0, :]
    face_embeds = face_embeds.astype(mx.float32)
    expected_dim = int(image_proj_require(image_proj, "proj.0.weight").shape[1])
    validate_ipadapter_embed_dim(face_embeds, expected_dim, "InsightFace-Embedding", source="insightface_embedding")

    y = linear_with_bias(
        face_embeds,
        image_proj_require(image_proj, "proj.0.weight"),
        image_proj_get(image_proj, "proj.0.bias"),
    )
    y = nn.gelu(y)
    y = linear_with_bias(
        y,
        image_proj_require(image_proj, "proj.2.weight"),
        image_proj_get(image_proj, "proj.2.bias"),
    )
    output_dim = int(ipadapter["output_cross_attention_dim"])
    tokens = int(y.shape[-1] // output_dim)
    y = y.reshape(y.shape[0], tokens, output_dim)
    return mlx_layer_norm(
        y,
        image_proj_require(image_proj, "norm.weight"),
        image_proj_get(image_proj, "norm.bias"),
    ).astype(mx.float16)


def project_faceid_plusv2_tokens(ipadapter, face_embeds, clip_embeds, shortcut_scale=2.0):
    image_proj = ipadapter["image_proj"]
    face_tokens = project_faceid_mlp_tokens(ipadapter, face_embeds).astype(mx.float32)
    proj_in_weight = image_proj_require(image_proj, "proj_in.weight")
    expected_dim = int(proj_in_weight.shape[1])
    validate_ipadapter_embed_dim(
        clip_embeds,
        expected_dim,
        "CLIP-Vision FaceID-PlusV2-Embeds",
        source="penultimate_hidden_states",
    )

    x = linear_with_bias(
        clip_embeds.astype(mx.float32),
        proj_in_weight,
        image_proj_get(image_proj, "proj_in.bias"),
    )
    latents = face_tokens
    depth = image_proj_resampler_depth(image_proj)
    if depth <= 0:
        raise ValueError("SDMLX: FaceID PlusV2 resampler layer not recognized.")
    heads = int(ipadapter["output_cross_attention_dim"] // 64)
    for layer_index in range(depth):
        latents = latents + ipadapter_plus_perceiver_attention(x, latents, image_proj, layer_index, heads)
        latents = latents + ipadapter_plus_feed_forward(latents, image_proj, layer_index)

    latents = linear_with_bias(
        latents,
        image_proj_require(image_proj, "proj_out.weight"),
        image_proj_get(image_proj, "proj_out.bias"),
    )
    latents = mlx_layer_norm(
        latents,
        image_proj_require(image_proj, "norm_out.weight"),
        image_proj_get(image_proj, "norm_out.bias"),
    )
    return (face_tokens + mx.array(float(shortcut_scale), dtype=mx.float32) * latents).astype(mx.float16)


def project_faceid_plusv2_embeds(
    ipadapter,
    face_embeds,
    clip_vision_output,
    negative_clip_vision_output=None,
    image_detail_transfer=1.0,
):
    if not ipadapter.get("is_sdxl"):
        raise ValueError("SDMLX: FaceID is currently only enabled for SDXL IP-Adapter.")
    if not ipadapter.get("is_faceid_plusv2"):
        raise ValueError("SDMLX: FaceID PlusV2 detail transfer was called for a non-PlusV2 model.")

    embeds = clip_vision_tensor(clip_vision_output, "penultimate_hidden_states")
    if embeds is None:
        raise ValueError(
            "SDMLX: FaceID PlusV2 additionally needs CLIP_VISION_OUTPUT.penultimate_hidden_states. "
            "Please connect SDMLX CLIP Vision directly to the FaceID node or use the SDMLX CLIP Vision Encode node."
        )
    negative_embeds = clip_vision_tensor(negative_clip_vision_output, "penultimate_hidden_states")
    if negative_embeds is None:
        negative_embeds = mx.zeros_like(embeds)
    cond = project_faceid_plusv2_tokens(
        ipadapter,
        face_embeds,
        embeds,
        shortcut_scale=float(image_detail_transfer),
    )
    uncond = project_faceid_plusv2_tokens(
        ipadapter,
        mx.zeros_like(face_embeds),
        negative_embeds,
        shortcut_scale=float(image_detail_transfer),
    )

    mx.eval(cond, uncond)
    return cond, uncond


def project_faceid_base_or_portrait_embeds(ipadapter, face_embeds):
    if not ipadapter.get("is_sdxl"):
        raise ValueError("SDMLX: FaceID is currently only enabled for SDXL IP-Adapter.")
    if ipadapter.get("is_faceid_plusv2"):
        raise ValueError("SDMLX: FaceID PlusV2 needs the separate CLIP Vision detail path.")

    if ipadapter.get("is_portrait_unnorm"):
        face_sequence = face_embeds[:, None, :] if len(face_embeds.shape) == 2 else face_embeds
        cond = project_plus_ipadapter_tokens(ipadapter, face_sequence)
        uncond = project_plus_ipadapter_tokens(ipadapter, mx.zeros_like(face_sequence))
    else:
        cond = project_faceid_mlp_tokens(ipadapter, face_embeds)
        uncond = project_faceid_mlp_tokens(ipadapter, mx.zeros_like(face_embeds))

    mx.eval(cond, uncond)
    return cond, uncond


def project_standard_ipadapter_embeds(ipadapter, clip_vision_output, negative_clip_vision_output=None):
    if ipadapter.get("is_faceid"):
        raise ValueError(
            "SDMLX: FaceID IP-Adapters need InsightFace embeds. Please use the SDMLX Apply IP-Adapter FaceID node."
        )
    if not ipadapter.get("is_sdxl"):
        raise ValueError("SDMLX: Currently only SDXL IP-Adapter is supported.")
    if ipadapter.get("is_full"):
        raise ValueError("SDMLX: IP-Adapter Full/MLP projection is not implemented yet.")
    if ipadapter.get("is_plus"):
        return project_plus_ipadapter_embeds(ipadapter, clip_vision_output, negative_clip_vision_output)

    embeds = clip_vision_tensor(clip_vision_output, "image_embeds")
    if embeds is None:
        raise ValueError("SDMLX: CLIP_VISION_OUTPUT does not contain image_embeds.")
    negative_embeds = clip_vision_tensor(negative_clip_vision_output, "image_embeds")
    if negative_embeds is None:
        negative_embeds = mx.zeros_like(embeds)

    image_proj = ipadapter["image_proj"]
    proj_weight = image_proj.get("proj.weight")
    proj_bias = image_proj.get("proj.bias")
    norm_weight = image_proj.get("norm.weight")
    norm_bias = image_proj.get("norm.bias")
    if proj_weight is None or norm_weight is None:
        raise ValueError("SDMLX: IP-Adapter image projection format not recognized.")
    expected_dim = int(proj_weight.shape[1])
    validate_ipadapter_embed_dim(embeds, expected_dim, "CLIP-Vision")
    validate_ipadapter_embed_dim(negative_embeds, expected_dim, "Negative-CLIP-Vision")

    def project(x):
        y = linear_with_bias(x, proj_weight, proj_bias)
        tokens = int(y.shape[-1] // ipadapter["output_cross_attention_dim"])
        y = y.reshape(y.shape[0], tokens, ipadapter["output_cross_attention_dim"])
        return mlx_layer_norm(y, norm_weight, norm_bias).astype(mx.float16)

    cond = project(embeds)
    uncond = project(negative_embeds)
    mx.eval(cond, uncond)
    return cond, uncond


def ipadapter_weight_value(weight, weight_composition, weight_type):
    if weight_type == "style transfer":
        return {6: float(weight)}
    if weight_type == "composition":
        return {3: float(weight_composition)}
    if weight_type == "strong style transfer":
        return {0: float(weight), 1: float(weight), 2: float(weight), 4: float(weight), 5: float(weight), 6: float(weight), 7: float(weight), 8: float(weight), 9: float(weight), 10: float(weight)}
    if weight_type == "style and composition":
        return {3: float(weight_composition), 6: float(weight)}
    if weight_type == "strong style and composition":
        return {0: float(weight), 1: float(weight), 2: float(weight), 3: float(weight_composition), 4: float(weight), 5: float(weight), 6: float(weight), 7: float(weight), 8: float(weight), 9: float(weight), 10: float(weight)}
    return float(weight)


def faceid_identity_base_weight(ipadapter):
    if ipadapter.get("is_portrait") or ipadapter.get("is_portrait_unnorm"):
        return 0.65
    return 1.0


def add_ipadapter_to_model(mlx_model, adapter):
    result = dict(mlx_model)
    adapters = list(result.get("ip_adapters", []))
    adapters.append(adapter)
    result["ip_adapters"] = adapters
    return result


def faceid_lora_candidates(ipadapter_name):
    lower = os.path.basename(str(ipadapter_name)).lower()
    if "faceid-plusv2" in lower:
        return ["ip-adapter-faceid-plusv2_sdxl_lora.safetensors"]
    if "faceid_sdxl" in lower or "faceid-sdxl" in lower:
        return ["ip-adapter-faceid_sdxl_lora.safetensors"]
    return []


def find_lora_model_by_basename(basenames):
    try:
        import folder_paths

        wanted = {name.lower() for name in basenames}
        matches = [
            name
            for name in folder_paths.get_filename_list("loras")
            if os.path.basename(name).lower() in wanted
        ]
        if not matches:
            return None, None
        matches.sort(key=lambda value: (0 if "ipadapter" in value.lower() else 1, value.lower()))
        name = matches[0]
        return name, folder_paths.get_full_path("loras", name)
    except Exception:
        return None, None


def add_lora_to_model_once(mlx_model, lora_name, path, strength_model, schedule=None, allow_partial=False):
    strength_model = lora_strength_for_item(strength_model, schedule)
    if path is None or strength_model == 0.0:
        return mlx_model, False
    abs_path = os.path.abspath(path)
    loras = list(mlx_model.get("loras", []))
    for item in loras:
        if os.path.abspath(item.get("path", "")) == abs_path:
            return mlx_model, False
    loras.append(
        {
            "name": lora_name,
            "path": abs_path,
            "strength_model": strength_model,
            "identity": lora_file_identity(abs_path),
            "schedule": schedule,
            "allow_partial": bool(allow_partial),
        }
    )
    return {**mlx_model, "loras": loras}, True


def sdmlx_sdxl_attention_module_bases():
    bases = []
    for block_index, transformer_count in ((1, 2), (2, 10)):
        for attention_index in range(2):
            for transformer_index in range(transformer_count):
                bases.append(
                    f"down_blocks.{block_index}.attentions.{attention_index}."
                    f"transformer_blocks.{transformer_index}"
                )
    for block_index, transformer_count in ((0, 10), (1, 2)):
        for attention_index in range(3):
            for transformer_index in range(transformer_count):
                bases.append(
                    f"up_blocks.{block_index}.attentions.{attention_index}."
                    f"transformer_blocks.{transformer_index}"
                )
    for transformer_index in range(10):
        bases.append(f"mid_blocks.1.transformer_blocks.{transformer_index}")
    return bases


def internal_faceid_lora_modules(ipadapter):
    modules = []
    ip_layers = ipadapter.get("ip_adapter", {})
    projection_map = {
        "q": "query_proj",
        "k": "key_proj",
        "v": "value_proj",
        "out": "out_proj",
    }
    for module_index, base in enumerate(sdmlx_sdxl_attention_module_bases()):
        for attention_offset, attention_name in ((0, "attn1"), (1, "attn2")):
            internal_index = module_index * 2 + attention_offset
            for source_name, target_name in projection_map.items():
                prefix = f"{internal_index}.to_{source_name}_lora"
                down = ip_layers.get(f"{prefix}.down.weight")
                up = ip_layers.get(f"{prefix}.up.weight")
                if down is None or up is None:
                    continue
                rank = int(down.shape[0])
                modules.append(
                    {
                        "target_base": f"{base}.{attention_name}.{target_name}",
                        "up": up,
                        "down": down,
                        "alpha": float(rank),
                        "rank": rank,
                    }
                )
    return modules


def add_internal_faceid_lora_to_model_once(mlx_model, ipadapter, strength_model, schedule=None):
    strength_model = lora_strength_for_item(strength_model, schedule)
    if strength_model == 0.0:
        return mlx_model, False
    modules = internal_faceid_lora_modules(ipadapter)
    if not modules:
        return mlx_model, False
    source_path = str(ipadapter.get("path") or ipadapter.get("name") or "faceid")
    lora_key = f"internal-faceid:{source_path}"
    loras = list(mlx_model.get("loras", []))
    for item in loras:
        if item.get("internal_lora_key") == lora_key:
            return mlx_model, False
    cache_key = ipadapter.get("cache_key")
    identity = {
        "path": lora_key,
        "size": len(modules),
        "mtime_ns": int(cache_key[2]) if isinstance(cache_key, tuple) and len(cache_key) > 2 else 0,
    }
    loras.append(
        {
            "name": f"{ipadapter.get('name', 'FaceID')} internal LoRA",
            "path": lora_key,
            "strength_model": strength_model,
            "identity": identity,
            "internal_lora_key": lora_key,
            "internal_lora_modules": modules,
            "schedule": schedule,
        }
    )
    return {**mlx_model, "loras": loras}, True


def lora_prefix_to_diffusers_key(prefix):
    if prefix.startswith("lora_unet_"):
        prefix = prefix[len("lora_unet_"):]
    key = prefix.replace("_", ".")
    fixes = {
        "input.blocks": "input_blocks",
        "middle.block": "middle_block",
        "output.blocks": "output_blocks",
        "down.blocks": "down_blocks",
        "up.blocks": "up_blocks",
        "mid.block": "mid_block",
        "in.layers": "in_layers",
        "out.layers": "out_layers",
        "emb.layers": "emb_layers",
        "skip.connection": "skip_connection",
        "transformer.blocks": "transformer_blocks",
        "to.k": "to_k",
        "to.q": "to_q",
        "to.v": "to_v",
        "to.out.0": "to_out.0",
        "time.emb.proj": "time_emb_proj",
        "time.embed": "time_embed",
        "conv.shortcut": "conv_shortcut",
        "proj.in": "proj_in",
        "proj.out": "proj_out",
        "conv.in": "conv_in",
        "conv.out": "conv_out",
    }
    for source, target in fixes.items():
        key = key.replace(source, target)
    key = f"{key}.weight"
    if key.startswith(("input_blocks.", "middle_block.", "output_blocks.")):
        mapped_key = ldm_unet_key_to_diffusers(key)
        if mapped_key is not None:
            return mapped_key
    return key


def diffusers_unet_key_to_sdmlx_targets(key):
    if "downsamplers" in key:
        key = key.replace("downsamplers.0.conv", "downsample")
    if "upsamplers" in key:
        key = key.replace("upsamplers.0.conv", "upsample")
    if "mid_block.resnets.0" in key:
        key = key.replace("mid_block.resnets.0", "mid_blocks.0")
    if "mid_block.attentions.0" in key:
        key = key.replace("mid_block.attentions.0", "mid_blocks.1")
    if "mid_block.resnets.1" in key:
        key = key.replace("mid_block.resnets.1", "mid_blocks.2")
    if "to_k" in key:
        key = key.replace("to_k", "key_proj")
    if "to_out.0" in key:
        key = key.replace("to_out.0", "out_proj")
    if "to_q" in key:
        key = key.replace("to_q", "query_proj")
    if "to_v" in key:
        key = key.replace("to_v", "value_proj")
    if "ff.net.2" in key:
        return [(key.replace("ff.net.2", "linear3"), None)]
    if "ff.net.0.proj" in key:
        return [
            (key.replace("ff.net.0.proj", "linear1"), 0),
            (key.replace("ff.net.0.proj", "linear2"), 1),
        ]
    return [(key, None)]


def mapped_lora_factor_arrays(up, down, split_index):
    def factor_array(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().float().numpy()
        return get_numpy_array(value)

    up_np = factor_array(up)
    down_np = factor_array(down)
    if split_index is not None:
        midpoint = up_np.shape[0] // 2
        up_np = up_np[:midpoint] if split_index == 0 else up_np[midpoint:]
    if up_np.ndim == 4 and up_np.shape[2:] == (1, 1):
        up_np = up_np[:, :, 0, 0]
    if down_np.ndim == 4 and down_np.shape[2:] == (1, 1):
        down_np = down_np[:, :, 0, 0]
    elif down_np.ndim == 4:
        down_np = down_np.transpose(0, 2, 3, 1)
    return up_np, down_np


def load_lora_modules(path):
    identity = lora_file_identity(path)
    cache_key = (identity["path"], identity["size"], identity["mtime_ns"])
    if cache_key in LORA_MODULES_CACHE:
        return LORA_MODULES_CACHE[cache_key]

    from comfy.utils import load_torch_file

    state = load_torch_file(path, safe_load=True)
    modules = []
    skipped_prefixes = []
    text_prefixes = 0
    prefixes = sorted(
        key[: -len(".lora_down.weight")]
        for key in state
        if key.endswith(".lora_down.weight")
    )
    for prefix in prefixes:
        if not prefix.startswith("lora_unet_"):
            if prefix.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
                text_prefixes += 1
            else:
                skipped_prefixes.append(prefix)
            continue

        up = state.get(f"{prefix}.lora_up.weight")
        down = state.get(f"{prefix}.lora_down.weight")
        if up is None or down is None:
            skipped_prefixes.append(prefix)
            continue

        rank = int(down.shape[0])
        alpha_tensor = state.get(f"{prefix}.alpha")
        alpha = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
        diffusers_key = lora_prefix_to_diffusers_key(prefix)
        for target_key, split_index in diffusers_unet_key_to_sdmlx_targets(diffusers_key):
            factor_base = target_key[: -len(".weight")] if target_key.endswith(".weight") else target_key
            target_base = normalize_speed_patch_target_base(factor_base)
            up_np, down_np = mapped_lora_factor_arrays(up, down, split_index)
            modules.append(
                {
                    "target_base": target_base,
                    "up": up_np,
                    "down": down_np,
                    "alpha": alpha,
                    "rank": int(down_np.shape[0]),
                }
            )

    result = {
        "identity": identity,
        "modules": modules,
        "skipped_prefixes": skipped_prefixes,
        "text_prefixes": text_prefixes,
        "state_prefix_count": len(prefixes),
    }
    LORA_MODULES_CACHE[cache_key] = result
    return result


def build_speed_patch_from_lora(source_path, patch_info, force_rebuild=False):
    from safetensors.numpy import save_file

    package_name = patch_info["package"]
    package_path = speed_patch_package_path(package_name)
    factors_path = os.path.join(package_path, "patch.safetensors")
    manifest = read_speed_patch_manifest(package_path)
    if (
        not force_rebuild
        and manifest
        and manifest.get("format") == "sdmlx-acceleration-patch-v1"
        and os.path.exists(factors_path)
    ):
        if package_path not in SPEED_PATCH_MARKED_PACKAGES:
            mark_macos_package(package_path)
            SPEED_PATCH_MARKED_PACKAGES.add(package_path)
        return {
            "name": package_name,
            "path": package_path,
            "label": speed_patch_label(package_name),
            "built": False,
            "modules": int(manifest.get("module_count", 0)),
        }

    lora_info = load_lora_modules(source_path)
    modules = lora_info.get("modules", [])
    if not modules:
        raise ValueError(
            "SDMLX: This speed LoRA contains no supported SDXL UNet LoRA modules "
            f"and cannot be converted into a Speed Patch: {source_path}"
        )

    if os.path.exists(package_path) and force_rebuild:
        shutil.rmtree(package_path)
    os.makedirs(package_path, exist_ok=True)

    factors = {}
    manifest_modules = []
    for module in modules:
        target_base = module["target_base"]
        up = np.ascontiguousarray(np.asarray(module["up"]))
        down = np.ascontiguousarray(np.asarray(module["down"]))
        alpha = float(module["alpha"])
        factors[f"{target_base}.lora_up"] = up
        factors[f"{target_base}.lora_down"] = down
        factors[f"{target_base}.alpha"] = np.array(alpha, dtype=np.float32)
        manifest_modules.append(
            {
                "target": f"{target_base}.weight",
                "rank": int(module["rank"]),
                "alpha": alpha,
                "up_shape": list(up.shape),
                "down_shape": list(down.shape),
            }
        )

    save_file(factors, factors_path)
    source_identity = lora_file_identity(source_path)
    manifest = {
        "format": "sdmlx-acceleration-patch-v1",
        "base_model_family": "sdxl",
        "created_at_unix": int(time.time()),
        "module_count": len(manifest_modules),
        "components": {"factors": "patch.safetensors"},
        "recommendations": patch_info.get("recommendations", {}),
        "source": {
            "repo": patch_info.get("source_repo"),
            "file": patch_info.get("source_file"),
            "license": patch_info.get("license"),
        },
        "source_identity": source_identity,
        "factor_layout": {
            "linear": {
                "lora_down": "[rank, in]",
                "lora_up": "[out, rank]",
            },
            "conv2d": {
                "lora_down": "[rank, kernel_h, kernel_w, in]",
                "lora_up": "[out, rank]",
            },
            "scale": "runtime_strength * alpha / rank",
        },
        "modules": manifest_modules,
    }
    with open(os.path.join(package_path, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    source_metadata = {
        "source_path": os.path.abspath(source_path),
        "source_repo": patch_info.get("source_repo"),
        "source_file": patch_info.get("source_file"),
        "license": patch_info.get("license"),
        "text_prefixes": lora_info.get("text_prefixes", 0),
        "skipped_prefixes": lora_info.get("skipped_prefixes", []),
        "source_identity": source_identity,
    }
    with open(os.path.join(package_path, "source_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(source_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    mark_macos_package(package_path)
    SPEED_PATCH_MARKED_PACKAGES.add(package_path)
    invalidate_speed_patch_options_cache()
    return {
        "name": package_name,
        "path": package_path,
        "label": speed_patch_label(package_name),
        "built": True,
        "modules": len(manifest_modules),
    }


def apply_loras_to_mapped_weights(mapped_weights, loras):
    if not loras:
        return mapped_weights, []

    weight_map = dict(mapped_weights)
    results = []
    for item in loras:
        if item.get("schedule"):
            continue
        strength = float(item.get("strength_model", 1.0))
        if strength == 0.0:
            continue
        internal_modules = item.get("internal_lora_modules")
        if internal_modules is not None:
            lora_info = {
                "modules": internal_modules,
                "skipped_prefixes": [],
                "text_prefixes": 0,
                "state_prefix_count": len(internal_modules),
            }
        else:
            lora_info = load_lora_modules(item["path"])
        applied = 0
        skipped = 0
        pending_eval = []
        for module in lora_info["modules"]:
            weight_key = f"{module['target_base']}.weight"
            if weight_key not in weight_map:
                skipped += 1
                continue
            base_weight = weight_map[weight_key]
            scale = strength * float(module["alpha"]) / float(module["rank"])
            try:
                delta = lora_delta(module["up"], module["down"], base_weight.shape, mx.float32)
            except Exception:
                skipped += 1
                continue
            updated = (base_weight.astype(mx.float32) + delta * scale).astype(base_weight.dtype)
            weight_map[weight_key] = updated
            pending_eval.append(updated)
            applied += 1
            if len(pending_eval) >= 24:
                mx.eval(*pending_eval)
                pending_eval.clear()
        if pending_eval:
            mx.eval(*pending_eval)
        lora_name = item.get("name", os.path.basename(item["path"]))
        module_count = len(lora_info["modules"])
        if module_count == 0:
            if lora_info["text_prefixes"]:
                reason = "it only contains text-encoder LoRA modules, which SDMLX currently does not patch"
            else:
                reason = "it contains no recognizable SDXL UNet LoRA modules"
            raise ValueError(
                f"SDMLX: LoRA `{lora_name}` cannot be applied to this SDXL UNet; {reason}. "
                "Check whether it is really an SDXL LoRA."
            )
        if applied == 0:
            raise ValueError(
                f"SDMLX: LoRA `{lora_name}` was not applied: 0/{module_count} UNet-modules match. "
                "This is very likely an SD1.5/Flux/other model-family LoRA instead of SDXL."
            )
        if skipped > applied and not bool(item.get("allow_partial", False)):
            raise ValueError(
                f"SDMLX: LoRA `{lora_name}` appears incompatible: {applied}/{module_count} modules applied, "
                f"{skipped} target modules skipped. Check whether this LoRA is meant for SDXL."
            )
        log_timing(
            f"SDMLX: LoRA {lora_name} applied "
            f"({applied} UNet modules, strength={strength:g})."
        )
        if lora_info["text_prefixes"]:
            log_timing(f"SDMLX: LoRA contains {lora_info['text_prefixes']} text-encoder prefixes; currently only UNet is patched.")
        if skipped or lora_info["skipped_prefixes"]:
            message = (
                "SDMLX: LoRA partially skipped: "
                f"{skipped} target weights, {len(lora_info['skipped_prefixes'])} Prefixes."
            )
            if skipped:
                print(message)
            else:
                log_timing(message)
        results.append(
            {
                "name": lora_name,
                "applied": applied,
                "skipped": skipped,
                "text_prefixes": lora_info["text_prefixes"],
            }
        )
    return list(weight_map.items()), results


def checkpoint_file_content_digest(path, stat=None):
    stat = os.stat(path) if stat is None else stat
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("utf-8"))
    chunk_size = 4 * 1024 * 1024
    offsets = [0]
    if stat.st_size > chunk_size:
        offsets.append(max(0, stat.st_size // 2 - chunk_size // 2))
        offsets.append(max(0, stat.st_size - chunk_size))
    seen = set()
    with open(path, "rb") as handle:
        for offset in offsets:
            if offset in seen:
                continue
            seen.add(offset)
            handle.seek(offset)
            digest.update(offset.to_bytes(8, "little", signed=False))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()[:24]


def checkpoint_cache_identity(path):
    stat = os.stat(path)
    source_path = os.path.abspath(path)
    source_content_digest = checkpoint_file_content_digest(path, stat)
    return {
        "cache_version": SDMLX_CACHE_VERSION,
        "source_name": os.path.basename(path),
        "source_path": source_path,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_content_digest": source_content_digest,
        "source_digest": source_content_digest[:16],
    }


def safe_package_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"[^A-Za-z0-9._ +()-]+", "_", base).strip(" .")
    return name or "checkpoint"


def manifest_path(package_path):
    return os.path.join(package_path, "manifest.json")


def read_cache_manifest(package_path):
    try:
        with open(manifest_path(package_path), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def manifest_matches_identity(manifest, identity):
    if not manifest:
        return False
    if manifest.get("source_content_digest") and identity.get("source_content_digest"):
        return all(manifest.get(key) == identity[key] for key in (
            "cache_version",
            "source_size",
            "source_content_digest",
        ))
    return all(manifest.get(key) == identity[key] for key in (
        "cache_version",
        "source_path",
        "source_size",
        "source_mtime_ns",
        "source_digest",
    ))


def manifest_matches_relocated_identity(manifest, identity):
    if not manifest:
        return False
    if not all(manifest.get(key) == identity[key] for key in ("cache_version", "source_name", "source_size")):
        return False

    manifest_content_digest = manifest.get("source_content_digest")
    if manifest_content_digest and identity.get("source_content_digest"):
        return manifest_content_digest == identity["source_content_digest"]

    old_path = manifest.get("source_path")
    if not old_path:
        return manifest.get("source_mtime_ns") == identity.get("source_mtime_ns")
    try:
        if os.path.exists(old_path):
            if os.path.realpath(old_path) == os.path.realpath(identity["source_path"]):
                return True
            if identity.get("source_content_digest"):
                old_stat = os.stat(old_path)
                if old_stat.st_size == identity["source_size"]:
                    return checkpoint_file_content_digest(old_path, old_stat) == identity["source_content_digest"]
            return False
        if manifest.get("source_mtime_ns") == identity.get("source_mtime_ns"):
            return True
        if not manifest.get("source_content_digest"):
            return True
    except Exception:
        return False
    return False


def cache_package_components_exist(package_path, manifest):
    components = (manifest or {}).get("components")
    required = ("unet", "clip_l", "clip_g", "vae")
    if not isinstance(components, dict):
        return False
    for component in required:
        entry = components.get(component)
        if entry is None:
            return False
        path = component_entry_path(package_path, component, entry)
        if not path or not os.path.exists(path):
            return False
    return True


def refresh_cache_manifest_identity(package_path, identity):
    manifest = read_cache_manifest(package_path)
    if not manifest or not cache_package_components_exist(package_path, manifest):
        return False
    manifest.update(identity)
    manifest.setdefault("package_format", "sdmlx-package-v3")
    with open(manifest_path(package_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    mark_macos_package(package_path)
    return True


MACOS_FINDER_FLAG_CUSTOM_ICON = 0x0400
MACOS_FINDER_FLAG_BUNDLE = 0x2000
MACOS_FINDER_FLAG_INVISIBLE = 0x4000
MACOS_CUSTOM_ICON_FILENAME = "Icon\r"


def resource_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", filename)


def add_macos_finder_flags(path, flags):
    try:
        finder_info = bytes(32)
        if hasattr(os, "getxattr"):
            try:
                finder_info = os.getxattr(path, "com.apple.FinderInfo")
            except OSError:
                finder_info = bytes(32)
        else:
            result = subprocess.run(
                ["xattr", "-px", "com.apple.FinderInfo", path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                finder_info = bytes.fromhex(result.stdout.strip().replace(" ", ""))
        data = bytearray(finder_info.ljust(32, b"\0")[:32])
        current = int.from_bytes(data[8:10], "big")
        data[8:10] = (current | flags).to_bytes(2, "big")
        if hasattr(os, "setxattr"):
            os.setxattr(path, "com.apple.FinderInfo", bytes(data))
        else:
            result = subprocess.run(
                ["xattr", "-wx", "com.apple.FinderInfo", bytes(data).hex(), path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode != 0:
                return False
        return True
    except Exception:
        return False


def macos_icon_has_resource_fork(path):
    try:
        if hasattr(os, "getxattr"):
            os.getxattr(path, "com.apple.ResourceFork")
            return True
        result = subprocess.run(
            ["xattr", "-p", "com.apple.ResourceFork", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def prepare_macos_custom_icon_file(path):
    try:
        result = subprocess.run(
            ["sips", "-i", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0 and macos_icon_has_resource_fork(path)
    except Exception:
        return False


def ensure_macos_custom_icon(path):
    icon_source = resource_path("sdmlx.icns")
    if not os.path.exists(icon_source):
        return False
    icon_target = os.path.join(path, MACOS_CUSTOM_ICON_FILENAME)
    try:
        needs_copy = (
            not os.path.exists(icon_target)
            or os.path.getsize(icon_target) != os.path.getsize(icon_source)
            or int(os.path.getmtime(icon_target)) < int(os.path.getmtime(icon_source))
        )
        if needs_copy:
            shutil.copyfile(icon_source, icon_target)
        if needs_copy or not macos_icon_has_resource_fork(icon_target):
            if not prepare_macos_custom_icon_file(icon_target):
                return False
        try:
            result = subprocess.run(
                ["SetFile", "-a", "V", icon_target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode != 0:
                add_macos_finder_flags(icon_target, MACOS_FINDER_FLAG_INVISIBLE)
        except Exception:
            add_macos_finder_flags(icon_target, MACOS_FINDER_FLAG_INVISIBLE)
        return True
    except Exception:
        return False


def mark_macos_package(path):
    if sys.platform != "darwin":
        return
    has_icon = ensure_macos_custom_icon(path)
    try:
        result = subprocess.run(
            ["SetFile", "-a", "BC" if has_icon else "B", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return
    except Exception:
        pass
    flags = MACOS_FINDER_FLAG_BUNDLE
    if has_icon:
        flags |= MACOS_FINDER_FLAG_CUSTOM_ICON
    add_macos_finder_flags(path, flags)


def checkpoint_cache_package(path):
    identity = checkpoint_cache_identity(path)
    base = safe_package_name(path)
    root = cache_dir()
    package_path = os.path.join(root, f"{base}.sdmlx")
    if os.path.exists(package_path):
        manifest = read_cache_manifest(package_path)
        if manifest_matches_identity(manifest, identity):
            return package_path, identity

        if manifest_matches_relocated_identity(manifest, identity) and refresh_cache_manifest_identity(package_path, identity):
            print(f"SDMLX: Reused existing package after checkpoint relocation: {os.path.basename(package_path)}")
            return package_path, identity

    for name in sorted(os.listdir(root)):
        candidate = os.path.join(root, name)
        if not name.endswith(".sdmlx") or not os.path.isdir(candidate) or candidate == package_path:
            continue
        manifest = read_cache_manifest(candidate)
        if manifest_matches_identity(manifest, identity):
            return candidate, identity
        if manifest_matches_relocated_identity(manifest, identity) and refresh_cache_manifest_identity(candidate, identity):
            print(f"SDMLX: Reused existing package after checkpoint relocation: {name}")
            return candidate, identity

    if not os.path.exists(package_path):
        return package_path, identity

    package_path = os.path.join(root, f"{base}-{identity['source_digest']}.sdmlx")
    return package_path, identity


def write_cache_manifest(package_path, identity, components):
    manifest = {
        **identity,
        "package_format": "sdmlx-package-v3",
        "components": components,
    }
    with open(manifest_path(package_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    mark_macos_package(package_path)


def weight_group_arrays(weights):
    return {k: get_numpy_array(v) for k, v in weights.items()}


def weight_group_digest(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()[:24]


def save_arrays_file(path, arrays):
    for suffix in (".safetensors", ".npz"):
        existing = path + suffix
        if os.path.exists(existing):
            os.remove(existing)
    try:
        from safetensors.numpy import save_file
        save_file(arrays, path + ".safetensors")
        return path + ".safetensors"
    except Exception:
        np.savez_compressed(path + ".npz", **arrays)
        return path + ".npz"


def load_weight_file(path):
    if path.endswith(".safetensors") and os.path.exists(path):
        return mx.load(path)
    if path.endswith(".npz") and os.path.exists(path):
        with np.load(path) as data:
            return {k: mx.array(data[k]) for k in data.files}
    return None


def load_weight_group(path):
    for suffix in (".safetensors", ".npz"):
        weights = load_weight_file(path + suffix)
        if weights is not None:
            return weights
    return None


def clone_or_copy_file(source_path, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        os.remove(target_path)
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["cp", "-c", source_path, target_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0 and os.path.exists(target_path):
                return "apfs_clone"
        except Exception:
            pass
    shutil.copy2(source_path, target_path)
    return "copy"


def reusable_package_component_path(component, digest, exclude_package_path=None):
    root = cache_dir()
    if not os.path.isdir(root):
        return None
    exclude = os.path.realpath(exclude_package_path) if exclude_package_path else None
    for name in sorted(os.listdir(root)):
        package_path = os.path.join(root, name)
        if not name.endswith(".sdmlx") or not os.path.isdir(package_path):
            continue
        if exclude and os.path.realpath(package_path) == exclude:
            continue
        manifest = read_cache_manifest(package_path) or {}
        entry = (manifest.get("components") or {}).get(component)
        if not isinstance(entry, dict) or entry.get("digest") != digest:
            continue
        path = component_entry_path(package_path, component, entry)
        if path and os.path.exists(path):
            return path
    return None


def save_package_component(package_path, component, weights, reuse_existing=False):
    arrays = weight_group_arrays(weights)
    digest = weight_group_digest(arrays)
    source_path = reusable_package_component_path(component, digest, package_path) if reuse_existing else None
    if source_path:
        suffix = ".safetensors" if source_path.endswith(".safetensors") else ".npz"
        target_path = os.path.join(package_path, component + suffix)
        clone_mode = clone_or_copy_file(source_path, target_path)
        saved_path = target_path
        log_timing(
            f"SDMLX: Package component reused: "
            f"{component}/{os.path.basename(saved_path)} ({clone_mode})"
        )
    else:
        saved_path = save_arrays_file(os.path.join(package_path, component), arrays)
        log_timing(f"SDMLX: Package component saved: {component}/{os.path.basename(saved_path)}")
    return {
        "storage": "package",
        "component": component,
        "digest": digest,
        "filename": os.path.basename(saved_path),
        "bytes": os.path.getsize(saved_path),
    }


def component_entry_path(package_path, component, entry):
    if not isinstance(entry, dict) or entry.get("storage") != "package":
        return None
    filename = entry.get("filename")
    if not filename:
        return None
    return os.path.join(package_path, filename)


def package_component_bytes(components):
    total = 0
    for entry in (components or {}).values():
        if isinstance(entry, dict):
            total += int(entry.get("bytes") or 0)
    return total


def load_cached_weight_group(package_path, component):
    manifest = read_cache_manifest(package_path)
    components = (manifest or {}).get("components")
    entry = components.get(component) if isinstance(components, dict) else None
    path = component_entry_path(package_path, component, entry)
    return load_weight_file(path) if path else None


def validate_text_encoder_weight_groups(groups, source_label, error_cls=ValueError):
    required = {
        "clip_l": (
            "text_model.embeddings.token_embedding.weight",
            "text_model.final_layer_norm.weight",
        ),
        "clip_g": (
            "text_model.embeddings.token_embedding.weight",
            "text_model.final_layer_norm.weight",
            "text_projection.weight",
        ),
    }
    missing = []
    for group_name, keys in required.items():
        weights = groups.get(group_name)
        if weights is None:
            missing.append(group_name)
            continue
        for key in keys:
            if key not in weights:
                missing.append(f"{group_name}.{key}")
    if missing:
        raise error_cls(
            "SDMLX: The text encoder weights are incomplete "
            f"({source_label}). Missing: {', '.join(missing[:6])}. "
            "Delete the affected .sdmlx package or reload the original checkpoint with SDMLX Loader Universal "
            "to rebuild it."
        )


def sdmlx_package_options():
    root = cache_dir()
    if not os.path.isdir(root):
        return []
    packages = sorted(
        name
        for name in os.listdir(root)
        if name.endswith(".sdmlx") and os.path.isdir(os.path.join(root, name))
    )
    for name in packages:
        mark_macos_package(os.path.join(root, name))
    return packages


def load_sdmlx_package(package_path, preload=False):
    manifest = read_cache_manifest(package_path)
    if manifest and manifest.get("cache_version") != SDMLX_CACHE_VERSION:
        package_name = os.path.basename(package_path)
        raise FileNotFoundError(
            f"SDMLX: Package {package_name} was created with an older converter "
            f"({manifest.get('cache_version', 'unknown')} != {SDMLX_CACHE_VERSION}) and must be rebuilt."
        )
    cached_unet = load_cached_weight_group(package_path, "unet")
    cached_clip_l = load_cached_weight_group(package_path, "clip_l")
    cached_clip_g = load_cached_weight_group(package_path, "clip_g")
    cached_vae = load_cached_weight_group(package_path, "vae")
    groups = {
        "unet": cached_unet,
        "clip_l": cached_clip_l,
        "clip_g": cached_clip_g,
        "vae": cached_vae,
    }
    missing = [name for name, weights in groups.items() if weights is None]
    if missing:
        package_name = os.path.basename(package_path)
        raise FileNotFoundError(
            f"SDMLX: Package {package_name} is incomplete: {', '.join(missing)} missing."
        )
    validate_text_encoder_weight_groups(groups, os.path.basename(package_path), error_cls=FileNotFoundError)

    mark_macos_package(package_path)
    log_timing(f"SDMLX: Loading MLX package from {os.path.basename(package_path)}...")
    log_timing(
        f"SDMLX STATUS: UNet={len(cached_unet)}, CLIP-L={len(cached_clip_l)}, "
        f"CLIP-G={len(cached_clip_g)}, VAE={len(cached_vae)}"
    )
    mlx_model = {"weights": cached_unet, "cache_key": package_path}
    mlx_clip = {"clip_l": cached_clip_l, "clip_g": cached_clip_g, "cache_key": package_path}
    mlx_vae = {"weights": cached_vae, "cache_key": package_path}
    if preload:
        preload_mlx_model(mlx_model, mlx_clip, mlx_vae, fast_mode=True, compute_dtype="float16", vae_dtype="float32")
    return mlx_model, mlx_clip, mlx_vae


def bytes_to_gb(value):
    return float(value) / (1024 ** 3)


def gb_to_bytes(value):
    return int(float(value) * (1024 ** 3))


def default_memory_reserve_gb():
    info = system_memory_info()
    total_gb = bytes_to_gb(info.get("total") or 0)
    if total_gb <= 18:
        return 8
    if total_gb <= 40:
        return 12
    if total_gb <= 80:
        return 24
    return 24


def active_memory_reserve_gb():
    reserve = MEMORY_CACHE_POLICY.get("reserve_gb")
    if reserve is None:
        reserve = default_memory_reserve_gb()
    return max(0, int(reserve))


def system_memory_info():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total": int(vm.total),
            "available": int(vm.available),
            "source": "psutil",
        }
    except Exception:
        pass

    total = 0
    available = 0
    try:
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except Exception:
        pass
    try:
        output = subprocess.check_output(["vm_stat"], text=True)
        page_size = 4096
        match = re.search(r"page size of (\d+) bytes", output)
        if match:
            page_size = int(match.group(1))
        fields = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            number = re.sub(r"[^0-9]", "", value)
            if number:
                fields[key.strip()] = int(number)
        available_pages = (
            fields.get("Pages free", 0)
            + fields.get("Pages inactive", 0)
            + fields.get("Pages speculative", 0)
            + fields.get("Pages purgeable", 0)
        )
        available = available_pages * page_size
    except Exception:
        pass
    return {"total": total, "available": available, "source": "vm_stat"}


def process_memory_bytes():
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


def mlx_dtype_itemsize(dtype):
    text = str(dtype).lower()
    if "float16" in text or "bfloat16" in text:
        return 2
    if "float32" in text:
        return 4
    if "float64" in text:
        return 8
    if "int8" in text or "uint8" in text or "bool" in text:
        return 1
    if "int16" in text or "uint16" in text:
        return 2
    if "int64" in text or "uint64" in text:
        return 8
    return 4


def estimate_array_bytes(value):
    if not is_mlx_array(value):
        return 0
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)
        except Exception:
            pass
    count = 1
    for dim in value.shape:
        count *= int(dim)
    return count * mlx_dtype_itemsize(value.dtype)


def estimate_tree_bytes(value):
    if is_mlx_array(value):
        return estimate_array_bytes(value)
    if isinstance(value, dict):
        return sum(estimate_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(estimate_tree_bytes(item) for item in value)
    return 0


def estimate_model_bytes(model):
    try:
        return estimate_tree_bytes(model.parameters())
    except Exception:
        return 0


def touch_model_cache(key):
    meta = MODEL_CACHE_META.get(key)
    if meta is not None:
        meta["last_used"] = time.monotonic()


def store_model_cache(key, model, kind, estimated_bytes=0):
    MODEL_CACHE[key] = model
    now = time.monotonic()
    MODEL_CACHE_META[key] = {
        "kind": kind,
        "bytes": int(estimated_bytes or 0),
        "created": now,
        "last_used": now,
        "source": str(key[0]) if isinstance(key, tuple) and key else "",
    }


def clear_compiled_step_denoisers_for_source(source):
    source = str(source)
    removed = 0
    for key in list(COMPILED_STEP_DENOISERS):
        if str(key[0]).startswith(source):
            COMPILED_STEP_DENOISERS.pop(key, None)
            removed += 1
    return removed


def release_mlx_cache_memory():
    gc.collect()
    try:
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is None:
            metal = getattr(mx, "metal", None)
            clear_cache = getattr(metal, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
    except Exception:
        pass


def mlx_memory_value(name):
    try:
        getter = getattr(mx, name, None)
        if getter is None:
            metal = getattr(mx, "metal", None)
            getter = getattr(metal, name, None)
        if getter is not None:
            return int(getter())
    except Exception:
        pass
    return 0


def mlx_cache_limit_bytes():
    mode = MEMORY_CACHE_POLICY.get("mode", "balanced")
    if mode == "manual":
        return None
    if mode == "low_memory":
        return gb_to_bytes(1)

    total = int(system_memory_info().get("total") or 0)
    total_gb = bytes_to_gb(total) if total else 64.0
    if mode == "keep_warm":
        return gb_to_bytes(min(24.0, max(8.0, total_gb * 0.28)))
    return gb_to_bytes(min(12.0, max(5.0, total_gb * 0.16)))


def configure_mlx_memory_limits():
    limit = mlx_cache_limit_bytes()
    key = (MEMORY_CACHE_POLICY.get("mode", "balanced"), limit)
    if MLX_MEMORY_LIMIT_STATE.get("cache_key") == key:
        return
    try:
        setter = getattr(mx, "set_cache_limit", None)
        if setter is None:
            metal = getattr(mx, "metal", None)
            setter = getattr(metal, "set_cache_limit", None)
        if setter is not None:
            if limit is None:
                original = MLX_MEMORY_LIMIT_STATE.get("original_cache_limit")
                if original is None:
                    MLX_MEMORY_LIMIT_STATE["cache_key"] = key
                    return
                previous = setter(int(original))
                log_timing(f"SDMLX: MLX Cache Limit aus (wiederhergestellt: {bytes_to_gb(original):.1f}GB).")
            else:
                previous = setter(int(limit))
                log_timing(f"SDMLX: MLX Cache Limit {bytes_to_gb(limit):.1f}GB ({key[0]}).")
            if MLX_MEMORY_LIMIT_STATE.get("original_cache_limit") is None:
                MLX_MEMORY_LIMIT_STATE["original_cache_limit"] = int(previous)
            MLX_MEMORY_LIMIT_STATE["cache_key"] = key
    except Exception:
        pass


def set_memory_assist(memory_assist):
    memory_assist = memory_assist if memory_assist in MEMORY_ASSIST_OPTIONS else "auto"
    mode_by_assist = {
        "auto": "balanced",
        "max_performance": "keep_warm",
        "low_memory": "low_memory",
        "off": "manual",
    }
    mode = mode_by_assist[memory_assist]
    MEMORY_CACHE_POLICY["mode"] = mode
    MEMORY_CACHE_POLICY["reserve_gb"] = default_memory_reserve_gb() if mode != "manual" else 0
    configure_mlx_memory_limits()
    log_timing(f"SDMLX Memory Assist: {memory_assist} ({mode}).")
    return memory_assist


def release_mlx_cache_memory_after_sampling():
    configure_mlx_memory_limits()
    mode = MEMORY_CACHE_POLICY.get("mode", "balanced")
    if mode == "manual":
        return
    if mode == "low_memory":
        release_mlx_cache_memory()
        return

    limit = mlx_cache_limit_bytes()
    cache_memory = mlx_memory_value("get_cache_memory")
    active_memory = mlx_memory_value("get_active_memory")
    hard_limit = int((limit or 0) * 2)
    if hard_limit and cache_memory > hard_limit:
        release_mlx_cache_memory()
        log_timing(
            "SDMLX: MLX Cache nach Sampling freigegeben "
            f"(active={bytes_to_gb(active_memory):.1f}GB, "
            f"cache={bytes_to_gb(cache_memory):.1f}GB > limit={bytes_to_gb(hard_limit):.1f}GB)."
        )


def release_mlx_cache_memory_after_decode():
    configure_mlx_memory_limits()
    mode = MEMORY_CACHE_POLICY.get("mode", "balanced")
    if mode == "manual":
        return
    if mode == "low_memory":
        release_mlx_cache_memory()
        return

    limit = mlx_cache_limit_bytes()
    cache_memory = mlx_memory_value("get_cache_memory")
    active_memory = mlx_memory_value("get_active_memory")
    hard_limit = int((limit or 0) * 2)
    if hard_limit and cache_memory > hard_limit:
        release_mlx_cache_memory()
        log_timing(
            "SDMLX: MLX Cache nach Decode freigegeben "
            f"(active={bytes_to_gb(active_memory):.1f}GB, "
            f"cache={bytes_to_gb(cache_memory):.1f}GB > limit={bytes_to_gb(hard_limit):.1f}GB)."
        )


def evict_model_cache_key(key):
    meta = MODEL_CACHE_META.pop(key, None) or {}
    MODEL_CACHE.pop(key, None)
    if meta.get("kind") == "unet" and isinstance(key, tuple) and key:
        clear_compiled_step_denoisers_for_source(key[0])
    release_mlx_cache_memory()
    return meta


def clear_sdmlx_model_cache(kinds=None):
    if kinds is not None:
        kinds = set(kinds)
    removed = {"unet": 0, "clip": 0, "vae": 0, "other": 0}
    freed = 0
    for key in list(MODEL_CACHE):
        meta = MODEL_CACHE_META.get(key, {})
        kind = meta.get("kind", "other")
        if kinds is not None and kind not in kinds:
            continue
        freed += int(meta.get("bytes", 0))
        removed[kind if kind in removed else "other"] += 1
        evict_model_cache_key(key)
    if kinds is None or "unet" in kinds:
        COMPILED_STEP_DENOISERS.clear()
    if kinds is None or "vae" in kinds:
        COMPILED_VAE_DECODERS.clear()
    release_mlx_cache_memory()
    return removed, freed


def is_plain_unet_cache_key(key):
    return (
        isinstance(key, tuple)
        and len(key) >= 12
        and key[1] == "unet"
        and not key[9]
        and not key[11]
    )


def evict_redundant_plain_unets(current_key):
    if not (isinstance(current_key, tuple) and len(current_key) >= 12 and current_key[1] == "unet"):
        return []
    current_source = current_key[0]
    current_has_variant = bool(current_key[9]) or bool(current_key[11])
    if not current_has_variant:
        return []
    evicted = []
    for key in list(MODEL_CACHE_META):
        if key == current_key:
            continue
        meta = MODEL_CACHE_META.get(key, {})
        if meta.get("kind") != "unet":
            continue
        if not is_plain_unet_cache_key(key):
            continue
        if key[0] != current_source:
            continue
        evicted.append((key, meta))
        evict_model_cache_key(key)
    return evicted


def enforce_memory_cache_policy(current_key=None):
    mode = MEMORY_CACHE_POLICY.get("mode", "balanced")
    reserve_gb = active_memory_reserve_gb()
    if mode == "manual" or reserve_gb <= 0:
        return []

    evicted = []
    candidates = [
        (meta.get("last_used", 0.0), key, meta)
        for key, meta in MODEL_CACHE_META.items()
        if meta.get("kind") == "unet" and key != current_key
    ]
    candidates.sort()

    if mode == "low_memory":
        for _, key, meta in candidates:
            evicted.append((key, meta))
            evict_model_cache_key(key)
        return evicted

    reserve_bytes = gb_to_bytes(reserve_gb)
    info = system_memory_info()
    total = int(info.get("total") or 0)
    available = int(info.get("available") or 0)
    process_memory = process_memory_bytes()
    process_limit = max(total - reserve_bytes, 0) if total else 0
    while candidates and (
        (available and available < reserve_bytes)
        or (process_limit and process_memory > process_limit)
    ):
        _, key, meta = candidates.pop(0)
        evicted.append((key, meta))
        evict_model_cache_key(key)
        info = system_memory_info()
        available = int(info.get("available") or 0)
        process_memory = process_memory_bytes()

    return evicted


def model_cache_summary():
    counts = {"unet": 0, "clip": 0, "vae": 0, "other": 0}
    bytes_by_kind = {"unet": 0, "clip": 0, "vae": 0, "other": 0}
    for meta in MODEL_CACHE_META.values():
        kind = meta.get("kind", "other")
        if kind not in counts:
            kind = "other"
        counts[kind] += 1
        bytes_by_kind[kind] += int(meta.get("bytes", 0))
    info = system_memory_info()
    total = int(info.get("total") or 0)
    available = int(info.get("available") or 0)
    process_memory = process_memory_bytes()
    reserve = active_memory_reserve_gb()
    lines = [
        f"mode={MEMORY_CACHE_POLICY.get('mode', 'balanced')}, reserve={reserve}GB",
        f"system available={bytes_to_gb(available):.1f}GB / total={bytes_to_gb(total):.1f}GB ({info.get('source', '-')})",
        f"python process rss={bytes_to_gb(process_memory):.1f}GB",
        (
            "mlx memory: "
            f"active={bytes_to_gb(mlx_memory_value('get_active_memory')):.1f}GB, "
            f"cache={bytes_to_gb(mlx_memory_value('get_cache_memory')):.1f}GB, "
            f"peak={bytes_to_gb(mlx_memory_value('get_peak_memory')):.1f}GB"
        ),
        (
            "cached models: "
            f"unet={counts['unet']} ({bytes_to_gb(bytes_by_kind['unet']):.1f}GB est), "
            f"clip={counts['clip']} ({bytes_to_gb(bytes_by_kind['clip']):.1f}GB est), "
            f"vae={counts['vae']} ({bytes_to_gb(bytes_by_kind['vae']):.1f}GB est)"
        ),
        f"compiled: step={len(COMPILED_STEP_DENOISERS)}, vae={len(COMPILED_VAE_DECODERS)}",
    ]
    return "\n".join(lines)


def tokenizer_is_usable(tokenizer):
    try:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer.get_vocab()))
    except Exception:
        vocab_size = 0
    vocab_file = tokenizer.init_kwargs.get("vocab_file") if hasattr(tokenizer, "init_kwargs") else None
    merges_file = tokenizer.init_kwargs.get("merges_file") if hasattr(tokenizer, "init_kwargs") else None
    return vocab_size > 1000 and bool(vocab_file) and bool(merges_file)


def get_tokenizer(model_id="openai/clip-vit-large-patch14", subfolder=None):
    cache_key = (model_id, subfolder)
    if cache_key not in TOKENIZER_CACHE:
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                model_id,
                subfolder=subfolder,
                local_files_only=True,
            )
            if not tokenizer_is_usable(tokenizer):
                raise RuntimeError(
                    f"Tokenizer cache for {model_id} is incomplete; downloading tokenizer files."
                )
        except Exception:
            tokenizer = CLIPTokenizer.from_pretrained(
                model_id,
                subfolder=subfolder,
            )
        if not tokenizer_is_usable(tokenizer):
            raise RuntimeError(
                f"SDMLX: CLIP tokenizer for {model_id} is incomplete or invalid."
            )
        TOKENIZER_CACHE[cache_key] = tokenizer
    return TOKENIZER_CACHE[cache_key]


def get_clip_l_tokenizer():
    return get_tokenizer("openai/clip-vit-large-patch14")


def get_clip_g_tokenizer():
    return get_tokenizer("openai/clip-vit-large-patch14")


def get_clip_model(cache_key, weights, is_g=False):
    from .mlx_sd.clip import CLIPTextModel
    from .mlx_sd.config import CLIPTextModelConfig

    key = (cache_key, "clip_g" if is_g else "clip_l")
    if key in MODEL_CACHE:
        touch_model_cache(key)
        return MODEL_CACHE[key]

    target_dim = 1280 if is_g else 768
    conf = CLIPTextModelConfig(
        num_layers=32 if is_g else 12,
        model_dims=target_dim,
        num_heads=20 if is_g else 12,
        projection_dim=1280 if is_g else None,
        hidden_act="gelu" if is_g else "quick_gelu",
    )
    clip = CLIPTextModel(conf)
    mapped_weights = map_clip_g_weights(weights) if is_g else map_clip_l_weights(weights)
    clip.update(tree_unflatten(mapped_weights))
    mx.eval(clip.parameters())
    store_model_cache(key, clip, "clip", estimate_model_bytes(clip))
    log_timing(f"SDMLX: CLIP-{'G' if is_g else 'L'} weights loaded ({len(mapped_weights)}).")
    return clip


def get_unet_model(
    cache_key,
    weights,
    quantize_unet=False,
    quant_bits=8,
    quant_group_size=64,
    fast_transformer=False,
    fast_ffn=False,
    fast_attention=False,
    compute_dtype="float32",
    speed_patch=SPEED_PATCH_NONE,
    speed_patch_strength=1.0,
    loras=None,
):
    from .mlx_sd.config import UNetConfig
    from .mlx_sd.unet import UNetModel

    quant_bits = max(2, int(quant_bits)) if quantize_unet else int(quant_bits)
    speed_patch_name = normalized_speed_patch_name(speed_patch)
    speed_patch_strength_key = round(float(speed_patch_strength), 6) if speed_patch_name else 0.0
    lora_key = lora_stack_key(loras)
    key = (
        cache_key,
        "unet",
        bool(quantize_unet),
        int(quant_bits),
        int(quant_group_size),
        bool(fast_transformer),
        bool(fast_ffn),
        bool(fast_attention),
        compute_dtype,
        speed_patch_name,
        speed_patch_strength_key,
        lora_key,
    )
    if key in MODEL_CACHE:
        touch_model_cache(key)
        return MODEL_CACHE[key]

    enforce_memory_cache_policy()
    conf = UNetConfig()
    conf.block_out_channels = [320, 640, 1280]
    conf.layers_per_block = [2, 2, 2]
    conf.transformer_layers_per_block = [1, 2, 10]
    conf.num_attention_heads = [5, 10, 20]
    conf.cross_attention_dim = [2048] * 3
    conf.down_block_types = [
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ]
    # Apple constructs up_blocks by iterating this list in reverse order.
    conf.up_block_types = [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ]
    conf.addition_embed_type = "text_time"
    conf.addition_time_embed_dim = 256
    conf.projection_class_embeddings_input_dim = 2816

    unet = UNetModel(conf)
    mapped_weights = []
    for weight_key, value in weights.items():
        mapped_weights.extend(map_unet_weights(weight_key, value))
    if compute_dtype == "float16":
        mapped_weights = cast_mapped_weights(mapped_weights, mx.float16)
        log_timing("SDMLX: UNet base weights prepared as float16 before patch/LoRA.")
    mapped_weights, patch_result = apply_speed_patch_to_mapped_weights(
        mapped_weights,
        speed_patch_name,
        speed_patch_strength_key,
    )
    mapped_weights, lora_results = apply_loras_to_mapped_weights(mapped_weights, loras)
    unet.update(tree_unflatten(mapped_weights))
    mapped_count = len(mapped_weights)
    if fast_transformer:
        patched = patch_unet_transformers(unet)
        log_timing(f"SDMLX: FastTransformer2D enabled ({patched} Module).")
    if fast_attention:
        patched = patch_unet_fast_attention(unet)
        log_timing(f"SDMLX: Fast Attention projections enabled ({patched} Attention-Module).")
    if fast_ffn:
        patched = patch_unet_fast_ffn(unet)
        log_timing(f"SDMLX: Fast FFN enabled ({patched} Transformer-Blocks).")
    if compute_dtype == "float16":
        unet.set_dtype(mx.float16)
        log_timing("SDMLX: UNet compute dtype float16 enabled.")
    elif compute_dtype == "conv_float16":
        patched = patch_unet_resnet_convs_float16(unet)
        log_timing(f"SDMLX: ResNet convs compute dtype float16 enabled ({patched} Module).")
    if quantize_unet:
        nn.quantize(
            unet,
            group_size=int(quant_group_size),
            bits=int(quant_bits),
            class_predicate=lambda _, module: isinstance(module, nn.Linear),
        )
        log_timing(f"SDMLX: UNet linear layers quantized ({quant_bits}-bit, group {quant_group_size}).")
    mx.eval(unet.parameters())
    store_model_cache(key, unet, "unet", estimate_model_bytes(unet))
    redundant_evicted = evict_redundant_plain_unets(key)
    evicted = enforce_memory_cache_policy(current_key=key)
    evicted = redundant_evicted + evicted
    if evicted:
        freed = sum(int(meta.get("bytes", 0)) for _, meta in evicted)
        log_timing(f"SDMLX: Warm cache released {len(evicted)} old UNet variant(s) (~{bytes_to_gb(freed):.1f}GB est).")
    log_timing(f"SDMLX: UNet weights loaded ({mapped_count}).")
    if patch_result and patch_result.get("recommendations"):
        rec = patch_result["recommendations"]
        log_timing(
            "SDMLX: Speed Patch recommendation: "
            f"steps={rec.get('steps', '-')}, "
            f"cfg={rec.get('cfg', '-')}, "
            f"sampler={rec.get('sampler_name', '-')}, "
            f"scheduler={rec.get('scheduler', '-')}, "
            f"force_no_cfg={rec.get('force_no_cfg', '-')}"
        )
    if lora_results:
        applied_total = sum(result["applied"] for result in lora_results)
        log_timing(f"SDMLX: LoRA stack active ({len(lora_results)} LoRA(s), {applied_total} UNet-modules applied).")
    return unet


def get_vae_model(cache_key, weights, compute_dtype="float32"):
    from .mlx_sd.config import AutoencoderConfig
    from .mlx_sd.vae import Autoencoder

    key = (cache_key, "vae", compute_dtype)
    if key in MODEL_CACHE:
        touch_model_cache(key)
        return MODEL_CACHE[key]

    vae = Autoencoder(
        AutoencoderConfig(
            in_channels=3,
            out_channels=3,
            latent_channels_out=8,
            latent_channels_in=4,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            norm_num_groups=32,
            scaling_factor=0.13025,
        )
    )
    mapped_count = apply_mapped_weights(vae, weights, map_vae_weights_for_apple)
    if compute_dtype == "float16":
        vae.set_dtype(mx.float16)
        log_timing("SDMLX: VAE compute dtype float16 enabled.")
    mx.eval(vae.parameters())
    store_model_cache(key, vae, "vae", estimate_model_bytes(vae))
    log_timing(f"SDMLX: VAE weights loaded ({mapped_count}).")
    return vae


def preload_mlx_model(mlx_model, mlx_clip, mlx_vae, fast_mode=True, compute_dtype="float16", vae_dtype="float32"):
    start_time = time.perf_counter()
    fast_mode = bool(fast_mode)
    static_loras, scheduled_loras = split_loras_by_schedule(mlx_model.get("loras", []))
    get_clip_model(mlx_clip["cache_key"], mlx_clip["clip_l"], is_g=False)
    get_clip_model(mlx_clip["cache_key"], mlx_clip["clip_g"], is_g=True)
    get_vae_model(mlx_vae["cache_key"], mlx_vae["weights"], vae_dtype)
    get_unet_model(
        mlx_model["cache_key"],
        mlx_model["weights"],
        False,
        8,
        64,
        fast_mode,
        fast_mode,
        fast_mode,
        compute_dtype,
        SPEED_PATCH_NONE,
        1.0,
        static_loras,
    )
    elapsed = time.perf_counter() - start_time
    message = (
        "SDMLX: Preload finished "
        f"(fast_mode={fast_mode}, compute_dtype={compute_dtype}, vae_dtype={vae_dtype}, "
        f"static_loras={len(static_loras)}, scheduled_loras={len(scheduled_loras)}, {elapsed:.2f}s)."
    )
    log_timing(message)
    return message


def log_timing(message):
    if TIMING_LOGS_ENABLED:
        print(message)


def resolve_size(size_preset, width, height):
    presets = {
        "1024x1024": (1024, 1024),
        "1152x896": (1152, 896),
        "896x1152": (896, 1152),
        "1216x832": (1216, 832),
        "832x1216": (832, 1216),
        "1344x768": (1344, 768),
        "768x1344": (768, 1344),
        "1536x640": (1536, 640),
        "640x1536": (640, 1536),
        "768x768": (768, 768),
        "832x832": (832, 832),
        "1024x768": (1024, 768),
        "768x1024": (768, 1024),
    }
    if size_preset in presets:
        return presets[size_preset]
    return width, height


def sigma_to_timestep_values(sampler, sigmas):
    sigma_table = np.array(sampler._sigmas, dtype=np.float32)
    indices = np.arange(len(sigma_table), dtype=np.float32)
    clipped = np.clip(np.array(sigmas, dtype=np.float32), sigma_table[0], sigma_table[-1])
    return np.interp(clipped, sigma_table, indices).astype(np.float32)


def karras_sigmas(sigma_min, sigma_max, steps, rho=7.0):
    ramp = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])


def exponential_sigmas(sigma_min, sigma_max, steps):
    sigma_min = max(float(sigma_min), 1e-6)
    sigma_max = max(float(sigma_max), sigma_min)
    sigmas = np.exp(
        np.linspace(
            math.log(sigma_max),
            math.log(sigma_min),
            steps,
            dtype=np.float32,
        )
    ).astype(np.float32)
    return np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])


def scheduler_timesteps(sampler, steps, scheduler_name):
    if scheduler_name == "normal":
        return sampler.timesteps(steps)

    sigma_table = np.array(sampler._sigmas, dtype=np.float32)
    sigma_max = float(sigma_table[-1])
    sigma_min = float(sigma_table[1])

    if scheduler_name == "simple":
        stride = len(sigma_table) / steps
        sigmas = [float(sigma_table[-(1 + int(index * stride))]) for index in range(steps)]
        sigmas.append(0.0)
    elif scheduler_name == "sgm_uniform":
        ts = np.linspace(sampler.max_time, 0.0, steps + 1, dtype=np.float32)[:-1]
        sigmas = np.array(sampler.sigmas(mx.array(ts)), dtype=np.float32)
        sigmas = np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])
    elif scheduler_name == "karras":
        sigmas = karras_sigmas(sigma_min, sigma_max, steps)
    elif scheduler_name == "exponential":
        sigmas = exponential_sigmas(sigma_min, sigma_max, steps)
    else:
        raise ValueError(f"Unknown SDMLX scheduler: {scheduler_name}")

    ts = sigma_to_timestep_values(sampler, sigmas)
    steps_array = mx.array(ts)
    return list(zip(steps_array[:-1], steps_array[1:]))


def scheduler_sigmas_and_timesteps(sampler, steps, scheduler_name):
    if scheduler_name == "normal":
        ts = np.linspace(sampler.max_time, 0.0, steps + 1, dtype=np.float32)
        sigmas = np.array(sampler.sigmas(mx.array(ts)), dtype=np.float32)
        return sigmas, ts

    sigma_table = np.array(sampler._sigmas, dtype=np.float32)
    sigma_max = float(sigma_table[-1])
    sigma_min = float(sigma_table[1])

    if scheduler_name == "simple":
        stride = len(sigma_table) / steps
        sigmas = np.array(
            [float(sigma_table[-(1 + int(index * stride))]) for index in range(steps)]
            + [0.0],
            dtype=np.float32,
        )
    elif scheduler_name == "sgm_uniform":
        ts_base = np.linspace(sampler.max_time, 0.0, steps + 1, dtype=np.float32)[:-1]
        sigmas = np.array(sampler.sigmas(mx.array(ts_base)), dtype=np.float32)
        sigmas = np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])
    elif scheduler_name == "karras":
        sigmas = karras_sigmas(sigma_min, sigma_max, steps)
    elif scheduler_name == "exponential":
        sigmas = exponential_sigmas(sigma_min, sigma_max, steps)
    else:
        raise ValueError(f"Unknown SDMLX scheduler: {scheduler_name}")

    ts = sigma_to_timestep_values(sampler, sigmas)
    return sigmas, ts


def scheduler_step_plan(sampler, steps, scheduler_name, sampler_name):
    sigmas, ts = scheduler_sigmas_and_timesteps(sampler, steps, scheduler_name)
    timesteps = mx.array(ts[:-1], dtype=mx.float32)
    next_timesteps = mx.array(ts[1:], dtype=mx.float32)
    plan = []
    for index in range(steps):
        sigma = float(sigmas[index])
        sigma_prev = float(sigmas[index + 1])
        sigma2 = sigma * sigma
        sigma_prev2 = sigma_prev * sigma_prev
        if sampler_name == "lcm":
            scale = math.sqrt(sigma2 + 1.0)
            dt = -sigma
            noise_scale = sigma_prev
            out_scale = 1.0 / math.sqrt(sigma_prev2 + 1.0)
        elif sampler_name in ("euler", "heun", "dpmpp_2m"):
            scale = math.sqrt(sigma2 + 1.0)
            dt = sigma_prev - sigma
            noise_scale = 0.0
            out_scale = 1.0 / math.sqrt(sigma_prev2 + 1.0)
        elif sampler_name == "euler_ancestral":
            if sigma == 0.0:
                sigma_up = 0.0
            else:
                sigma_up = math.sqrt(max(sigma_prev2 * (sigma2 - sigma_prev2) / sigma2, 0.0))
            sigma_down = math.sqrt(max(sigma_prev2 - sigma_up * sigma_up, 0.0))
            scale = math.sqrt(sigma2 + 1.0)
            dt = sigma_down - sigma
            noise_scale = sigma_up
            out_scale = 1.0 / math.sqrt(sigma_prev2 + 1.0)
        else:
            raise ValueError(f"Unknown SDMLX sampler: {sampler_name}")
        plan.append(
            (
                timesteps[index],
                next_timesteps[index],
                mx.array(scale, dtype=mx.float32),
                mx.array(dt, dtype=mx.float32),
                None if sampler_name == "euler" or noise_scale == 0.0 else mx.array(noise_scale, dtype=mx.float32),
                mx.array(out_scale, dtype=mx.float32),
                mx.array(sigma, dtype=mx.float32),
                mx.array(sigma_prev, dtype=mx.float32),
            )
        )
    return plan


def scheduler_step_plan_for_denoise(sampler, steps, scheduler_name, sampler_name, denoise):
    denoise = max(0.0, min(1.0, float(denoise)))
    steps = max(1, int(steps))
    if denoise >= 0.9999:
        return scheduler_step_plan(sampler, steps, scheduler_name, sampler_name)
    if denoise <= 0.0:
        return []
    full_steps = max(steps, int(steps / denoise))
    plan = scheduler_step_plan(sampler, full_steps, scheduler_name, sampler_name)
    return plan[-steps:]


def apply_sampler_step(noise_pred, latents, scale, dt, noise_scale, out_scale):
    latents = latents * (scale * out_scale) + noise_pred * (dt * out_scale)
    if noise_scale is not None:
        latents = latents + mx.random.normal(latents.shape).astype(latents.dtype) * (noise_scale * out_scale)
    return latents


def unscaled_latents(latents, sigma):
    return latents * mx.sqrt(sigma * sigma + 1.0)


def normalized_latents(latents, sigma):
    return latents * mx.rsqrt(sigma * sigma + 1.0)


def mx_scalar_float(value):
    return float(np.asarray(value).reshape(()))


def heun_sampler_step(latents, noise_pred, sigma, sigma_next, denoise_next):
    if mx_scalar_float(sigma_next) <= 0.0:
        return normalized_latents(unscaled_latents(latents, sigma) + noise_pred * (sigma_next - sigma), sigma_next)

    latents_unscaled = unscaled_latents(latents, sigma)
    euler_unscaled = latents_unscaled + noise_pred * (sigma_next - sigma)
    euler_latents = normalized_latents(euler_unscaled, sigma_next)
    noise_pred_next = denoise_next(euler_latents)
    heun_unscaled = latents_unscaled + (noise_pred + noise_pred_next) * 0.5 * (sigma_next - sigma)
    return normalized_latents(heun_unscaled, sigma_next)


def dpmpp_2m_sampler_step(latents, denoised, old_denoised, sigma, sigma_next, old_sigma):
    sigma_value = max(mx_scalar_float(sigma), 1e-12)
    sigma_next_value = max(mx_scalar_float(sigma_next), 0.0)
    if sigma_next_value <= 0.0:
        return normalized_latents(denoised, sigma_next)

    denoised_d = denoised
    if old_denoised is not None and old_sigma is not None:
        old_sigma_value = max(float(old_sigma), sigma_value + 1e-12)
        h = max(math.log(sigma_value / sigma_next_value), 1e-12)
        h_last = max(math.log(old_sigma_value / sigma_value), 1e-12)
        r = h_last / h
        denoised_d = denoised * (1.0 + 1.0 / (2.0 * r)) - old_denoised * (1.0 / (2.0 * r))

    ratio = sigma_next_value / sigma_value
    latents_unscaled = unscaled_latents(latents, sigma)
    next_unscaled = latents_unscaled * ratio + denoised_d * (1.0 - ratio)
    return normalized_latents(next_unscaled, sigma_next)


def noise_latents_at_sigma(latents, noise, sigma):
    sigma2 = sigma * sigma
    return (latents + noise * sigma) * mx.rsqrt(sigma2 + 1.0)


def denoised_latents_estimate(latents, noise_pred, scale, sigma):
    return latents * scale - noise_pred * sigma


def noise_pred_from_denoised(latents, denoised, scale, sigma):
    return (latents * scale - denoised) / sigma


def controlnet_residuals_for_step(
    controlnets,
    x_model,
    timestep,
    context,
    pooled,
    time_ids,
    width,
    height,
    step_percent,
    dtype,
    fast_transformer=False,
    fast_ffn=False,
    fast_attention=False,
):
    down_sum = None
    mid_sum = None
    for control in controlnets:
        strength = control_effective_strength(control, step_percent)
        if abs(strength) <= CONTROL_STRENGTH_EPSILON:
            continue
        model = get_controlnet_union_model(
            control["controlnet"],
            fast_transformer=fast_transformer,
            fast_ffn=fast_ffn,
            fast_attention=fast_attention,
        )
        control_image = control.get("prepared_image")
        if control_image is None:
            control_image = control_image_to_mlx(
                control["image"],
                width,
                height,
                len(x_model),
                dtype=dtype,
            )
        down, mid = model(
            x_model,
            timestep=timestep,
            encoder_x=context,
            control_image=control_image,
            control_type_idx=[int(control["control_type"])],
            conditioning_scale=mx.array(strength, dtype=dtype),
            text_time=(pooled, time_ids),
        )
        if down_sum is None:
            down_sum = down
            mid_sum = mid
        else:
            down_sum = [a + b for a, b in zip(down_sum, down)]
            mid_sum = mid_sum + mid
    return down_sum, mid_sum


def get_step_denoiser(cache_key, unet, compute_dtype, use_cfg, use_compiled):
    if not use_compiled:
        return None

    key = (cache_key, "step_denoise", compute_dtype, bool(use_cfg))
    if key not in COMPILED_STEP_DENOISERS:
        dtype = precision_dtype(compute_dtype)

        def step_denoise(latents, timestep, context, pooled, time_ids, cfg_value):
            x_in = mx.concatenate([latents] * 2) if use_cfg else latents
            x_model = x_in.astype(dtype) if compute_dtype == "float16" else x_in
            t_in = mx.broadcast_to(timestep, [len(x_in)])
            noise_pred = unet(
                x_model,
                timestep=t_in,
                encoder_x=context,
                text_time=(pooled, time_ids),
            ).astype(mx.float32)
            if use_cfg:
                eps_pos, eps_neg = mx.split(noise_pred, 2)
                noise_pred = eps_neg + cfg_value * (eps_pos - eps_neg)
            return noise_pred

        COMPILED_STEP_DENOISERS[key] = mx.compile(step_denoise)
        log_timing("SDMLX: Compiled step denoiser created.")

    return COMPILED_STEP_DENOISERS[key]


def mlx_scalar_float(value):
    mx.eval(value)
    return float(np.asarray(value).item())


def mlx_rms(value):
    value = value.astype(mx.float32)
    return mx.sqrt(mx.mean(mx.square(value)))


def sdmlx_conditioning_diagnostics_full():
    return SDMLX_CONDITIONING_DIAGNOSTICS_MODE in {"full", "all"}


def sdmlx_diagnostic_device_report():
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else {}
    except Exception:
        try:
            info = mx.metal.device_info()
        except Exception:
            info = {}
    device = str(info.get("device_name", "unknown"))
    architecture = str(info.get("architecture", "unknown"))
    memory = int(info.get("memory_size") or 0)
    working_set = int(info.get("max_recommended_working_set_size") or 0)
    return (
        f"mlx={getattr(mx, '__version__', 'unknown')}, "
        f"device={device}, architecture={architecture}, "
        f"memory={bytes_to_gb(memory):.1f}GB, recommended_working_set={bytes_to_gb(working_set):.1f}GB, "
        f"diagnostics={SDMLX_CONDITIONING_DIAGNOSTICS_MODE or 'off'}, "
        f"safe_mode={SDMLX_SAFE_MODE}, disable_compile={SDMLX_DISABLE_STEP_COMPILE}, "
        f"disable_fast_attention={SDMLX_DISABLE_FAST_ATTENTION}"
    )


def unet_cfg_probe(unet, latents, timestep, context, pooled, time_ids, cfg_value, compute_dtype):
    dtype = precision_dtype(compute_dtype)
    x_in = mx.concatenate([latents] * 2)
    x_model = x_in.astype(dtype) if compute_dtype == "float16" else x_in
    t_in = mx.broadcast_to(timestep, [len(x_in)])
    raw = unet(
        x_model,
        timestep=t_in,
        encoder_x=context,
        text_time=(pooled, time_ids),
    ).astype(mx.float32)
    eps_pos, eps_neg = mx.split(raw, 2)
    cfg_output = eps_neg + cfg_value * (eps_pos - eps_neg)
    cfg_delta = mlx_rms(eps_pos - eps_neg)
    return cfg_output, cfg_delta


def conditioning_probe_status(metrics):
    cond_delta = metrics.get("cond_delta", 0.0)
    pooled_delta = metrics.get("pooled_delta", 0.0)
    uncompiled_delta = metrics.get("uncompiled_cfg_delta")
    compiled_delta = metrics.get("compiled_cfg_delta")
    compiled_ratio = metrics.get("compiled_delta_ratio")
    compiled_diff = metrics.get("compiled_vs_uncompiled")
    fast_attention_off_delta = metrics.get("fast_attention_off_cfg_delta")

    conditioning_identical = cond_delta < 1e-5 and pooled_delta < 1e-5
    if conditioning_identical:
        return "conditioning-identical"
    if uncompiled_delta is not None and uncompiled_delta < 1e-5:
        if fast_attention_off_delta is not None and fast_attention_off_delta > 1e-5:
            return "fast-attention-conditioning-collapse"
        return "unet-conditioning-weak"
    if (
        compiled_ratio is not None
        and uncompiled_delta is not None
        and uncompiled_delta > 1e-5
        and compiled_ratio < 0.05
    ):
        return "compiled-conditioning-collapse"
    if compiled_delta is not None and compiled_delta < 1e-5:
        return "fast-path-conditioning-collapse"
    if compiled_diff is not None and compiled_diff > 0.05:
        return "compiled-output-drift"
    return "ok"


def evaluate_conditioning_probe(
    unet,
    step_denoiser,
    latents,
    timestep,
    context,
    pooled,
    time_ids,
    cfg_value,
    compute_dtype,
    include_uncompiled=True,
    include_compiled=True,
    return_uncompiled_cfg=False,
    return_compiled_cfg=False,
):
    cond_pos, cond_neg = mx.split(context, 2)
    pooled_pos, pooled_neg = mx.split(pooled, 2)
    cond_delta = mlx_rms(cond_pos - cond_neg)
    pooled_delta = mlx_rms(pooled_pos - pooled_neg)

    uncompiled_cfg = None
    uncompiled_cfg_delta = None
    if include_uncompiled:
        uncompiled_cfg, uncompiled_cfg_delta = unet_cfg_probe(
            unet,
            latents,
            timestep,
            context,
            pooled,
            time_ids,
            cfg_value,
            compute_dtype,
        )

    compiled_cfg_delta = None
    compiled_vs_uncompiled = None
    compiled_cfg = None
    if include_compiled and step_denoiser is not None:
        compiled_cfg = step_denoiser(latents, timestep, context, pooled, time_ids, cfg_value)
        compiled_neg = step_denoiser(
            latents,
            timestep,
            context,
            pooled,
            time_ids,
            mx.array(0.0, dtype=mx.float32),
        )
        cfg_scale = mx.maximum(mx.abs(cfg_value), mx.array(1e-6, dtype=mx.float32))
        compiled_cfg_delta = mlx_rms((compiled_cfg - compiled_neg) / cfg_scale)
        if uncompiled_cfg is not None:
            compiled_vs_uncompiled = mlx_rms(compiled_cfg - uncompiled_cfg) / mx.maximum(
                mlx_rms(uncompiled_cfg),
                mx.array(1e-8, dtype=mx.float32),
            )

    eval_values = [cond_delta, pooled_delta]
    if uncompiled_cfg_delta is not None:
        eval_values.append(uncompiled_cfg_delta)
    if compiled_cfg_delta is not None:
        eval_values.append(compiled_cfg_delta)
    if compiled_vs_uncompiled is not None:
        eval_values.append(compiled_vs_uncompiled)
    mx.eval(*eval_values)

    cond_delta_f = mlx_scalar_float(cond_delta)
    pooled_delta_f = mlx_scalar_float(pooled_delta)
    uncompiled_delta_f = (
        mlx_scalar_float(uncompiled_cfg_delta)
        if uncompiled_cfg_delta is not None
        else None
    )
    compiled_delta_f = (
        mlx_scalar_float(compiled_cfg_delta)
        if compiled_cfg_delta is not None
        else None
    )
    compiled_diff_f = (
        mlx_scalar_float(compiled_vs_uncompiled)
        if compiled_vs_uncompiled is not None
        else None
    )
    compiled_ratio_f = (
        compiled_delta_f / uncompiled_delta_f
        if compiled_delta_f is not None and uncompiled_delta_f is not None and uncompiled_delta_f > 1e-8
        else None
    )

    metrics = {
        "cond_delta": cond_delta_f,
        "pooled_delta": pooled_delta_f,
        "uncompiled_cfg_delta": uncompiled_delta_f,
        "compiled_cfg_delta": compiled_delta_f,
        "compiled_delta_ratio": compiled_ratio_f,
        "compiled_vs_uncompiled": compiled_diff_f,
    }
    if return_uncompiled_cfg:
        metrics["uncompiled_cfg"] = uncompiled_cfg
    if return_compiled_cfg:
        metrics["compiled_cfg"] = compiled_cfg
    metrics["status"] = conditioning_probe_status(metrics)
    return metrics


def conditioning_probe_summary(metrics):
    parts = [
        f"status={metrics.get('status', 'unknown')}",
        f"cond_delta={metrics.get('cond_delta', 0.0):.6g}",
        f"pooled_delta={metrics.get('pooled_delta', 0.0):.6g}",
    ]
    uncompiled_delta = metrics.get("uncompiled_cfg_delta")
    if uncompiled_delta is not None:
        parts.append(f"uncompiled_cfg_delta={uncompiled_delta:.6g}")
    compiled_delta = metrics.get("compiled_cfg_delta")
    if compiled_delta is None:
        parts.append("compiled=inactive")
    else:
        parts.append(f"compiled_cfg_delta={compiled_delta:.6g}")
        ratio = metrics.get("compiled_delta_ratio")
        if ratio is not None:
            parts.append(f"compiled_delta_ratio={ratio:.4f}")
        diff = metrics.get("compiled_vs_uncompiled")
        if diff is not None:
            parts.append(f"compiled_vs_uncompiled={diff:.6g}")
    fast_attention_off_delta = metrics.get("fast_attention_off_cfg_delta")
    if fast_attention_off_delta is not None:
        parts.append(f"fast_attention_off_cfg_delta={fast_attention_off_delta:.6g}")
        parts.append(f"fast_attention_off_vs_current={metrics.get('fast_attention_off_vs_current', 0.0):.6g}")
    return ", ".join(parts)


def run_conditioning_diagnostics(
    mlx_model,
    unet,
    step_denoiser,
    latents,
    timestep,
    context,
    pooled,
    time_ids,
    cfg_value,
    compute_dtype,
    quantize_unet,
    quant_bits,
    quant_group_size,
    fast_transformer,
    effective_fast_ffn,
    effective_fast_attention,
    speed_patch,
    speed_patch_strength,
    static_loras,
):
    global SDMLX_CONDITIONING_DIAGNOSTICS_HEADER_PRINTED
    if not SDMLX_CONDITIONING_DIAGNOSTICS:
        return None
    try:
        if not SDMLX_CONDITIONING_DIAGNOSTICS_HEADER_PRINTED:
            print(f"SDMLX Conditioning Diagnostics: {sdmlx_diagnostic_device_report()}")
            SDMLX_CONDITIONING_DIAGNOSTICS_HEADER_PRINTED = True

        metrics = evaluate_conditioning_probe(
            unet,
            step_denoiser,
            latents,
            timestep,
            context,
            pooled,
            time_ids,
            cfg_value,
            compute_dtype,
            include_uncompiled=True,
            include_compiled=True,
            return_uncompiled_cfg=True,
        )
        diagnostic_cache_keys = []
        uncompiled_cfg = metrics.get("uncompiled_cfg")
        if sdmlx_conditioning_diagnostics_full() and effective_fast_attention:
            cache_before = set(MODEL_CACHE)
            safe_unet = get_unet_model(
                mlx_model["cache_key"],
                mlx_model["weights"],
                quantize_unet,
                quant_bits,
                quant_group_size,
                fast_transformer,
                effective_fast_ffn,
                False,
                compute_dtype,
                speed_patch,
                speed_patch_strength,
                static_loras,
            )
            safe_cfg, fast_attention_off_delta = unet_cfg_probe(
                safe_unet,
                latents,
                timestep,
                context,
                pooled,
                time_ids,
                cfg_value,
                compute_dtype,
            )
            fast_attention_off_vs_current = mlx_rms(safe_cfg - uncompiled_cfg) / mx.maximum(
                mlx_rms(uncompiled_cfg),
                mx.array(1e-8, dtype=mx.float32),
            )
            diagnostic_cache_keys = [
                key
                for key in set(MODEL_CACHE) - cache_before
                if MODEL_CACHE_META.get(key, {}).get("kind") == "unet"
            ]
            mx.eval(fast_attention_off_delta, fast_attention_off_vs_current)
            metrics["fast_attention_off_cfg_delta"] = mlx_scalar_float(fast_attention_off_delta)
            metrics["fast_attention_off_vs_current"] = mlx_scalar_float(fast_attention_off_vs_current)
            metrics["status"] = conditioning_probe_status(metrics)
        for key in diagnostic_cache_keys:
            evict_model_cache_key(key)

        status = metrics.get("status", "unknown")
        print(f"SDMLX Conditioning Diagnostics: {conditioning_probe_summary(metrics)}")
        if status in {"compiled-conditioning-collapse", "fast-path-conditioning-collapse"}:
            print(
                "SDMLX Conditioning Diagnostics: compiled denoiser appears to ignore text conditioning. "
                "Retest with SDMLX_DISABLE_STEP_COMPILE=1."
            )
        elif status == "fast-attention-conditioning-collapse":
            print(
                "SDMLX Conditioning Diagnostics: Fast Attention appears to ignore text conditioning. "
                "Retest with SDMLX_DISABLE_FAST_ATTENTION=1."
            )
        elif status == "unet-conditioning-weak":
            print(
                "SDMLX Conditioning Diagnostics: uncompiled UNet output barely changes between positive "
                "and negative conditioning. Retest with SDMLX_DISABLE_FAST_ATTENTION=1."
            )
        return metrics
    except Exception as exc:
        print(f"SDMLX Conditioning Diagnostics: failed ({exc}).")
        return None


def conditioning_guard_should_fail(metrics):
    return metrics.get("status") in {
        "conditioning-identical",
        "fast-path-conditioning-collapse",
        "compiled-conditioning-collapse",
        "fast-attention-conditioning-collapse",
        "unet-conditioning-weak",
        "compiled-output-drift",
    }


def conditioning_guard_cache_result(guard_key, metrics):
    cached = dict(metrics)
    cached.pop("uncompiled_cfg", None)
    cached.pop("compiled_cfg", None)
    CONDITIONING_GUARD_CACHE[guard_key] = cached
    return cached


def conditioning_guard_error(metrics):
    return "\n".join([
        "SDMLX Conditioning Guard stopped this run.",
        "",
        "The fast MLX sampler did not show a reliable text-conditioning response. "
        "This usually means the generated image would ignore the prompt or look effectively unconditional, "
        "so SDMLX stops before wasting a full run.",
        "",
        f"Guard metrics: {conditioning_probe_summary(metrics)}",
        f"Device: {sdmlx_diagnostic_device_report()}",
        "",
        "Next steps:",
        "1. Share the Guard metrics and Device line in the GitHub issue/community thread.",
        "2. Run diagnostics and share the two 'SDMLX Conditioning Diagnostics' lines:",
        "   SDMLX_CONDITIONING_DIAGNOSTICS=full /Applications/ComfyUI.app/Contents/MacOS/ComfyUI",
        "   For shell installs: SDMLX_CONDITIONING_DIAGNOSTICS=full python main.py",
        "3. To verify the compatibility path, restart with SDMLX_SAFE_MODE=1. "
        "This is slower and is never enabled silently.",
    ])


def conditioning_texts_identical(positive, negative):
    positive_text = positive.get("text") if isinstance(positive, dict) else None
    negative_text = negative.get("text") if isinstance(negative, dict) else None
    if isinstance(positive_text, str) and isinstance(negative_text, str):
        return positive_text.strip() == negative_text.strip()
    return False


def maybe_run_conditioning_guard(
    guard_key,
    unet,
    step_denoiser,
    latents,
    timestep,
    context,
    pooled,
    time_ids,
    cfg_value,
    compute_dtype,
    diagnostic_metrics=None,
):
    if not SDMLX_CONDITIONING_GUARD or step_denoiser is None:
        return
    cached = CONDITIONING_GUARD_CACHE.get(guard_key)
    if cached is not None:
        if conditioning_guard_should_fail(cached):
            raise RuntimeError(conditioning_guard_error(cached))
        return

    metrics = diagnostic_metrics
    if metrics is None or metrics.get("compiled_cfg_delta") is None:
        metrics = evaluate_conditioning_probe(
            unet,
            step_denoiser,
            latents,
            timestep,
            context,
            pooled,
            time_ids,
            cfg_value,
            compute_dtype,
            include_uncompiled=False,
            include_compiled=True,
            return_compiled_cfg=True,
        )

    cached = conditioning_guard_cache_result(guard_key, metrics)
    if conditioning_guard_should_fail(cached):
        print(f"SDMLX Conditioning Guard: failed ({conditioning_probe_summary(cached)}).")
        raise RuntimeError(conditioning_guard_error(cached))

    ratio = cached.get("compiled_delta_ratio")
    ratio_text = f", ratio={ratio:.4f}" if ratio is not None else ""
    print(
        "SDMLX: Conditioning guard ok "
        f"(fast_cfg_delta={cached.get('compiled_cfg_delta', 0.0):.6g}{ratio_text})."
    )
    return metrics.get("compiled_cfg")


def profiled_unet_block(block, x, profile, label, encoder_x=None, temb=None, residual_hidden_states=None):
    from .mlx_sd.unet import upsample_nearest

    output_states = []
    for index in range(len(block.resnets)):
        if residual_hidden_states is not None:
            x = mx.concatenate([x, residual_hidden_states.pop()], axis=-1)

        x = timed_detail(
            profile,
            f"{label}.resnets.{index}",
            lambda index=index, x=x: block.resnets[index](x, temb),
        )

        if "attentions" in block:
            x = timed_detail(
                profile,
                f"{label}.attentions.{index}",
                lambda index=index, x=x: profiled_transformer2d(
                    block.attentions[index],
                    x,
                    encoder_x,
                    profile,
                    f"{label}.attentions.{index}",
                ),
            )

        output_states.append(x)

    if "downsample" in block:
        x = timed_detail(profile, f"{label}.downsample", lambda: block.downsample(x))
        output_states.append(x)

    if "upsample" in block:
        x = timed_detail(profile, f"{label}.upsample", lambda: block.upsample(upsample_nearest(x)))
        output_states.append(x)

    return x, output_states


def profiled_transformer_block(block, x, memory, profile, label):
    y_norm = timed_detail(profile, f"{label}.norm1", lambda: block.norm1(x))
    y = timed_detail(profile, f"{label}.attn1", lambda: block.attn1(y_norm, y_norm, y_norm, None))
    x = x + y

    y_norm = timed_detail(profile, f"{label}.norm2", lambda: block.norm2(x))
    y = timed_detail(profile, f"{label}.attn2", lambda: block.attn2(y_norm, memory, memory, None))
    x = x + y

    y_norm = timed_detail(profile, f"{label}.norm3", lambda: block.norm3(x))

    def ffn():
        if "ffn" in block:
            return block.ffn(y_norm)
        y_a = block.linear1(y_norm)
        y_b = block.linear2(y_norm)
        return block.linear3(y_a * nn.gelu(y_b))

    y = timed_detail(profile, f"{label}.ffn", ffn)
    return x + y


def profiled_transformer2d(transformer, x, encoder_x, profile, label):
    input_x = x
    B, H, W, C = x.shape

    x = timed_detail(
        profile,
        f"{label}.norm_proj_in",
        lambda: transformer.proj_in(transformer.norm(x).reshape(B, -1, C)),
    )

    for index, block in enumerate(transformer.transformer_blocks):
        x = profiled_transformer_block(block, x, encoder_x, profile, f"{label}.transformer_blocks.{index}")

    x = timed_detail(profile, f"{label}.proj_out", lambda: transformer.proj_out(x))
    x = x.reshape(B, H, W, C)
    return x + input_x


def profiled_unet_forward(unet, x, timestep, encoder_x, pooled, time_ids, profile):
    profile["calls"] += 1

    temb = unet.timesteps(timestep).astype(x.dtype)
    temb = unet.time_embedding(temb)
    emb = unet.add_time_proj(time_ids).flatten(1).astype(x.dtype)
    emb = mx.concatenate([pooled, emb], axis=-1)
    emb = unet.add_embedding(emb)
    temb = temb + emb

    x = timed_eval("conv_in", lambda: unet.conv_in(x), profile, "conv_in")

    residuals = [x]
    for index, block in enumerate(unet.down_blocks):
        x, res = timed_eval(
            f"down_blocks.{index}",
            lambda block=block, x=x, index=index: profiled_unet_block(
                block,
                x,
                profile,
                f"down_blocks.{index}",
                encoder_x=encoder_x,
                temb=temb,
            ),
            profile,
            "down_blocks",
            index,
        )
        residuals.extend(res)

    x = timed_eval("mid_blocks.0", lambda: unet.mid_blocks[0](x, temb), profile, "mid_blocks", 0)
    x = timed_eval(
        "mid_blocks.1",
        lambda: unet.mid_blocks[1](x, encoder_x, None, None),
        profile,
        "mid_blocks",
        1,
    )
    x = timed_eval("mid_blocks.2", lambda: unet.mid_blocks[2](x, temb), profile, "mid_blocks", 2)

    for index, block in enumerate(unet.up_blocks):
        x, _ = timed_eval(
            f"up_blocks.{index}",
            lambda block=block, x=x, index=index: profiled_unet_block(
                block,
                x,
                profile,
                f"up_blocks.{index}",
                encoder_x=encoder_x,
                temb=temb,
                residual_hidden_states=residuals,
            ),
            profile,
            "up_blocks",
            index,
        )

    def finish():
        y = unet.conv_norm_out(x)
        y = nn.silu(y)
        return unet.conv_out(y)

    return timed_eval("out", finish, profile, "out")


def encode_text_pair(mlx_clip, positive_text, negative_text, conditioning_mode="normal"):
    cache_key = (CONDITIONING_CACHE_VERSION, mlx_clip["cache_key"], positive_text, negative_text, conditioning_mode)
    if cache_key in CONDITIONING_CACHE:
        pos, neg = CONDITIONING_CACHE[cache_key]
        log_timing("SDMLX: Text pair loaded from RAM cache.")
        return pos, neg

    clip_l = get_clip_model(mlx_clip["cache_key"], mlx_clip["clip_l"], is_g=False)
    clip_g = get_clip_model(mlx_clip["cache_key"], mlx_clip["clip_g"], is_g=True)
    tokenizer_l = get_clip_l_tokenizer()
    tokenizer_g = get_clip_g_tokenizer()
    tokens_l = mx.array(
        tokenizer_l(
            [positive_text, negative_text],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )["input_ids"]
    )
    tokens_g = mx.array(
        tokenizer_g(
            [positive_text, negative_text],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )["input_ids"]
    )

    output_l = clip_l(tokens_l)
    output_g = clip_g(tokens_g)
    hidden_l = output_l.hidden_states[-2] if output_l.hidden_states and len(output_l.hidden_states) >= 2 else output_l.last_hidden_state
    if conditioning_mode == "clip_g_last":
        hidden_g = output_g.last_hidden_state
    else:
        hidden_g = output_g.hidden_states[-2] if output_g.hidden_states and len(output_g.hidden_states) >= 2 else output_g.last_hidden_state

    if conditioning_mode == "clip_l_only":
        hidden_g = mx.zeros_like(hidden_g)
    elif conditioning_mode == "clip_g_only":
        hidden_l = mx.zeros_like(hidden_l)

    cond = mx.concatenate([hidden_l, hidden_g], axis=2)
    pooled = output_g.pooled_output if hasattr(output_g, "pooled_output") else mx.zeros((2, 1280))
    if conditioning_mode == "zero_pooled":
        pooled = mx.zeros_like(pooled)
    mx.eval(cond, pooled)

    positive = {"cond": cond[0:1], "pooled": pooled[0:1], "text": positive_text}
    negative = {"cond": cond[1:2], "pooled": pooled[1:2], "text": negative_text}
    CONDITIONING_CACHE[cache_key] = (positive, negative)
    return positive, negative


def make_comfy_progress_bar(total_steps):
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(total_steps)
    except Exception as exc:
        log_timing(f"SDMLX: Comfy ProgressBar not available: {exc}")
        return None


def make_terminal_progress_bar(total_steps, description="SDMLX Sampling", unit="step"):
    try:
        from tqdm.auto import tqdm

        kwargs = {
            "total": total_steps,
            "desc": description,
            "unit": unit,
            "dynamic_ncols": True,
            "leave": True,
            "smoothing": 0.3,
        }
        try:
            return tqdm(**kwargs, colour="green")
        except TypeError:
            return tqdm(**kwargs)
    except Exception as exc:
        log_timing(f"SDMLX: Terminal ProgressBar not available: {exc}")
        return None


def preview_tuple_from_image(preview_image):
    try:
        import latent_preview

        return ("JPEG", preview_image, latent_preview.MAX_PREVIEW_RESOLUTION)
    except Exception:
        return ("JPEG", preview_image, 512)


def torch_image_to_pil_first(image):
    image_t = image.detach().cpu().float() if hasattr(image, "detach") else torch.from_numpy(get_numpy_array(image).astype(np.float32))
    if image_t.ndim != 4 or image_t.shape[-1] != 3:
        return None
    pixels = torch.clamp(image_t[0], 0.0, 1.0).numpy()
    return Image.fromarray((pixels * 255.0).astype(np.uint8))


def compose_preview_image(preview_image, preview_mode, crop_info):
    if preview_mode == "crop" and crop_info and "preview_crop" in crop_info:
        try:
            left, top, width, height = [int(value) for value in crop_info["preview_crop"]]
            source_w, source_h = crop_info.get("target_size", preview_image.size)
            if source_w > 0 and source_h > 0 and (source_w, source_h) != preview_image.size:
                scale_x = float(preview_image.size[0]) / float(source_w)
                scale_y = float(preview_image.size[1]) / float(source_h)
                left = int(round(left * scale_x))
                top = int(round(top * scale_y))
                width = int(round(width * scale_x))
                height = int(round(height * scale_y))
            right = min(preview_image.size[0], left + max(1, width))
            bottom = min(preview_image.size[1], top + max(1, height))
            left = max(0, min(left, right - 1))
            top = max(0, min(top, bottom - 1))
            return preview_image.crop((left, top, right, bottom))
        except Exception as exc:
            if not TAESD_PREVIEWER_CACHE.get("crop_compose_error_logged"):
                print(f"SDMLX: Crop inpaint preview could not be cropped: {exc}")
                TAESD_PREVIEWER_CACHE["crop_compose_error_logged"] = True
            return preview_image

    if preview_mode != "full_image" or not crop_info:
        return preview_image

    original = crop_info.get("original_image")
    if original is None:
        return preview_image

    try:
        original_image = torch_image_to_pil_first(original)
        if original_image is None:
            return preview_image

        x0, y0, x1, y1 = crop_info["bbox"]
        crop_w, crop_h = crop_info["crop_size"]
        resample = getattr(Image, "Resampling", Image).BICUBIC
        crop_preview = preview_image.resize((crop_w, crop_h), resample)

        mask_crop = crop_info["mask_crop"]
        mask_t = mask_crop.detach().cpu().float() if hasattr(mask_crop, "detach") else torch.from_numpy(get_numpy_array(mask_crop).astype(np.float32))
        if mask_t.ndim == 3:
            mask_t = mask_t[0]
        mask_np = torch.clamp(mask_t, 0.0, 1.0).numpy()
        mask_image = Image.fromarray((mask_np * 255.0).astype(np.uint8)).resize((crop_w, crop_h), resample)

        composed = original_image.copy()
        composed.paste(crop_preview, (x0, y0), mask_image)
        return composed
    except Exception as exc:
        if not TAESD_PREVIEWER_CACHE.get("compose_error_logged"):
            print(f"SDMLX: Full-image inpaint preview could not be composed: {exc}")
            TAESD_PREVIEWER_CACHE["compose_error_logged"] = True
        return preview_image


def get_sdxl_system_previewer():
    try:
        import comfy.latent_formats
        import comfy.model_management
        import latent_preview
        from comfy.cli_args import LatentPreviewMethod

        device = comfy.model_management.get_torch_device()
        preview_method = latent_preview.args.preview_method
        preview_method_name = str(preview_method)
        cache_key = f"sdxl_system:{preview_method}:{device}"
        if cache_key in TAESD_PREVIEWER_CACHE:
            return TAESD_PREVIEWER_CACHE[cache_key]

        latent_format = comfy.latent_formats.SDXL()
        if preview_method == LatentPreviewMethod.NoPreviews:
            preview_method_name = "default_latent2rgb"
            cache_key = f"sdxl_system:{preview_method_name}:{device}"
            if cache_key in TAESD_PREVIEWER_CACHE:
                return TAESD_PREVIEWER_CACHE[cache_key]
            previewer = latent_preview.Latent2RGBPreviewer(
                latent_format.latent_rgb_factors,
                latent_format.latent_rgb_factors_bias,
                latent_format.latent_rgb_factors_reshape,
            )
        else:
            previewer = latent_preview.get_previewer(device, latent_format)
        TAESD_PREVIEWER_CACHE[cache_key] = (previewer, device)
        return TAESD_PREVIEWER_CACHE[cache_key]
    except Exception as exc:
        if not TAESD_PREVIEWER_CACHE.get("previewer_error_logged"):
            print(f"SDMLX: Comfy preview not available: {exc}")
            TAESD_PREVIEWER_CACHE["previewer_error_logged"] = True
        return (None, None)


def decode_system_preview_bytes(latents, previewer, device, preview_mode="crop", crop_info=None):
    if previewer is None or device is None:
        return None

    try:
        mx.eval(latents)
        preview_latents = np.array(latents.transpose(0, 3, 1, 2)).astype(np.float32)
        preview_latents = torch.from_numpy(preview_latents).to(device=device)
        preview_bytes = previewer.decode_latent_to_preview_image("JPEG", preview_latents)
        if (
            crop_info
            and isinstance(preview_bytes, tuple)
            and len(preview_bytes) >= 2
            and hasattr(preview_bytes[1], "crop")
        ):
            preview_image = compose_preview_image(preview_bytes[1], preview_mode, crop_info)
            return preview_tuple_from_image(preview_image)
        return preview_bytes
    except Exception as exc:
        if not TAESD_PREVIEWER_CACHE.get("decode_error_logged"):
            print(f"SDMLX: Comfy preview could not be decoded: {exc}")
            TAESD_PREVIEWER_CACHE["decode_error_logged"] = True
        return None


def sample_latents(
    mlx_model,
    positive,
    negative,
    width,
    height,
    seed,
    steps,
    cfg,
    scheduler,
    sampler_name,
    force_no_cfg,
    compile_step,
    sync_each_step,
    debug_timing,
    preview=False,
    quantize_unet=False,
    quant_bits=8,
    quant_group_size=64,
    profile_unet=False,
    fast_transformer=False,
    fast_ffn=False,
    fast_attention=False,
    compute_dtype="float32",
    speed_patch=SPEED_PATCH_NONE,
    speed_patch_strength=1.0,
    initial_latents=None,
    noise_mask=None,
    denoise=1.0,
    preview_mode="crop",
    preview_crop_info=None,
    terminal_progress=True,
    terminal_progress_interval=None,
    sdxl_time_ids=None,
    differential_mask=False,
    differential_mask_strength=1.0,
):
    from .mlx_sd.config import DiffusionConfig
    from .mlx_sd.sampler import SimpleEulerAncestralSampler, SimpleEulerSampler

    configure_mlx_memory_limits()
    if width % 64 != 0 or height % 64 != 0:
        raise ValueError(
            f"SDMLX requires width and height to be divisible by 64, got {width}x{height}. "
            "Use a preset size or choose Custom dimensions such as 768, 1024, or 1344."
        )

    terminal_progress = bool(terminal_progress)
    start_time = time.perf_counter()
    loras = mlx_model.get("loras", [])
    static_loras, scheduled_loras = split_loras_by_schedule(loras)
    ip_adapters = list(mlx_model.get("ip_adapters", []))
    controlnets = collect_conditioning_controlnets(mlx_model, positive, negative)
    if controlnets and profile_unet:
        raise ValueError("SDMLX: profile_unet is not currently combined with ControlNet.")
    speed_patch_name = normalized_speed_patch_name(speed_patch)
    effective_fast_ffn = bool(fast_ffn)
    effective_fast_attention = bool(fast_attention) and not SDMLX_DISABLE_FAST_ATTENTION
    if fast_attention and not effective_fast_attention:
        print("SDMLX: Fast Attention disabled by environment.")
    if scheduled_loras and (effective_fast_ffn or effective_fast_attention):
        print(
            "SDMLX: Scheduled LoRA active: Fast FFN/Fast Attention are disabled for this run "
            "so the dynamic LoRA target modules remain individually controllable."
        )
        effective_fast_ffn = False
        effective_fast_attention = False
    unet = get_unet_model(
        mlx_model["cache_key"],
        mlx_model["weights"],
        quantize_unet,
        quant_bits,
        quant_group_size,
        fast_transformer,
        effective_fast_ffn,
        effective_fast_attention,
        compute_dtype,
        speed_patch,
        speed_patch_strength,
        static_loras,
    )
    scheduled_lora_stats = prepare_scheduled_loras_for_unet(
        unet,
        scheduled_loras,
        precision_dtype(compute_dtype),
    )
    speed_patch_strength_key = round(float(speed_patch_strength), 6) if speed_patch_name else 0.0
    lora_key = lora_stack_key(static_loras)
    scheduled_lora_key = lora_schedule_key(scheduled_loras)
    denoiser_key = (
        f"{mlx_model['cache_key']}:q{int(quantize_unet)}:{int(quant_bits)}:"
        f"{int(quant_group_size)}:ft{int(fast_transformer)}:"
        f"ffn{int(effective_fast_ffn)}:"
        f"fattn{int(effective_fast_attention)}:dtype{compute_dtype}:"
        f"gelu{SDMLX_GELU_MODE}:"
        f"patch{speed_patch_name}:pstrength{speed_patch_strength_key}:"
        f"loras{lora_key}:"
        f"scheduled_loras{scheduled_lora_key}:"
        f"sched{scheduler}"
    )
    profile = new_profile() if profile_unet else None
    mx.random.seed(seed)
    sampler_cls = SimpleEulerAncestralSampler if sampler_name == "euler_ancestral" else SimpleEulerSampler
    sampler = sampler_cls(
        DiffusionConfig(
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            num_train_steps=1000,
        )
    )

    latent_height = height // 8
    latent_width = width // 8
    dtype = precision_dtype(compute_dtype)
    initial_latents = initial_latents.astype(mx.float32) if initial_latents is not None else None
    if initial_latents is not None and tuple(initial_latents.shape[1:3]) != (latent_height, latent_width):
        raise ValueError(
            "SDMLX: initial_latents do not match the target size: "
            f"{tuple(initial_latents.shape)} vs {(latent_height, latent_width)}."
        )

    if sdxl_time_ids is None:
        time_id_values = [float(height), float(width), 0.0, 0.0, float(height), float(width)]
        time_id_source = "default"
    else:
        time_id_values = [float(value) for value in sdxl_time_ids]
        if len(time_id_values) != 6:
            raise ValueError(f"SDMLX: sdxl_time_ids must contain 6 values, got {time_id_values}.")
        time_id_source = "crop"
    time_id = mx.array([time_id_values])
    use_cfg = cfg > 1.0 and not force_no_cfg
    portrait_ip_adapters = [
        adapter
        for adapter in ip_adapters
        if adapter.get("is_portrait") or adapter.get("is_portrait_unnorm")
    ]
    if portrait_ip_adapters:
        portrait_names = ", ".join(
            str(adapter.get("name", "FaceID Portrait")) for adapter in portrait_ip_adapters
        )
        non_linear_weight_types = sorted({
            str(adapter.get("weight_type", "linear"))
            for adapter in portrait_ip_adapters
            if str(adapter.get("weight_type", "linear")) != "linear"
        })
        if not use_cfg:
            print(
                "SDMLX: Note: FaceID Portrait is running with CFG off "
                f"({portrait_names}). These adapters are very dominant style-transfer models; "
                "for cropping, blur, or color casts, first test CFG 3-4 with force_no_cfg=False."
            )
        if speed_patch_name:
            print(
                "SDMLX: Note: FaceID Portrait is used with a Speed Patch "
                f"({speed_patch_name}). If Portrait/Unnorm looks unstable, "
                "first check a reference run without a patch."
            )
        if non_linear_weight_types:
            print(
                "SDMLX: Note: FaceID Portrait uses non-linear weight type(s) "
                f"({', '.join(non_linear_weight_types)}). Linear is the most stable starting point."
            )
    if use_cfg:
        context = mx.concatenate([positive["cond"], negative["cond"]], axis=0)
        pooled = mx.concatenate([positive["pooled"], negative["pooled"]], axis=0)
        t_ids = mx.concatenate([time_id] * 2, axis=0)
    else:
        context = positive["cond"]
        pooled = positive["pooled"]
        t_ids = time_id

    context = context.astype(dtype)
    pooled = pooled.astype(dtype)
    t_ids = t_ids.astype(dtype)
    cfg_value = mx.array(cfg, dtype=mx.float32)
    denoise = max(0.0, min(1.0, float(denoise)))
    step_plan = scheduler_step_plan_for_denoise(sampler, steps, scheduler, sampler_name, denoise)
    progress_steps = len(step_plan)
    dynamic_step_context = bool(ip_adapters or scheduled_loras)
    control_active_by_step = [
        controlnets_active_at_percent(controlnets, i / max(progress_steps - 1, 1))
        for i in range(progress_steps)
    ] if controlnets else []
    control_free_steps = control_active_by_step.count(False) if controlnets else 0
    effective_compile_step = (
        compile_step
        and not SDMLX_DISABLE_STEP_COMPILE
        and not scheduled_loras
        and not ip_adapters
        and not profile_unet
        and (not controlnets or control_free_steps > 0)
    )
    if compile_step and SDMLX_DISABLE_STEP_COMPILE:
        print("SDMLX: Step denoiser compile disabled by environment.")
    if controlnets:
        controlnets = prepare_controlnets_for_sampling(
            controlnets,
            width,
            height,
            2 if use_cfg else 1,
            dtype,
        )
    mx.eval(context, pooled, t_ids)
    step_denoiser = get_step_denoiser(
        denoiser_key,
        unet,
        compute_dtype,
        use_cfg,
        effective_compile_step,
    )
    ip_adapter_kv_layers = 0
    if ip_adapters:
        patched = ensure_unet_ipadapter_wrapped(unet)
        if patched:
            log_timing(f"SDMLX: IP-Adapter cross-attention enabled ({patched} Transformer-Blocks).")
        ip_adapters, ip_adapter_kv_layers = prepare_ipadapter_kv_cache(
            ip_adapters,
            2 if use_cfg else 1,
            dtype,
            use_cfg,
        )
    progress_sync_interval = 0
    if terminal_progress and not sync_each_step and not preview:
        if terminal_progress_interval is not None:
            progress_sync_interval = max(1, int(terminal_progress_interval))
        else:
            progress_sync_interval = 2 if progress_steps <= 12 else 4
    comfy_progress = bool(preview or terminal_progress or debug_timing)
    if controlnets:
        for control in controlnets:
            get_controlnet_union_model(
                control["controlnet"],
                fast_transformer=fast_transformer,
                fast_ffn=fast_ffn,
                fast_attention=effective_fast_attention,
            )
    log_timing(
        "SDMLX: Runtime Config: "
        f"sampler_name={sampler_name}, "
        f"sync_each_step={sync_each_step}, "
        f"terminal_progress={terminal_progress}, "
        f"comfy_progress={comfy_progress}, "
        f"progress_sync_interval={progress_sync_interval}, "
        f"compile_step={compile_step}, "
        f"effective_compile_step={effective_compile_step}, "
        f"fast_transformer={fast_transformer}, "
        f"fast_ffn={effective_fast_ffn}, "
        f"fast_attention={effective_fast_attention}, "
        f"gelu_mode={SDMLX_GELU_MODE}, "
        f"quantize_unet={quantize_unet}, "
        f"profile_unet={profile_unet}, "
        f"dynamic_step_context={dynamic_step_context}, "
        f"debug_timing={debug_timing}, "
        f"preview={preview}, "
        f"preview_mode={preview_mode}, "
        f"speed_patch={speed_patch_name or SPEED_PATCH_NONE}, "
        f"patch_strength={speed_patch_strength_key:g}, "
        f"loras={len(static_loras)}, "
        f"scheduled_loras={len(scheduled_loras)}, "
        f"scheduled_lora_modules={scheduled_lora_stats.get('modules', 0)}, "
        f"ip_adapters={len(ip_adapters)}, "
        f"ip_adapter_kv_layers={ip_adapter_kv_layers}, "
        f"controlnets={len(controlnets)}, "
        f"control_free_steps={control_free_steps}, "
        f"compute_dtype={compute_dtype}, "
        f"time_ids={time_id_source}, "
        f"differential_mask={bool(differential_mask)}, "
        "step_plan=precomputed"
    )
    if terminal_progress or debug_timing:
        print(
            f"SDMLX: Fast Sampling ({steps} Steps, {width}x{height}, "
            f"CFG {'an' if use_cfg else 'aus'}, "
            f"Scheduler {scheduler}, Sampler {sampler_name}, "
            f"DType {compute_dtype})..."
        )

    if denoise == 0.0 and initial_latents is not None:
        print("SDMLX: denoise=0, Initial latents are returned unchanged.")
        mx.eval(initial_latents)
        return initial_latents

    if initial_latents is not None:
        init_noise = mx.random.normal(initial_latents.shape).astype(mx.float32)
        start_sigma = step_plan[0][6]
        latents = noise_latents_at_sigma(initial_latents, init_noise, start_sigma)
    else:
        latents = sampler.sample_prior((1, latent_height, latent_width, 4), dtype=mx.float32)

    mask = None
    if noise_mask is not None:
        mask = noise_mask.astype(mx.float32)
        if len(mask.shape) == 3:
            mask = mask[..., None]
        if mask.shape[1] != latent_height or mask.shape[2] != latent_width:
            raise ValueError(
                "SDMLX: noise_mask does not match the latent size: "
                f"{tuple(mask.shape)} vs {(latent_height, latent_width)}."
            )
        if initial_latents is None:
            raise ValueError("SDMLX: noise_mask requires initial_latents.")
        differential_mask_strength = max(0.0, min(1.0, float(differential_mask_strength)))
        start_t = step_plan[0][0]
        end_t = mx.array(0.0, dtype=mx.float32)
        differential_denom = mx.maximum(start_t - end_t, mx.array(1e-6, dtype=mx.float32))

        def active_mask_for_t(timestep):
            if not differential_mask:
                return mask
            threshold = mx.clip((timestep - end_t) / differential_denom, 0.0, 1.0).astype(mx.float32)
            threshold = mx.maximum(threshold, mx.array(1e-6, dtype=mx.float32))
            binary_mask = (mask >= threshold).astype(mx.float32)
            if differential_mask_strength < 1.0:
                strength = mx.array(differential_mask_strength, dtype=mx.float32)
                return strength * binary_mask + (1.0 - strength) * mask
            return binary_mask

        initial_mask = active_mask_for_t(step_plan[0][0])
        preserved = noise_latents_at_sigma(initial_latents, init_noise, step_plan[0][6])
        latents = latents * initial_mask + preserved * (1.0 - initial_mask)
        log_timing("SDMLX: Inpaint Sampler: Comfy baseline with noised original and X0 masking active.")
        if differential_mask:
            log_timing(
                "SDMLX: Inpaint Sampler: Differential diffusion mask active "
                f"(strength={differential_mask_strength:g})."
            )

    guard_supported = use_cfg and not dynamic_step_context and not profile_unet and not controlnets and mask is None
    guard_auto_supported = guard_supported and not conditioning_texts_identical(positive, negative)
    diagnostic_metrics = None
    guard_first_noise_pred = None
    if SDMLX_CONDITIONING_DIAGNOSTICS:
        if guard_supported:
            diagnostic_metrics = run_conditioning_diagnostics(
                mlx_model,
                unet,
                step_denoiser,
                latents,
                step_plan[0][0],
                context,
                pooled,
                t_ids,
                cfg_value,
                compute_dtype,
                quantize_unet,
                quant_bits,
                quant_group_size,
                fast_transformer,
                effective_fast_ffn,
                effective_fast_attention,
                speed_patch,
                speed_patch_strength,
                static_loras,
            )
        else:
            reasons = []
            if not use_cfg:
                reasons.append("CFG is off")
            if dynamic_step_context:
                reasons.append("dynamic step context")
            if profile_unet:
                reasons.append("UNet profiling")
            if controlnets:
                reasons.append("ControlNet")
            if mask is not None:
                reasons.append("inpaint mask")
            print(f"SDMLX Conditioning Diagnostics: skipped ({', '.join(reasons) or 'unsupported path'}).")
    if guard_auto_supported:
        guard_key = (
            "conditioning_guard_v1",
            denoiser_key,
            width,
            height,
            progress_steps,
            compute_dtype,
        )
        guard_first_noise_pred = maybe_run_conditioning_guard(
            guard_key,
            unet,
            step_denoiser,
            latents,
            step_plan[0][0],
            context,
            pooled,
            t_ids,
            cfg_value,
            compute_dtype,
            diagnostic_metrics=diagnostic_metrics,
        )

    pbar = make_comfy_progress_bar(progress_steps) if comfy_progress else None
    terminal_pbar = make_terminal_progress_bar(progress_steps) if terminal_progress else None
    previewer, preview_device = get_sdxl_system_previewer() if preview else (None, None)

    def run_denoiser(latents_in, timestep, step_percent, step_has_control):
        if dynamic_step_context:
            if ip_adapters:
                SDMLX_IPADAPTER_CONTEXT["adapters"] = ip_adapters
                SDMLX_IPADAPTER_CONTEXT["step_percent"] = step_percent
                SDMLX_IPADAPTER_CONTEXT["use_cfg"] = use_cfg
            if scheduled_loras:
                SDMLX_LORA_CONTEXT["step_percent"] = step_percent
        if step_denoiser is not None and not step_has_control:
            return step_denoiser(latents_in, timestep, context, pooled, t_ids, cfg_value)

        x_in = mx.concatenate([latents_in] * 2) if use_cfg else latents_in
        x_model = x_in.astype(dtype) if compute_dtype == "float16" else x_in
        t_in = mx.broadcast_to(timestep, [len(x_in)])
        control_down, control_mid = (
            controlnet_residuals_for_step(
                controlnets,
                x_model,
                t_in,
                context,
                pooled,
                t_ids,
                width,
                height,
                step_percent,
                dtype,
                fast_transformer,
                effective_fast_ffn,
                effective_fast_attention,
            )
            if step_has_control
            else (None, None)
        )
        if profile_unet:
            noise_pred_local = profiled_unet_forward(unet, x_model, t_in, context, pooled, t_ids, profile)
        else:
            noise_pred_local = unet(
                x_model,
                timestep=t_in,
                encoder_x=context,
                text_time=(pooled, t_ids),
                control_down_residuals=control_down,
                control_mid_residual=control_mid,
            )
        noise_pred_local = noise_pred_local.astype(mx.float32)

        if use_cfg:
            eps_pos, eps_neg = mx.split(noise_pred_local, 2)
            noise_pred_local = eps_neg + cfg_value * (eps_pos - eps_neg)
        return noise_pred_local

    old_dpmpp_denoised = None
    old_dpmpp_sigma = None
    try:
        for i, (t, next_t, step_scale, step_dt, step_noise_scale, step_out_scale, step_sigma, next_sigma) in enumerate(step_plan):
            step_percent = i / max(progress_steps - 1, 1)
            step_has_control = control_active_by_step[i] if control_active_by_step else False
            if mask is not None:
                step_mask = active_mask_for_t(t)
                preserved = noise_latents_at_sigma(initial_latents, init_noise, step_sigma)
                latents = latents * step_mask + preserved * (1.0 - step_mask)
            if i == 0 and guard_first_noise_pred is not None and not step_has_control:
                noise_pred = guard_first_noise_pred
                guard_first_noise_pred = None
            else:
                noise_pred = run_denoiser(latents, t, step_percent, step_has_control)

            denoised_latents = None
            if mask is not None or preview or sampler_name == "dpmpp_2m":
                denoised_latents = denoised_latents_estimate(latents, noise_pred, step_scale, step_sigma)
            if mask is not None:
                denoised_latents = denoised_latents * step_mask + initial_latents * (1.0 - step_mask)
                noise_pred = noise_pred_from_denoised(latents, denoised_latents, step_scale, step_sigma)

            preview_latents = None
            if preview:
                preview_latents = denoised_latents

            if sampler_name == "heun":
                def denoise_next(trial_latents):
                    next_noise_pred = run_denoiser(trial_latents, next_t, step_percent, step_has_control)
                    if mask is not None:
                        next_scale = mx.sqrt(next_sigma * next_sigma + 1.0)
                        next_denoised = denoised_latents_estimate(trial_latents, next_noise_pred, next_scale, next_sigma)
                        next_denoised = next_denoised * step_mask + initial_latents * (1.0 - step_mask)
                        next_noise_pred = noise_pred_from_denoised(trial_latents, next_denoised, next_scale, next_sigma)
                    return next_noise_pred

                latents = heun_sampler_step(latents, noise_pred, step_sigma, next_sigma, denoise_next)
            elif sampler_name == "dpmpp_2m":
                latents = dpmpp_2m_sampler_step(
                    latents,
                    denoised_latents,
                    old_dpmpp_denoised,
                    step_sigma,
                    next_sigma,
                    old_dpmpp_sigma,
                )
                old_dpmpp_denoised = denoised_latents
                old_dpmpp_sigma = mx_scalar_float(step_sigma)
            else:
                latents = apply_sampler_step(noise_pred, latents, step_scale, step_dt, step_noise_scale, step_out_scale)
            if mask is not None:
                step_mask = active_mask_for_t(next_t)
                preserved = noise_latents_at_sigma(initial_latents, init_noise, next_sigma)
                latents = latents * step_mask + preserved * (1.0 - step_mask)
            should_sync_for_progress = bool(
                progress_sync_interval
                and ((i + 1) % progress_sync_interval == 0 or (i + 1) == progress_steps)
            )
            if sync_each_step or should_sync_for_progress:
                mx.eval(latents)
            preview_bytes = (
                decode_system_preview_bytes(
                    preview_latents,
                    previewer,
                    preview_device,
                    preview_mode,
                    preview_crop_info,
                )
                if preview
                else None
            )
            should_publish_progress = bool(
                preview_bytes is not None
                or sync_each_step
                or should_sync_for_progress
                or (terminal_progress and not progress_sync_interval)
            )
            if pbar is not None and should_publish_progress:
                pbar.update_absolute(i + 1, progress_steps, preview_bytes)
            if terminal_pbar is not None and should_publish_progress:
                delta = (i + 1) - terminal_pbar.n
                if delta > 0:
                    terminal_pbar.update(delta)
            if debug_timing and ((i + 1) % 5 == 0 or i == 0):
                print(f"   Step {i + 1}/{progress_steps}...")
    finally:
        for adapter in ip_adapters:
            stats = adapter.get("debug_stats") or []
            if stats:
                avg_ratio = sum(item["ratio"] for item in stats) / len(stats)
                max_ratio = max(item["ratio"] for item in stats)
                first = stats[0]
                log_timing(
                    "SDMLX: FaceID Attention Delta "
                    f"(samples={len(stats)}, avg_delta/base={avg_ratio:.3f}, "
                    f"max_delta/base={max_ratio:.3f}, first_module={first['module']}, "
                    f"first_delta={first['delta_rms']:.4f}, first_base={first['base_rms']:.4f})."
                )
            elif adapter.get("is_faceid"):
                log_timing(
                    "SDMLX: FaceID Attention Delta: no samples recorded "
                    "(IP-Adapter-Wrapper was not reached in this sampler run)."
                )
        if dynamic_step_context:
            if ip_adapters:
                SDMLX_IPADAPTER_CONTEXT["adapters"] = []
                SDMLX_IPADAPTER_CONTEXT["step_percent"] = 0.0
                SDMLX_IPADAPTER_CONTEXT["use_cfg"] = False
            if scheduled_loras:
                SDMLX_LORA_CONTEXT["step_percent"] = 0.0
        if terminal_pbar is not None:
            terminal_pbar.close()

    if not sync_each_step and not progress_sync_interval:
        mx.eval(latents)
    release_mlx_cache_memory_after_sampling()
    if terminal_progress or debug_timing:
        print(f"SDMLX: Fast Sampling finished in {time.perf_counter() - start_time:.2f}s.")
    print_unet_profile(profile)
    return latents


def get_vae_decoder(cache_key, vae, compute_dtype, use_compiled):
    if not use_compiled:
        return vae.decode

    key = (cache_key, "vae_decode", compute_dtype)
    if key not in COMPILED_VAE_DECODERS:
        def decode(z):
            pixels = vae.decode(z)
            if isinstance(pixels, tuple):
                pixels = pixels[0]
            return pixels

        COMPILED_VAE_DECODERS[key] = mx.compile(decode)
        log_timing("SDMLX: Compiled VAE decoder created.")

    return COMPILED_VAE_DECODERS[key]


def decode_latents(mlx_vae, latents, compile_vae=False, vae_dtype="float32", debug_timing=False):
    configure_mlx_memory_limits()
    start_time = time.perf_counter()
    vae = get_vae_model(mlx_vae["cache_key"], mlx_vae["weights"], vae_dtype)
    decoder = get_vae_decoder(mlx_vae["cache_key"], vae, vae_dtype, compile_vae)

    if debug_timing:
        vae_start = time.perf_counter()
        pixels = decoder(latents)
        if isinstance(pixels, tuple):
            pixels = pixels[0]
        pixels = mx.clip((pixels + 1.0) / 2.0, 0.0, 1.0)
        mx.eval(pixels)
        vae_elapsed = time.perf_counter() - vae_start

        normalize_start = time.perf_counter()
        mx.eval(pixels)
        normalize_elapsed = time.perf_counter() - normalize_start

        transfer_start = time.perf_counter()
        pixels_np = np.array(pixels).astype(np.float32)
        transfer_elapsed = time.perf_counter() - transfer_start

        torch_start = time.perf_counter()
        image = torch.from_numpy(pixels_np)
        torch_elapsed = time.perf_counter() - torch_start
        release_mlx_cache_memory_after_decode()

        total_elapsed = time.perf_counter() - start_time
        print(
            "SDMLX: Fast Decode Split: "
            f"compile_vae={compile_vae}, "
            f"vae_dtype={vae_dtype}, "
            f"vae={vae_elapsed:.2f}s, "
            f"normalize={normalize_elapsed:.2f}s, "
            f"transfer={transfer_elapsed:.2f}s, "
            f"torch={torch_elapsed:.2f}s, "
            f"total={total_elapsed:.2f}s"
        )
        print(f"SDMLX: Fast Decode finished in {total_elapsed:.2f}s.")
        return image

    pixels = decoder(latents)
    if isinstance(pixels, tuple):
        pixels = pixels[0]
    pixels = mx.clip((pixels + 1.0) / 2.0, 0.0, 1.0)
    image = torch.from_numpy(np.array(pixels).astype(np.float32))
    release_mlx_cache_memory_after_decode()
    log_timing(
        f"SDMLX: Fast Decode finished in {time.perf_counter() - start_time:.2f}s "
        f"(compile_vae={compile_vae}, vae_dtype={vae_dtype})."
    )
    return image


def tiled_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = [0]
    while starts[-1] + tile < length:
        next_start = min(length - tile, starts[-1] + stride)
        if next_start <= starts[-1]:
            break
        starts.append(next_start)
    return starts


def tile_blend_mask(height, width, top_edge, bottom_edge, left_edge, right_edge, overlap):
    y = torch.ones((height,), dtype=torch.float32)
    x = torch.ones((width,), dtype=torch.float32)
    overlap = int(max(0, overlap))
    if overlap > 0:
        y_overlap = min(overlap, height)
        x_overlap = min(overlap, width)
        if not top_edge:
            y[:y_overlap] = torch.minimum(y[:y_overlap], torch.linspace(0.0, 1.0, y_overlap))
        if not bottom_edge:
            y[-y_overlap:] = torch.minimum(y[-y_overlap:], torch.linspace(1.0, 0.0, y_overlap))
        if not left_edge:
            x[:x_overlap] = torch.minimum(x[:x_overlap], torch.linspace(0.0, 1.0, x_overlap))
        if not right_edge:
            x[-x_overlap:] = torch.minimum(x[-x_overlap:], torch.linspace(1.0, 0.0, x_overlap))
    return (y[:, None] * x[None, :])[None, :, :, None]


def decode_latents_tiled(
    mlx_vae,
    latents,
    compile_vae=False,
    vae_dtype="float32",
    tile_size=1024,
    overlap=128,
):
    configure_mlx_memory_limits()
    start_time = time.perf_counter()
    vae = get_vae_model(mlx_vae["cache_key"], mlx_vae["weights"], vae_dtype)
    decoder = get_vae_decoder(mlx_vae["cache_key"], vae, vae_dtype, compile_vae)

    latent_height = int(latents.shape[1])
    latent_width = int(latents.shape[2])
    batch = int(latents.shape[0])
    pixel_height = latent_height * 8
    pixel_width = latent_width * 8
    tile_latents = max(64, int(tile_size) // 8)
    overlap_latents = max(0, int(overlap) // 8)
    overlap_latents = min(overlap_latents, max(0, tile_latents // 2 - 1))
    tile_latents = min(tile_latents, max(latent_height, latent_width))

    y_starts = tiled_starts(latent_height, min(tile_latents, latent_height), overlap_latents)
    x_starts = tiled_starts(latent_width, min(tile_latents, latent_width), overlap_latents)
    out = torch.zeros((batch, pixel_height, pixel_width, 3), dtype=torch.float32)
    weight = torch.zeros((1, pixel_height, pixel_width, 1), dtype=torch.float32)
    decoded_tiles = 0

    for y0 in y_starts:
        y1 = min(latent_height, y0 + tile_latents)
        for x0 in x_starts:
            x1 = min(latent_width, x0 + tile_latents)
            tile = latents[:, y0:y1, x0:x1, :]
            pixels = decoder(tile)
            if isinstance(pixels, tuple):
                pixels = pixels[0]
            pixels = mx.clip((pixels + 1.0) / 2.0, 0.0, 1.0)
            tile_np = np.array(pixels).astype(np.float32)
            tile_t = torch.from_numpy(tile_np)
            py0 = y0 * 8
            py1 = y1 * 8
            px0 = x0 * 8
            px1 = x1 * 8
            mask = tile_blend_mask(
                py1 - py0,
                px1 - px0,
                y0 == 0,
                y1 == latent_height,
                x0 == 0,
                x1 == latent_width,
                overlap_latents * 8,
            )
            out[:, py0:py1, px0:px1, :] += tile_t * mask
            weight[:, py0:py1, px0:px1, :] += mask
            decoded_tiles += 1
            del pixels, tile_np, tile_t, tile

    out = out / torch.clamp(weight, min=1e-6)
    release_mlx_cache_memory_after_decode()
    log_timing(
        f"SDMLX: Tiled Decode finished in {time.perf_counter() - start_time:.2f}s "
        f"(tiles={decoded_tiles}, tile={tile_latents * 8}px, overlap={overlap_latents * 8}px, "
        f"compile_vae={compile_vae}, vae_dtype={vae_dtype})."
    )
    return torch.clamp(out, 0.0, 1.0)


def decode_latents_quiet(mlx_vae, latents, compile_vae=True, vae_dtype="float32"):
    vae = get_vae_model(mlx_vae["cache_key"], mlx_vae["weights"], vae_dtype)
    decoder = get_vae_decoder(mlx_vae["cache_key"], vae, vae_dtype, compile_vae)
    pixels = decoder(latents)
    if isinstance(pixels, tuple):
        pixels = pixels[0]
    pixels = mx.clip((pixels + 1.0) / 2.0, 0.0, 1.0)
    image = torch.from_numpy(np.array(pixels).astype(np.float32))
    release_mlx_cache_memory_after_decode()
    return image


def should_use_tiled_decode(mode, width, height):
    if mode == "tiled":
        return True
    if mode == "full":
        return False
    return int(width) * int(height) >= 2_500_000


def encode_pixels_to_latents(mlx_vae, pixels, vae_dtype="float32"):
    vae = get_vae_model(mlx_vae["cache_key"], mlx_vae["weights"], vae_dtype)
    if hasattr(pixels, "detach"):
        pixels_np = pixels.detach().cpu().float().numpy()
    else:
        pixels_np = get_numpy_array(pixels).astype(np.float32)
    if pixels_np.ndim != 4 or pixels_np.shape[-1] != 3:
        raise ValueError(f"SDMLX: IMAGE must be [B,H,W,3], got {tuple(pixels_np.shape)}.")
    x = mx.array(pixels_np).astype(mx.float32) * 2.0 - 1.0
    mean, _ = vae.encode(x)
    mx.eval(mean)
    return mean


def inpaint_work_size(width, height, multiple=64):
    target_width = max(multiple, int(math.ceil(int(width) / multiple) * multiple))
    target_height = max(multiple, int(math.ceil(int(height) / multiple) * multiple))
    return target_width, target_height


def pad_inpaint_pixels_and_mask(pixels, mask_chw=None, target_width=None, target_height=None, pad_alignment=1):
    if hasattr(pixels, "detach"):
        pixels_t = pixels.detach().cpu().float()
    else:
        pixels_t = torch.from_numpy(get_numpy_array(pixels).astype(np.float32))
    if pixels_t.ndim != 4 or pixels_t.shape[-1] != 3:
        raise ValueError(f"SDMLX: IMAGE must be [B,H,W,3], got {tuple(pixels_t.shape)}.")

    height = int(pixels_t.shape[1])
    width = int(pixels_t.shape[2])
    target_width = width if target_width is None else int(target_width)
    target_height = height if target_height is None else int(target_height)
    pad_width = int(target_width) - width
    pad_height = int(target_height) - height
    if pad_width < 0 or pad_height < 0:
        raise ValueError(f"SDMLX Inpaint: target size {target_width}x{target_height} is smaller than image {width}x{height}.")
    if pad_width == 0 and pad_height == 0:
        return pixels_t, mask_chw, (0, 0, width, height)

    pad_alignment = max(1, int(pad_alignment))
    if pad_alignment > 1:
        left = int(round((pad_width * 0.5) / pad_alignment) * pad_alignment)
        top = int(round((pad_height * 0.5) / pad_alignment) * pad_alignment)
        left = min(max(0, left), pad_width)
        top = min(max(0, top), pad_height)
    else:
        left = pad_width // 2
        top = pad_height // 2
    right = pad_width - left
    bottom = pad_height - top

    pixels_nchw = pixels_t.permute(0, 3, 1, 2)
    pixels_nchw = torch.nn.functional.pad(pixels_nchw, (left, right, top, bottom), mode="replicate")
    pixels_t = pixels_nchw.permute(0, 2, 3, 1).contiguous()
    if mask_chw is not None:
        mask_chw = torch.nn.functional.pad(mask_chw, (left, right, top, bottom), mode="constant", value=0.0)
        mask_chw = mask_chw.contiguous()
    return pixels_t, mask_chw, (left, top, width, height)


def resize_image_tensor(image, target_width, target_height, method="lanczos"):
    if hasattr(image, "detach"):
        tensor = image.detach().cpu().float()
    else:
        tensor = torch.from_numpy(get_numpy_array(image).astype(np.float32))
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(f"SDMLX: IMAGE must be [B,H,W,3], got {tuple(tensor.shape)}.")

    if int(tensor.shape[2]) == int(target_width) and int(tensor.shape[1]) == int(target_height):
        return torch.clamp(tensor, 0.0, 1.0).contiguous()

    method = method if method in HIRES_RESIZE_METHODS else "lanczos"
    if method == "lanczos":
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = []
        for img in tensor:
            arr = (torch.clamp(img, 0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
            pil = pil.resize((int(target_width), int(target_height)), resampling)
            resized.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))
        return torch.stack(resized, dim=0).contiguous()

    chw = tensor.permute(0, 3, 1, 2).contiguous()
    resized = torch.nn.functional.interpolate(
        chw,
        size=(int(target_height), int(target_width)),
        mode=method,
        align_corners=False,
    )
    return torch.clamp(resized.permute(0, 2, 3, 1).contiguous(), 0.0, 1.0)


def resolve_hires_target_size(image, scale, custom_width, custom_height, multiple_of=64):
    if hasattr(image, "shape"):
        source_height = int(image.shape[1])
        source_width = int(image.shape[2])
    else:
        arr = get_numpy_array(image)
        source_height = int(arr.shape[1])
        source_width = int(arr.shape[2])

    if scale == "custom":
        target_width = int(custom_width)
        target_height = int(custom_height)
    else:
        factor = float(str(scale).rstrip("x"))
        target_width = int(round(source_width * factor))
        target_height = int(round(source_height * factor))

    target_width = max(multiple_of, round_up(target_width, multiple_of))
    target_height = max(multiple_of, round_up(target_height, multiple_of))
    return target_width, target_height


def resolve_hires_target_size_from_factor(image, scale_factor, custom_width, custom_height, multiple_of=64, custom_size=False):
    scale = "custom" if custom_size else f"{float(scale_factor)}x"
    return resolve_hires_target_size(image, scale, custom_width, custom_height, multiple_of)


def validate_hires_target_size(width, height, max_pixels=HIRES_MAX_PIXELS):
    pixels = int(width) * int(height)
    if pixels <= max_pixels:
        return
    raise ValueError(
        "SDMLX Hires Fix: target image is too large for the full-frame MLX path "
        f"({width}x{height} = {pixels / 1_000_000:.1f}MP, limit {max_pixels / 1_000_000:.1f}MP). "
        "Up to 2x at typical SDXL sizes is currently the stable range. "
        "True 4x needs a separate tiled upscale/sampling node."
    )


def validate_tiled_upscale_target_size(width, height, max_megapixels):
    pixels = int(width) * int(height)
    limit = int(min(TILED_UPSCALE_MAX_PIXELS, max(1.0, float(max_megapixels)) * 1_000_000))
    if pixels <= limit:
        return
    raise ValueError(
        "SDMLX Tiled Upscale: target image is larger than the configured limit "
        f"({width}x{height} = {pixels / 1_000_000:.1f}MP, limit {limit / 1_000_000:.1f}MP). "
        "Increase max_megapixels intentionally if you really want to test this."
    )


def mask_to_chw(mask, height=None, width=None):
    if hasattr(mask, "detach"):
        mask_t = mask.detach().cpu().float()
    else:
        mask_t = torch.from_numpy(get_numpy_array(mask).astype(np.float32))
    if mask_t.ndim == 2:
        mask_t = mask_t.unsqueeze(0)
    if mask_t.ndim == 3:
        mask_t = mask_t.unsqueeze(1)
    elif mask_t.ndim == 4:
        if mask_t.shape[-1] == 1 and mask_t.shape[1] != 1:
            mask_t = mask_t.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"SDMLX: MASK muss 2D/3D/4D sein, ist aber {tuple(mask_t.shape)}.")
    if height is not None and width is not None and (mask_t.shape[-2] != height or mask_t.shape[-1] != width):
        mask_t = torch.nn.functional.interpolate(mask_t, size=(height, width), mode="bilinear", align_corners=False)
    return torch.clamp(mask_t, 0.0, 1.0)


def normalize_nonempty_mask(mask_t, min_peak=0.99):
    mask_t = torch.clamp(mask_t, 0.0, 1.0)
    peak = float(torch.max(mask_t).item()) if mask_t.numel() else 0.0
    if 0.0 < peak < float(min_peak):
        mask_t = torch.clamp(mask_t / peak, 0.0, 1.0)
    return mask_t


def sdmlx_make_3d_mask(mask):
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    elif is_mlx_array(mask):
        mask = torch.from_numpy(np.array(mask))
    elif hasattr(mask, "detach"):
        mask = mask.detach().cpu().float()
    else:
        mask = torch.from_numpy(get_numpy_array(mask).astype(np.float32))

    if len(mask.shape) == 4:
        return mask.squeeze(0)
    if len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask


def sdmlx_tensor_check_mask(mask):
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels")


def sdmlx_tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    elif is_mlx_array(mask):
        mask = torch.from_numpy(np.array(mask))

    if mask.ndim == 2:
        mask = mask[None, ..., None]
    elif mask.ndim == 3:
        mask = mask[..., None]

    sdmlx_tensor_check_mask(mask)

    kernel_size = int(kernel_size)
    if kernel_size <= 0:
        return mask

    kernel_size = kernel_size * 2 + 1
    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask

    import torchvision.transforms
    mask = mask.float()
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size,
        sigma=float(sigma),
    )(mask)
    return blurred_mask[:, 0, ..., None]


def grow_mask_tensor(mask_t, grow_mask_by=0):
    grow_mask_by = int(grow_mask_by)
    if grow_mask_by > 0:
        kernel = grow_mask_by * 2 + 1
        mask_t = torch.nn.functional.max_pool2d(mask_t, kernel_size=kernel, stride=1, padding=grow_mask_by)
    return torch.clamp(mask_t, 0.0, 1.0)


def impact_gaussian_blur_mask_tensor(mask_t, kernel_size, sigma=10.0):
    kernel_size = int(kernel_size)
    if kernel_size <= 0:
        return torch.clamp(mask_t, 0.0, 1.0)
    kernel_size = kernel_size * 2 + 1
    shortest = min(int(mask_t.shape[-2]), int(mask_t.shape[-1]))
    if shortest <= kernel_size:
        kernel_size = int(shortest / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return torch.clamp(mask_t, 0.0, 1.0)
    try:
        import torchvision.transforms
        blurred = torchvision.transforms.GaussianBlur(
            kernel_size=kernel_size,
            sigma=max(float(sigma), 0.1),
        )(mask_t)
    except Exception:
        radius = kernel_size // 2
        blurred = torch.nn.functional.avg_pool2d(mask_t, kernel_size=kernel_size, stride=1, padding=radius)
    return torch.clamp(blurred, 0.0, 1.0)


def feather_mask_tensor(mask_t, feather_radius=0, feather_mode="gaussian", feather_sigma=10.0):
    feather_radius = int(feather_radius)
    base_mask = torch.clamp(mask_t, 0.0, 1.0)
    if feather_radius <= 0:
        return base_mask
    feather_mode = feather_mode if feather_mode in MASK_FEATHER_MODES else "gaussian"
    if feather_mode == "box":
        kernel = feather_radius * 2 + 1
        mask_t = torch.nn.functional.avg_pool2d(base_mask, kernel_size=kernel, stride=1, padding=feather_radius)
        mask_t = torch.nn.functional.avg_pool2d(mask_t, kernel_size=kernel, stride=1, padding=feather_radius)
        return torch.clamp(mask_t, 0.0, 1.0)

    return impact_gaussian_blur_mask_tensor(base_mask, feather_radius, feather_sigma)


def prepare_inpaint_mask(
    mask,
    pixel_height,
    pixel_width,
    latent_height,
    latent_width,
    grow_mask_by=4,
    feather_radius=0,
    feather_mode="gaussian",
    feather_sigma=10.0,
):
    mask_t = mask_to_chw(mask, pixel_height, pixel_width)
    mask_t = grow_mask_tensor(mask_t, grow_mask_by)
    mask_t = feather_mask_tensor(mask_t, feather_radius, feather_mode, feather_sigma)

    latent_mask = torch.nn.functional.interpolate(mask_t, size=(latent_height, latent_width), mode="area")
    latent_mask = torch.clamp(latent_mask, 0.0, 1.0).permute(0, 2, 3, 1).contiguous()
    return mx.array(latent_mask.numpy()).astype(mx.float32)


def prepare_padded_latent_mask_from_crop(mask_crop, render_width, render_height, decode_crop, latent_height, latent_width):
    if hasattr(mask_crop, "detach"):
        mask_t = mask_crop.detach().cpu().float()
    else:
        mask_t = torch.from_numpy(get_numpy_array(mask_crop).astype(np.float32))
    if mask_t.ndim == 2:
        mask_t = mask_t.unsqueeze(0)
    if mask_t.ndim == 3:
        mask_t = mask_t.unsqueeze(1)
    elif mask_t.ndim == 4:
        if mask_t.shape[-1] == 1 and mask_t.shape[1] != 1:
            mask_t = mask_t.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"SDMLX: Crop mask must be 2D/3D/4D, got {tuple(mask_t.shape)}.")

    render_width = int(render_width)
    render_height = int(render_height)
    if render_width % 8 != 0 or render_height % 8 != 0:
        raise ValueError(f"SDMLX Inpaint: Render crop must be aligned to an 8px grid, got {render_width}x{render_height}.")

    left, top, crop_width, crop_height = [int(value) for value in decode_crop]
    if left % 8 != 0 or top % 8 != 0:
        raise ValueError(f"SDMLX Inpaint: Decode crop is not latent-aligned: left={left}, top={top}.")
    if crop_width != render_width or crop_height != render_height:
        raise ValueError(
            "SDMLX Inpaint: Decode crop does not match render size: "
            f"{crop_width}x{crop_height} vs {render_width}x{render_height}."
        )

    render_latent_width = max(1, render_width // 8)
    render_latent_height = max(1, render_height // 8)
    latent_left = left // 8
    latent_top = top // 8
    latent_right = min(int(latent_width), latent_left + render_latent_width)
    latent_bottom = min(int(latent_height), latent_top + render_latent_height)
    if latent_right <= latent_left or latent_bottom <= latent_top:
        raise ValueError("SDMLX Inpaint: Latent mask area is empty.")

    crop_latent_mask = torch.nn.functional.interpolate(
        torch.clamp(mask_t, 0.0, 1.0),
        size=(render_latent_height, render_latent_width),
        mode="bilinear",
        align_corners=False,
    )
    latent_mask = torch.zeros(
        (crop_latent_mask.shape[0], 1, int(latent_height), int(latent_width)),
        dtype=crop_latent_mask.dtype,
    )
    latent_mask[
        :,
        :,
        latent_top:latent_bottom,
        latent_left:latent_right,
    ] = crop_latent_mask[
        :,
        :,
        :latent_bottom - latent_top,
        :latent_right - latent_left,
    ]
    latent_mask = torch.clamp(latent_mask, 0.0, 1.0).permute(0, 2, 3, 1).contiguous()
    return mx.array(latent_mask.numpy()).astype(mx.float32)


def precision_dtype(name):
    return mx.float16 if name == "float16" else mx.float32


# --- NODES ---


class SDMLX_GaussianBlurMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mask": ("MASK",),
            "kernel_size": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
            "sigma": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    CATEGORY = "SDMLX/Mask"

    def doit(self, mask, kernel_size, sigma):
        mask = sdmlx_make_3d_mask(mask)
        mask = torch.unsqueeze(mask, dim=-1)
        mask = sdmlx_tensor_gaussian_blur_mask(mask, kernel_size, sigma)
        mask = torch.squeeze(mask, dim=-1)
        return (mask,)


class SDMLX_LoaderUniversal:
    @classmethod
    def INPUT_TYPES(s):
        try:
            import folder_paths
            checkpoint_names = [SDMLX_SELECT_CHECKPOINT] + list(folder_paths.get_filename_list("checkpoints"))
        except Exception:
            checkpoint_names = [SDMLX_SELECT_CHECKPOINT]
        return {"required": {
            "ckpt_name": (checkpoint_names,),
            "memory_assist": (MEMORY_ASSIST_OPTIONS, {"default": "auto"}),
        }}

    RETURN_TYPES = ("MLX_MODEL", "MLX_CLIP", "MLX_VAE")
    FUNCTION = "load_ckpt"
    CATEGORY = "SDMLX/Loaders"

    def load_ckpt(self, ckpt_name, memory_assist="auto"):
        if not ckpt_name or ckpt_name == SDMLX_SELECT_CHECKPOINT:
            raise ValueError("SDMLX: Please select an SDXL checkpoint first.")
        import folder_paths
        from comfy.utils import load_torch_file

        total_start = time.perf_counter()
        memory_assist = set_memory_assist(memory_assist)
        preload = memory_assist == "max_performance"
        path = folder_paths.get_full_path("checkpoints", ckpt_name)
        package_path, identity = checkpoint_cache_package(path)
        try:
            return load_sdmlx_package(package_path, preload=preload)
        except FileNotFoundError:
            pass

        print(f"SDMLX: Analyzing and converting weights from {ckpt_name}...")
        load_start = time.perf_counter()
        torch_sd = load_torch_file(path)
        load_time = time.perf_counter() - load_start
        source_dtype = summarize_source_dtypes(torch_sd)
        split_start = time.perf_counter()
        validate_sdxl_checkpoint_keys(torch_sd)
        groups = split_sdxl_checkpoint(torch_sd)
        validate_text_encoder_weight_groups(groups, ckpt_name)
        split_time = time.perf_counter() - split_start

        os.makedirs(package_path, exist_ok=True)
        save_start = time.perf_counter()
        components = {
            "unet": save_package_component(package_path, "unet", groups["unet"]),
            "clip_l": save_package_component(package_path, "clip_l", groups["clip_l"], reuse_existing=True),
            "clip_g": save_package_component(package_path, "clip_g", groups["clip_g"], reuse_existing=True),
            "vae": save_package_component(package_path, "vae", groups["vae"], reuse_existing=True),
        }
        write_cache_manifest(package_path, identity, components)
        save_time = time.perf_counter() - save_start
        package_bytes = package_component_bytes(components)
        total_time = time.perf_counter() - total_start
        print(f"SDMLX: MLX checkpoint package created: {package_path}")
        print(
            "SDMLX: Conversion Timing "
            f"(load={load_time:.2f}s, map={split_time:.2f}s, save={save_time:.2f}s, total={total_time:.2f}s, "
            f"source={format_bytes(identity['source_size'])}, package={format_bytes(package_bytes)}, "
            f"source_dtype={source_dtype}, package_dtype=fp16)."
        )

        log_timing(
            f"SDMLX STATUS: UNet={len(groups['unet'])}, "
            f"CLIP-L={len(groups['clip_l'])}, CLIP-G={len(groups['clip_g'])}, "
            f"VAE={len(groups['vae'])}"
        )
        mlx_model = {"weights": groups["unet"], "cache_key": package_path}
        mlx_clip = {"clip_l": groups["clip_l"], "clip_g": groups["clip_g"], "cache_key": package_path}
        mlx_vae = {"weights": groups["vae"], "cache_key": package_path}
        if preload:
            preload_mlx_model(mlx_model, mlx_clip, mlx_vae, fast_mode=True, compute_dtype="float16", vae_dtype="float32")
        return (mlx_model, mlx_clip, mlx_vae)


class SDMLX_Loader:
    @classmethod
    def INPUT_TYPES(s):
        packages = sdmlx_package_options()
        if not packages:
            packages = ["<no .sdmlx packages found>"]
        return {"required": {
            "package_name": (packages,),
            "memory_assist": (MEMORY_ASSIST_OPTIONS, {"default": "auto"}),
        }}

    RETURN_TYPES = ("MLX_MODEL", "MLX_CLIP", "MLX_VAE")
    FUNCTION = "load_package"
    CATEGORY = "SDMLX/Loaders"

    def load_package(self, package_name, memory_assist="auto"):
        if package_name.startswith("<"):
            raise FileNotFoundError("SDMLX: No .sdmlx packages found.")
        memory_assist = set_memory_assist(memory_assist)
        preload = memory_assist == "max_performance"
        package_path = os.path.join(cache_dir(), package_name)
        if not os.path.isdir(package_path):
            raise FileNotFoundError(f"SDMLX: Package not found: {package_name}")
        return load_sdmlx_package(package_path, preload=preload)


class SDMLX_ControlNetUnionLoader:
    @classmethod
    def INPUT_TYPES(s):
        models = controlnet_file_options()
        return {"required": {
            "control_net_name": (models,),
        }}

    RETURN_TYPES = ("MLX_CONTROLNET",)
    RETURN_NAMES = ("mlx_controlnet",)
    FUNCTION = "load_controlnet"
    CATEGORY = "SDMLX/ControlNet"

    @classmethod
    def VALIDATE_INPUTS(s, control_net_name):
        return True

    def load_controlnet(self, control_net_name):
        path, resolved_name = resolve_controlnet_path(control_net_name)
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"SDMLX: ControlNet not found: {control_net_name}")
        weights = load_weight_file(path)
        if weights is None:
            raise ValueError(f"SDMLX: ControlNet could not be loaded: {control_net_name}")
        validate_controlnet_union_promax_weights(weights, resolved_name)
        identity = lora_file_identity(path)
        log_timing(f"SDMLX: ControlNet Union weights loaded: {resolved_name} ({len(weights)} Tensoren).")
        return ({
            "weights": weights,
            "identity": identity,
            "cache_key": path,
            "dtype": "float16",
            "name": resolved_name,
        },)


class SDMLX_ApplyControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("MLX_CONDITIONING",),
            "negative": ("MLX_CONDITIONING",),
            "mlx_controlnet": ("MLX_CONTROLNET",),
            "control_image": ("IMAGE",),
            "control_type": (list(UNION_CONTROL_TYPES.keys()), {"default": "line to canny"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "resize_mode": (["crop center", "crop top", "fit pad", "stretch"], {"default": "crop center"}),
        }, "optional": {
            "controlnet_scheduler": ("SDMLX_SCHEDULE",),
        }}

    RETURN_TYPES = ("MLX_CONDITIONING", "MLX_CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    CATEGORY = "SDMLX/ControlNet"

    def apply_controlnet(
        self,
        positive,
        negative,
        mlx_controlnet,
        control_image,
        control_type,
        strength,
        start_percent,
        end_percent,
        resize_mode="crop center",
        controlnet_scheduler=None,
    ):
        schedule = controlnet_scheduler if isinstance(controlnet_scheduler, dict) else None
        control = {
            "controlnet": mlx_controlnet,
            "image": control_image,
            "control_type": UNION_CONTROL_TYPES[control_type],
            "control_type_name": control_type,
            "strength": float(strength),
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "resize_mode": resize_mode,
        }
        if schedule is not None:
            control["schedule"] = schedule
        positive = add_controlnet_to_conditioning(positive, control)
        negative = add_controlnet_to_conditioning(negative, control)
        schedule_text = ""
        strength_text = f"{float(strength):g}"
        if schedule is not None:
            strength_text = "schedule"
            schedule_text = (
                f", schedule={schedule['mode']}/{schedule['curve']} "
                f"{schedule['start_percent']:g}-{schedule['end_percent']:g}, "
                f"range={schedule['minimum_strength']:g}..{schedule['maximum_strength']:g}, "
                "strength/start/end ignored"
            )
        log_timing(
            "SDMLX: ControlNet applied "
            f"(type={control_type}, strength={strength_text}, "
            f"start={float(start_percent):g}, end={float(end_percent):g}, "
            f"resize={resize_mode}{schedule_text})."
        )
        return (positive, negative)


class SDMLX_LoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        try:
            import folder_paths
            lora_names = [MULTI_LORA_NONE] + list(folder_paths.get_filename_list("loras"))
        except Exception:
            lora_names = [MULTI_LORA_NONE]
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "lora_name": (lora_names,),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora_scheduler": ("SDMLX_SCHEDULE",),
            },
        }

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("mlx_model",)
    FUNCTION = "load_lora"
    CATEGORY = "SDMLX"

    def load_lora(
        self,
        mlx_model,
        lora_name=None,
        strength=1.0,
        enabled=True,
        lora_scheduler=None,
    ):
        import folder_paths

        if not enabled or not lora_name or lora_name == MULTI_LORA_NONE:
            log_timing(f"SDMLX: LoRA disabled: {lora_name}")
            return (mlx_model,)

        if hasattr(folder_paths, "get_full_path_or_raise"):
            path = folder_paths.get_full_path_or_raise("loras", lora_name)
        else:
            path = folder_paths.get_full_path("loras", lora_name)
            if path is None:
                raise FileNotFoundError(f"SDMLX: LoRA not found: {lora_name}")

        strength = float(strength)
        schedule = lora_scheduler if isinstance(lora_scheduler, dict) else None
        effective_strength_model = lora_strength_for_item(strength, schedule)
        model_loras = list(mlx_model.get("loras", []))
        if effective_strength_model != 0.0:
            item = {
                "name": lora_name,
                "path": os.path.abspath(path),
                "strength_model": effective_strength_model,
                "identity": lora_file_identity(path),
            }
            if schedule is not None:
                item["schedule"] = schedule
            model_loras.append(item)
        patched_model = {**mlx_model, "loras": model_loras}
        schedule_text = ""
        model_text = f"{effective_strength_model:g}"
        if schedule is not None:
            model_text = "schedule"
            schedule_text = (
                f", schedule={schedule['mode']}/{schedule['curve']} {schedule['start_percent']:g}-{schedule['end_percent']:g}, "
                f"range={schedule['minimum_strength']:g}..{schedule['maximum_strength']:g}, "
                f"strength ignored"
            )
        log_timing(
            f"SDMLX: LoRA Stack erweitert: {lora_name} "
            f"(strength={model_text}{schedule_text}, total={len(model_loras)})."
        )
        return (patched_model,)


MULTI_LORA_SLOT_COUNT = 12
MULTI_LORA_NONE = ""


def lora_file_options_with_none():
    try:
        import folder_paths
        return [MULTI_LORA_NONE] + list(folder_paths.get_filename_list("loras"))
    except Exception:
        return [MULTI_LORA_NONE]


def resolve_lora_path_or_raise(lora_name):
    import folder_paths
    if hasattr(folder_paths, "get_full_path_or_raise"):
        return folder_paths.get_full_path_or_raise("loras", lora_name)
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise FileNotFoundError(f"SDMLX: LoRA not found: {lora_name}")
    return path


class SDMLX_MultiLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        required = {
            "mlx_model": ("MLX_MODEL",),
        }
        loras = lora_file_options_with_none()
        required["slot_count"] = ("INT", {"default": 1, "min": 1, "max": MULTI_LORA_SLOT_COUNT, "socketless": True})
        for index in range(1, MULTI_LORA_SLOT_COUNT + 1):
            required[f"enabled_{index}"] = ("BOOLEAN", {"default": True, "socketless": True})
            required[f"lora_{index}"] = (loras, {"socketless": True})
            required[f"strength_{index}"] = ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05, "socketless": True})
        return {"required": required}

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("mlx_model",)
    FUNCTION = "load_loras"
    CATEGORY = "SDMLX"

    def load_loras(self, mlx_model, **kwargs):
        model_loras = list(mlx_model.get("loras", []))
        added = []
        slot_count = int(kwargs.get("slot_count", 1))
        slot_count = max(1, min(MULTI_LORA_SLOT_COUNT, slot_count))

        for index in range(1, slot_count + 1):
            if not bool(kwargs.get(f"enabled_{index}", True)):
                continue
            lora_name = kwargs.get(f"lora_{index}", MULTI_LORA_NONE)
            if not lora_name or lora_name == MULTI_LORA_NONE:
                continue
            strength_model = float(kwargs.get(f"strength_{index}", 1.0))
            if strength_model == 0.0:
                continue

            path = resolve_lora_path_or_raise(lora_name)
            item = {
                "name": lora_name,
                "path": os.path.abspath(path),
                "strength_model": strength_model,
                "identity": lora_file_identity(path),
            }
            model_loras.append(item)
            added.append(
                {
                    "slot": index,
                    "name": lora_name,
                    "strength": strength_model,
                }
            )

        patched_model = {**mlx_model, "loras": model_loras}
        if added:
            details = []
            for item in added:
                details.append(
                    f"#{item['slot']} {item['name']} strength={item['strength']:g}"
                )
            log_timing(
                "SDMLX: Multi LoRA stack extended "
                f"({len(added)} LoRAs, total={len(model_loras)}): "
                + "; ".join(details)
            )
        else:
            log_timing("SDMLX: Multi LoRA Loader: no active LoRA selected.")
        return (patched_model,)


class SDMLX_SpeedPatchConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "speed_lora": (supported_speed_lora_options(),),
            "force_rebuild": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("SDMLX_SPEED_PATCH", "STRING")
    RETURN_NAMES = ("speed_patch", "message")
    FUNCTION = "convert"
    CATEGORY = "SDMLX/Loaders"

    def convert(self, speed_lora, force_rebuild=False):
        if not speed_lora or speed_lora == SUPPORTED_SPEED_LORA_PLACEHOLDER:
            accepted = ", ".join(sorted(SUPPORTED_SPEED_LORA_PATCHES.keys(), key=str.lower))
            raise ValueError(
                "SDMLX: No supported speed LoRA found. "
                "Place one of the accepted LoRAs in models/loras: "
                f"{accepted}"
            )

        patch_info = supported_speed_lora_info(speed_lora)
        if patch_info is None:
            raise ValueError(
                "SDMLX: This LoRA is not approved as a speed patch. "
                f"Selected: {speed_lora}"
            )

        path = resolve_lora_path_or_raise(speed_lora)
        result = build_speed_patch_from_lora(path, patch_info, bool(force_rebuild))
        action = "built" if result["built"] else "already exists"
        message = (
            f"{result['label']} {action}: {result['modules']} Module, "
            f"{result['path']}"
        )
        print(f"SDMLX: Speed Patch Converter: {message}")
        return (result["label"], message)


class SDMLX_SpectrumBoost:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "degree": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
            "ridge": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            "window_size": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            "flex_window": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005}),
            "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 20, "step": 1}),
            "final_real_steps": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
        }}

    RETURN_TYPES = ("SDMLX_SPECTRUM_ACCELERATION",)
    RETURN_NAMES = ("spectrum_acceleration",)
    FUNCTION = "build"
    CATEGORY = "SDMLX/Sampling"

    def build(
        self,
        weight,
        degree,
        ridge,
        window_size,
        flex_window,
        warmup_steps,
        final_real_steps,
    ):
        return ({
            "profile": "advanced",
            "advanced": True,
            "final_real_steps": int(final_real_steps),
            "weight": float(weight),
            "degree": int(degree),
            "ridge": float(ridge),
            "window_size": float(window_size),
            "flex_window": float(flex_window),
            "warmup_steps": int(warmup_steps),
        },)


class SDMLX_LoraSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mode": (LORA_SCHEDULE_MODES, {"default": "blend in"}),
            "minimum_strength": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "maximum_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "curve": (LORA_SCHEDULE_CURVES, {"default": "linear"}),
            "advanced": ("BOOLEAN", {"default": False}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
        }}

    RETURN_TYPES = ("SDMLX_SCHEDULE",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "create"
    CATEGORY = "SDMLX"

    def create(
        self,
        mode="blend in",
        minimum_strength=0.0,
        maximum_strength=1.0,
        curve="linear",
        advanced=False,
        start_percent=0.0,
        end_percent=1.0,
    ):
        schedule = make_lora_schedule(
            mode=mode,
            minimum_strength=minimum_strength,
            maximum_strength=maximum_strength,
            curve=curve,
            advanced=advanced,
            start_percent=start_percent,
            end_percent=end_percent,
        )
        print(
            "SDMLX: Scheduler created "
            f"(mode={schedule['mode']}, curve={schedule['curve']}, "
            f"window={schedule['start_percent']:g}-{schedule['end_percent']:g}, "
            f"range={schedule['minimum_strength']:g}..{schedule['maximum_strength']:g}, "
            f"advanced={bool(advanced)})."
        )
        return (schedule,)


class SDMLX_IPAdapterLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ipadapter_name": (ipadapter_model_options(),),
        }}

    RETURN_TYPES = ("SDMLX_IPADAPTER",)
    RETURN_NAMES = ("ipadapter",)
    FUNCTION = "load"
    CATEGORY = "SDMLX/IPAdapter"

    @classmethod
    def VALIDATE_INPUTS(s, ipadapter_name):
        return True

    def load(self, ipadapter_name):
        return (load_sdmlx_ipadapter_model(ipadapter_name),)


class SDMLX_CLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision_name": (clip_vision_model_options(),),
            "compute_dtype": (["float16", "float32"], {"default": "float16"}),
        }}

    RETURN_TYPES = ("SDMLX_CLIP_VISION",)
    RETURN_NAMES = ("sdmlx_clip_vision",)
    FUNCTION = "load"
    CATEGORY = "SDMLX/IPAdapter"

    @classmethod
    def VALIDATE_INPUTS(s, clip_vision_name):
        return True

    def load(self, clip_vision_name, compute_dtype="float16"):
        return (load_sdmlx_clip_vision_model(clip_vision_name, compute_dtype),)


class SDMLX_IPAdapterMLXCLIPVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "sdmlx_clip_vision": ("SDMLX_CLIP_VISION",),
            "image": ("IMAGE",),
            "resize_mode": (["crop center", "crop top", "fit pad", "stretch"], {"default": "crop center"}),
        }}

    RETURN_TYPES = ("CLIP_VISION_OUTPUT", "IMAGE")
    RETURN_NAMES = ("clip_vision_output", "image")
    FUNCTION = "encode"
    CATEGORY = "SDMLX/IPAdapter"

    def encode(self, sdmlx_clip_vision, image, resize_mode="crop center"):
        return (encode_sdmlx_clip_vision_for_ipadapter(sdmlx_clip_vision, image, resize_mode), image)


class SDMLX_InsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "provider": (["CPU", "CoreML"], {"default": "CPU"}),
        }}

    RETURN_TYPES = ("SDMLX_INSIGHTFACE",)
    RETURN_NAMES = ("insightface",)
    FUNCTION = "load"
    CATEGORY = "SDMLX/IPAdapter"

    def load(self, provider="CoreML"):
        return (load_sdmlx_insightface(provider),)


class SDMLX_InsightFaceAlignCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "insightface": ("SDMLX_INSIGHTFACE",),
            "image": ("IMAGE",),
            "size": ("INT", {"default": 256, "min": 128, "max": 512, "step": 16}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_image",)
    FUNCTION = "crop"
    CATEGORY = "SDMLX/IPAdapter"

    def crop(self, insightface, image, size=256):
        _, face_image = extract_insightface_face_data(
            insightface,
            image,
            unnorm=False,
            aligned_crop=True,
            aligned_size=int(size),
        )
        print(f"SDMLX: InsightFace align crop created ({int(size)}x{int(size)}, batch={face_image.shape[0]}).")
        return (face_image,)


class SDMLX_ApplyIPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "ipadapter": ("SDMLX_IPADAPTER",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 3.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (IPADAPTER_WEIGHT_TYPES, {"default": "linear"}),
                "embeds_scaling": (IPADAPTER_EMBEDS_SCALING, {"default": "V only"}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "ipadapter_scheduler": ("SDMLX_SCHEDULE",),
            },
        }

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("mlx_model",)
    FUNCTION = "apply"
    CATEGORY = "SDMLX/IPAdapter"

    def apply(
        self,
        mlx_model,
        ipadapter,
        clip_vision_output,
        weight=1.0,
        start_at=0.0,
        end_at=1.0,
        weight_type="linear",
        embeds_scaling="V only",
        enabled=True,
        negative_clip_vision_output=None,
        ipadapter_scheduler=None,
    ):
        if not enabled or float(weight) == 0.0:
            print("SDMLX: IP-Adapter disabled.")
            return (mlx_model,)
        schedule = ipadapter_scheduler if isinstance(ipadapter_scheduler, dict) else None
        if schedule is None and end_at <= start_at:
            raise ValueError("SDMLX: IP-Adapter end_at must be greater than start_at.")

        cond, uncond = project_standard_ipadapter_embeds(ipadapter, clip_vision_output, negative_clip_vision_output)
        start_value = 0.0 if schedule is not None else float(start_at)
        end_value = 1.0 if schedule is not None else float(end_at)
        adapter = {
            "name": ipadapter.get("name", "ipadapter"),
            "ip_adapter": ipadapter["ip_adapter"],
            "cond": cond,
            "uncond": uncond,
            "weight": ipadapter_weight_value(float(weight), float(weight), weight_type),
            "weight_type": weight_type,
            "start_at": start_value,
            "end_at": end_value,
            "embeds_scaling": embeds_scaling,
            "layers": int(ipadapter.get("layers", 11)),
        }
        if schedule is not None:
            adapter["schedule"] = schedule
        patched_model = add_ipadapter_to_model(mlx_model, adapter)
        schedule_text = ""
        if schedule is not None:
            schedule_text = (
                f", schedule={schedule['mode']}/{schedule['curve']} "
                f"{schedule['start_percent']:g}-{schedule['end_percent']:g}, "
                f"range={schedule['minimum_strength']:g}..{schedule['maximum_strength']:g}, "
                "start/end ignored"
            )
        print(
            "SDMLX: IP-Adapter applied "
            f"({adapter['name']}, weight={float(weight):g}, start={start_value:g}, "
            f"end={end_value:g}, type={weight_type}, scaling={embeds_scaling}{schedule_text}, "
            f"tokens={tuple(cond.shape)})."
        )
        return (patched_model,)


class SDMLX_ApplyIPAdapterFaceID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "ipadapter": ("SDMLX_IPADAPTER",),
                "insightface": ("SDMLX_INSIGHTFACE",),
                "image": ("IMAGE",),
                "identity_bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "img_details_v2_only": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (IPADAPTER_WEIGHT_TYPES, {"default": "linear"}),
                "embeds_scaling": (IPADAPTER_EMBEDS_SCALING, {"default": "V only"}),
                "auto_lora": ("BOOLEAN", {"default": True}),
                "lora_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sdmlx_clip_vision": ("SDMLX_CLIP_VISION",),
                "ipadapter_scheduler": ("SDMLX_SCHEDULE",),
                "lora_scheduler": ("SDMLX_SCHEDULE",),
            },
        }

    RETURN_TYPES = ("MLX_MODEL", "IMAGE")
    RETURN_NAMES = ("mlx_model", "face_image")
    FUNCTION = "apply"
    CATEGORY = "SDMLX/IPAdapter"

    def apply(
        self,
        mlx_model,
        ipadapter,
        insightface,
        image,
        identity_bias=0.0,
        img_details_v2_only=1.0,
        start_at=0.0,
        end_at=1.0,
        weight_type="linear",
        embeds_scaling="V only",
        auto_lora=True,
        lora_strength=0.6,
        enabled=True,
        sdmlx_clip_vision=None,
        ipadapter_scheduler=None,
        lora_scheduler=None,
    ):
        identity_base_weight = faceid_identity_base_weight(ipadapter)
        identity_weight = identity_base_weight + float(identity_bias)
        effective_weight_type = weight_type
        ip_schedule = ipadapter_scheduler if isinstance(ipadapter_scheduler, dict) else None
        if not enabled or abs(identity_weight) <= 1e-6:
            print("SDMLX: FaceID IP-Adapter disabled.")
            return (mlx_model, image)
        if not ipadapter.get("is_faceid"):
            raise ValueError("SDMLX: This node expects a FaceID/Portrait IP-Adapter model.")
        if ip_schedule is None and end_at <= start_at:
            raise ValueError("SDMLX: FaceID IP-Adapter end_at must be greater than start_at.")

        is_plusv2_adapter = bool(ipadapter.get("is_faceid_plusv2"))
        use_unnorm = bool(ipadapter.get("is_portrait_unnorm"))
        face_embeds, aligned_face_image = extract_insightface_face_data(
            insightface,
            image,
            unnorm=use_unnorm,
            aligned_crop=True,
            aligned_size=256,
        )
        clip_source = "not used"
        detail_transfer_requested = float(img_details_v2_only)
        detail_transfer_log = "img_details_v2_only=not used"
        if is_plusv2_adapter and sdmlx_clip_vision is not None:
            clip_vision_output = encode_sdmlx_clip_vision_for_ipadapter(sdmlx_clip_vision, aligned_face_image, "crop center")
            zero_size = int(sdmlx_clip_vision["config"].get("image_size", 224))
            zero_face = torch.zeros((1, zero_size, zero_size, 3), dtype=image.dtype if hasattr(image, "dtype") else torch.float32)
            negative_clip_vision_output = encode_sdmlx_clip_vision_for_ipadapter(sdmlx_clip_vision, zero_face, "stretch")
            clip_source = "SDMLX MLX face crop"
        elif is_plusv2_adapter:
            clip_vision_output = None
            negative_clip_vision_output = None

        if is_plusv2_adapter:
            cond, uncond = project_faceid_plusv2_embeds(
                ipadapter,
                face_embeds,
                clip_vision_output=clip_vision_output,
                negative_clip_vision_output=negative_clip_vision_output,
                image_detail_transfer=detail_transfer_requested,
            )
            detail_transfer_log = f"img_details_v2_only={detail_transfer_requested:g}"
        else:
            cond, uncond = project_faceid_base_or_portrait_embeds(ipadapter, face_embeds)
        cond_f = cond.astype(mx.float32)
        uncond_f = uncond.astype(mx.float32)
        faceid_signal = float(mx.sqrt(mx.mean(mx.square(cond_f - uncond_f))).item())
        start_value = 0.0 if ip_schedule is not None else float(start_at)
        end_value = 1.0 if ip_schedule is not None else float(end_at)
        adapter = {
            "name": ipadapter.get("name", "faceid"),
            "ip_adapter": ipadapter["ip_adapter"],
            "cond": cond,
            "uncond": uncond,
            "weight": ipadapter_weight_value(identity_weight, identity_weight, effective_weight_type),
            "weight_type": effective_weight_type,
            "start_at": start_value,
            "end_at": end_value,
            "embeds_scaling": embeds_scaling,
            "layers": int(ipadapter.get("layers", 11)),
            "is_faceid": True,
            "is_faceid_plusv2": bool(ipadapter.get("is_faceid_plusv2")),
            "is_portrait": bool(ipadapter.get("is_portrait")),
            "is_portrait_unnorm": bool(ipadapter.get("is_portrait_unnorm")),
        }
        if ip_schedule is not None:
            adapter["schedule"] = ip_schedule
        patched_model = add_ipadapter_to_model(mlx_model, adapter)

        lora_state = "none"
        if auto_lora and ipadapter.get("has_lora_ip_weights"):
            lora_schedule = lora_scheduler if isinstance(lora_scheduler, dict) else None
            face_lora_label = "schedule" if lora_schedule is not None else f"{float(lora_strength):g}"
            patched_model, added = add_internal_faceid_lora_to_model_once(patched_model, ipadapter, float(lora_strength), lora_schedule)
            if added:
                lora_state = f"internal@{face_lora_label}"
            else:
                lora_name, lora_path = find_lora_model_by_basename(faceid_lora_candidates(ipadapter.get("name", "")))
                if lora_path is not None:
                    patched_model, added = add_lora_to_model_once(patched_model, lora_name, lora_path, float(lora_strength), lora_schedule)
                    lora_state = f"{lora_name}@{face_lora_label}" if added else f"{lora_name} already present"
                else:
                    lora_state = "missing"
                    print("SDMLX: FaceID LoRA not found; FaceID runs without additional UNet LoRA.")
            if lora_schedule is not None and lora_state not in ("none", "missing"):
                lora_state = (
                    f"{lora_state}, schedule={lora_schedule['mode']}/{lora_schedule['curve']} "
                    f"{lora_schedule['start_percent']:g}-{lora_schedule['end_percent']:g}, "
                    f"range={lora_schedule['minimum_strength']:g}..{lora_schedule['maximum_strength']:g}"
                )

        if ipadapter.get("is_portrait") or ipadapter.get("is_portrait_unnorm"):
            print(
                "SDMLX: Note: FaceID Portrait adapters are style-transfer models without LoRA; "
                "FaceID PlusV2 is usually the better choice for robust identity transfer."
            )
            if abs(detail_transfer_requested - 1.0) > 1e-6:
                print(
                    "SDMLX: FaceID Portrait Safe Mode active "
                    "(PlusV2 image-detail path skipped; img_details_v2_only is not read)."
                )
        schedule_text = ""
        if ip_schedule is not None:
            schedule_text = (
                f", ip_schedule={ip_schedule['mode']}/{ip_schedule['curve']} "
                f"{ip_schedule['start_percent']:g}-{ip_schedule['end_percent']:g}, "
                f"range={ip_schedule['minimum_strength']:g}..{ip_schedule['maximum_strength']:g}, "
                "start/end ignored"
            )
        print(
            "SDMLX: FaceID IP-Adapter applied "
            f"({adapter['name']}, faces={face_embeds.shape[0]}, tokens={tuple(cond.shape)}, "
            f"identity_bias={float(identity_bias):g}, identity_base={identity_base_weight:g}, "
            f"identity_weight={identity_weight:g}, "
            f"start={start_value:g}, end={end_value:g}, "
            f"type={effective_weight_type}, scaling={embeds_scaling}, {detail_transfer_log}, "
            f"clip={clip_source}, signal={faceid_signal:.3f}{schedule_text}, lora={lora_state})."
        )
        return (patched_model, aligned_face_image)


class SDMLX_ApplyIPAdapterFaceIDAIO:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mlx_model": ("MLX_MODEL",),
            "image": ("IMAGE",),
            "ipadapter_name": (faceid_ipadapter_model_options(),),
            "clip_vision_name": (faceid_clip_vision_model_options(), {"default": AUTO_CLIP_VISION_OPTION}),
            "insightface_provider": (["CPU", "CoreML"], {"default": "CPU"}),
            "identity_bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            "img_details_v2_only": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
            "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "weight_type": (IPADAPTER_WEIGHT_TYPES, {"default": "linear"}),
            "embeds_scaling": (IPADAPTER_EMBEDS_SCALING, {"default": "V only"}),
            "auto_lora": ("BOOLEAN", {"default": True}),
            "lora_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05}),
            "enabled": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "ipadapter_scheduler": ("SDMLX_SCHEDULE",),
            "lora_scheduler": ("SDMLX_SCHEDULE",),
        }}

    RETURN_TYPES = ("MLX_MODEL", "IMAGE")
    RETURN_NAMES = ("mlx_model", "face_image")
    FUNCTION = "apply"
    CATEGORY = "SDMLX/IPAdapter"

    @classmethod
    def VALIDATE_INPUTS(s, ipadapter_name, clip_vision_name):
        return True

    def apply(
        self,
        mlx_model,
        image,
        ipadapter_name,
        clip_vision_name=AUTO_CLIP_VISION_OPTION,
        insightface_provider="CPU",
        identity_bias=0.0,
        img_details_v2_only=1.0,
        start_at=0.0,
        end_at=1.0,
        weight_type="linear",
        embeds_scaling="V only",
        auto_lora=True,
        lora_strength=0.6,
        enabled=True,
        ipadapter_scheduler=None,
        lora_scheduler=None,
    ):
        if not enabled:
            print("SDMLX: FaceID AIO disabled.")
            return (mlx_model, image)

        ipadapter = load_sdmlx_ipadapter_model(ipadapter_name)
        if not ipadapter.get("is_faceid"):
            raise ValueError("SDMLX: FaceID AIO erwartet ein FaceID/Portrait IP-Adapter-Modell.")
        ip_schedule = ipadapter_scheduler if isinstance(ipadapter_scheduler, dict) else None
        if ip_schedule is None and end_at <= start_at:
            raise ValueError("SDMLX: FaceID AIO end_at must be greater than start_at.")

        identity_base_weight = faceid_identity_base_weight(ipadapter)
        identity_weight = identity_base_weight + float(identity_bias)
        if abs(identity_weight) <= 1e-6:
            print("SDMLX: FaceID AIO disabled, because identity_bias sets the identity weight to 0.")
            return (mlx_model, image)

        sdmlx_clip_vision = None
        selected_clip_vision = "not used"
        if ipadapter.get("is_faceid_plusv2"):
            selected_clip_vision = resolve_faceid_clip_vision_name(clip_vision_name)
            if selected_clip_vision == CLIP_VISION_PLACEHOLDER:
                raise ValueError(
                    "SDMLX: FaceID PlusV2 braucht ein CLIP-Vision-Modell. "
                    "Please place CLIP-ViT-H-14-laion2B-s32B-b79K in models/clip_vision."
                )
            sdmlx_clip_vision = load_sdmlx_clip_vision_model(selected_clip_vision, "float16")
            expected_hidden = int(ipadapter.get("clip_vision_dim") or 0)
            actual_hidden = int(sdmlx_clip_vision["config"].get("hidden_size") or 0)
            if expected_hidden and actual_hidden and expected_hidden != actual_hidden:
                raise ValueError(
                    "SDMLX: FaceID AIO CLIP Vision does not match the adapter "
                    f"(adapter expects hidden={expected_hidden}, loaded hidden={actual_hidden}: "
                    f"{selected_clip_vision}). Fuer FaceID PlusV2 meist CLIP-ViT-H-14 nutzen."
                )

        insightface = load_sdmlx_insightface(insightface_provider)
        patched_model, face_image = SDMLX_ApplyIPAdapterFaceID().apply(
            mlx_model,
            ipadapter,
            insightface,
            image,
            identity_bias=identity_bias,
            img_details_v2_only=img_details_v2_only,
            start_at=start_at,
            end_at=end_at,
            weight_type=weight_type,
            embeds_scaling=embeds_scaling,
            auto_lora=auto_lora,
            lora_strength=lora_strength,
            enabled=enabled,
            sdmlx_clip_vision=sdmlx_clip_vision,
            ipadapter_scheduler=ipadapter_scheduler,
            lora_scheduler=lora_scheduler,
        )
        print(
            "SDMLX: FaceID AIO finished "
            f"(ipadapter={ipadapter_name}, clip_vision={selected_clip_vision}, "
            f"insightface={insightface_provider})."
        )
        return (patched_model, face_image)


class SDMLX_DifferentialDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mlx_model": ("MLX_MODEL",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "enabled": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("mlx_model",)
    FUNCTION = "apply"
    CATEGORY = "SDMLX/Inpaint"

    def apply(self, mlx_model, strength=1.0, enabled=True):
        strength = max(0.0, min(1.0, float(strength)))
        patched_model = {
            **mlx_model,
            "differential_mask": bool(enabled),
            "differential_mask_strength": strength,
        }
        state = "active" if enabled else "disabled"
        print(
            "SDMLX: Differential Diffusion "
            f"{state} (strength={strength:g}, applies to latents with noise_mask)."
        )
        return (patched_model,)


class SDMLX_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mlx_clip": ("MLX_CLIP", ), "text": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("MLX_CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "SDMLX"

    def encode(self, mlx_clip, text):
        start_time = time.perf_counter()
        conditioning_key = (CONDITIONING_CACHE_VERSION, mlx_clip["cache_key"], text)
        if conditioning_key in CONDITIONING_CACHE:
            cached = CONDITIONING_CACHE[conditioning_key]
            log_timing("SDMLX: CLIP conditioning loaded from RAM cache.")
            if isinstance(cached, dict):
                return (cached,)
            cond, pooled = cached
            return ({"cond": cond, "pooled": pooled, "text": text},)

        def run_clip(data, is_g=False):
            if not data: return None

            clip = get_clip_model(mlx_clip["cache_key"], data, is_g=is_g)
            tokenizer = get_clip_g_tokenizer() if is_g else get_clip_l_tokenizer()
            tokens = mx.array(tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="np")["input_ids"])
            output = clip(tokens)
            if output.hidden_states and len(output.hidden_states) >= 2:
                output.last_hidden_state = output.hidden_states[-2]
            return output

        res_l = run_clip(mlx_clip["clip_l"], is_g=False)
        res_g = run_clip(mlx_clip["clip_g"], is_g=True)
        
        cond = mx.concatenate([res_l.last_hidden_state, res_g.last_hidden_state], axis=2)
        pooled = res_g.pooled_output if hasattr(res_g, "pooled_output") else mx.zeros((1, 1280))
        mx.eval(cond, pooled)
        conditioning = {"cond": cond, "pooled": pooled, "text": text}
        CONDITIONING_CACHE[conditioning_key] = conditioning
        log_timing(f"SDMLX: CLIP encode finished in {time.perf_counter() - start_time:.2f}s.")
        return (conditioning,)


class SDMLX_InpaintConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("MLX_CONDITIONING",),
            "negative": ("MLX_CONDITIONING",),
            "pixels": ("IMAGE",),
            "mlx_vae": ("MLX_VAE",),
            "mask": ("MASK",),
            "noise_mask": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MLX_CONDITIONING", "MLX_CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "SDMLX/Inpaint"

    def encode(
        self,
        positive,
        negative,
        pixels,
        mlx_vae,
        mask,
        noise_mask=True,
    ):
        start_time = time.perf_counter()
        if hasattr(pixels, "detach"):
            pixel_shape = tuple(pixels.shape)
            pixel_height = int(pixels.shape[1])
            pixel_width = int(pixels.shape[2])
        else:
            pixels_np = get_numpy_array(pixels)
            pixel_shape = tuple(pixels_np.shape)
            pixel_height = int(pixels_np.shape[1])
            pixel_width = int(pixels_np.shape[2])

        mask_chw = mask_to_chw(mask, pixel_height, pixel_width)
        target_width, target_height = inpaint_work_size(pixel_width, pixel_height, multiple=64)
        pixels, mask_chw, decode_crop = pad_inpaint_pixels_and_mask(pixels, mask_chw, target_width, target_height)
        padded = target_width != pixel_width or target_height != pixel_height
        work_width = target_width
        work_height = target_height

        latents = encode_pixels_to_latents(mlx_vae, pixels, "float32")
        out = {"samples": latents}
        if padded:
            out["sdmlx_decode_crop"] = decode_crop
            out["sdmlx_original_size"] = (pixel_width, pixel_height)
            out["sdmlx_padded_size"] = (work_width, work_height)
        if noise_mask:
            out["noise_mask"] = prepare_inpaint_mask(
                mask_chw,
                work_height,
                work_width,
                latents.shape[1],
                latents.shape[2],
            )
        print(
            "SDMLX: Inpaint conditioning created "
            f"(image={pixel_shape}, latent={tuple(latents.shape)}, "
            f"work_size={work_width}x{work_height}, padded={padded}, "
            f"noise_mask={noise_mask}, mode=comfy_baseline, "
            "masked_latent=original, mask_coverage=grow_area, sdxl_time_ids=default, "
            f"{time.perf_counter() - start_time:.2f}s)."
        )
        return (positive, negative, out)


def mask_bounds(mask_t, threshold=0.01):
    coords = torch.nonzero(mask_t > threshold, as_tuple=False)
    if coords.numel() == 0:
        height = int(mask_t.shape[-2])
        width = int(mask_t.shape[-1])
        return 0, 0, width, height
    y0 = int(coords[:, -2].min().item())
    y1 = int(coords[:, -2].max().item()) + 1
    x0 = int(coords[:, -1].min().item())
    x1 = int(coords[:, -1].max().item()) + 1
    return x0, y0, x1, y1


def round_up(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def round_up_to_multiple(value, multiple):
    multiple = max(1, int(multiple))
    return int(max(multiple, math.ceil(float(value) / multiple) * multiple))


def crop_bbox_by_factor(mask_bbox, image_width, image_height, crop=2.0, crop_aspect=0.0):
    x0, y0, x1, y1 = mask_bbox
    mask_w = max(1, x1 - x0)
    mask_h = max(1, y1 - y0)
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    crop = max(1.0, min(10.0, float(crop)))
    crop_aspect = max(-1.0, min(1.0, float(crop_aspect)))
    width_factor = crop * (1.0 + max(crop_aspect, 0.0))
    height_factor = crop * (1.0 + max(-crop_aspect, 0.0))
    crop_w = min(int(image_width), max(mask_w, int(round(mask_w * width_factor))))
    crop_h = min(int(image_height), max(mask_h, int(round(mask_h * height_factor))))

    x0 = int(round(center_x - crop_w * 0.5))
    y0 = int(round(center_y - crop_h * 0.5))
    x0 = min(max(0, x0), max(0, int(image_width) - crop_w))
    y0 = min(max(0, y0), max(0, int(image_height) - crop_h))
    return x0, y0, x0 + crop_w, y0 + crop_h


def align_bbox_size_to_multiple(bbox, image_width, image_height, multiple=8):
    x0, y0, x1, y1 = [int(value) for value in bbox]
    image_width = int(image_width)
    image_height = int(image_height)
    multiple = max(1, int(multiple))
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    target_w = min(image_width, round_up_to_multiple(crop_w, multiple))
    target_h = min(image_height, round_up_to_multiple(crop_h, multiple))
    if target_w == crop_w and target_h == crop_h:
        return x0, y0, x1, y1

    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    new_x0 = int(round(center_x - target_w * 0.5))
    new_y0 = int(round(center_y - target_h * 0.5))
    new_x0 = min(max(0, new_x0), max(0, image_width - target_w))
    new_y0 = min(max(0, new_y0), max(0, image_height - target_h))
    return new_x0, new_y0, new_x0 + target_w, new_y0 + target_h


def parse_scale_factor(scale):
    return float(str(scale).rstrip("x"))


def detailer_guide_render_size(crop_w, crop_h, mask_bbox, guide_size, guide_size_for_bbox, max_size):
    crop_w = max(1, int(crop_w))
    crop_h = max(1, int(crop_h))
    bbox_w = max(1, int(mask_bbox[2] - mask_bbox[0]))
    bbox_h = max(1, int(mask_bbox[3] - mask_bbox[1]))
    guide_size = max(64.0, float(guide_size))
    max_size = max(64.0, float(max_size))
    guide_base = min(bbox_w, bbox_h) if guide_size_for_bbox else min(crop_w, crop_h)
    upscale = guide_size / max(1.0, float(guide_base))
    render_w = int(crop_w * upscale)
    render_h = int(crop_h * upscale)

    if render_w > max_size or render_h > max_size:
        upscale *= max_size / max(float(render_w), float(render_h))
        render_w = int(crop_w * upscale)
        render_h = int(crop_h * upscale)

    if upscale <= 1.0 or render_w <= 0 or render_h <= 0:
        upscale = 1.0
        render_w = crop_w
        render_h = crop_h

    return max(1, render_w), max(1, render_h), float(upscale)


class SDMLX_InpaintDetailer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mlx_model": ("MLX_MODEL",),
            "mlx_vae": ("MLX_VAE",),
            "positive": ("MLX_CONDITIONING",),
            "negative": ("MLX_CONDITIONING",),
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "crop": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            "guide_size": ("FLOAT", {"default": 512.0, "min": 64.0, "max": 4096.0, "step": 8.0}),
            "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "mask bbox", "label_off": "crop region"}),
            "max_size": ("FLOAT", {"default": 1024.0, "min": 64.0, "max": 4096.0, "step": 8.0}),
            "resize_method": (HIRES_RESIZE_METHODS, {"default": "lanczos"}),
            "soft_mask": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "soft_mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "crop_blend": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
            "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
            "steps": ("INT", {"default": 12, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 7.0}),
            "sampler_name": (SAMPLERS, {"default": "euler"}),
            "scheduler": (SCHEDULERS, {"default": "simple"}),
            "force_no_cfg": ("BOOLEAN", {"default": False}),
            "speed_patch": (speed_patch_options(), {"default": SPEED_PATCH_NONE}),
            "patch_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "spectrum_acceleration": (["off", "fast", "standard"], {"default": "off"}),
            "max_megapixels": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 16.0, "step": 0.5}),
            "preview": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "speed_patch_input": ("SDMLX_SPEED_PATCH",),
            "spectrum_acceleration_advanced": ("SDMLX_SPECTRUM_ACCELERATION",),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "detail"
    CATEGORY = "SDMLX/Inpaint"

    def detail(
        self,
        mlx_model,
        mlx_vae,
        positive,
        negative,
        image,
        mask,
        crop,
        guide_size,
        guide_size_for,
        max_size,
        resize_method,
        soft_mask,
        soft_mask_strength,
        crop_blend,
        denoise,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        force_no_cfg,
        speed_patch,
        patch_strength,
        spectrum_acceleration,
        max_megapixels,
        preview,
        speed_patch_input=None,
        spectrum_acceleration_advanced=None,
    ):
        global TIMING_LOGS_ENABLED
        TIMING_LOGS_ENABLED = SDMLX_VERBOSE_LOGS
        configure_mlx_memory_limits()
        start_time = time.perf_counter()
        fast_mode = True
        compute_dtype = "float16"
        effective_patch = speed_patch_override(speed_patch, speed_patch_input)

        image_t = image.detach().cpu().float() if hasattr(image, "detach") else torch.from_numpy(get_numpy_array(image).astype(np.float32))
        if image_t.ndim != 4 or image_t.shape[-1] != 3:
            raise ValueError(f"SDMLX Inpaint Detailer: IMAGE must be [B,H,W,3], got {tuple(image_t.shape)}.")
        height = int(image_t.shape[1])
        width = int(image_t.shape[2])
        mask_chw = mask_to_chw(mask, height, width)
        mask_peak_before_normalize = float(torch.max(mask_chw).item()) if mask_chw.numel() else 0.0
        mask_chw = normalize_nonempty_mask(mask_chw)
        mask_peak_after_normalize = float(torch.max(mask_chw).item()) if mask_chw.numel() else 0.0
        if torch.count_nonzero(mask_chw > 0.01).item() == 0:
            print("SDMLX: Inpaint Detailer skipped, mask is empty.")
            return (torch.clamp(image_t, 0.0, 1.0),)

        crop_aspect = 0.0
        mask_bbox = mask_bounds(mask_chw[0, 0])
        raw_crop_bbox = crop_bbox_by_factor(mask_bbox, width, height, crop, crop_aspect)
        x0, y0, x1, y1 = align_bbox_size_to_multiple(raw_crop_bbox, width, height, 8)
        crop_image = image_t[:, y0:y1, x0:x1, :]
        crop_mask = mask_chw[:, :, y0:y1, x0:x1]
        crop_h = int(crop_image.shape[1])
        crop_w = int(crop_image.shape[2])

        render_mask = crop_mask[:, 0, :, :].contiguous()
        render_mask = torch.clamp(render_mask, 0.0, 1.0).contiguous()

        render_w, render_h, effective_scale_factor = detailer_guide_render_size(
            crop_w,
            crop_h,
            mask_bbox,
            guide_size,
            bool(guide_size_for),
            max_size,
        )
        render_w = round_up_to_multiple(render_w, 8)
        render_h = round_up_to_multiple(render_h, 8)
        effective_scale_x = float(render_w) / max(float(crop_w), 1.0)
        effective_scale_y = float(render_h) / max(float(crop_h), 1.0)
        work_w = round_up(max(64, render_w), 64)
        work_h = round_up(max(64, render_h), 64)
        limit = int(max(1.0, float(max_megapixels)) * 1_000_000)
        work_pixels = work_w * work_h
        if work_pixels > limit:
            raise ValueError(
                "SDMLX Inpaint Detailer: work crop is larger than the configured limit "
                f"({work_w}x{work_h} = {work_pixels / 1_000_000:.1f}MP, limit {limit / 1_000_000:.1f}MP). "
                "Reduce crop/guide_size/max_size or intentionally increase max_megapixels."
            )

        resize_start = time.perf_counter()
        crop_render = resize_image_tensor(crop_image, render_w, render_h, resize_method)
        resize_elapsed = time.perf_counter() - resize_start

        mask_start = time.perf_counter()
        soft_mask_amount = int(soft_mask)
        soft_mask_crop = render_mask
        if soft_mask_amount > 0:
            soft_mask_crop = sdmlx_tensor_gaussian_blur_mask(
                soft_mask_crop[..., None],
                soft_mask_amount,
                10.0,
            )
            soft_mask_crop = torch.squeeze(soft_mask_crop, dim=-1)
        soft_mask_crop = torch.clamp(soft_mask_crop, 0.0, 1.0).contiguous()
        mask_elapsed = time.perf_counter() - mask_start

        pad_start = time.perf_counter()
        crop_up, _, render_decode_crop = pad_inpaint_pixels_and_mask(
            crop_render,
            target_width=work_w,
            target_height=work_h,
            pad_alignment=8,
        )
        pad_elapsed = time.perf_counter() - pad_start

        encode_start = time.perf_counter()
        initial_latents = encode_pixels_to_latents(mlx_vae, crop_up, "float32")
        noise_mask = prepare_padded_latent_mask_from_crop(
            soft_mask_crop,
            render_w,
            render_h,
            render_decode_crop,
            initial_latents.shape[1],
            initial_latents.shape[2],
        )
        encode_elapsed = time.perf_counter() - encode_start

        crop_info = {
            "bbox": (x0, y0, x1, y1),
            "raw_bbox": raw_crop_bbox,
            "mask_bbox": mask_bbox,
            "original_size": (width, height),
            "crop_size": (crop_w, crop_h),
            "target_size": (work_w, work_h),
            "render_size": (render_w, render_h),
            "preview_crop": render_decode_crop,
            "crop": float(crop),
            "crop_aspect": float(crop_aspect),
            "guide_size": float(guide_size),
            "guide_size_for": "mask_bbox" if guide_size_for else "crop_region",
            "max_size": float(max_size),
            "effective_scale_factor": effective_scale_factor,
            "effective_scale_xy": (effective_scale_x, effective_scale_y),
        }
        sdxl_time_ids = None
        time_id_mode = "default"

        spectrum_preset = None
        spectrum_label = "off"
        spectrum_reason = "disabled"
        spectrum_choice = str(spectrum_acceleration or "off")
        if isinstance(spectrum_acceleration, bool):
            spectrum_choice = "standard" if spectrum_acceleration else "off"
        use_spectrum = spectrum_choice != "off" or isinstance(spectrum_acceleration_advanced, dict)
        if use_spectrum:
            conditioning_controlnets = collect_conditioning_controlnets(mlx_model, positive, negative)
            if any(controlnets_active_at_percent(conditioning_controlnets, i / 20.0) for i in range(21)):
                spectrum_reason = "ControlNet active"
            else:
                from . import spectrum as spectrum_engine

                if isinstance(spectrum_acceleration_advanced, dict):
                    spectrum_preset, spectrum_label, spectrum_reason = spectrum_engine.resolve_spectrum_config(
                        spectrum_acceleration_advanced,
                        effective_patch,
                        steps,
                        sampler_name,
                    )
                else:
                    spectrum_preset, spectrum_label, spectrum_reason = spectrum_engine.resolve_spectrum_auto(
                        effective_patch,
                        steps,
                        sampler_name,
                        mode=spectrum_choice,
                    )

        sample_start = time.perf_counter()
        if spectrum_preset is None:
            if use_spectrum:
                print(f"SDMLX: Spectrum mode: off ({spectrum_reason}).")
            latents = sample_latents(
                mlx_model=mlx_model,
                positive=positive,
                negative=negative,
                width=work_w,
                height=work_h,
                seed=seed,
                steps=steps,
                cfg=cfg,
                scheduler=scheduler,
                sampler_name=sampler_name,
                force_no_cfg=force_no_cfg,
                compile_step=fast_mode,
                sync_each_step=False,
                debug_timing=False,
                preview=preview,
                quantize_unet=False,
                quant_bits=8,
                quant_group_size=64,
                profile_unet=False,
                fast_transformer=fast_mode,
                fast_ffn=fast_mode,
                fast_attention=fast_mode,
                compute_dtype=compute_dtype,
                speed_patch=effective_patch,
                speed_patch_strength=patch_strength,
                initial_latents=initial_latents,
                noise_mask=noise_mask,
                denoise=denoise,
                preview_mode="crop",
                preview_crop_info=crop_info,
                terminal_progress=True,
                sdxl_time_ids=sdxl_time_ids,
                differential_mask=soft_mask_amount > 0,
                differential_mask_strength=soft_mask_strength,
            )
        else:
            from . import spectrum as spectrum_engine

            if isinstance(spectrum_acceleration_advanced, dict):
                print("SDMLX: Spectrum mode: advanced.")
            else:
                terminal_label = spectrum_engine.terminal_profile_label(spectrum_label, spectrum_choice)
                print(f"SDMLX: Spectrum mode: {terminal_label}.")
            latents = spectrum_engine.sample_latents_spectrum(
                mlx_model,
                positive,
                negative,
                work_w,
                work_h,
                seed,
                steps,
                float(cfg),
                scheduler,
                sampler_name,
                force_no_cfg,
                preview=preview,
                compute_dtype=compute_dtype,
                speed_patch=effective_patch,
                speed_patch_strength=patch_strength,
                initial_latents=initial_latents,
                noise_mask=noise_mask,
                denoise=denoise,
                sdxl_time_ids=sdxl_time_ids,
                differential_mask=soft_mask_amount > 0,
                differential_mask_strength=soft_mask_strength,
                preview_mode="crop",
                preview_crop_info=crop_info,
                spectrum_verbose=False,
                **spectrum_preset,
            )
        sample_elapsed = time.perf_counter() - sample_start

        decode_start = time.perf_counter()
        if should_use_tiled_decode("auto", work_w, work_h):
            rendered_crop = decode_latents_tiled(
                mlx_vae,
                latents,
                compile_vae=True,
                vae_dtype="float32",
                tile_size=1024,
                overlap=128,
            )
            decode_mode = "tiled"
        else:
            rendered_crop = decode_latents(mlx_vae, latents, compile_vae=True, vae_dtype="float32", debug_timing=False)
            decode_mode = "full"
        decode_elapsed = time.perf_counter() - decode_start

        composite_start = time.perf_counter()
        render_left, render_top, render_width, render_height = [int(value) for value in render_decode_crop]
        rendered_crop = rendered_crop[
            :,
            render_top:render_top + render_height,
            render_left:render_left + render_width,
            :,
        ].contiguous()
        rendered_crop = resize_image_tensor(rendered_crop, crop_w, crop_h, resize_method)
        composite_mask = torch.ones(
            (rendered_crop.shape[0], crop_h, crop_w),
            dtype=rendered_crop.dtype,
            device=rendered_crop.device,
        )
        crop_blend = int(crop_blend)
        if crop_blend > 0:
            border = min(crop_blend, max(0, crop_h // 2), max(0, crop_w // 2))
            if border > 0:
                composite_mask[:, :border, :] = 0.0
                composite_mask[:, crop_h - border:, :] = 0.0
                composite_mask[:, :, :border] = 0.0
                composite_mask[:, :, crop_w - border:] = 0.0
            composite_mask = sdmlx_tensor_gaussian_blur_mask(composite_mask[..., None], crop_blend, 10.0)
            composite_mask = torch.squeeze(composite_mask, dim=-1)
        composite_mask = torch.clamp(composite_mask, 0.0, 1.0).unsqueeze(-1).contiguous()

        output = image_t.clone()
        target = output[:, y0:y1, x0:x1, :]
        if rendered_crop.shape[0] != target.shape[0]:
            rendered_crop = rendered_crop[:1].repeat(target.shape[0], 1, 1, 1)
            composite_mask = composite_mask[:1].repeat(target.shape[0], 1, 1, 1)
        output[:, y0:y1, x0:x1, :] = rendered_crop * composite_mask + target * (1.0 - composite_mask)
        composite_elapsed = time.perf_counter() - composite_start

        total_elapsed = time.perf_counter() - start_time
        sampler_mask_mode = "hard_mask"
        if soft_mask_amount > 0:
            sampler_mask_mode = "soft_mask_dynamic"
        differential_mode = soft_mask_amount > 0
        mask_margin = (
            int(mask_bbox[0] - x0),
            int(mask_bbox[1] - y0),
            int(x1 - mask_bbox[2]),
            int(y1 - mask_bbox[3]),
        )
        pad_left, pad_top, _, _ = [int(value) for value in render_decode_crop]
        pad_ltrb = (
            pad_left,
            pad_top,
            int(work_w - render_w - pad_left),
            int(work_h - render_h - pad_top),
        )
        print(
            "SDMLX: Inpaint Detailer finished "
            f"in {total_elapsed:.2f}s "
            f"(render={render_w}x{render_h}, sample={sample_elapsed:.2f}s, "
            f"decode={decode_elapsed:.2f}s/{decode_mode})."
        )
        log_timing(
            "SDMLX: Inpaint Detailer Details "
            f"(bbox={(x0, y0, x1, y1)}, raw_bbox={raw_crop_bbox}, mask_bbox={mask_bbox}, "
            f"mask_margin_ltrb={mask_margin}, "
            f"crop={float(crop):g}, aspect={float(crop_aspect):g}, "
            f"guide={float(guide_size):g}/{('mask_bbox' if guide_size_for else 'crop_region')}, "
            f"max={float(max_size):g}, upscale={effective_scale_factor:.3g}, "
            f"scale_xy=({effective_scale_x:.3g},{effective_scale_y:.3g}), "
            f"render={render_w}x{render_h}, work={work_w}x{work_h}, "
            f"pad_ltrb={pad_ltrb}, "
            f"mask_peak={mask_peak_before_normalize:.3g}->{mask_peak_after_normalize:.3g}, "
            f"soft_mask={soft_mask_amount}, "
            f"sampler_mask={sampler_mask_mode}, differential_mask={differential_mode}, "
            f"soft_mask_strength={float(soft_mask_strength):g}, "
            f"composite_mask=full_crop_plus_crop_blend, crop_blend={crop_blend}, "
            f"preview={bool(preview)}, time_ids={time_id_mode}, "
            f"mask={mask_elapsed:.2f}s, resize={resize_elapsed:.2f}s, pad={pad_elapsed:.2f}s, encode={encode_elapsed:.2f}s, "
            f"sample={sample_elapsed:.2f}s, decode={decode_elapsed:.2f}s/{decode_mode}, "
            f"composite={composite_elapsed:.2f}s, total={total_elapsed:.2f}s)."
        )
        return (torch.clamp(output, 0.0, 1.0),)


class SDMLX_HiresFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mlx_model": ("MLX_MODEL",),
            "mlx_vae": ("MLX_VAE",),
            "positive": ("MLX_CONDITIONING",),
            "negative": ("MLX_CONDITIONING",),
            "image": ("IMAGE",),
            "scale_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.05}),
            "custom_size": ("BOOLEAN", {"default": False}),
            "custom_width": ("INT", {"default": 1536, "min": 64, "max": 4096, "step": 64}),
            "custom_height": ("INT", {"default": 1536, "min": 64, "max": 4096, "step": 64}),
            "resize_method": (HIRES_RESIZE_METHODS, {"default": "lanczos"}),
            "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 7.0}),
            "sampler_name": (SAMPLERS, {"default": "euler"}),
            "scheduler": (SCHEDULERS, {"default": "simple"}),
            "force_no_cfg": ("BOOLEAN", {"default": False}),
            "speed_patch": (speed_patch_options(), {"default": SPEED_PATCH_NONE}),
            "patch_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "spectrum_acceleration": (["off", "fast", "standard"], {"default": "fast"}),
            "decode_mode": (VAE_DECODE_MODES, {"default": "auto"}),
            "decode_tile_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
            "decode_overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 64}),
            "preview": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "speed_patch_input": ("SDMLX_SPEED_PATCH",),
            "spectrum_acceleration_advanced": ("SDMLX_SPECTRUM_ACCELERATION",),
        }}

    RETURN_TYPES = ("IMAGE", "MLX_LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "upscale"
    CATEGORY = "SDMLX/Upscale"

    def upscale(
        self,
        mlx_model,
        mlx_vae,
        positive,
        negative,
        image,
        scale_factor,
        custom_size,
        custom_width,
        custom_height,
        resize_method,
        denoise,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        force_no_cfg,
        speed_patch,
        patch_strength,
        spectrum_acceleration,
        decode_mode,
        decode_tile_size,
        decode_overlap,
        preview,
        speed_patch_input=None,
        spectrum_acceleration_advanced=None,
    ):
        global TIMING_LOGS_ENABLED
        TIMING_LOGS_ENABLED = SDMLX_VERBOSE_LOGS
        configure_mlx_memory_limits()
        start_time = time.perf_counter()
        fast_mode = True
        compute_dtype = "float16"
        effective_patch = speed_patch_override(speed_patch, speed_patch_input)

        target_width, target_height = resolve_hires_target_size_from_factor(
            image,
            scale_factor,
            custom_width,
            custom_height,
            multiple_of=64,
            custom_size=custom_size,
        )
        validate_hires_target_size(target_width, target_height)
        source_height = int(image.shape[1])
        source_width = int(image.shape[2])
        print(
            "SDMLX: Hires Fix started "
            f"({source_width}x{source_height} -> {target_width}x{target_height})."
        )

        resize_start = time.perf_counter()
        upscaled_image = resize_image_tensor(image, target_width, target_height, resize_method)
        resize_elapsed = time.perf_counter() - resize_start

        encode_start = time.perf_counter()
        initial_latents = encode_pixels_to_latents(mlx_vae, upscaled_image, "float32")
        encode_elapsed = time.perf_counter() - encode_start

        spectrum_preset = None
        spectrum_label = "off"
        spectrum_reason = "disabled"
        spectrum_choice = str(spectrum_acceleration or "off")
        if isinstance(spectrum_acceleration, bool):
            spectrum_choice = "standard" if spectrum_acceleration else "off"
        use_spectrum = spectrum_choice != "off" or isinstance(spectrum_acceleration_advanced, dict)
        if use_spectrum:
            conditioning_controlnets = collect_conditioning_controlnets(mlx_model, positive, negative)
            if any(controlnets_active_at_percent(conditioning_controlnets, i / 20.0) for i in range(21)):
                spectrum_reason = "ControlNet active"
            else:
                from . import spectrum as spectrum_engine

                if isinstance(spectrum_acceleration_advanced, dict):
                    spectrum_preset, spectrum_label, spectrum_reason = spectrum_engine.resolve_spectrum_config(
                        spectrum_acceleration_advanced,
                        effective_patch,
                        steps,
                        sampler_name,
                    )
                else:
                    spectrum_preset, spectrum_label, spectrum_reason = spectrum_engine.resolve_spectrum_auto(
                        effective_patch,
                        steps,
                        sampler_name,
                        mode=spectrum_choice,
                    )

        sample_start = time.perf_counter()
        if spectrum_preset is None:
            if use_spectrum:
                print(f"SDMLX: Spectrum mode: off ({spectrum_reason}).")
            latents = sample_latents(
                mlx_model=mlx_model,
                positive=positive,
                negative=negative,
                width=target_width,
                height=target_height,
                seed=seed,
                steps=steps,
                cfg=cfg,
                scheduler=scheduler,
                sampler_name=sampler_name,
                force_no_cfg=force_no_cfg,
                compile_step=fast_mode,
                sync_each_step=False,
                debug_timing=False,
                preview=preview,
                quantize_unet=False,
                quant_bits=8,
                quant_group_size=64,
                profile_unet=False,
                fast_transformer=fast_mode,
                fast_ffn=fast_mode,
                fast_attention=fast_mode,
                compute_dtype=compute_dtype,
                speed_patch=effective_patch,
                speed_patch_strength=patch_strength,
                initial_latents=initial_latents,
                noise_mask=None,
                denoise=denoise,
                preview_mode="crop",
                preview_crop_info=None,
                terminal_progress=True,
                terminal_progress_interval=2,
            )
        else:
            from . import spectrum as spectrum_engine

            if isinstance(spectrum_acceleration_advanced, dict):
                print("SDMLX: Spectrum mode: advanced.")
            else:
                terminal_label = spectrum_engine.terminal_profile_label(spectrum_label, spectrum_choice)
                print(f"SDMLX: Spectrum mode: {terminal_label}.")
            latents = spectrum_engine.sample_latents_spectrum(
                mlx_model,
                positive,
                negative,
                target_width,
                target_height,
                seed,
                steps,
                float(cfg),
                scheduler,
                sampler_name,
                force_no_cfg,
                preview=preview,
                compute_dtype=compute_dtype,
                speed_patch=effective_patch,
                speed_patch_strength=patch_strength,
                initial_latents=initial_latents,
                noise_mask=None,
                denoise=denoise,
                spectrum_verbose=False,
                **spectrum_preset,
            )
        sample_elapsed = time.perf_counter() - sample_start

        decode_start = time.perf_counter()
        if should_use_tiled_decode(decode_mode, target_width, target_height):
            out_image = decode_latents_tiled(
                mlx_vae,
                latents,
                compile_vae=True,
                vae_dtype="float32",
                tile_size=decode_tile_size,
                overlap=decode_overlap,
            )
            effective_decode_mode = "tiled"
        else:
            out_image = decode_latents(mlx_vae, latents, compile_vae=True, vae_dtype="float32", debug_timing=False)
            effective_decode_mode = "full"
        decode_elapsed = time.perf_counter() - decode_start
        total_elapsed = time.perf_counter() - start_time
        print(
            "SDMLX: Hires Fix finished "
            f"(sample={sample_elapsed:.2f}s, decode={decode_elapsed:.2f}s/{effective_decode_mode}, "
            f"total={total_elapsed:.2f}s)."
        )
        return (out_image, {"samples": latents})


class SDMLX_TiledUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "mlx_vae": ("MLX_VAE",),
                "positive": ("MLX_CONDITIONING",),
                "negative": ("MLX_CONDITIONING",),
                "image": ("IMAGE",),
                "scale": (TILED_UPSCALE_SCALE_OPTIONS, {"default": "2x"}),
                "custom_width": ("INT", {"default": 3072, "min": 64, "max": 8192, "step": 64}),
                "custom_height": ("INT", {"default": 3072, "min": 64, "max": 8192, "step": 64}),
                "resize_method": (HIRES_RESIZE_METHODS, {"default": "lanczos"}),
                "tile_size": ("INT", {"default": 1024, "min": 512, "max": 1536, "step": 64}),
                "tile_overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 64}),
                "tile_control_strength": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 2.0, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0}),
                "sampler_name": (SAMPLERS, {"default": "euler"}),
                "scheduler": (SCHEDULERS, {"default": "simple"}),
                "force_no_cfg": ("BOOLEAN", {"default": False}),
                "speed_patch": (speed_patch_options(), {"default": SPEED_PATCH_NONE}),
                "patch_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "max_megapixels": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 32.0, "step": 0.5}),
                "preview": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mlx_controlnet": ("MLX_CONTROLNET",),
                "speed_patch_input": ("SDMLX_SPEED_PATCH",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "SDMLX/Upscale"

    def upscale(
        self,
        mlx_model,
        mlx_vae,
        positive,
        negative,
        image,
        scale,
        custom_width,
        custom_height,
        resize_method,
        tile_size,
        tile_overlap,
        tile_control_strength,
        denoise,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        force_no_cfg,
        speed_patch,
        patch_strength,
        max_megapixels,
        preview,
        mlx_controlnet=None,
        speed_patch_input=None,
    ):
        global TIMING_LOGS_ENABLED
        TIMING_LOGS_ENABLED = SDMLX_VERBOSE_LOGS
        configure_mlx_memory_limits()
        start_time = time.perf_counter()
        fast_mode = True
        compute_dtype = "float16"

        target_width, target_height = resolve_hires_target_size(
            image,
            scale,
            custom_width,
            custom_height,
            multiple_of=64,
        )
        validate_tiled_upscale_target_size(target_width, target_height, max_megapixels)
        tile_size = max(64, int(tile_size) // 64 * 64)
        tile_overlap = max(0, int(tile_overlap) // 64 * 64)
        tile_overlap = min(tile_overlap, max(0, tile_size - 64))
        tile_control_strength = max(0.0, float(tile_control_strength))
        use_tile_control = mlx_controlnet is not None and tile_control_strength > CONTROL_STRENGTH_EPSILON
        tile_w = min(tile_size, target_width)
        tile_h = min(tile_size, target_height)
        x_starts = tiled_starts(target_width, tile_w, tile_overlap)
        y_starts = tiled_starts(target_height, tile_h, tile_overlap)
        total_tiles = len(x_starts) * len(y_starts)
        source_height = int(image.shape[1])
        source_width = int(image.shape[2])
        print(
            "SDMLX: Tiled Upscale started "
            f"({source_width}x{source_height} -> {target_width}x{target_height}, {total_tiles} tiles)."
        )

        resize_start = time.perf_counter()
        upscaled_image = resize_image_tensor(image, target_width, target_height, resize_method)
        resize_elapsed = time.perf_counter() - resize_start
        if float(denoise) == 0.0:
            print(
                "SDMLX: Tiled Upscale denoise=0, resize only "
                f"(resize={resize_elapsed:.2f}s, total={time.perf_counter() - start_time:.2f}s)."
            )
            return (upscaled_image,)

        output = torch.zeros_like(upscaled_image)
        weight = torch.zeros((1, target_height, target_width, 1), dtype=torch.float32)
        pbar = make_comfy_progress_bar(total_tiles)
        terminal_pbar = make_terminal_progress_bar(total_tiles, "SDMLX Tiled Upscale", unit="tile")
        encode_elapsed = 0.0
        sample_elapsed = 0.0
        decode_elapsed = 0.0
        tile_index = 0

        try:
            for y0 in y_starts:
                y1 = min(target_height, y0 + tile_h)
                for x0 in x_starts:
                    x1 = min(target_width, x0 + tile_w)
                    tile_index += 1
                    tile_image = upscaled_image[:, y0:y1, x0:x1, :].contiguous()
                    tile_positive = positive
                    tile_negative = negative
                    if use_tile_control:
                        tile_control = {
                            "controlnet": mlx_controlnet,
                            "image": tile_image,
                            "control_type": UNION_CONTROL_TYPES["tile"],
                            "control_type_name": "tile",
                            "strength": tile_control_strength,
                            "start_percent": 0.0,
                            "end_percent": 1.0,
                            "curve": "constant",
                            "resize_mode": "stretch",
                        }
                        tile_positive = add_controlnet_to_conditioning(positive, tile_control)
                        tile_negative = add_controlnet_to_conditioning(negative, tile_control)

                    encode_start = time.perf_counter()
                    initial_latents = encode_pixels_to_latents(mlx_vae, tile_image, "float32")
                    encode_elapsed += time.perf_counter() - encode_start

                    sample_start = time.perf_counter()
                    latents = sample_latents(
                        mlx_model=mlx_model,
                        positive=tile_positive,
                        negative=tile_negative,
                        width=x1 - x0,
                        height=y1 - y0,
                        seed=int(seed) + tile_index - 1,
                        steps=steps,
                        cfg=cfg,
                        scheduler=scheduler,
                        sampler_name=sampler_name,
                        force_no_cfg=force_no_cfg,
                        compile_step=fast_mode,
                        sync_each_step=False,
                        debug_timing=False,
                        preview=preview,
                        quantize_unet=False,
                        quant_bits=8,
                        quant_group_size=64,
                        profile_unet=False,
                        fast_transformer=fast_mode,
                        fast_ffn=fast_mode,
                        fast_attention=fast_mode,
                        compute_dtype=compute_dtype,
                        speed_patch=speed_patch_override(speed_patch, speed_patch_input),
                        speed_patch_strength=patch_strength,
                        initial_latents=initial_latents,
                        noise_mask=None,
                        denoise=denoise,
                        preview_mode="crop",
                        preview_crop_info=None,
                        terminal_progress=False,
                    )
                    sample_elapsed += time.perf_counter() - sample_start

                    decode_start = time.perf_counter()
                    tile_out = decode_latents_quiet(mlx_vae, latents, compile_vae=True, vae_dtype="float32")
                    decode_elapsed += time.perf_counter() - decode_start

                    mask = tile_blend_mask(
                        y1 - y0,
                        x1 - x0,
                        y0 == 0,
                        y1 == target_height,
                        x0 == 0,
                        x1 == target_width,
                        tile_overlap,
                    )
                    output[:, y0:y1, x0:x1, :] += tile_out * mask
                    weight[:, y0:y1, x0:x1, :] += mask
                    if pbar is not None:
                        pbar.update_absolute(tile_index, total_tiles)
                    if terminal_pbar is not None:
                        terminal_pbar.update(1)
                    else:
                        print(f"SDMLX: Tiled Upscale {tile_index}/{total_tiles}")
                    del tile_image, initial_latents, latents, tile_out, mask
        finally:
            if terminal_pbar is not None:
                terminal_pbar.close()

        output = torch.clamp(output / torch.clamp(weight, min=1e-6), 0.0, 1.0)
        total_elapsed = time.perf_counter() - start_time
        print(
            "SDMLX: Tiled Upscale finished "
            f"({total_tiles} tiles, sample={sample_elapsed:.2f}s, decode={decode_elapsed:.2f}s, "
            f"total={total_elapsed:.2f}s)."
        )
        return (output,)


class SDMLX_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mlx_model": ("MLX_MODEL",),
            "mlx_vae": ("MLX_VAE",),
            "positive": ("MLX_CONDITIONING",),
            "negative": ("MLX_CONDITIONING",),
            "latent_image": ("LATENT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 7.0}),
            "sampler_name": (SAMPLERS, {"default": "euler"}),
            "scheduler": (SCHEDULERS, {"default": "simple"}),
            "force_no_cfg": ("BOOLEAN", {"default": False}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "speed_patch": (speed_patch_options(), {"default": SPEED_PATCH_NONE}),
            "patch_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "spectrum_acceleration": (["off", "fast", "standard"], {"default": "off"}),
            "preview": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "speed_patch_input": ("SDMLX_SPEED_PATCH",),
            "spectrum_acceleration_advanced": ("SDMLX_SPECTRUM_ACCELERATION",),
        }, "hidden": {
            "unique_id": "UNIQUE_ID",
            "prompt": "PROMPT",
        }}

    RETURN_TYPES = ("IMAGE", "MLX_LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "sample"
    CATEGORY = "SDMLX"

    def sample(
        self,
        mlx_model,
        mlx_vae,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        force_no_cfg,
        denoise,
        speed_patch,
        patch_strength,
        spectrum_acceleration,
        preview,
        speed_patch_input=None,
        spectrum_acceleration_advanced=None,
        unique_id=None,
        prompt=None,
    ):
        global TIMING_LOGS_ENABLED
        TIMING_LOGS_ENABLED = SDMLX_VERBOSE_LOGS
        fast_mode = True
        compute_dtype = "float16"
        latents = get_mlx_array(latent_image["samples"])
        if latents.shape[1] == 4:
            latents = latents.transpose(0, 2, 3, 1)
        height = int(latents.shape[1] * 8)
        width = int(latents.shape[2] * 8)
        noise_mask = get_mlx_array(latent_image["noise_mask"]) if "noise_mask" in latent_image else None
        initial_latents = latents if noise_mask is not None or float(denoise) < 1.0 else None
        differential_mask = bool(mlx_model.get("differential_mask", False)) and noise_mask is not None
        differential_mask_strength = float(mlx_model.get("differential_mask_strength", 1.0))
        sdxl_time_ids = latent_image.get("sdxl_time_ids")
        effective_patch = speed_patch_override(speed_patch, speed_patch_input)
        spectrum_preset = None
        spectrum_label = "off"
        spectrum_reason = "disabled"
        spectrum_choice = str(spectrum_acceleration or "off")
        if isinstance(spectrum_acceleration, bool):
            spectrum_choice = "standard" if spectrum_acceleration else "off"
        use_spectrum = spectrum_choice != "off" or isinstance(spectrum_acceleration_advanced, dict)
        if use_spectrum:
            conditioning_controlnets = collect_conditioning_controlnets(mlx_model, positive, negative)
            if any(controlnets_active_at_percent(conditioning_controlnets, i / 20.0) for i in range(21)):
                spectrum_reason = "ControlNet active"
            else:
                from . import spectrum as spectrum_engine

                if isinstance(spectrum_acceleration_advanced, dict):
                    spectrum_preset, spectrum_label, spectrum_reason = spectrum_engine.resolve_spectrum_config(
                        spectrum_acceleration_advanced,
                        effective_patch,
                        steps,
                        sampler_name,
                    )
                else:
                    spectrum_preset, spectrum_label, spectrum_reason = spectrum_engine.resolve_spectrum_auto(
                        effective_patch,
                        steps,
                        sampler_name,
                        mode=spectrum_choice,
                    )
        if spectrum_preset is None:
            if use_spectrum:
                print(f"SDMLX: Spectrum mode: off ({spectrum_reason}).")
            samples = sample_latents(
                mlx_model,
                positive,
                negative,
                width,
                height,
                seed,
                steps,
                cfg,
                scheduler,
                sampler_name,
                force_no_cfg,
                fast_mode,
                False,
                False,
                preview,
                False,
                8,
                64,
                False,
                fast_mode,
                fast_mode,
                fast_mode,
                compute_dtype,
                effective_patch,
                patch_strength,
                initial_latents,
                noise_mask,
                denoise,
                "crop",
                None,
                sdxl_time_ids=sdxl_time_ids,
                differential_mask=differential_mask,
                differential_mask_strength=differential_mask_strength,
            )
        else:
            from . import spectrum as spectrum_engine

            if isinstance(spectrum_acceleration_advanced, dict):
                print("SDMLX: Spectrum mode: advanced.")
            else:
                terminal_label = spectrum_engine.terminal_profile_label(spectrum_label, spectrum_choice)
                print(f"SDMLX: Spectrum mode: {terminal_label}.")
            effective_cfg = float(cfg)
            samples = spectrum_engine.sample_latents_spectrum(
                mlx_model,
                positive,
                negative,
                width,
                height,
                seed,
                steps,
                effective_cfg,
                scheduler,
                sampler_name,
                force_no_cfg,
                preview=preview,
                compute_dtype=compute_dtype,
                speed_patch=effective_patch,
                speed_patch_strength=patch_strength,
                initial_latents=initial_latents,
                noise_mask=noise_mask,
                denoise=denoise,
                sdxl_time_ids=sdxl_time_ids,
                differential_mask=differential_mask,
                differential_mask_strength=differential_mask_strength,
                spectrum_verbose=False,
                **spectrum_preset,
            )
        out = {"samples": samples}
        for key in ("sdmlx_decode_crop", "sdmlx_original_size", "sdmlx_padded_size"):
            if key in latent_image:
                out[key] = latent_image[key]
        image = decode_mlx_latent_to_image(out, mlx_vae) if output_is_connected(prompt, unique_id, 0) else None
        return (image, out)


def output_is_connected(prompt, unique_id, output_index):
    if prompt is None or unique_id is None:
        return True
    source_id = str(unique_id)
    try:
        for node in prompt.values():
            inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
            for value in inputs.values():
                if (
                    isinstance(value, (list, tuple))
                    and len(value) >= 2
                    and str(value[0]) == source_id
                    and int(value[1]) == int(output_index)
                ):
                    return True
    except Exception:
        return True
    return False


def decode_mlx_latent_to_image(mlx_latent, mlx_vae):
    latents = mlx_latent["samples"]
    if latents.shape[-1] != 4:
        latents = latents.transpose(0, 2, 3, 1)
    image = decode_latents(mlx_vae, latents, compile_vae=True, vae_dtype="float32", debug_timing=False)
    decode_crop = mlx_latent.get("sdmlx_decode_crop")
    if decode_crop is not None:
        left, top, width, height = [int(value) for value in decode_crop]
        image = image[:, top:top + height, left:left + width, :].contiguous()
        padded_size = mlx_latent.get("sdmlx_padded_size", (int(latents.shape[2] * 8), int(latents.shape[1] * 8)))
        print(
            "SDMLX: Decode crop to original size "
            f"({padded_size[0]}x{padded_size[1]} -> {width}x{height}, offset={left},{top})."
        )
    return image


class SDMLX_VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mlx_latent": ("MLX_LATENT",), "mlx_vae": ("MLX_VAE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "SDMLX"

    def decode(self, mlx_latent, mlx_vae):
        return (decode_mlx_latent_to_image(mlx_latent, mlx_vae),)


class SDMLX_NumberPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "selected": (["20", "25", "30"], {"default": "20"}),
            "edit_values": ("BOOLEAN", {"default": False}),
            "values": ("STRING", {"default": "20, 25, 30", "multiline": False}),
        }}

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int", "float")
    FUNCTION = "select"
    CATEGORY = "SDMLX/Utilities"

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    @staticmethod
    def _parse_values(values):
        parsed = []
        seen = set()
        for token in re.split(r"[\s,;]+", str(values or "")):
            label = token.strip()
            if not label:
                continue
            try:
                value = float(label)
            except ValueError:
                continue
            if not math.isfinite(value) or label in seen:
                continue
            parsed.append((label, value))
            seen.add(label)
        return parsed or [("0", 0.0)]

    def select(self, selected, edit_values=False, values="20, 25, 30"):
        parsed = self._parse_values(values)
        selected_label = str(selected)
        value = None
        for label, candidate in parsed:
            if label == selected_label:
                value = candidate
                break
        if value is None:
            try:
                selected_value = float(selected_label)
            except ValueError:
                selected_value = None
            value = selected_value if selected_value is not None and math.isfinite(selected_value) else parsed[0][1]
        return (int(round(value)), value)


NODE_CLASS_MAPPINGS = {
    "SDMLX_GaussianBlurMask": SDMLX_GaussianBlurMask,
    "SDMLX_CheckpointLoader": SDMLX_LoaderUniversal,
    "SDMLX_Loader": SDMLX_Loader,
    "SDMLX_CLIPTextEncode": SDMLX_CLIPTextEncode,
    "SDMLX_LoraLoader": SDMLX_LoraLoader,
    "SDMLX_MultiLoraLoader": SDMLX_MultiLoraLoader,
    "SDMLX_SpeedPatchConverter": SDMLX_SpeedPatchConverter,
    "SDMLX_SpectrumBoost": SDMLX_SpectrumBoost,
    "SDMLX_LoraSchedule": SDMLX_LoraSchedule,
    "SDMLX_IPAdapterLoader": SDMLX_IPAdapterLoader,
    "SDMLX_CLIPVisionLoader": SDMLX_CLIPVisionLoader,
    "SDMLX_IPAdapterMLXCLIPVisionEncode": SDMLX_IPAdapterMLXCLIPVisionEncode,
    "SDMLX_InsightFaceLoader": SDMLX_InsightFaceLoader,
    "SDMLX_InsightFaceAlignCrop": SDMLX_InsightFaceAlignCrop,
    "SDMLX_ApplyIPAdapter": SDMLX_ApplyIPAdapter,
    "SDMLX_ApplyIPAdapterFaceID": SDMLX_ApplyIPAdapterFaceID,
    "SDMLX_ApplyIPAdapterFaceIDAIO": SDMLX_ApplyIPAdapterFaceIDAIO,
    "SDMLX_DifferentialDiffusion": SDMLX_DifferentialDiffusion,
    "SDMLX_KSampler": SDMLX_KSampler,
    "SDMLX_VAEDecode": SDMLX_VAEDecode,
    "SDMLX_ControlNetUnionLoader": SDMLX_ControlNetUnionLoader,
    "SDMLX_ApplyControlNet": SDMLX_ApplyControlNet,
    "SDMLX_InpaintConditioning": SDMLX_InpaintConditioning,
    "SDMLX_InpaintDetailer": SDMLX_InpaintDetailer,
    "SDMLX_HiresFix": SDMLX_HiresFix,
    "SDMLX_TiledUpscale": SDMLX_TiledUpscale,
    "SDMLX_NumberPicker": SDMLX_NumberPicker,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **{k: "🍏 " + k.replace("_", " ") for k in NODE_CLASS_MAPPINGS.keys()},
    "SDMLX_GaussianBlurMask": "🍏 SDMLX Gaussian Blur Mask",
    "SDMLX_CheckpointLoader": "🍏 SDMLX Loader Universal",
    "SDMLX_Loader": "🍏 SDMLX Loader",
    "SDMLX_CLIPTextEncode": "🍏 SDMLX CLIP Text Encode",
    "SDMLX_LoraLoader": "🍏 SDMLX LoRA Loader",
    "SDMLX_MultiLoraLoader": "🍏 SDMLX Multi LoRA Loader",
    "SDMLX_SpeedPatchConverter": "🍏 SDMLX Speed Patch Converter",
    "SDMLX_SpectrumBoost": "🍏 SDMLX Spectrum Advanced",
    "SDMLX_LoraSchedule": "🍏 SDMLX Scheduler",
    "SDMLX_IPAdapterLoader": "🍏 SDMLX IP-Adapter Loader",
    "SDMLX_CLIPVisionLoader": "🍏 SDMLX CLIP Vision Loader",
    "SDMLX_IPAdapterMLXCLIPVisionEncode": "🍏 SDMLX CLIP Vision Encode",
    "SDMLX_InsightFaceLoader": "🍏 SDMLX InsightFace Loader",
    "SDMLX_InsightFaceAlignCrop": "🍏 SDMLX InsightFace Align Crop",
    "SDMLX_ApplyIPAdapter": "🍏 SDMLX Apply IP-Adapter",
    "SDMLX_ApplyIPAdapterFaceID": "🍏 SDMLX Apply IP-Adapter FaceID",
    "SDMLX_ApplyIPAdapterFaceIDAIO": "🍏 SDMLX FaceID AIO",
    "SDMLX_DifferentialDiffusion": "🍏 SDMLX Differential Diffusion",
    "SDMLX_KSampler": "🍏 SDMLX KSampler",
    "SDMLX_VAEDecode": "🍏 SDMLX VAE Decode",
    "SDMLX_ControlNetUnionLoader": "🍏 SDMLX ControlNet Union ProMax Loader",
    "SDMLX_ApplyControlNet": "🍏 SDMLX Apply ControlNet",
    "SDMLX_InpaintConditioning": "🍏 SDMLX Inpaint Conditioning",
    "SDMLX_InpaintDetailer": "🍏 SDMLX Inpaint Detailer",
    "SDMLX_HiresFix": "🍏 SDMLX Hires Fix",
    "SDMLX_TiledUpscale": "🍏 SDMLX Tiled Upscale",
    "SDMLX_NumberPicker": "🍏 SDMLX Number Picker",
}
