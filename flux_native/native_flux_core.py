from __future__ import annotations

import math
import os
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open


COMFY_ROOT = Path(os.environ.get("SDMLX_COMFY_ROOT", Path(__file__).resolve().parents[2] / "ComfyUI")).expanduser()
FLUX_PATH = COMFY_ROOT / "models/diffusion_models/flux1-schnell.safetensors"

HIDDEN_DIM = 3072
HEADS = 24
HEAD_DIM = 128
MLP_DIM = 12288
DOUBLE_BLOCKS = 19
SINGLE_BLOCKS = 38
USE_FAST_SDPA = False
SDPA_MEMORY_EFFICIENT_THRESHOLD: int | None = None
FUSED_ADALN = False
CAST_MODULATED_NORM = False
CAST_ATTENTION_OUT = False
CAST_BLOCK_OUTPUT = False
NAN_TO_NUM_FP16 = False
ROPE_FLOAT32 = False
QK_NORM_MODE = "fast"
SINGLE_LINEAR1_FLAT = False
SINGLE_LINEAR2_FLAT = False
SINGLE_LINEAR2_CONTIG = False
SINGLE_LINEAR2_CAST = False
FP8_FUSED_LINEAR_NAME = "off"
FP8_FUSED_LINEAR_FN = None
FP8_FUSED_LINEAR_LUT = None


def linear(x: mx.array, w: mx.array, b: mx.array | None = None) -> mx.array:
    if b is None:
        return x @ w.T
    return mx.addmm(b, x, w.T)


def layer_norm(x: mx.array) -> mx.array:
    return mx.fast.layer_norm(x, None, None, 1e-6)


ModulationDims = tuple[tuple[int, int, int], ...]


def apply_token_modulation(
    x: mx.array,
    mult: mx.array,
    add: mx.array | None = None,
    modulation_dims: ModulationDims | None = None,
) -> mx.array:
    if modulation_dims is None:
        out = x * mult[:, None]
        if add is not None:
            out = out + add[:, None]
        return out

    parts: list[mx.array] = []
    cursor = 0
    seq_len = int(x.shape[1])
    for start, end, index in modulation_dims:
        start = max(0, min(seq_len, int(start)))
        end = max(start, min(seq_len, int(end)))
        index = int(index)
        if start > cursor:
            part = x[:, cursor:start] * mult[0:1, None]
            if add is not None:
                part = part + add[0:1, None]
            parts.append(part)
        if end > start:
            part = x[:, start:end] * mult[index : index + 1, None]
            if add is not None:
                part = part + add[index : index + 1, None]
            parts.append(part)
        cursor = end
    if cursor < seq_len:
        part = x[:, cursor:] * mult[0:1, None]
        if add is not None:
            part = part + add[0:1, None]
        parts.append(part)
    if not parts:
        return x
    if len(parts) == 1:
        return parts[0]
    return mx.concatenate(parts, axis=1)


def modulated_layer_norm(
    x: mx.array,
    scale: mx.array,
    shift: mx.array,
    modulation_dims: ModulationDims | None = None,
) -> mx.array:
    if modulation_dims is None:
        if FUSED_ADALN and x.shape[0] == 1:
            out = mx.fast.layer_norm(x, 1 + scale.squeeze(0), shift.squeeze(0), 1e-6)
        else:
            out = layer_norm(x) * (1 + scale[:, None]) + shift[:, None]
    else:
        out = apply_token_modulation(layer_norm(x), 1 + scale, shift, modulation_dims)
    if CAST_MODULATED_NORM:
        out = out.astype(scale.dtype)
    return out


def _sea_filter_from_sigma(
    x: mx.array,
    sigma: float,
    *,
    power_exp: float = 2.0,
    eps: float = 1e-16,
) -> mx.array:
    sigma = max(1e-6, min(1.0 - 1e-6, float(sigma)))
    a = 1.0 - sigma
    b = sigma
    source_dtype = x.dtype
    x32 = mx.contiguous(x).astype(mx.float32)
    spectrum = mx.fft.fftn(x32, axes=(1, 2))
    filt: mx.array | None = None
    for axis in (1, 2):
        n = int(x32.shape[axis])
        freq = mx.fft.fftfreq(n).astype(mx.float32)
        rad = mx.abs(freq)
        power = 1.0 / ((rad ** power_exp) + eps)
        gain_1d = (a * power) / ((a * a * power) + (b * b) + eps)
        shape = [1] * len(x32.shape)
        shape[axis] = n
        gain_1d = mx.reshape(gain_1d, tuple(shape))
        filt = gain_1d if filt is None else filt * gain_1d
    if filt is None:
        return x
    filt = filt / (mx.mean(filt) + eps)
    filtered = mx.real(mx.fft.ifftn(spectrum * filt, axes=(1, 2)))
    return filtered.astype(source_dtype)


def rms_norm(x: mx.array, weight: mx.array) -> mx.array:
    return mx.fast.rms_norm(x, weight, 1e-6)


def nan_to_num_fp16(x: mx.array) -> mx.array:
    if NAN_TO_NUM_FP16 and x.dtype == mx.float16:
        return mx.nan_to_num(x, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return x


def sanitize_for_final_layer(x: mx.array, dtype: mx.Dtype) -> mx.array:
    limit = 65504.0 if dtype == mx.float16 else 3.38953139e38
    x = mx.nan_to_num(x.astype(mx.float32), nan=0.0, posinf=limit, neginf=-limit)
    x = mx.minimum(mx.maximum(x, -limit), limit)
    return x.astype(dtype)


def nan_to_num_flux_fp16(x: mx.array) -> mx.array:
    if x.dtype == mx.float16:
        return mx.nan_to_num(x, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return x


_qk_rms_norm_kernel = mx.fast.metal_kernel(
    name="sdmlx_flux_qk_rms_norm",
    input_names=["q", "k", "q_weight", "k_weight"],
    output_names=["q_out", "k_out"],
    source="""
        constexpr uint head_dim = 128;
        uint row = thread_position_in_grid.x;
        uint base = row * head_dim;

        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            float qv = float(q[base + i]);
            float kv = float(k[base + i]);
            q_sum += qv * qv;
            k_sum += kv * kv;
        }

        float q_scale = metal::rsqrt(q_sum / float(head_dim) + 1e-6f);
        float k_scale = metal::rsqrt(k_sum / float(head_dim) + 1e-6f);
        for (uint i = 0; i < head_dim; ++i) {
            q_out[base + i] = T(float(q[base + i]) * q_scale * float(q_weight[i]));
            k_out[base + i] = T(float(k[base + i]) * k_scale * float(k_weight[i]));
        }
    """,
)


def qk_norm(q: mx.array, k: mx.array, q_weight: mx.array, k_weight: mx.array) -> tuple[mx.array, mx.array]:
    if QK_NORM_MODE != "metal":
        return rms_norm(q, q_weight), rms_norm(k, k_weight)
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM or q.dtype != k.dtype:
        return rms_norm(q, q_weight), rms_norm(k, k_weight)
    if q_weight.dtype != q.dtype:
        q_weight = q_weight.astype(q.dtype)
    if k_weight.dtype != k.dtype:
        k_weight = k_weight.astype(k.dtype)
    rows = q.size // HEAD_DIM
    q_out, k_out = _qk_rms_norm_kernel(
        inputs=[q, k, q_weight, k_weight],
        template=[("T", q.dtype)],
        grid=(rows, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
        stream=mx.gpu,
    )
    return q_out, k_out


def time_proj(time_steps: mx.array) -> mx.array:
    max_period = 10000
    half_dim = 128
    exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, dtype=mx.float32)
    exponent = exponent / half_dim
    emb = mx.exp(exponent)
    emb = time_steps[:, None].astype(mx.float32) * emb[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    return mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)


def rope(pos: mx.array, dim: int, theta: float = 10000) -> mx.array:
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta**scale)
    batch_size, _ = pos.shape
    out = mx.expand_dims(pos, axis=-1) * mx.expand_dims(omega, axis=0)
    stacked = mx.stack([mx.cos(out), -mx.sin(out), mx.sin(out), mx.cos(out)], axis=-1)
    return mx.reshape(stacked, (batch_size, -1, dim // 2, 2, 2))


def prepare_text_ids(seq_len: int) -> mx.array:
    return mx.zeros((1, seq_len, 3))


def prepare_latent_image_ids(height: int, width: int) -> mx.array:
    latent_width = width // 16
    latent_height = height // 16
    ids = mx.zeros((latent_height, latent_width, 3))
    ids = ids.at[:, :, 1].add(mx.arange(0, latent_height)[:, None])
    ids = ids.at[:, :, 2].add(mx.arange(0, latent_width)[None, :])
    ids = mx.repeat(ids[None, :], 1, axis=0)
    return mx.reshape(ids, (1, latent_width * latent_height, 3))


def prepare_kontext_image_ids(height: int, width: int) -> mx.array:
    ids = prepare_latent_image_ids(height, width)
    return ids.at[..., 0].add(1)


def embed_nd(ids: mx.array) -> mx.array:
    axes_dim = (16, 56, 56)
    emb = mx.concatenate([rope(ids[..., i], axes_dim[i]) for i in range(3)], axis=-3)
    return mx.expand_dims(emb, axis=1)


@partial(mx.compile, shapeless=True)
def ab_plus_cd(a: mx.array, b: mx.array, c: mx.array, d: mx.array) -> mx.array:
    return a * b + c * d


def apply_rope(q: mx.array, k: mx.array, freqs: mx.array) -> tuple[mx.array, mx.array]:
    q_dtype = q.dtype
    k_dtype = k.dtype
    q_ = q.astype(mx.float32).reshape(*q.shape[:-1], -1, 1, 2)
    k_ = k.astype(mx.float32).reshape(*k.shape[:-1], -1, 1, 2)
    q_out = ab_plus_cd(freqs[..., 0], q_[..., 0], freqs[..., 1], q_[..., 1])
    k_out = ab_plus_cd(freqs[..., 0], k_[..., 0], freqs[..., 1], k_[..., 1])
    if ROPE_FLOAT32:
        return q_out.reshape(*q.shape).astype(mx.float32), k_out.reshape(*k.shape).astype(mx.float32)
    return q_out.reshape(*q.shape).astype(q_dtype), k_out.reshape(*k.shape).astype(k_dtype)


def attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    scale = HEAD_DIM**-0.5
    if USE_FAST_SDPA:
        kwargs = {"scale": scale}
        return mx.fast.scaled_dot_product_attention(q, k, v, **kwargs)
    scores = (q * scale) @ k.transpose(0, 1, 3, 2)
    attn = mx.softmax(scores, axis=-1)
    return attn @ v


def as_heads(x: mx.array) -> mx.array:
    x = mx.reshape(x, (1, -1, HEADS, HEAD_DIM))
    return mx.transpose(x, (0, 2, 1, 3))


def from_heads(x: mx.array) -> mx.array:
    x = mx.transpose(x, (0, 2, 1, 3))
    return mx.reshape(x, (1, -1, HEADS * HEAD_DIM))


def normalize_flux_weight_key(key: str) -> str | None:
    if key.startswith(("text_encoders.", "vae.")):
        return None
    prefix = "model.diffusion_model."
    if key.startswith(prefix):
        key = key[len(prefix) :]
    if key.startswith(("text_encoders.", "vae.")):
        return None
    if ".norm.key_norm.weight" in key:
        key = key.replace(".norm.key_norm.weight", ".norm.key_norm.scale")
    if ".norm.query_norm.weight" in key:
        key = key.replace(".norm.query_norm.weight", ".norm.query_norm.scale")
    return key


def _has_fp8_weights(path: Path) -> bool:
    if path.suffix.lower() == ".gguf":
        return False
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            normalized = normalize_flux_weight_key(key)
            if normalized is None:
                continue
            dtype = f.get_slice(key).get_dtype()
            if dtype.startswith("F8_"):
                return True
    return False


def _load_regular_flux_weights(path: Path) -> dict[str, mx.array]:
    raw = mx.load(str(path))
    weights: dict[str, mx.array] = {}
    for key, value in raw.items():
        normalized = normalize_flux_weight_key(key)
        if normalized is None:
            continue
        weights[normalized] = value
    return weights


def _gguf_get_field(reader, field_name: str, field_type):
    import gguf

    field = reader.get_field(field_name)
    if field is None:
        return None
    if field_type is str:
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad GGUF field type for {field_name}: expected STRING, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    if field_type in (int, float, bool):
        return field_type(field.parts[field.data[-1]].item())
    raise TypeError(f"Unsupported GGUF field type: {field_type!r}")


def _gguf_orig_shape(reader, tensor_name: str) -> tuple[int, ...] | None:
    import gguf

    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad GGUF original-shape metadata for {field_key}: {field.types!r}")
    return tuple(int(field.parts[part_idx][0]) for part_idx in field.data)


def _gguf_tensor_shape(reader, tensor) -> tuple[int, ...]:
    shape = _gguf_orig_shape(reader, tensor.name)
    if shape is not None:
        return shape
    return tuple(int(dim) for dim in reversed(tensor.shape))


def _gguf_tensor_to_mx(reader, tensor, precision) -> mx.array:
    import gguf

    qtype = tensor.tensor_type
    shape = _gguf_tensor_shape(reader, tensor)
    data = np.asarray(tensor.data)
    if qtype in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
        array_np = data.reshape(shape)
    else:
        array_np = gguf.quants.dequantize(data, qtype).reshape(shape)
    array = mx.array(array_np)
    if array.dtype in (mx.float16, mx.float32, mx.bfloat16):
        array = array.astype(precision)
    return array


def _gguf_q8_0_to_mx(reader, tensor, precision) -> tuple[mx.array, mx.array, mx.array, tuple[int, int]]:
    shape = _gguf_tensor_shape(reader, tensor)
    if len(shape) != 2:
        raise ValueError(f"Q8_0 GGUF tensor must be 2D, got {shape} for {tensor.name}")
    out_dim, in_dim = (int(dim) for dim in shape)
    if in_dim % 32 != 0:
        raise ValueError(f"Q8_0 GGUF tensor input dim must be divisible by 32, got {shape} for {tensor.name}")
    blocks = np.asarray(tensor.data).reshape(out_dim, in_dim // 32, 34)
    scales = blocks[..., :2].copy().view(np.float16).reshape(out_dim, in_dim // 32)
    values = blocks[..., 2:].copy().view(np.int8).reshape(out_dim, in_dim)
    affine_values = (values.astype(np.int16) + 128).astype(np.uint8)
    packed_u8 = affine_values.reshape(out_dim, in_dim // 4, 4).astype(np.uint32)
    packed = packed_u8[..., 0] | (packed_u8[..., 1] << 8) | (packed_u8[..., 2] << 16) | (packed_u8[..., 3] << 24)
    scales_mx = mx.array(scales).astype(precision)
    biases_mx = (-128.0 * scales_mx).astype(precision)
    return mx.array(packed), scales_mx, biases_mx, (out_dim, in_dim)


def _gguf_q4_0_to_mx(reader, tensor, precision) -> tuple[mx.array, mx.array, mx.array, tuple[int, int]]:
    shape = _gguf_tensor_shape(reader, tensor)
    if len(shape) != 2:
        raise ValueError(f"Q4_0 GGUF tensor must be 2D, got {shape} for {tensor.name}")
    out_dim, in_dim = (int(dim) for dim in shape)
    if in_dim % 32 != 0:
        raise ValueError(f"Q4_0 GGUF tensor input dim must be divisible by 32, got {shape} for {tensor.name}")
    blocks = np.asarray(tensor.data).reshape(out_dim, in_dim // 32, 18)
    scales = blocks[..., :2].copy().view(np.float16).reshape(out_dim, in_dim // 32)
    packed_nibbles = blocks[..., 2:].copy().reshape(out_dim, in_dim // 32, 16)
    low = packed_nibbles & 0x0F
    high = packed_nibbles >> 4
    # GGUF Q4_0 stores the low nibbles for the first 16 values in the block
    # and the high nibbles for the second 16 values.
    affine_values = np.concatenate([low, high], axis=-1).reshape(out_dim, in_dim).astype(np.uint8)
    packed_u4 = affine_values.reshape(out_dim, in_dim // 8, 8).astype(np.uint32)
    packed = (
        packed_u4[..., 0]
        | (packed_u4[..., 1] << 4)
        | (packed_u4[..., 2] << 8)
        | (packed_u4[..., 3] << 12)
        | (packed_u4[..., 4] << 16)
        | (packed_u4[..., 5] << 20)
        | (packed_u4[..., 6] << 24)
        | (packed_u4[..., 7] << 28)
    )
    scales_mx = mx.array(scales).astype(precision)
    biases_mx = (-8.0 * scales_mx).astype(precision)
    return mx.array(packed), scales_mx, biases_mx, (out_dim, in_dim)


def _load_gguf_flux_weights(path: Path, precision) -> tuple[dict[str, mx.array], str, dict[str, tuple[mx.array, mx.array, mx.array, tuple[int, int]]]]:
    import gguf

    reader = gguf.GGUFReader(str(path))
    arch = _gguf_get_field(reader, "general.architecture", str)
    if arch not in {None, "flux"}:
        raise RuntimeError(f"SDMLX FLUX: unsupported GGUF architecture {arch!r} in {path.name}; expected 'flux'.")

    weights: dict[str, mx.array] = {}
    gguf_q8: dict[str, tuple[mx.array, mx.array, mx.array, tuple[int, int]]] = {}
    qtype_counts: dict[str, int] = {}
    for tensor in reader.tensors:
        normalized = normalize_flux_weight_key(tensor.name)
        if normalized is None:
            continue
        qtype_name = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
        qtype_counts[qtype_name] = qtype_counts.get(qtype_name, 0) + 1
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0 and normalized.endswith(".weight"):
            gguf_q8[normalized] = _gguf_q8_0_to_mx(reader, tensor, precision)
            continue
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0 and normalized.endswith(".weight"):
            gguf_q8[normalized] = _gguf_q4_0_to_mx(reader, tensor, precision)
            continue
        weights[normalized] = _gguf_tensor_to_mx(reader, tensor, precision)
    qtypes = ",".join(f"{name}:{count}" for name, count in sorted(qtype_counts.items()))
    load_mode = f"gguf_native_affine:{qtypes or 'none'}" if gguf_q8 else f"gguf_dequant:{qtypes or 'none'}"
    return weights, load_mode, gguf_q8


def _load_fp8_flux_weights(path: Path, precision) -> dict[str, mx.array]:
    import torch

    if precision == mx.float32:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    weights: dict[str, mx.array] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            normalized = normalize_flux_weight_key(key)
            if normalized is None:
                continue
            tensor = f.get_tensor(key)
            if tensor.is_floating_point():
                tensor = tensor.to(torch_dtype)
            array = mx.array(tensor.cpu().contiguous().numpy())
            if array.dtype in (mx.float16, mx.float32, mx.bfloat16):
                array = array.astype(precision)
            weights[normalized] = array
    return weights


def _load_native_fp8_flux_weights(path: Path, precision) -> tuple[dict[str, mx.array], set[str]]:
    import torch

    if precision == mx.float32:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    weights: dict[str, mx.array] = {}
    fp8_weight_keys: set[str] = set()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            normalized = normalize_flux_weight_key(key)
            if normalized is None:
                continue
            dtype = f.get_slice(key).get_dtype()
            tensor = f.get_tensor(key)
            if dtype.startswith("F8_"):
                if normalized.endswith(".weight") and len(tensor.shape) == 2:
                    raw = tensor.view(torch.uint8).cpu().contiguous().numpy()
                    weights[normalized] = mx.array(raw)
                    fp8_weight_keys.add(normalized)
                else:
                    weights[normalized] = mx.array(tensor.to(torch_dtype).cpu().contiguous().numpy()).astype(precision)
                continue
            if tensor.is_floating_point():
                tensor = tensor.to(torch_dtype)
            array = mx.array(tensor.cpu().contiguous().numpy())
            if array.dtype in (mx.float16, mx.float32, mx.bfloat16):
                array = array.astype(precision)
            weights[normalized] = array
    return weights, fp8_weight_keys


def _pack_fp8_u8_to_u32(raw: np.ndarray) -> np.ndarray:
    if raw.dtype != np.uint8:
        raw = raw.astype(np.uint8, copy=False)
    if raw.shape[-1] % 4 != 0:
        raise ValueError(f"Cannot pack FP8 weight with non-multiple-of-4 last dim: {raw.shape}")
    packed = raw.reshape(*raw.shape[:-1], raw.shape[-1] // 4, 4).astype(np.uint32)
    return packed[..., 0] | (packed[..., 1] << 8) | (packed[..., 2] << 16) | (packed[..., 3] << 24)


def _load_packed_mxfp8_flux_weights(path: Path, precision) -> tuple[dict[str, mx.array], dict[str, tuple[mx.array, mx.array]]]:
    import torch

    if precision == mx.float32:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    weights: dict[str, mx.array] = {}
    fp8_mxfp8: dict[str, tuple[mx.array, mx.array]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            normalized = normalize_flux_weight_key(key)
            if normalized is None:
                continue
            dtype = f.get_slice(key).get_dtype()
            tensor = f.get_tensor(key)
            if dtype.startswith("F8_") and normalized.endswith(".weight") and len(tensor.shape) == 2:
                raw = tensor.view(torch.uint8).cpu().contiguous().numpy()
                packed = _pack_fp8_u8_to_u32(raw)
                scales = np.full((raw.shape[0], raw.shape[1] // 32), 127, dtype=np.uint8)
                fp8_mxfp8[normalized] = (mx.array(packed), mx.array(scales))
                continue
            if dtype.startswith("F8_"):
                weights[normalized] = mx.array(tensor.to(torch_dtype).cpu().contiguous().numpy()).astype(precision)
                continue
            if tensor.is_floating_point():
                tensor = tensor.to(torch_dtype)
            array = mx.array(tensor.cpu().contiguous().numpy())
            if array.dtype in (mx.float16, mx.float32, mx.bfloat16):
                array = array.astype(precision)
            weights[normalized] = array
    return weights, fp8_mxfp8


def load_flux_weights(
    path: Path,
    precision,
    fp8_mode: str = "dequant",
) -> tuple[
    dict[str, mx.array],
    str,
    set[str],
    dict[str, tuple[mx.array, mx.array]],
    dict[str, tuple[mx.array, mx.array, mx.array, tuple[int, int]]],
]:
    fp8_mxfp8: dict[str, tuple[mx.array, mx.array]] = {}
    gguf_q8: dict[str, tuple[mx.array, mx.array, mx.array, tuple[int, int]]] = {}
    if path.suffix.lower() == ".gguf":
        weights, load_mode, gguf_q8 = _load_gguf_flux_weights(path, precision)
        fp8_weight_keys = set()
    elif _has_fp8_weights(path):
        if fp8_mode in {"native", "native_fp8", "mlx_fp8_native"}:
            weights, fp8_weight_keys = _load_native_fp8_flux_weights(path, precision)
            load_mode = "mlx_fp8_native"
        elif fp8_mode == "packed_mxfp8":
            weights, fp8_mxfp8 = _load_packed_mxfp8_flux_weights(path, precision)
            fp8_weight_keys = set()
            load_mode = "mlx_fp8_packed_mxfp8"
        else:
            weights = _load_fp8_flux_weights(path, precision)
            fp8_weight_keys = set()
            load_mode = "torch_fp8_dequant"
    else:
        weights = _load_regular_flux_weights(path)
        fp8_weight_keys = set()
        load_mode = "mlx"

    if "txt_in.weight" not in weights and "txt_in.weight" not in fp8_mxfp8 and "txt_in.weight" not in gguf_q8:
        sample = ", ".join(list(weights)[:8])
        raise KeyError(
            "txt_in.weight missing after FLUX weight normalization. "
            f"Loaded {len(weights)} transformer keys from {path.name}. "
            f"First keys: {sample}"
        )
    return weights, load_mode, fp8_weight_keys, fp8_mxfp8, gguf_q8


class FluxNativeTransformer:
    def __init__(
        self,
        weights: dict[str, mx.array],
        precision=mx.bfloat16,
        fp8_weight_keys: set[str] | None = None,
        fp8_mxfp8: dict[str, tuple[mx.array, mx.array]] | None = None,
        gguf_q8: dict[str, tuple[mx.array, mx.array, mx.array, tuple[int, int]]] | None = None,
    ):
        self.w = weights
        self.load_mode = "unknown"
        self.wt: dict[str, mx.array] = {}
        self.q: dict[str, tuple[mx.array, mx.array, mx.array | None, int, int]] = {}
        self.fp8_weight_keys = fp8_weight_keys or set()
        self.fp8_mxfp8 = fp8_mxfp8 or {}
        self.gguf_q8 = gguf_q8 or {}
        self.lora_adapters: dict[str, list[tuple[mx.array, mx.array, int | None, int | None]]] = {}
        self.lora_sources: list[str] = []
        self.precision = precision
        self._rotary_cache: dict[tuple, mx.array] = {}
        self.profile_enabled = False
        self.detail_single_block: int | None = None
        self.detail_double_block: int | None = None
        self.approx_image_mlp = False
        self.profile: dict[str, list[float]] = {}
        self.profile_steps: set[int] = set()
        self.profile_by_step = False
        self.detail_profile: dict[str, list[float]] = {}
        self._detail_order: list[str] = []
        self._detail_dtype_printed = False
        self.modulation_cache: dict[str, object] | None = None
        self.shadow_enabled = False
        self.shadow_double_blocks: set[int] = set()
        self.shadow_single_blocks: set[int] = set()
        self.shadow_step_index = 0
        self.shadow_previous: dict[str, mx.array] = {}
        self.shadow_metrics: list[dict[str, float | int | str]] = []
        self.teacache_mode = "off"
        self.teacache_threshold = 0.08
        self.teacache_threshold_end: float | None = None
        self.teacache_warmup_steps = 5
        self.teacache_final_steps = 3
        self.teacache_total_steps = 0
        self.teacache_previous_probe: mx.array | None = None
        self.teacache_last_feature: mx.array | None = None
        self.seacache_previous_residual: mx.array | None = None
        self.teacache_accumulated = 0.0
        self.teacache_hits = 0
        self.teacache_real_steps = 0
        self.teacache_metrics: list[dict[str, float | int | str]] = []
        self.forecast_step_index = 0
        self.forecast_single_mode = "off"
        self.forecast_single_scope = "residual"
        self.forecast_single_steps: set[int] = set()
        self.forecast_single_blocks: set[int] = set()
        self.forecast_single_plan: dict[int, tuple[str, set[int]]] = {}
        self.forecast_single_attention_plan: dict[int, tuple[str, set[int]]] = {}
        self.forecast_single_linear2_plan: dict[int, tuple[str, set[int]]] = {}
        self.forecast_single_linear2_late_plan: dict[int, tuple[str, set[int]]] = {}
        self.forecast_single_linear2_late_split = "all"
        self.forecast_single_adaptive = False
        self.forecast_single_adaptive_blocks: set[int] = set()
        self.forecast_single_adaptive_step2_blocks: set[int] = set()
        self.forecast_single_adaptive_sensitivity = 0.5
        self.forecast_single_adaptive_step_sensitivity: dict[int, float] = {}
        self.forecast_single_adaptive_step_lowest_count: dict[int, int] = {}
        self.forecast_single_adaptive_low_pool_split = False
        self.forecast_single_adaptive_low_pool_source_step = 0
        self.forecast_single_adaptive_low_pool_steps: dict[int, set[int]] = {}
        self.forecast_single_adaptive_lowest_cache: dict[int, set[int]] = {}
        self.forecast_single_adaptive_spread = 1.7
        self.forecast_single_adaptive_top_count = 10
        self.forecast_single_gain = 1.0
        self.forecast_single_step_gain: dict[int, float] = {}
        self.forecast_single_delta_clamp = 0.0
        self.forecast_single_step_delta_clamp: dict[int, float] = {}
        self.forecast_single_total_steps = 0
        self.forecast_single_last_weather_step = 0
        self.forecast_single_weather_debug = False
        self.forecast_single_history: dict[int, list[tuple[int, mx.array]]] = {}
        self.forecast_single_attention_history: dict[int, list[tuple[int, mx.array]]] = {}
        self.forecast_single_linear2_history: dict[int, list[tuple[int, mx.array]]] = {}
        self.forecast_single_linear2_norm_history: dict[int, list[tuple[int, mx.array]]] = {}
        self.forecast_single_linear2_gate_history: dict[int, list[tuple[int, mx.array]]] = {}
        self.forecast_single_linear2_shape = "plain"
        self.forecast_single_linear2_shape_blocks: set[int] | None = None
        self.forecast_single_linear2_shape_clamp = 0.0
        self.forecast_single_linear2_block_gain: dict[int, float] = {}
        self.forecast_single_linear2_scout_plan: dict[int, tuple[str, set[int]]] = {}
        self.forecast_single_linear2_scout_metrics: list[dict[str, float | int | str]] = []
        self.forecast_single_weather_latest: dict[int, dict[str, float | int]] = {}
        self.forecast_single_weather_metrics: list[dict[str, float | int]] = []
        self.forecast_single_linear2_partial_hits = 0
        self.forecast_single_hits: list[dict[str, float | int | str]] = []
        self.forecast_single_current_step = 0
        self.forecast_single_locked_blocks: set[int] = set()
        self.forecast_single_next_locked_blocks: set[int] = set()
        self.forecast_single_storm_ref: float | None = None
        self.forecast_single_storm_metrics: list[dict[str, float | int]] = []
        self.forecast_double_img_mlp_mode = "off"
        self.forecast_double_img_mlp_steps: set[int] = set()
        self.forecast_double_img_mlp_blocks: set[int] = set()
        self.forecast_double_img_mlp_plan: dict[int, tuple[str, set[int]]] = {}
        self.forecast_double_img_mlp_gain = 1.0
        self.forecast_double_img_mlp_step_gain: dict[int, float] = {}
        self.forecast_double_img_mlp_history: dict[int, list[tuple[int, mx.array]]] = {}
        self.forecast_double_img_mlp_hits: list[dict[str, float | int | str]] = []
        self.forecast_double_txt_mode = "off"
        self.forecast_double_txt_steps: set[int] = set()
        self.forecast_double_txt_blocks: set[int] = set()
        self.forecast_double_txt_plan: dict[int, tuple[str, set[int]]] = {}
        self.forecast_double_txt_gain = 1.0
        self.forecast_double_txt_step_gain: dict[int, float] = {}
        self.forecast_double_txt_history: dict[int, list[tuple[int, mx.array]]] = {}
        self.forecast_double_txt_hits: list[dict[str, float | int | str]] = []
        self.token_scout_enabled = False
        self.token_scout_double_blocks: set[int] = set()
        self.token_scout_single_blocks: set[int] = set()
        self.token_scout_txt_len = 0
        self.token_scout_previous: dict[str, mx.array] = {}
        self.token_scout_metrics: list[dict[str, float | int | str]] = []
        self.attention_scout_enabled = False
        self.attention_scout_steps: set[int] = set()
        self.attention_scout_double_blocks: set[int] = set()
        self.attention_scout_single_blocks: set[int] = set()
        self.attention_scout_chunk = 512
        self.attention_scout_metrics: list[dict[str, float | int | str]] = []
        self.kontext_kv_cache_enabled = False
        self.kontext_kv_cache: dict[str, tuple[mx.array, mx.array]] = {}
        self.kontext_kv_cache_reference_tokens = 0
        self.kontext_kv_cache_hits = 0
        self.kontext_kv_cache_stores = 0
        self.kontext_reference_zero_calls = 0
        self.kontext_reference_zero_last: dict[str, object] = {}

    @classmethod
    def load(cls, path: Path = FLUX_PATH, precision=mx.bfloat16, fp8_mode: str = "dequant") -> "FluxNativeTransformer":
        weights, load_mode, fp8_weight_keys, fp8_mxfp8, gguf_q8 = load_flux_weights(path, precision, fp8_mode=fp8_mode)
        transformer = cls(
            weights,
            precision=precision,
            fp8_weight_keys=fp8_weight_keys,
            fp8_mxfp8=fp8_mxfp8,
            gguf_q8=gguf_q8,
        )
        transformer.load_mode = load_mode
        return transformer

    def k(self, key: str) -> mx.array:
        return self.w[key]

    def cast_weights(self, dtype) -> int:
        count = 0
        casted = {}
        for key, value in list(self.w.items()):
            if value.dtype in (mx.float16, mx.bfloat16, mx.float32):
                casted[key] = value.astype(dtype)
                count += 1
            else:
                casted[key] = value
        self.w = casted
        mx.eval(self.w)
        return count

    def prepare_transposed_linears(
        self,
        *,
        scope: str = "all",
        fp8_dequant: bool = False,
        drop_raw: str = "off",
    ) -> int:
        count = 0
        dequantized_fp8_keys = set()
        pending_eval: list[mx.array] = []
        pending_raw_keys: list[str] = []

        def flush_pending() -> None:
            if not pending_eval:
                return
            mx.eval(*pending_eval)
            for raw_key in pending_raw_keys:
                self.w.pop(raw_key, None)
            pending_eval.clear()
            pending_raw_keys.clear()

        def should_drop_raw(raw_key: str) -> bool:
            if drop_raw == "all":
                return True
            if drop_raw == "fp8" and raw_key in self.fp8_weight_keys:
                return True
            return False

        for key, value in list(self.w.items()):
            if not key.endswith(".weight") or len(value.shape) != 2:
                continue
            if key in self.q:
                continue
            if not self.quant_scope_matches(key, scope):
                continue
            if fp8_dequant and key in self.fp8_weight_keys:
                self.wt[key] = mx.from_fp8(value.T, self.precision)
                dequantized_fp8_keys.add(key)
                pending_eval.append(self.wt[key])
                pending_raw_keys.append(key)
                if len(pending_eval) >= 16:
                    flush_pending()
            else:
                if should_drop_raw(key):
                    self.wt[key] = mx.contiguous(value.T)
                    pending_eval.append(self.wt[key])
                    pending_raw_keys.append(key)
                    if len(pending_eval) >= 16:
                        flush_pending()
                else:
                    self.wt[key] = value.T
            count += 1
        flush_pending()
        if dequantized_fp8_keys:
            self.fp8_weight_keys.difference_update(dequantized_fp8_keys)
        mx.eval(self.wt)
        return count

    def release_model_weights(self) -> None:
        self.w.clear()
        self.wt.clear()
        self.q.clear()
        self.fp8_mxfp8.clear()
        self.gguf_q8.clear()
        self.lora_adapters.clear()
        self.lora_sources.clear()
        self.fp8_weight_keys.clear()

    def profile_eval(self, label: str, start: float, *arrays: mx.array) -> None:
        if not self.profile_enabled:
            return
        step = int(getattr(self, "forecast_step_index", 0))
        if self.profile_steps and step not in self.profile_steps:
            return
        mx.eval(*arrays)
        if self.profile_by_step and step > 0:
            label = f"step{step}.{label}"
        self.profile.setdefault(label, []).append(time.perf_counter() - start)

    def profile_report(self) -> list[tuple[str, int, float, float]]:
        report = []
        for label, values in self.profile.items():
            total = sum(values)
            report.append((label, len(values), total, total / len(values)))
        return sorted(report, key=lambda item: item[2], reverse=True)

    def detail_record(self, label: str, value: float) -> None:
        if label not in self.detail_profile:
            self._detail_order.append(label)
        self.detail_profile.setdefault(label, []).append(value)

    def detail_eval(self, label: str, start: float, *arrays: mx.array) -> None:
        mx.eval(*arrays)
        self.detail_record(label, time.perf_counter() - start)

    def detail_report(self) -> list[tuple[str, int, float, float]]:
        report = []
        for label in self._detail_order:
            values = self.detail_profile[label]
            total = sum(values)
            report.append((label, len(values), total, total / len(values)))
        return report

    def shadow_probe(self, label: str, value: mx.array) -> None:
        if not self.shadow_enabled:
            return
        t0 = time.perf_counter()
        value = value.astype(mx.float32)
        value_shape = tuple(int(dim) for dim in value.shape)
        value_mean_abs = mx.mean(mx.abs(value))
        previous = self.shadow_previous.get(label)
        if previous is None:
            mx.eval(value, value_mean_abs)
            self.shadow_metrics.append(
                {
                    "step": self.shadow_step_index,
                    "label": label,
                    "mean_abs": float(value_mean_abs.item()),
                    "delta_mean_abs": 0.0,
                    "rel_l1": 0.0,
                    "max_abs_delta": 0.0,
                    "shape": str(value_shape),
                    "shape_changed": 0,
                    "probe_s": time.perf_counter() - t0,
                }
            )
            self.shadow_previous[label] = value
            return

        if tuple(int(dim) for dim in previous.shape) != value_shape:
            mx.eval(value, value_mean_abs)
            self.shadow_metrics.append(
                {
                    "step": self.shadow_step_index,
                    "label": label,
                    "mean_abs": float(value_mean_abs.item()),
                    "delta_mean_abs": 0.0,
                    "rel_l1": 0.0,
                    "max_abs_delta": 0.0,
                    "shape": str(value_shape),
                    "shape_changed": 1,
                    "probe_s": time.perf_counter() - t0,
                }
            )
            self.shadow_previous[label] = value
            return

        delta = mx.abs(value - previous)
        delta_mean_abs = mx.mean(delta)
        max_abs_delta = mx.max(delta)
        rel_l1 = delta_mean_abs / (mx.mean(mx.abs(previous)) + 1e-6)
        mx.eval(value, value_mean_abs, delta_mean_abs, max_abs_delta, rel_l1)
        self.shadow_metrics.append(
            {
                "step": self.shadow_step_index,
                "label": label,
                "mean_abs": float(value_mean_abs.item()),
                "delta_mean_abs": float(delta_mean_abs.item()),
                "rel_l1": float(rel_l1.item()),
                "max_abs_delta": float(max_abs_delta.item()),
                "shape": str(value_shape),
                "shape_changed": 0,
                "probe_s": time.perf_counter() - t0,
            }
        )
        self.shadow_previous[label] = value

    def shadow_report(self) -> list[dict[str, float | int | str]]:
        return self.shadow_metrics

    def reset_teacache_gate(self) -> None:
        self.teacache_previous_probe = None
        self.teacache_last_feature = None
        self.seacache_previous_residual = None
        self.teacache_accumulated = 0.0
        self.teacache_hits = 0
        self.teacache_real_steps = 0
        self.teacache_metrics = []

    def teacache_gate_active(self) -> bool:
        return str(self.teacache_mode or "off") in {"double0_txt", "sea_img"}

    def seacache_gate_active(self) -> bool:
        return str(self.teacache_mode or "off") == "sea_img"

    def teacache_step_threshold(self, step: int) -> float:
        start = float(self.teacache_threshold)
        end_raw = getattr(self, "teacache_threshold_end", None)
        if end_raw is None:
            return start
        end = float(end_raw)
        total_steps = int(self.teacache_total_steps or 0)
        if total_steps <= 0:
            return start
        active_start = max(1, int(self.teacache_warmup_steps) + 1)
        active_end = max(active_start, total_steps - int(self.teacache_final_steps))
        progress = (int(step) - active_start) / max(1, active_end - active_start)
        progress = max(0.0, min(1.0, float(progress)))
        return start + (end - start) * progress

    def teacache_try_reuse_feature(
        self,
        *,
        txt: mx.array,
        vec: mx.array,
        modulation_index: int | None = None,
    ) -> mx.array | None:
        if not self.teacache_gate_active():
            return None

        step = int(self.forecast_step_index)
        total_steps = int(self.teacache_total_steps or 0)
        if step <= 0:
            return None

        base = "double_blocks.0"
        if self.modulation_cache is not None and modulation_index is not None:
            _img_mod, txt_mod = self.modulation_cache["double"][0]  # type: ignore[index]
            txt_shift, txt_scale, *_ = (chunk[modulation_index : modulation_index + 1] for chunk in txt_mod)
        else:
            txt_shift, txt_scale, *_ = self.modulate_double(vec, f"{base}.txt_mod")
        probe = modulated_layer_norm(txt, txt_scale, txt_shift).astype(mx.float32)
        mean_abs = mx.mean(mx.abs(probe))

        previous = self.teacache_previous_probe
        reason = "ok"
        rel_l1_value = 0.0
        shape_changed = 0
        if previous is None:
            mx.eval(probe, mean_abs)
            reason = "first_probe"
        elif tuple(int(dim) for dim in previous.shape) != tuple(int(dim) for dim in probe.shape):
            mx.eval(probe, mean_abs)
            reason = "shape_changed"
            shape_changed = 1
        else:
            delta = mx.mean(mx.abs(probe - previous))
            rel_l1 = delta / (mx.mean(mx.abs(previous)) + 1e-6)
            mx.eval(probe, mean_abs, delta, rel_l1)
            rel_l1_value = float(rel_l1.item())
        self.teacache_previous_probe = probe

        force_real = False
        if reason != "ok":
            force_real = True
        elif self.teacache_last_feature is None:
            reason = "no_feature"
            force_real = True
        elif step <= int(self.teacache_warmup_steps):
            reason = "warmup"
            force_real = True
        elif total_steps > 0 and step > total_steps - int(self.teacache_final_steps):
            reason = "final"
            force_real = True

        threshold = self.teacache_step_threshold(step)
        accumulated_next = float(self.teacache_accumulated) + float(rel_l1_value)
        if force_real:
            self.teacache_accumulated = 0.0
            self.teacache_metrics.append(
                {
                    "step": step,
                    "action": "real",
                    "reason": reason,
                    "rel_l1": rel_l1_value,
                    "accumulated": 0.0,
                    "threshold": threshold,
                    "shape_changed": shape_changed,
                    "mean_abs": float(mean_abs.item()),
                }
            )
            return None

        if accumulated_next <= threshold:
            self.teacache_accumulated = accumulated_next
            self.teacache_hits += 1
            self.teacache_metrics.append(
                {
                    "step": step,
                    "action": "reuse",
                    "reason": "under_threshold",
                    "rel_l1": rel_l1_value,
                    "accumulated": accumulated_next,
                    "threshold": threshold,
                    "shape_changed": shape_changed,
                    "mean_abs": float(mean_abs.item()),
                }
            )
            return self.teacache_last_feature

        self.teacache_accumulated = 0.0
        self.teacache_metrics.append(
            {
                "step": step,
                "action": "real",
                "reason": "threshold",
                "rel_l1": rel_l1_value,
                "accumulated": accumulated_next,
                "threshold": threshold,
                "shape_changed": shape_changed,
                "mean_abs": float(mean_abs.item()),
            }
        )
        return None

    def teacache_record_real_feature(self, feature: mx.array) -> None:
        if not self.teacache_gate_active() or self.seacache_gate_active():
            return
        self.teacache_last_feature = feature
        self.teacache_real_steps += 1

    def seacache_try_reuse_feature(
        self,
        *,
        img: mx.array,
        vec: mx.array,
        height: int,
        width: int,
        target_len: int,
        sigma: float,
        modulation_index: int | None = None,
        img_modulation_dims: ModulationDims | None = None,
    ) -> mx.array | None:
        if not self.seacache_gate_active():
            return None

        step = int(self.forecast_step_index)
        total_steps = int(self.teacache_total_steps or 0)
        if step <= 0:
            return None

        base = "double_blocks.0"
        if self.modulation_cache is not None and modulation_index is not None:
            img_mod, _txt_mod = self.modulation_cache["double"][0]  # type: ignore[index]
            img_shift, img_scale, *_ = (chunk[modulation_index : modulation_index + 1] for chunk in img_mod)
        else:
            img_shift, img_scale, *_ = self.modulate_double(vec, f"{base}.img_mod")
        probe = modulated_layer_norm(img, img_scale, img_shift, img_modulation_dims)[:, :target_len].astype(mx.float32)
        latent_h = int(height) // 16
        latent_w = int(width) // 16
        mean_abs = mx.mean(mx.abs(probe))

        reason = "ok"
        rel_l1_value = 0.0
        shape_changed = 0
        if latent_h * latent_w != int(target_len):
            mx.eval(probe, mean_abs)
            filtered_probe = probe
            reason = "shape_mismatch"
            shape_changed = 1
        else:
            filtered_grid = _sea_filter_from_sigma(
                mx.reshape(probe, (probe.shape[0], latent_h, latent_w, probe.shape[-1])),
                sigma,
            )
            filtered_probe = mx.reshape(filtered_grid, probe.shape)

        previous = self.teacache_previous_probe
        if reason == "ok" and previous is None:
            mx.eval(filtered_probe, mean_abs)
            reason = "first_probe"
        elif reason == "ok" and tuple(int(dim) for dim in previous.shape) != tuple(int(dim) for dim in filtered_probe.shape):
            mx.eval(filtered_probe, mean_abs)
            reason = "shape_changed"
            shape_changed = 1
        elif reason == "ok":
            delta = mx.mean(mx.abs(filtered_probe - previous))
            rel_l1 = delta / (mx.mean(mx.abs(previous)) + 1e-16)
            mx.eval(filtered_probe, mean_abs, delta, rel_l1)
            rel_l1_value = float(rel_l1.item())
        self.teacache_previous_probe = filtered_probe

        force_real = False
        if reason != "ok":
            force_real = True
        elif self.seacache_previous_residual is None:
            reason = "no_residual"
            force_real = True
        elif step <= int(self.teacache_warmup_steps):
            reason = "warmup"
            force_real = True
        elif total_steps > 0 and step > total_steps - int(self.teacache_final_steps):
            reason = "final"
            force_real = True

        threshold = self.teacache_step_threshold(step)
        accumulated_next = float(self.teacache_accumulated) + float(rel_l1_value)
        if force_real:
            self.teacache_accumulated = 0.0
            self.teacache_metrics.append(
                {
                    "step": step,
                    "action": "real",
                    "reason": reason,
                    "rel_l1": rel_l1_value,
                    "accumulated": 0.0,
                    "threshold": threshold,
                    "shape_changed": shape_changed,
                    "mean_abs": float(mean_abs.item()),
                    "sigma": float(sigma),
                }
            )
            return None

        if accumulated_next < threshold:
            self.teacache_accumulated = accumulated_next
            self.teacache_hits += 1
            self.teacache_metrics.append(
                {
                    "step": step,
                    "action": "reuse",
                    "reason": "under_threshold",
                    "rel_l1": rel_l1_value,
                    "accumulated": accumulated_next,
                    "threshold": threshold,
                    "shape_changed": shape_changed,
                    "mean_abs": float(mean_abs.item()),
                    "sigma": float(sigma),
                }
            )
            return img[:, :target_len] + self.seacache_previous_residual

        self.teacache_accumulated = 0.0
        self.teacache_metrics.append(
            {
                "step": step,
                "action": "real",
                "reason": "threshold",
                "rel_l1": rel_l1_value,
                "accumulated": accumulated_next,
                "threshold": threshold,
                "shape_changed": shape_changed,
                "mean_abs": float(mean_abs.item()),
                "sigma": float(sigma),
            }
        )
        return None

    def seacache_record_real_feature(self, feature: mx.array, initial_img: mx.array) -> None:
        if not self.seacache_gate_active():
            return
        self.seacache_previous_residual = feature - initial_img[:, : feature.shape[1]]
        self.teacache_real_steps += 1

    def token_scout_before_single(self, index: int, x: mx.array) -> mx.array | None:
        if not self.token_scout_enabled:
            return None
        if index not in self.token_scout_single_blocks:
            return None
        if self.token_scout_txt_len <= 0:
            return None
        return x[:, : self.token_scout_txt_len]

    def token_scout_probe(self, stream: str, block: int, before: mx.array, after: mx.array) -> None:
        if not self.token_scout_enabled:
            return
        t0 = time.perf_counter()
        before = before.astype(mx.float32)
        after = after.astype(mx.float32)
        mean_abs = mx.mean(mx.abs(after), axis=-1).reshape(-1)
        before_abs = mx.mean(mx.abs(before), axis=-1).reshape(-1)
        block_delta = mx.mean(mx.abs(after - before), axis=-1).reshape(-1)
        block_rel = block_delta / (before_abs + 1e-6)
        block_dot = mx.sum(after * before, axis=-1).reshape(-1)
        block_norm = mx.sqrt(mx.sum(after * after, axis=-1) * mx.sum(before * before, axis=-1)).reshape(-1)
        block_cos_dist = 1.0 - (block_dot / (block_norm + 1e-6))

        key = f"{stream}:{block}"
        previous = self.token_scout_previous.get(key)
        if previous is None or previous.shape != after.shape:
            step_delta = mx.zeros_like(block_delta)
            step_rel = mx.zeros_like(block_delta)
            step_cos_dist = mx.zeros_like(block_delta)
        else:
            previous_abs = mx.mean(mx.abs(previous), axis=-1).reshape(-1)
            step_delta = mx.mean(mx.abs(after - previous), axis=-1).reshape(-1)
            step_rel = step_delta / (previous_abs + 1e-6)
            step_dot = mx.sum(after * previous, axis=-1).reshape(-1)
            step_norm = mx.sqrt(mx.sum(after * after, axis=-1) * mx.sum(previous * previous, axis=-1)).reshape(-1)
            step_cos_dist = 1.0 - (step_dot / (step_norm + 1e-6))

        mx.eval(mean_abs, block_delta, block_rel, block_cos_dist, step_delta, step_rel, step_cos_dist, after)
        mean_abs_values = mean_abs.tolist()
        block_delta_values = block_delta.tolist()
        block_rel_values = block_rel.tolist()
        block_cos_values = block_cos_dist.tolist()
        step_delta_values = step_delta.tolist()
        step_rel_values = step_rel.tolist()
        step_cos_values = step_cos_dist.tolist()
        for token, values in enumerate(
            zip(
                mean_abs_values,
                block_delta_values,
                block_rel_values,
                block_cos_values,
                step_delta_values,
                step_rel_values,
                step_cos_values,
            )
        ):
            (
                token_mean_abs,
                token_block_delta,
                token_block_rel,
                token_block_cos,
                token_step_delta,
                token_step_rel,
                token_step_cos,
            ) = values
            self.token_scout_metrics.append(
                {
                    "step": self.forecast_step_index,
                    "stream": stream,
                    "block": block,
                    "token": token,
                    "mean_abs": float(token_mean_abs),
                    "block_delta": float(token_block_delta),
                    "block_rel": float(token_block_rel),
                    "block_cos_dist": float(token_block_cos),
                    "step_delta": float(token_step_delta),
                    "step_rel": float(token_step_rel),
                    "step_cos_dist": float(token_step_cos),
                    "probe_s": time.perf_counter() - t0,
                }
            )
        self.token_scout_previous[key] = after

    def token_scout_report(self) -> list[dict[str, float | int | str]]:
        return self.token_scout_metrics

    def attention_scout_wants(self, stream: str, block: int) -> bool:
        if not self.attention_scout_enabled:
            return False
        if self.forecast_step_index not in self.attention_scout_steps:
            return False
        if stream == "double_img_to_txt":
            return block in self.attention_scout_double_blocks
        if stream == "single_img_to_txt":
            return block in self.attention_scout_single_blocks
        return False

    def attention_scout_probe(self, stream: str, block: int, q: mx.array, k: mx.array, txt_len: int) -> None:
        if not self.attention_scout_wants(stream, block):
            return
        t0 = time.perf_counter()
        seq_len = int(q.shape[2])
        txt_len = int(txt_len)
        if txt_len <= 0 or txt_len >= seq_len:
            return
        chunk_size = max(1, int(self.attention_scout_chunk))
        scale = HEAD_DIM**-0.5
        k_t = k.transpose(0, 1, 3, 2)
        token_mass = mx.zeros((txt_len,), dtype=mx.float32)
        query_rows = 0
        for start in range(txt_len, seq_len, chunk_size):
            stop = min(seq_len, start + chunk_size)
            q_chunk = q[:, :, start:stop, :]
            scores = (q_chunk * scale) @ k_t
            attn = mx.softmax(scores, axis=-1)
            txt_attn = attn[..., :txt_len].astype(mx.float32)
            chunk_mass = mx.sum(txt_attn, axis=0)
            chunk_mass = mx.sum(chunk_mass, axis=0)
            chunk_mass = mx.sum(chunk_mass, axis=0)
            token_mass = token_mass + chunk_mass
            query_rows += HEADS * (stop - start)
            mx.eval(token_mass)
        if query_rows <= 0:
            return
        token_mass = token_mass / float(query_rows)
        text_mass_fraction = mx.sum(token_mass)
        token_share = token_mass / (text_mass_fraction + 1e-12)
        mx.eval(token_mass, token_share, text_mass_fraction)
        mass_values = token_mass.tolist()
        share_values = token_share.tolist()
        text_mass_value = float(text_mass_fraction.item())
        probe_s = time.perf_counter() - t0
        for token, (mass, share) in enumerate(zip(mass_values, share_values)):
            self.attention_scout_metrics.append(
                {
                    "step": self.forecast_step_index,
                    "stream": stream,
                    "block": block,
                    "token": token,
                    "token_mass": float(mass),
                    "token_share": float(share),
                    "text_mass_fraction": text_mass_value,
                    "query_rows": query_rows,
                    "probe_s": probe_s,
                }
            )

    def attention_scout_report(self) -> list[dict[str, float | int | str]]:
        return self.attention_scout_metrics

    def reset_kontext_kv_cache(self) -> None:
        self.kontext_kv_cache = {}
        self.kontext_kv_cache_reference_tokens = 0
        self.kontext_kv_cache_hits = 0
        self.kontext_kv_cache_stores = 0

    def set_kontext_kv_cache(self, enabled: bool, reference_tokens: int = 0) -> None:
        self.kontext_kv_cache_enabled = bool(enabled and reference_tokens > 0)
        self.reset_kontext_kv_cache()
        self.kontext_reference_zero_calls = 0
        self.kontext_reference_zero_last = {}
        if self.kontext_kv_cache_enabled:
            self.kontext_kv_cache_reference_tokens = int(reference_tokens)

    def kontext_kv_cache_ready(self) -> bool:
        return self.kontext_kv_cache_enabled and bool(self.kontext_kv_cache)

    def kontext_kv_attention(
        self,
        scope: str,
        index: int,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        reference_tokens: int,
    ) -> mx.array:
        if not self.kontext_kv_cache_enabled or reference_tokens <= 0:
            return attention(q, k, v)
        cache_key = f"{scope}_{index}"
        cached = self.kontext_kv_cache.get(cache_key)
        if cached is not None:
            kk, vv = cached
            self.kontext_kv_cache_hits += 1
            return attention(q, mx.concatenate([k, kk], axis=2), mx.concatenate([v, vv], axis=2))
        if int(k.shape[2]) < reference_tokens:
            return attention(q, k, v)
        kk = mx.contiguous(k[:, :, -reference_tokens:, :])
        vv = mx.contiguous(v[:, :, -reference_tokens:, :])
        mx.eval(kk, vv)
        self.kontext_kv_cache[cache_key] = (kk, vv)
        self.kontext_kv_cache_stores += 1
        return attention(q, k, v)

    def set_forecast_step(self, step: int) -> None:
        self.forecast_step_index = step
        if step == self.forecast_single_current_step:
            return
        self.forecast_single_current_step = step
        self.forecast_single_locked_blocks = self.forecast_single_next_locked_blocks
        self.forecast_single_next_locked_blocks = set()
        self.forecast_single_adaptive_lowest_cache = {}
        self.forecast_single_storm_ref = self.compute_single_storm_ref(step)

    def forecast_single_sensitivity_for_step(self, step: int) -> float:
        return self.forecast_single_adaptive_step_sensitivity.get(
            step, self.forecast_single_adaptive_sensitivity
        )

    def forecast_single_gain_for_step(self, step: int) -> float:
        return self.forecast_single_step_gain.get(step, self.forecast_single_gain)

    def forecast_single_gain_for_scope(self, step: int, index: int, scope: str) -> float:
        if scope == "linear2" and index in self.forecast_single_linear2_block_gain:
            return self.forecast_single_linear2_block_gain[index]
        return self.forecast_single_gain_for_step(step)

    def forecast_single_delta_clamp_for_step(self, step: int) -> float:
        return self.forecast_single_step_delta_clamp.get(step, self.forecast_single_delta_clamp)

    def clamp_single_forecast_delta(self, last_residual: mx.array, delta: mx.array) -> mx.array:
        clamp = self.forecast_single_delta_clamp_for_step(self.forecast_step_index)
        if clamp <= 0:
            return delta
        last_abs = mx.mean(mx.abs(last_residual.astype(mx.float32)))
        delta_abs = mx.mean(mx.abs(delta.astype(mx.float32)))
        limit = last_abs * clamp
        factor = mx.minimum(mx.array(1.0, dtype=mx.float32), limit / (delta_abs + 1e-6))
        return delta * factor

    def single_linear2_norm_summary(self, norm_x: mx.array) -> mx.array:
        mode = self.forecast_single_linear2_shape
        norm_f32 = norm_x.astype(mx.float32)
        if mode == "scalar_rms":
            return mx.sqrt(mx.mean(norm_f32 * norm_f32) + 1e-6)
        if mode == "token_rms":
            return mx.sqrt(mx.mean(norm_f32 * norm_f32, axis=-1, keepdims=True) + 1e-6)
        return mx.array(1.0, dtype=mx.float32)

    def shape_single_linear2_forecast(
        self,
        index: int,
        value: mx.array,
        norm_x: mx.array | None,
        gate: mx.array | None = None,
    ) -> mx.array:
        mode = self.forecast_single_linear2_shape
        if mode == "plain":
            return value
        if self.forecast_single_linear2_shape_blocks is not None and index not in self.forecast_single_linear2_shape_blocks:
            return value
        ratio = None
        if mode in {"scalar_rms", "token_rms"}:
            if norm_x is None:
                return value
            history = self.forecast_single_linear2_norm_history.get(index, [])
            if not history:
                return value
            _, previous_norm = history[-1]
            current_norm = self.single_linear2_norm_summary(norm_x)
            ratio = current_norm / (previous_norm + 1e-6)
        elif mode == "gate_scalar":
            if gate is None:
                return value
            history = self.forecast_single_linear2_gate_history.get(index, [])
            if not history:
                return value
            _, previous_gate = history[-1]
            previous_abs = mx.mean(mx.abs(previous_gate.astype(mx.float32)))
            current_abs = mx.mean(mx.abs(gate.astype(mx.float32)))
            ratio = previous_abs / (current_abs + 1e-6)
        elif mode == "gate_channel":
            if gate is None:
                return value
            history = self.forecast_single_linear2_gate_history.get(index, [])
            if not history:
                return value
            _, previous_gate = history[-1]
            previous_gate = previous_gate.astype(mx.float32)
            current_gate = gate.astype(mx.float32)
            safe_current = mx.where(mx.abs(current_gate) > 1e-4, current_gate, mx.ones_like(current_gate))
            ratio = previous_gate / safe_current
        else:
            return value
        clamp = self.forecast_single_linear2_shape_clamp
        if clamp > 0:
            low = mx.array(max(0.0, 1.0 - clamp), dtype=mx.float32)
            high = mx.array(1.0 + clamp, dtype=mx.float32)
            ratio = mx.minimum(mx.maximum(ratio, low), high)
        if self.forecast_single_hits:
            hit = self.forecast_single_hits[-1]
            if (
                int(hit.get("step", -1)) == self.forecast_step_index
                and int(hit.get("block", -1)) == index
                and str(hit.get("scope", "")) == "linear2"
            ):
                hit["shape"] = mode
                hit["shape_clamp"] = clamp
        if mode == "gate_channel" and ratio.ndim == 2:
            ratio = ratio[:, None, :]
        return value * ratio.astype(value.dtype)

    def align_single_sequence_forecast(self, value: mx.array, target_len: int) -> mx.array:
        source_len = int(value.shape[1])
        if source_len == target_len:
            return value
        current_txt_len = max(0, int(self.token_scout_txt_len))
        image_len = max(0, target_len - current_txt_len)
        source_txt_len = max(0, source_len - image_len)
        if image_len <= 0 or source_txt_len <= 0 or image_len > source_len:
            return value
        txt = value[:, :source_txt_len]
        img = value[:, source_txt_len:]
        if current_txt_len < source_txt_len:
            txt = txt[:, :current_txt_len]
        elif current_txt_len > source_txt_len:
            pad_shape = (int(value.shape[0]), current_txt_len - source_txt_len, int(value.shape[2]))
            txt = mx.concatenate([txt, mx.zeros(pad_shape, dtype=value.dtype)], axis=1)
        return mx.concatenate([txt, img], axis=1)

    def raw_single_forecast_value(
        self, history: list[tuple[int, mx.array]], mode: str
    ) -> tuple[mx.array | None, float]:
        if mode == "hold":
            if not history:
                return None, 0.0
            return history[-1][1], 0.0
        if mode == "linear":
            if len(history) < 2:
                return None, 0.0
            previous_step, previous_value = history[-2]
            last_step, last_value = history[-1]
            previous_value = self.align_single_sequence_forecast(previous_value, int(last_value.shape[1]))
            step_delta = max(1, last_step - previous_step)
            scale = (self.forecast_step_index - last_step) / step_delta
            forecast_gain = self.forecast_single_gain_for_scope(
                self.forecast_step_index, index, scope
            )
            forecast_delta = (last_value - previous_value) * scale * forecast_gain
            forecast_delta = self.clamp_single_forecast_delta(last_value, forecast_delta)
            return last_value + forecast_delta, forecast_gain
        return None, 0.0

    def scout_single_linear2_forecast(
        self,
        index: int,
        actual: mx.array,
        norm_x: mx.array | None,
        gate: mx.array | None = None,
    ) -> None:
        planned = self.forecast_single_linear2_scout_plan.get(self.forecast_step_index)
        if planned is None:
            return
        mode, blocks = planned
        if index not in blocks:
            return
        history = self.forecast_single_linear2_history.get(index, [])
        forecast, forecast_gain = self.raw_single_forecast_value(history, mode)
        if forecast is None:
            return
        optimal_gain = mx.array(0.0, dtype=mx.float32)
        residual_optimal_gain = mx.array(0.0, dtype=mx.float32)
        if mode == "linear" and len(history) >= 2:
            previous_step, previous_value = history[-2]
            last_step, last_value = history[-1]
            step_delta = max(1, last_step - previous_step)
            scale = (self.forecast_step_index - last_step) / step_delta
            direction = (last_value.astype(mx.float32) - previous_value.astype(mx.float32)) * scale
            target = actual.astype(mx.float32) - last_value.astype(mx.float32)
            direction_energy = mx.sum(direction * direction)
            optimal_gain = mx.sum(target * direction) / (direction_energy + 1e-6)
            if gate is not None:
                gate_f32 = gate[:, None].astype(mx.float32)
                residual_direction = direction * gate_f32
                residual_target = target * gate_f32
                residual_energy = mx.sum(residual_direction * residual_direction)
                residual_optimal_gain = mx.sum(residual_target * residual_direction) / (residual_energy + 1e-6)
        forecast = self.shape_single_linear2_forecast(index, forecast, norm_x, gate)
        actual_f32 = actual.astype(mx.float32)
        forecast_f32 = forecast.astype(mx.float32)
        delta = mx.abs(actual_f32 - forecast_f32)
        actual_mean = mx.mean(mx.abs(actual_f32))
        delta_mean = mx.mean(delta)
        rel = delta_mean / (actual_mean + 1e-6)
        txt_len = max(0, min(int(self.token_scout_txt_len), int(actual.shape[1])))
        if txt_len > 0:
            txt_delta = mx.mean(delta[:, :txt_len])
            txt_abs = mx.mean(mx.abs(actual_f32[:, :txt_len]))
            txt_rel = txt_delta / (txt_abs + 1e-6)
        else:
            txt_delta = mx.array(0.0, dtype=mx.float32)
            txt_rel = mx.array(0.0, dtype=mx.float32)
        if txt_len < int(actual.shape[1]):
            img_delta = mx.mean(delta[:, txt_len:])
            img_abs = mx.mean(mx.abs(actual_f32[:, txt_len:]))
            img_rel = img_delta / (img_abs + 1e-6)
        else:
            img_delta = mx.array(0.0, dtype=mx.float32)
            img_rel = mx.array(0.0, dtype=mx.float32)
        if gate is not None:
            gate_f32 = gate[:, None].astype(mx.float32)
            actual_residual = actual_f32 * gate_f32
            forecast_residual = forecast_f32 * gate_f32
            residual_delta = mx.abs(actual_residual - forecast_residual)
            residual_actual_mean = mx.mean(mx.abs(actual_residual))
            residual_delta_mean = mx.mean(residual_delta)
            residual_rel = residual_delta_mean / (residual_actual_mean + 1e-6)
        else:
            residual_actual_mean = mx.array(0.0, dtype=mx.float32)
            residual_delta_mean = mx.array(0.0, dtype=mx.float32)
            residual_rel = mx.array(0.0, dtype=mx.float32)
        mx.eval(
            actual_mean,
            delta_mean,
            rel,
            txt_delta,
            txt_rel,
            img_delta,
            img_rel,
            residual_actual_mean,
            residual_delta_mean,
            residual_rel,
            optimal_gain,
            residual_optimal_gain,
        )
        self.forecast_single_linear2_scout_metrics.append(
            {
                "step": self.forecast_step_index,
                "block": index,
                "mode": mode,
                "shape": self.forecast_single_linear2_shape,
                "gain": forecast_gain,
                "actual_mean_abs": float(actual_mean.item()),
                "delta_mean_abs": float(delta_mean.item()),
                "rel_l1": float(rel.item()),
                "txt_delta_mean_abs": float(txt_delta.item()),
                "txt_rel_l1": float(txt_rel.item()),
                "img_delta_mean_abs": float(img_delta.item()),
                "img_rel_l1": float(img_rel.item()),
                "residual_actual_mean_abs": float(residual_actual_mean.item()),
                "residual_delta_mean_abs": float(residual_delta_mean.item()),
                "residual_rel_l1": float(residual_rel.item()),
                "optimal_gain": float(optimal_gain.item()),
                "residual_optimal_gain": float(residual_optimal_gain.item()),
            }
        )

    def single_storm_ref_for_weather_step(self, weather_step: int) -> float | None:
        rel_values = [
            float(metric["rel_l1"])
            for block, metric in self.forecast_single_weather_latest.items()
            if block in self.forecast_single_adaptive_blocks and int(metric["step"]) == weather_step
        ]
        if not rel_values:
            return None
        top_count = max(1, min(len(rel_values), self.forecast_single_adaptive_top_count))
        top_values = sorted(rel_values, reverse=True)[:top_count]
        return sum(top_values) / len(top_values)

    def compute_single_storm_ref(self, step: int) -> float | None:
        previous_step = step - 1
        storm_ref = self.single_storm_ref_for_weather_step(previous_step)
        if storm_ref is None:
            return None
        rel_values = [
            float(metric["rel_l1"])
            for block, metric in self.forecast_single_weather_latest.items()
            if block in self.forecast_single_adaptive_blocks and int(metric["step"]) == previous_step
        ]
        top_count = max(1, min(len(rel_values), self.forecast_single_adaptive_top_count))
        sensitivity = self.forecast_single_sensitivity_for_step(step)
        self.forecast_single_storm_metrics.append(
            {
                "step": step,
                "weather_step": previous_step,
                "storm_ref": storm_ref,
                "top_count": top_count,
                "allowed": storm_ref * sensitivity,
                "sensitivity": sensitivity,
                "spread": self.forecast_single_adaptive_spread,
            }
        )
        return storm_ref

    def spread_single_rel_l1(self, rel_l1: float, storm_ref: float | None) -> float:
        if storm_ref is None or storm_ref <= 1e-6:
            return rel_l1
        spread = max(0.01, self.forecast_single_adaptive_spread)
        ratio = max(0.0, rel_l1 / storm_ref)
        return storm_ref * (ratio**spread)

    def forecast_single_lowest_blocks_for_step(self, step: int) -> set[int] | None:
        count = self.forecast_single_adaptive_step_lowest_count.get(step)
        if count is None:
            return None
        if count <= 0:
            return set()
        if self.forecast_single_adaptive_low_pool_split:
            if not self.forecast_single_adaptive_low_pool_steps:
                self.build_single_low_pool_split()
            return self.forecast_single_adaptive_low_pool_steps.get(step, set())
        cached = self.forecast_single_adaptive_lowest_cache.get(step)
        if cached is not None:
            return cached
        previous_step = step - 1
        values = []
        for block, metric in self.forecast_single_weather_latest.items():
            if block in self.forecast_single_locked_blocks:
                continue
            if block not in self.forecast_single_adaptive_blocks:
                continue
            if int(metric["step"]) != previous_step:
                continue
            rel_l1 = float(metric["rel_l1"])
            if math.isfinite(rel_l1):
                values.append((rel_l1, block))
        values.sort()
        selected = {block for _, block in values[:count]}
        self.forecast_single_adaptive_lowest_cache[step] = selected
        return selected

    def build_single_low_pool_split(self) -> None:
        if not self.forecast_single_adaptive_step_lowest_count:
            return
        source_step = self.forecast_single_adaptive_low_pool_source_step
        if source_step <= 0:
            first_forecast_step = min(self.forecast_single_adaptive_step_lowest_count)
            source_step = first_forecast_step - 1
            self.forecast_single_adaptive_low_pool_source_step = source_step
        values = []
        for block, metric in self.forecast_single_weather_latest.items():
            if block not in self.forecast_single_adaptive_blocks:
                continue
            if int(metric["step"]) != source_step:
                continue
            rel_l1 = float(metric["rel_l1"])
            if math.isfinite(rel_l1):
                values.append((rel_l1, block))
        values.sort()
        offset = 0
        for forecast_step, count in sorted(self.forecast_single_adaptive_step_lowest_count.items()):
            if count <= 0:
                self.forecast_single_adaptive_low_pool_steps[forecast_step] = set()
                continue
            group = values[offset : offset + count]
            self.forecast_single_adaptive_low_pool_steps[forecast_step] = {block for _, block in group}
            offset += count

    def forecast_single_weather_step_for_forecast(self, step: int) -> int:
        if self.forecast_single_adaptive_low_pool_split:
            source_step = self.forecast_single_adaptive_low_pool_source_step
            if source_step > 0 and step in self.forecast_single_adaptive_low_pool_steps:
                return source_step
        return step - 1

    def forecast_single_tracked_blocks(self) -> set[int]:
        blocks = set(self.forecast_single_blocks)
        for _, planned_blocks in self.forecast_single_plan.values():
            blocks.update(planned_blocks)
        for _, planned_blocks in self.forecast_single_attention_plan.values():
            blocks.update(planned_blocks)
        for _, planned_blocks in self.forecast_single_linear2_plan.values():
            blocks.update(planned_blocks)
        for _, planned_blocks in self.forecast_single_linear2_late_plan.values():
            blocks.update(planned_blocks)
        blocks.update(self.forecast_single_adaptive_blocks)
        blocks.update(self.forecast_single_adaptive_step2_blocks)
        return blocks

    def forecast_double_img_mlp_gain_for_step(self, step: int) -> float:
        return self.forecast_double_img_mlp_step_gain.get(step, self.forecast_double_img_mlp_gain)

    def forecast_double_img_mlp_tracked_blocks(self) -> set[int]:
        blocks = set(self.forecast_double_img_mlp_blocks)
        for _, planned_blocks in self.forecast_double_img_mlp_plan.values():
            blocks.update(planned_blocks)
        return blocks

    def record_double_img_mlp_residual(self, index: int, residual: mx.array) -> None:
        if index not in self.forecast_double_img_mlp_tracked_blocks():
            return
        history = self.forecast_double_img_mlp_history.setdefault(index, [])
        history.append((self.forecast_step_index, residual))
        if len(history) > 3:
            del history[:-3]

    def forecast_double_img_mlp_residual(self, index: int) -> mx.array | None:
        scoped_plan = self.forecast_double_img_mlp_plan
        if scoped_plan:
            planned = scoped_plan.get(self.forecast_step_index)
            if planned is None:
                return None
            mode, blocks = planned
            if index not in blocks:
                return None
        else:
            mode = self.forecast_double_img_mlp_mode
            if mode == "off":
                return None
            if index not in self.forecast_double_img_mlp_blocks:
                return None
            if self.forecast_step_index not in self.forecast_double_img_mlp_steps:
                return None
        history = self.forecast_double_img_mlp_history.get(index, [])
        if mode == "hold":
            if not history:
                return None
            self.forecast_double_img_mlp_hits.append(
                {"step": self.forecast_step_index, "block": index, "mode": mode, "gain": 0.0}
            )
            return history[-1][1]
        if mode == "linear":
            if len(history) < 2:
                return None
            previous_step, previous_residual = history[-2]
            last_step, last_residual = history[-1]
            step_delta = max(1, last_step - previous_step)
            scale = (self.forecast_step_index - last_step) / step_delta
            gain = self.forecast_double_img_mlp_gain_for_step(self.forecast_step_index)
            forecast = last_residual + (last_residual - previous_residual) * scale * gain
            self.forecast_double_img_mlp_hits.append(
                {"step": self.forecast_step_index, "block": index, "mode": mode, "gain": gain}
            )
            return forecast
        return None

    def forecast_double_txt_gain_for_step(self, step: int) -> float:
        return self.forecast_double_txt_step_gain.get(step, self.forecast_double_txt_gain)

    def forecast_double_txt_tracked_blocks(self) -> set[int]:
        blocks = set(self.forecast_double_txt_blocks)
        for _, planned_blocks in self.forecast_double_txt_plan.values():
            blocks.update(planned_blocks)
        return blocks

    def record_double_txt_residual(self, index: int, residual: mx.array) -> None:
        if index not in self.forecast_double_txt_tracked_blocks():
            return
        history = self.forecast_double_txt_history.setdefault(index, [])
        history.append((self.forecast_step_index, residual))
        if len(history) > 3:
            del history[:-3]

    def forecast_double_txt_residual(self, index: int) -> mx.array | None:
        scoped_plan = self.forecast_double_txt_plan
        if scoped_plan:
            planned = scoped_plan.get(self.forecast_step_index)
            if planned is None:
                return None
            mode, blocks = planned
            if index not in blocks:
                return None
        else:
            mode = self.forecast_double_txt_mode
            if mode == "off":
                return None
            if index not in self.forecast_double_txt_blocks:
                return None
            if self.forecast_step_index not in self.forecast_double_txt_steps:
                return None
        history = self.forecast_double_txt_history.get(index, [])
        if mode == "hold":
            if not history:
                return None
            self.forecast_double_txt_hits.append(
                {"step": self.forecast_step_index, "block": index, "mode": mode, "gain": 0.0}
            )
            return history[-1][1]
        if mode == "linear":
            if len(history) < 2:
                return None
            previous_step, previous_residual = history[-2]
            last_step, last_residual = history[-1]
            step_delta = max(1, last_step - previous_step)
            scale = (self.forecast_step_index - last_step) / step_delta
            gain = self.forecast_double_txt_gain_for_step(self.forecast_step_index)
            forecast = last_residual + (last_residual - previous_residual) * scale * gain
            self.forecast_double_txt_hits.append(
                {"step": self.forecast_step_index, "block": index, "mode": mode, "gain": gain}
            )
            return forecast
        return None

    def adaptive_single_value(
        self, index: int, history: list[tuple[int, mx.array]], scope: str
    ) -> mx.array | None:
        if index in self.forecast_single_locked_blocks:
            return None
        mode = "off"
        rel_l1 = None
        risk_score = None
        allowed = None
        storm_ref = self.forecast_single_storm_ref
        sensitivity = self.forecast_single_sensitivity_for_step(self.forecast_step_index)
        forecast_gain = self.forecast_single_gain_for_scope(
            self.forecast_step_index, index, scope
        )
        if sensitivity <= 0:
            return None
        if self.forecast_step_index == 2 and index in self.forecast_single_adaptive_step2_blocks:
            mode = "hold"
        elif self.forecast_step_index >= 3 and index in self.forecast_single_adaptive_blocks:
            lowest_blocks = self.forecast_single_lowest_blocks_for_step(self.forecast_step_index)
            if self.forecast_single_adaptive_step_lowest_count and lowest_blocks is None:
                return None
            if lowest_blocks is not None and index not in lowest_blocks:
                return None
            weather_step = self.forecast_single_weather_step_for_forecast(self.forecast_step_index)
            if weather_step != self.forecast_step_index - 1:
                storm_ref = self.single_storm_ref_for_weather_step(weather_step)
            if storm_ref is None:
                return None
            weather = self.forecast_single_weather_latest.get(index)
            if weather is None or int(weather["step"]) != weather_step:
                return None
            rel_l1 = float(weather["rel_l1"])
            risk_score = self.spread_single_rel_l1(rel_l1, storm_ref)
            allowed = storm_ref * sensitivity
            if risk_score > allowed:
                return None
            mode = "linear"
        if mode == "hold":
            if not history:
                return None
            self.forecast_single_next_locked_blocks.add(index)
            self.forecast_single_hits.append(
                {
                    "step": self.forecast_step_index,
                    "block": index,
                    "scope": scope,
                    "mode": mode,
                    "rel_l1": -1.0,
                    "risk_score": -1.0,
                    "storm_ref": storm_ref if storm_ref is not None else -1.0,
                    "allowed": allowed if allowed is not None else -1.0,
                }
            )
            return history[-1][1]
        if mode == "linear":
            if len(history) < 2:
                return None
            previous_step, previous_residual = history[-2]
            last_step, last_residual = history[-1]
            step_delta = max(1, last_step - previous_step)
            scale = (self.forecast_step_index - last_step) / step_delta
            forecast_delta = (last_residual - previous_residual) * scale * forecast_gain
            forecast_delta = self.clamp_single_forecast_delta(last_residual, forecast_delta)
            self.forecast_single_next_locked_blocks.add(index)
            self.forecast_single_hits.append(
                {
                    "step": self.forecast_step_index,
                    "block": index,
                    "scope": scope,
                    "mode": mode,
                    "rel_l1": rel_l1 if rel_l1 is not None else -1.0,
                    "risk_score": risk_score if risk_score is not None else -1.0,
                    "storm_ref": storm_ref if storm_ref is not None else -1.0,
                    "allowed": allowed if allowed is not None else -1.0,
                    "gain": forecast_gain,
                    "delta_clamp": self.forecast_single_delta_clamp_for_step(self.forecast_step_index),
                }
            )
            return last_residual + forecast_delta
        return None

    def planned_single_value(
        self, index: int, history: list[tuple[int, mx.array]], scope: str
    ) -> mx.array | None:
        scoped_plan = self.forecast_single_plan
        if scope == "attention" and self.forecast_single_attention_plan:
            scoped_plan = self.forecast_single_attention_plan
        elif scope == "linear2" and self.forecast_single_linear2_plan:
            scoped_plan = self.forecast_single_linear2_plan
        elif scope == "linear2_late" and self.forecast_single_linear2_late_plan:
            scoped_plan = self.forecast_single_linear2_late_plan
        if scoped_plan:
            planned = scoped_plan.get(self.forecast_step_index)
            if planned is None:
                return None
            mode, blocks = planned
            if index not in blocks:
                return None
        else:
            mode = self.forecast_single_mode
            blocks = self.forecast_single_blocks
            if mode == "off":
                return None
            if index not in blocks:
                return None
            if self.forecast_step_index not in self.forecast_single_steps:
                return None
        if mode == "hold":
            if not history:
                return None
            self.forecast_single_hits.append(
                {"step": self.forecast_step_index, "block": index, "scope": scope, "mode": mode}
            )
            return history[-1][1]
        if mode == "linear":
            if len(history) < 2:
                return None
            previous_step, previous_value = history[-2]
            last_step, last_value = history[-1]
            previous_value = self.align_single_sequence_forecast(previous_value, int(last_value.shape[1]))
            step_delta = max(1, last_step - previous_step)
            scale = (self.forecast_step_index - last_step) / step_delta
            forecast_gain = self.forecast_single_gain_for_scope(
                self.forecast_step_index, index, scope
            )
            forecast_delta = (last_value - previous_value) * scale * forecast_gain
            forecast_delta = self.clamp_single_forecast_delta(last_value, forecast_delta)
            self.forecast_single_hits.append(
                {
                    "step": self.forecast_step_index,
                    "block": index,
                    "scope": scope,
                    "mode": mode,
                    "gain": forecast_gain,
                    "delta_clamp": self.forecast_single_delta_clamp_for_step(self.forecast_step_index),
                }
            )
            return last_value + forecast_delta
        return None

    def adaptive_single_residual(self, index: int) -> mx.array | None:
        history = self.forecast_single_history.get(index, [])
        return self.adaptive_single_value(index, history, "residual")

    def forecast_single_linear2(
        self,
        index: int,
        norm_x: mx.array | None = None,
        gate: mx.array | None = None,
    ) -> mx.array | None:
        if self.forecast_single_scope not in {"linear2", "attention_linear2"}:
            return None
        history = self.forecast_single_linear2_history.get(index, [])
        if self.forecast_single_adaptive:
            value = self.adaptive_single_value(index, history, "linear2")
        else:
            value = self.planned_single_value(index, history, "linear2")
        if value is None:
            return None
        if norm_x is not None:
            value = self.align_single_sequence_forecast(value, int(norm_x.shape[1]))
        return self.shape_single_linear2_forecast(index, value, norm_x, gate)

    def forecast_single_linear2_late(
        self,
        index: int,
        sequence_len: int,
        gate: mx.array | None = None,
    ) -> mx.array | None:
        if self.forecast_single_scope != "linear2_late":
            return None
        history = self.forecast_single_linear2_history.get(index, [])
        value = self.planned_single_value(index, history, "linear2_late")
        if value is None:
            return None
        value = self.align_single_sequence_forecast(value, sequence_len)
        return self.shape_single_linear2_forecast(index, value, None, gate)

    def blend_single_linear2_late(self, actual: mx.array, forecast: mx.array) -> mx.array:
        split = str(self.forecast_single_linear2_late_split or "all")
        if split == "all":
            return forecast
        txt_len = max(0, min(int(self.token_scout_txt_len), int(actual.shape[1]), int(forecast.shape[1])))
        if txt_len <= 0:
            return forecast
        if split == "img":
            return mx.concatenate([actual[:, :txt_len], forecast[:, txt_len:]], axis=1)
        if split == "txt":
            return mx.concatenate([forecast[:, :txt_len], actual[:, txt_len:]], axis=1)
        return forecast

    def partial_single_linear2_text_real(self, attn_mlp: mx.array, forecast: mx.array, prefix: str) -> mx.array:
        txt_len = max(0, min(int(self.token_scout_txt_len), int(attn_mlp.shape[1]), int(forecast.shape[1])))
        if txt_len <= 0:
            return forecast
        text_out = self.linear(attn_mlp[:, :txt_len], prefix)
        self.forecast_single_linear2_partial_hits += 1
        return mx.concatenate([text_out, forecast[:, txt_len:]], axis=1)

    def record_single_linear2(
        self,
        index: int,
        value: mx.array,
        norm_x: mx.array | None = None,
        gate: mx.array | None = None,
    ) -> None:
        if self.forecast_single_scope not in {"linear2", "attention_linear2", "linear2_late"}:
            return
        tracked_blocks = self.forecast_single_tracked_blocks()
        scout_blocks = set()
        for _, planned_blocks in self.forecast_single_linear2_scout_plan.values():
            scout_blocks.update(planned_blocks)
        if index not in tracked_blocks and index not in scout_blocks:
            return
        self.scout_single_linear2_forecast(index, value, norm_x, gate)
        history = self.forecast_single_linear2_history.setdefault(index, [])
        history.append((self.forecast_step_index, value))
        if len(history) > 2:
            del history[:-2]
        if self.forecast_single_linear2_shape != "plain" and norm_x is not None:
            norm_history = self.forecast_single_linear2_norm_history.setdefault(index, [])
            norm_history.append((self.forecast_step_index, self.single_linear2_norm_summary(norm_x)))
            if len(norm_history) > 2:
                del norm_history[:-2]
        if gate is not None:
            gate_history = self.forecast_single_linear2_gate_history.setdefault(index, [])
            gate_history.append((self.forecast_step_index, gate))
            if len(gate_history) > 2:
                del gate_history[:-2]

    def forecast_single_attention(self, index: int) -> mx.array | None:
        if self.forecast_single_scope not in {"attention", "attention_linear2"}:
            return None
        history = self.forecast_single_attention_history.get(index, [])
        if self.forecast_single_adaptive:
            return self.adaptive_single_value(index, history, "attention")
        return self.planned_single_value(index, history, "attention")

    def record_single_attention(self, index: int, value: mx.array) -> None:
        if self.forecast_single_scope not in {"attention", "attention_linear2"}:
            return
        tracked_blocks = self.forecast_single_tracked_blocks()
        if index not in tracked_blocks:
            return
        history = self.forecast_single_attention_history.setdefault(index, [])
        history.append((self.forecast_step_index, value))
        if len(history) > 2:
            del history[:-2]

    def forecast_single_residual(self, index: int) -> mx.array | None:
        if self.forecast_single_scope != "residual":
            return None
        if self.forecast_single_adaptive:
            return self.adaptive_single_residual(index)
        history = self.forecast_single_history.get(index, [])
        return self.planned_single_value(index, history, "residual")

    def record_single_residual(self, index: int, residual: mx.array) -> None:
        tracked_blocks = self.forecast_single_tracked_blocks()
        if (
            self.forecast_single_mode == "off"
            and not self.forecast_single_plan
            and not self.forecast_single_adaptive
            and not self.forecast_single_weather_debug
        ):
            return
        if index not in tracked_blocks:
            return
        needs_weather = not (
            self.forecast_single_last_weather_step
            and self.forecast_step_index > self.forecast_single_last_weather_step
            and not self.forecast_single_weather_debug
        )
        history = self.forecast_single_history.setdefault(index, [])
        if (self.forecast_single_adaptive or self.forecast_single_weather_debug) and needs_weather:
            residual_f32 = residual.astype(mx.float32)
            mean_abs = mx.mean(mx.abs(residual_f32))
            previous = history[-1][1].astype(mx.float32) if history else None
            if previous is None:
                mx.eval(mean_abs)
                metric = {
                    "step": self.forecast_step_index,
                    "block": index,
                    "mean_abs": float(mean_abs.item()),
                    "delta_mean_abs": 0.0,
                    "rel_l1": 0.0,
                }
            else:
                previous = self.align_single_sequence_forecast(previous, int(residual_f32.shape[1]))
                if tuple(previous.shape) != tuple(residual_f32.shape):
                    mx.eval(mean_abs)
                    metric = {
                        "step": self.forecast_step_index,
                        "block": index,
                        "mean_abs": float(mean_abs.item()),
                        "delta_mean_abs": 0.0,
                        "rel_l1": 0.0,
                    }
                    history.clear()
                    self.forecast_single_weather_latest.pop(index, None)
                    self.forecast_single_storm_ref = None
                    self.forecast_single_adaptive_lowest_cache.clear()
                    self.forecast_single_next_locked_blocks.discard(index)
                    if self.forecast_single_weather_debug:
                        self.forecast_single_weather_metrics.append(metric)
                    history.append((self.forecast_step_index, residual))
                    return
                delta = mx.abs(residual_f32 - previous)
                delta_mean_abs = mx.mean(delta)
                previous_metric = self.forecast_single_weather_latest.get(index)
                previous_mean_abs = (
                    float(previous_metric["mean_abs"])
                    if previous_metric is not None
                    and int(previous_metric["step"]) == self.forecast_step_index - 1
                    else None
                )
                if previous_mean_abs is None:
                    denominator = mx.mean(mx.abs(previous))
                else:
                    denominator = previous_mean_abs
                rel_l1 = delta_mean_abs / (denominator + 1e-6)
                mx.eval(mean_abs, delta_mean_abs, rel_l1)
                metric = {
                    "step": self.forecast_step_index,
                    "block": index,
                    "mean_abs": float(mean_abs.item()),
                    "delta_mean_abs": float(delta_mean_abs.item()),
                    "rel_l1": float(rel_l1.item()),
                }
            self.forecast_single_weather_latest[index] = metric
            if self.forecast_single_weather_debug:
                self.forecast_single_weather_metrics.append(metric)
        history.append((self.forecast_step_index, residual))
        if len(history) > 2:
            del history[:-2]

    def apply_lora_adapters(self, output: mx.array, input_value: mx.array, weight_key: str) -> mx.array:
        adapters = self.lora_adapters.get(weight_key)
        if not adapters:
            return output
        for down_t, up_t, start, length in adapters:
            delta = (input_value @ down_t) @ up_t
            if delta.dtype != output.dtype:
                delta = delta.astype(output.dtype)
            if start is None:
                output = output + delta
                continue
            end = start + (length or int(delta.shape[-1]))
            output = output.at[..., start:end].add(delta)
        return output

    def linear(self, x: mx.array, prefix: str) -> mx.array:
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        input_value = x

        def with_lora(output: mx.array) -> mx.array:
            return self.apply_lora_adapters(output, input_value, weight_key)

        if weight_key in self.fp8_mxfp8:
            q_weight, scales = self.fp8_mxfp8[weight_key]
            x = mx.quantized_matmul(
                x,
                q_weight,
                scales=scales,
                transpose=True,
                group_size=32,
                bits=8,
                mode="mxfp8",
            )
            if bias_key in self.w:
                x = x + self.w[bias_key]
            return with_lora(x)
        if weight_key in self.gguf_q8:
            q_weight, scales, quant_biases, _shape = self.gguf_q8[weight_key]
            in_dim = int(_shape[1])
            packed_cols = int(q_weight.shape[1])
            values_per_pack = in_dim // packed_cols
            bits = 32 // values_per_pack
            if (
                SINGLE_LINEAR1_FLAT
                and prefix.startswith("single_blocks.")
                and prefix.endswith(".linear1")
                and x.ndim == 3
                and x.shape[0] == 1
            ):
                x2 = mx.reshape(x, (x.shape[1], x.shape[2]))
                out = mx.quantized_matmul(
                    x2,
                    q_weight,
                    scales=scales,
                    biases=quant_biases,
                    transpose=True,
                    group_size=32,
                    bits=bits,
                    mode="affine",
                )
                if bias_key in self.w:
                    out = out + self.w[bias_key]
                return with_lora(mx.reshape(out, (1, x.shape[1], -1)))
            if (
                SINGLE_LINEAR2_FLAT
                and prefix.startswith("single_blocks.")
                and prefix.endswith(".linear2")
                and x.ndim == 3
                and x.shape[0] == 1
            ):
                x2 = mx.reshape(x, (x.shape[1], x.shape[2]))
                out = mx.quantized_matmul(
                    x2,
                    q_weight,
                    scales=scales,
                    biases=quant_biases,
                    transpose=True,
                    group_size=32,
                    bits=bits,
                    mode="affine",
                )
                if bias_key in self.w:
                    out = out + self.w[bias_key]
                return with_lora(mx.reshape(out, (1, x.shape[1], -1)))
            x = mx.quantized_matmul(
                x,
                q_weight,
                scales=scales,
                biases=quant_biases,
                transpose=True,
                group_size=32,
                bits=bits,
                mode="affine",
            )
            if bias_key in self.w:
                x = x + self.w[bias_key]
            return with_lora(x)
        if weight_key in self.fp8_weight_keys:
            if weight_key in self.wt:
                if (
                    FP8_FUSED_LINEAR_FN is not None
                    and self.precision == mx.float16
                    and bias_key in self.w
                    and x.ndim in {2, 3}
                ):
                    if x.ndim == 3 and x.shape[0] == 1:
                        m, k = int(x.shape[1]), int(x.shape[2])
                        x2 = mx.reshape(x, (m, k))
                        k2, n = (int(dim) for dim in self.wt[weight_key].shape)
                        if m >= 64 and k == k2 and k % 32 == 0 and n % 64 == 0:
                            out = FP8_FUSED_LINEAR_FN(x2, self.wt[weight_key], self.w[bias_key], FP8_FUSED_LINEAR_LUT)
                            return with_lora(mx.reshape(out, (1, m, n)))
                    elif x.ndim == 2:
                        m, k = (int(dim) for dim in x.shape)
                        k2, n = (int(dim) for dim in self.wt[weight_key].shape)
                        if m >= 64 and k == k2 and k % 32 == 0 and n % 64 == 0:
                            return with_lora(FP8_FUSED_LINEAR_FN(x, self.wt[weight_key], self.w[bias_key], FP8_FUSED_LINEAR_LUT))
                weight = mx.from_fp8(self.wt[weight_key], self.precision)
                if (
                    SINGLE_LINEAR1_FLAT
                    and prefix.startswith("single_blocks.")
                    and prefix.endswith(".linear1")
                    and x.ndim == 3
                    and x.shape[0] == 1
                ):
                    x2 = mx.reshape(x, (x.shape[1], x.shape[2]))
                    if bias_key in self.w:
                        return with_lora(mx.reshape(mx.addmm(self.w[bias_key], x2, weight), (1, x.shape[1], -1)))
                    return with_lora(mx.reshape(x2 @ weight, (1, x.shape[1], -1)))
                if (
                    SINGLE_LINEAR2_FLAT
                    and prefix.startswith("single_blocks.")
                    and prefix.endswith(".linear2")
                    and x.ndim == 3
                    and x.shape[0] == 1
                ):
                    x2 = mx.reshape(x, (x.shape[1], x.shape[2]))
                    if bias_key in self.w:
                        return with_lora(mx.reshape(mx.addmm(self.w[bias_key], x2, weight), (1, x.shape[1], -1)))
                    return with_lora(mx.reshape(x2 @ weight, (1, x.shape[1], -1)))
                if bias_key in self.w:
                    return with_lora(mx.addmm(self.w[bias_key], x, weight))
                return with_lora(x @ weight)
            weight = mx.from_fp8(self.w[weight_key], self.precision)
            if bias_key in self.w:
                return with_lora(mx.addmm(self.w[bias_key], x, weight.T))
            return with_lora(x @ weight.T)
        if weight_key in self.q:
            weight, scales, quant_biases, group_size, bits = self.q[weight_key]
            x = mx.quantized_matmul(
                x,
                weight,
                scales=scales,
                biases=quant_biases,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode="affine",
            )
            if bias_key in self.w:
                x = x + self.w[bias_key]
            return with_lora(x)
        if weight_key in self.wt:
            if (
                SINGLE_LINEAR1_FLAT
                and prefix.startswith("single_blocks.")
                and prefix.endswith(".linear1")
                and x.ndim == 3
                and x.shape[0] == 1
            ):
                x2 = mx.reshape(x, (x.shape[1], x.shape[2]))
                if bias_key in self.w:
                    return with_lora(mx.reshape(mx.addmm(self.w[bias_key], x2, self.wt[weight_key]), (1, x.shape[1], -1)))
                return with_lora(mx.reshape(x2 @ self.wt[weight_key], (1, x.shape[1], -1)))
            if (
                SINGLE_LINEAR2_FLAT
                and prefix.startswith("single_blocks.")
                and prefix.endswith(".linear2")
                and x.ndim == 3
                and x.shape[0] == 1
            ):
                x2 = mx.reshape(x, (x.shape[1], x.shape[2]))
                if bias_key in self.w:
                    return with_lora(mx.reshape(mx.addmm(self.w[bias_key], x2, self.wt[weight_key]), (1, x.shape[1], -1)))
                return with_lora(mx.reshape(x2 @ self.wt[weight_key], (1, x.shape[1], -1)))
            if bias_key in self.w:
                return with_lora(mx.addmm(self.w[bias_key], x, self.wt[weight_key]))
            return with_lora(x @ self.wt[weight_key])
        return with_lora(linear(x, self.k(weight_key), self.w.get(bias_key)))

    @staticmethod
    def quant_scope_matches(key: str, scope: str) -> bool:
        if scope == "all":
            return True
        if scope == "blocks":
            return key.startswith(("double_blocks.", "single_blocks."))
        if scope == "heavy":
            if key.startswith("double_blocks."):
                return any(
                    marker in key
                    for marker in (
                        ".img_attn.qkv.",
                        ".img_attn.proj.",
                        ".txt_attn.qkv.",
                        ".txt_attn.proj.",
                        ".img_mlp.",
                        ".txt_mlp.",
                    )
                )
            if key.startswith("single_blocks."):
                return ".linear1." in key or ".linear2." in key
            return False
        if scope == "double":
            return key.startswith("double_blocks.")
        if scope == "single":
            return key.startswith("single_blocks.")
        if scope == "io":
            return key.startswith(("img_in.", "txt_in.", "time_in.", "vector_in.", "final_layer."))
        raise ValueError(f"Unsupported quantization scope: {scope}")

    def quantize_linears(self, *, bits: int, group_size: int = 64, scope: str = "all") -> int:
        count = 0
        for key, value in self.w.items():
            if not key.endswith(".weight") or len(value.shape) != 2:
                continue
            if key in self.fp8_weight_keys:
                continue
            if not self.quant_scope_matches(key, scope):
                continue
            if value.shape[1] <= 64:
                continue
            q_weight, scales, *quant_biases = mx.quantize(value, group_size, bits, mode="affine")
            self.q[key] = (q_weight, scales, quant_biases[0] if quant_biases else None, group_size, bits)
            count += 1
        mx.eval(self.q)
        return count

    def mlp(self, x: mx.array, prefix: str, *, approx: bool = False) -> mx.array:
        x = self.linear(x, f"{prefix}.0")
        x = nn.gelu_approx(x) if approx else nn.gelu(x)
        return self.linear(x, f"{prefix}.2")

    def modulate_double(self, vec: mx.array, prefix: str) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        y = self.linear(nn.silu(vec), f"{prefix}.lin")
        chunk = HIDDEN_DIM
        return (
            y[:, 0 * chunk : 1 * chunk],
            y[:, 1 * chunk : 2 * chunk],
            y[:, 2 * chunk : 3 * chunk],
            y[:, 3 * chunk : 4 * chunk],
            y[:, 4 * chunk : 5 * chunk],
            y[:, 5 * chunk : 6 * chunk],
        )

    def modulate_single(self, vec: mx.array, prefix: str) -> tuple[mx.array, mx.array, mx.array]:
        y = self.linear(nn.silu(vec), f"{prefix}.lin")
        chunk = HIDDEN_DIM
        return (
            y[:, 0 * chunk : 1 * chunk],
            y[:, 1 * chunk : 2 * chunk],
            y[:, 2 * chunk : 3 * chunk],
        )

    def build_modulation_cache(
        self,
        *,
        step_values: mx.array,
        pooled_prompt_embeds: mx.array,
        guidance: float = 0.0,
    ) -> None:
        steps = int(step_values.shape[0])
        pooled = mx.broadcast_to(pooled_prompt_embeds, (steps, pooled_prompt_embeds.shape[-1]))
        guidance_arr = mx.full((steps,), guidance, dtype=self.precision)
        vec = self.time_text_embed(step_values.astype(self.precision), pooled, guidance_arr)
        double = []
        for i in range(DOUBLE_BLOCKS):
            base = f"double_blocks.{i}"
            double.append(
                (
                    self.modulate_double(vec, f"{base}.img_mod"),
                    self.modulate_double(vec, f"{base}.txt_mod"),
                )
            )
        single = []
        for i in range(SINGLE_BLOCKS):
            single.append(self.modulate_single(vec, f"single_blocks.{i}.modulation"))
        final = mx.split(self.linear(nn.silu(vec), "final_layer.adaLN_modulation.1"), 2, axis=-1)
        self.modulation_cache = {"vec": vec, "double": double, "single": single, "final": final}
        mx.eval(self.modulation_cache)

    def pooled_text_embed(self, pooled_projection: mx.array) -> mx.array:
        return self.linear(nn.silu(self.linear(pooled_projection, "vector_in.in_layer")), "vector_in.out_layer")

    def time_text_embed(
        self,
        time_step: mx.array,
        pooled_projection: mx.array,
        guidance: mx.array | None = None,
        pooled_embed: mx.array | None = None,
    ) -> mx.array:
        time_steps = time_proj(time_step)
        time_emb = self.linear(nn.silu(self.linear(time_steps, "time_in.in_layer")), "time_in.out_layer")
        pooled = pooled_embed if pooled_embed is not None else self.pooled_text_embed(pooled_projection)
        if guidance is not None and "guidance_in.in_layer.weight" in self.w:
            guidance_emb = time_proj(guidance)
            guidance_emb = self.linear(nn.silu(self.linear(guidance_emb, "guidance_in.in_layer")), "guidance_in.out_layer")
            time_emb = time_emb + guidance_emb
        return (time_emb + pooled).astype(self.precision)

    def time_text_embed_with_reference_zero(
        self,
        time_step: mx.array,
        pooled_projection: mx.array,
        guidance: mx.array,
        pooled_embed: mx.array | None = None,
    ) -> mx.array:
        step_values = mx.concatenate([time_step, time_step * 0], axis=0).astype(self.precision)
        guidance_values = mx.broadcast_to(guidance[0], (2,)).astype(self.precision)
        return self.time_text_embed(step_values, pooled_projection, guidance_values, pooled_embed=pooled_embed)

    def rotary(self, height: int, width: int, txt_len: int) -> mx.array:
        key = (height, width, txt_len)
        if key not in self._rotary_cache:
            ids = mx.concatenate((prepare_text_ids(txt_len), prepare_latent_image_ids(height, width)), axis=1)
            emb = embed_nd(ids)
            mx.eval(emb)
            self._rotary_cache[key] = emb
        return self._rotary_cache[key]

    def rotary_kontext(
        self,
        height: int,
        width: int,
        txt_len: int,
        reference_height: int,
        reference_width: int,
    ) -> mx.array:
        key = ("kontext", height, width, txt_len, reference_height, reference_width)
        if key not in self._rotary_cache:
            ids = mx.concatenate(
                (
                    prepare_text_ids(txt_len),
                    prepare_latent_image_ids(height, width),
                    prepare_kontext_image_ids(reference_height, reference_width),
                ),
                axis=1,
            )
            emb = embed_nd(ids)
            mx.eval(emb)
            self._rotary_cache[key] = emb
        return self._rotary_cache[key]

    def double_block(
        self,
        index: int,
        img: mx.array,
        txt: mx.array,
        vec: mx.array,
        rotary_emb: mx.array,
        modulation_index: int | None = None,
        reference_tokens: int = 0,
        img_modulation_dims: ModulationDims | None = None,
    ) -> tuple[mx.array, mx.array]:
        if self.detail_double_block == index:
            return self.double_block_detail(
                index,
                img,
                txt,
                vec,
                rotary_emb,
                modulation_index=modulation_index,
                reference_tokens=reference_tokens,
                img_modulation_dims=img_modulation_dims,
            )
        base = f"double_blocks.{index}"
        txt_before_scout = txt if self.token_scout_enabled and index in self.token_scout_double_blocks else None
        t0 = time.perf_counter()
        if self.modulation_cache is not None and modulation_index is not None:
            img_mod, txt_mod = self.modulation_cache["double"][index]  # type: ignore[index]
            img_shift, img_scale, img_gate, img_shift_mlp, img_scale_mlp, img_gate_mlp = (
                chunk[modulation_index : modulation_index + 1] for chunk in img_mod
            )
            txt_shift, txt_scale, txt_gate, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = (
                chunk[modulation_index : modulation_index + 1] for chunk in txt_mod
            )
        else:
            img_shift, img_scale, img_gate, img_shift_mlp, img_scale_mlp, img_gate_mlp = self.modulate_double(
                vec, f"{base}.img_mod"
            )
            txt_shift, txt_scale, txt_gate, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = self.modulate_double(
                vec, f"{base}.txt_mod"
            )
        if img_modulation_dims is not None:
            txt_shift = txt_shift[0:1]
            txt_scale = txt_scale[0:1]
            txt_gate = txt_gate[0:1]
            txt_shift_mlp = txt_shift_mlp[0:1]
            txt_scale_mlp = txt_scale_mlp[0:1]
            txt_gate_mlp = txt_gate_mlp[0:1]
        self.profile_eval("double.modulation", t0, img_shift, txt_shift)

        img_norm = modulated_layer_norm(img, img_scale, img_shift, img_modulation_dims)
        txt_norm = modulated_layer_norm(txt, txt_scale, txt_shift)
        if index in self.shadow_double_blocks:
            self.shadow_probe(f"double{index}.img_mod_in", img_norm)
            self.shadow_probe(f"double{index}.txt_mod_in", txt_norm)

        t0 = time.perf_counter()
        img_qkv = self.linear(img_norm, f"{base}.img_attn.qkv")
        txt_qkv = self.linear(txt_norm, f"{base}.txt_attn.qkv")
        self.profile_eval("double.qkv", t0, img_qkv, txt_qkv)
        img_q, img_k, img_v = mx.split(img_qkv, 3, axis=-1)
        txt_q, txt_k, txt_v = mx.split(txt_qkv, 3, axis=-1)

        t0 = time.perf_counter()
        img_q, img_k = qk_norm(
            as_heads(img_q),
            as_heads(img_k),
            self.k(f"{base}.img_attn.norm.query_norm.scale"),
            self.k(f"{base}.img_attn.norm.key_norm.scale"),
        )
        img_v = as_heads(img_v)
        txt_q, txt_k = qk_norm(
            as_heads(txt_q),
            as_heads(txt_k),
            self.k(f"{base}.txt_attn.norm.query_norm.scale"),
            self.k(f"{base}.txt_attn.norm.key_norm.scale"),
        )
        txt_v = as_heads(txt_v)

        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)
        q, k = apply_rope(q, k, rotary_emb)
        self.attention_scout_probe("double_img_to_txt", index, q, k, int(txt.shape[1]))
        attn_out = from_heads(self.kontext_kv_attention("double", index, q, k, v, reference_tokens))
        if CAST_ATTENTION_OUT:
            attn_out = attn_out.astype(self.precision)
        self.profile_eval("double.attention", t0, attn_out)
        txt_attn, img_attn = attn_out[:, : txt.shape[1]], attn_out[:, txt.shape[1] :]

        t0 = time.perf_counter()
        img = img + apply_token_modulation(
            self.linear(img_attn, f"{base}.img_attn.proj"),
            img_gate,
            modulation_dims=img_modulation_dims,
        )
        img_mlp_residual = self.forecast_double_img_mlp_residual(index)
        if img_mlp_residual is None:
            img_ff = modulated_layer_norm(img, img_scale_mlp, img_shift_mlp, img_modulation_dims)
            img_mlp_residual = apply_token_modulation(
                self.mlp(img_ff, f"{base}.img_mlp", approx=self.approx_image_mlp),
                img_gate_mlp,
                modulation_dims=img_modulation_dims,
            )
            self.record_double_img_mlp_residual(index, img_mlp_residual)
        img = img + img_mlp_residual
        if CAST_BLOCK_OUTPUT:
            img = img.astype(self.precision)
        self.profile_eval("double.img_out_mlp", t0, img)

        t0 = time.perf_counter()
        txt_residual = self.forecast_double_txt_residual(index)
        if txt_residual is None:
            txt_attn_residual = txt_gate[:, None] * self.linear(txt_attn, f"{base}.txt_attn.proj")
            txt_after_attn = txt + txt_attn_residual
            txt_ff = modulated_layer_norm(txt_after_attn, txt_scale_mlp, txt_shift_mlp)
            txt_residual = txt_attn_residual + txt_gate_mlp[:, None] * self.mlp(txt_ff, f"{base}.txt_mlp", approx=True)
            self.record_double_txt_residual(index, txt_residual)
        txt = txt + txt_residual
        if CAST_BLOCK_OUTPUT:
            txt = txt.astype(self.precision)
        txt = nan_to_num_fp16(txt)
        self.profile_eval("double.txt_out_mlp", t0, txt)
        if txt_before_scout is not None:
            self.token_scout_probe("double_txt", index, txt_before_scout, txt)
        return img, txt

    def double_block_detail(
        self,
        index: int,
        img: mx.array,
        txt: mx.array,
        vec: mx.array,
        rotary_emb: mx.array,
        modulation_index: int | None = None,
        reference_tokens: int = 0,
        img_modulation_dims: ModulationDims | None = None,
    ) -> tuple[mx.array, mx.array]:
        base = f"double_blocks.{index}"
        prefix = f"double{index}"
        sync_t0 = time.perf_counter()
        mx.eval(img, txt, vec, rotary_emb)
        self.detail_record(f"{prefix}.sync_before", time.perf_counter() - sync_t0)
        total_t0 = time.perf_counter()

        t0 = time.perf_counter()
        if self.modulation_cache is not None and modulation_index is not None:
            img_mod, txt_mod = self.modulation_cache["double"][index]  # type: ignore[index]
            img_shift, img_scale, img_gate, img_shift_mlp, img_scale_mlp, img_gate_mlp = (
                chunk[modulation_index : modulation_index + 1] for chunk in img_mod
            )
            txt_shift, txt_scale, txt_gate, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = (
                chunk[modulation_index : modulation_index + 1] for chunk in txt_mod
            )
        else:
            img_shift, img_scale, img_gate, img_shift_mlp, img_scale_mlp, img_gate_mlp = self.modulate_double(
                vec, f"{base}.img_mod"
            )
            txt_shift, txt_scale, txt_gate, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = self.modulate_double(
                vec, f"{base}.txt_mod"
            )
        if img_modulation_dims is not None:
            txt_shift = txt_shift[0:1]
            txt_scale = txt_scale[0:1]
            txt_gate = txt_gate[0:1]
            txt_shift_mlp = txt_shift_mlp[0:1]
            txt_scale_mlp = txt_scale_mlp[0:1]
            txt_gate_mlp = txt_gate_mlp[0:1]
        self.detail_eval(f"{prefix}.modulation", t0, img_shift, txt_shift)

        t0 = time.perf_counter()
        img_norm = modulated_layer_norm(img, img_scale, img_shift, img_modulation_dims)
        txt_norm = modulated_layer_norm(txt, txt_scale, txt_shift)
        self.detail_eval(f"{prefix}.pre_attn_norm", t0, img_norm, txt_norm)
        if index in self.shadow_double_blocks:
            self.shadow_probe(f"double{index}.img_mod_in", img_norm)
            self.shadow_probe(f"double{index}.txt_mod_in", txt_norm)

        t0 = time.perf_counter()
        img_qkv = self.linear(img_norm, f"{base}.img_attn.qkv")
        txt_qkv = self.linear(txt_norm, f"{base}.txt_attn.qkv")
        self.detail_eval(f"{prefix}.qkv", t0, img_qkv, txt_qkv)

        t0 = time.perf_counter()
        img_q, img_k, img_v = mx.split(img_qkv, 3, axis=-1)
        txt_q, txt_k, txt_v = mx.split(txt_qkv, 3, axis=-1)
        self.detail_eval(f"{prefix}.split_qkv", t0, img_q, img_k, img_v, txt_q, txt_k, txt_v)

        t0 = time.perf_counter()
        img_q, img_k = qk_norm(
            as_heads(img_q),
            as_heads(img_k),
            self.k(f"{base}.img_attn.norm.query_norm.scale"),
            self.k(f"{base}.img_attn.norm.key_norm.scale"),
        )
        img_v = as_heads(img_v)
        txt_q, txt_k = qk_norm(
            as_heads(txt_q),
            as_heads(txt_k),
            self.k(f"{base}.txt_attn.norm.query_norm.scale"),
            self.k(f"{base}.txt_attn.norm.key_norm.scale"),
        )
        txt_v = as_heads(txt_v)
        self.detail_eval(f"{prefix}.qk_norm_heads", t0, img_q, img_k, img_v, txt_q, txt_k, txt_v)

        t0 = time.perf_counter()
        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)
        q, k = apply_rope(q, k, rotary_emb)
        self.detail_eval(f"{prefix}.cat_rope", t0, q, k, v)

        t0 = time.perf_counter()
        attn_out = from_heads(attention(q, k, v))
        if CAST_ATTENTION_OUT:
            attn_out = attn_out.astype(self.precision)
        self.detail_eval(f"{prefix}.attention", t0, attn_out)

        t0 = time.perf_counter()
        txt_attn, img_attn = attn_out[:, : txt.shape[1]], attn_out[:, txt.shape[1] :]
        self.detail_eval(f"{prefix}.split_attn_out", t0, txt_attn, img_attn)

        t0 = time.perf_counter()
        img_attn_proj = self.linear(img_attn, f"{base}.img_attn.proj")
        self.detail_eval(f"{prefix}.img_attn_proj", t0, img_attn_proj)

        t0 = time.perf_counter()
        img = img + apply_token_modulation(img_attn_proj, img_gate, modulation_dims=img_modulation_dims)
        self.detail_eval(f"{prefix}.img_attn_residual", t0, img)

        t0 = time.perf_counter()
        img_ff = modulated_layer_norm(img, img_scale_mlp, img_shift_mlp, img_modulation_dims)
        self.detail_eval(f"{prefix}.img_mlp_norm", t0, img_ff)

        t0 = time.perf_counter()
        img_mlp0 = self.linear(img_ff, f"{base}.img_mlp.0")
        self.detail_eval(f"{prefix}.img_mlp_linear0", t0, img_mlp0)

        t0 = time.perf_counter()
        img_mlp_act = nn.gelu_approx(img_mlp0) if self.approx_image_mlp else nn.gelu(img_mlp0)
        self.detail_eval(f"{prefix}.img_mlp_act", t0, img_mlp_act)

        t0 = time.perf_counter()
        img_mlp2 = self.linear(img_mlp_act, f"{base}.img_mlp.2")
        self.detail_eval(f"{prefix}.img_mlp_linear2", t0, img_mlp2)

        t0 = time.perf_counter()
        img = img + apply_token_modulation(img_mlp2, img_gate_mlp, modulation_dims=img_modulation_dims)
        if CAST_BLOCK_OUTPUT:
            img = img.astype(self.precision)
        self.detail_eval(f"{prefix}.img_mlp_residual_cast", t0, img)

        t0 = time.perf_counter()
        txt_attn_proj = self.linear(txt_attn, f"{base}.txt_attn.proj")
        self.detail_eval(f"{prefix}.txt_attn_proj", t0, txt_attn_proj)

        t0 = time.perf_counter()
        txt = txt + txt_gate[:, None] * txt_attn_proj
        self.detail_eval(f"{prefix}.txt_attn_residual", t0, txt)

        t0 = time.perf_counter()
        txt_ff = modulated_layer_norm(txt, txt_scale_mlp, txt_shift_mlp)
        self.detail_eval(f"{prefix}.txt_mlp_norm", t0, txt_ff)

        t0 = time.perf_counter()
        txt_mlp0 = self.linear(txt_ff, f"{base}.txt_mlp.0")
        self.detail_eval(f"{prefix}.txt_mlp_linear0", t0, txt_mlp0)

        t0 = time.perf_counter()
        txt_mlp_act = nn.gelu_approx(txt_mlp0)
        self.detail_eval(f"{prefix}.txt_mlp_act", t0, txt_mlp_act)

        t0 = time.perf_counter()
        txt_mlp2 = self.linear(txt_mlp_act, f"{base}.txt_mlp.2")
        self.detail_eval(f"{prefix}.txt_mlp_linear2", t0, txt_mlp2)

        t0 = time.perf_counter()
        txt = txt + txt_gate_mlp[:, None] * txt_mlp2
        if CAST_BLOCK_OUTPUT:
            txt = txt.astype(self.precision)
        self.detail_eval(f"{prefix}.txt_mlp_residual_cast", t0, txt)

        self.detail_record(f"{prefix}.total", time.perf_counter() - total_t0)
        return img, txt

    def single_block(
        self,
        index: int,
        x: mx.array,
        vec: mx.array,
        rotary_emb: mx.array,
        modulation_index: int | None = None,
        reference_tokens: int = 0,
        modulation_dims: ModulationDims | None = None,
    ) -> mx.array:
        if self.detail_single_block == index:
            return self.single_block_detail(
                index,
                x,
                vec,
                rotary_emb,
                modulation_index=modulation_index,
                reference_tokens=reference_tokens,
                modulation_dims=modulation_dims,
            )
        base = f"single_blocks.{index}"
        txt_before_scout = self.token_scout_before_single(index, x)
        t0 = time.perf_counter()
        if self.modulation_cache is not None and modulation_index is not None:
            cached = self.modulation_cache["single"][index]  # type: ignore[index]
            shift, scale, gate = (chunk[modulation_index : modulation_index + 1] for chunk in cached)
        else:
            shift, scale, gate = self.modulate_single(vec, f"{base}.modulation")
        self.profile_eval("single.modulation", t0, shift, scale, gate)
        norm_x = modulated_layer_norm(x, scale, shift, modulation_dims)
        if index in self.shadow_single_blocks:
            self.shadow_probe(f"single{index}.mod_in", norm_x)
        forecast_residual = self.forecast_single_residual(index)
        if forecast_residual is not None:
            x = x + forecast_residual
            if CAST_BLOCK_OUTPUT:
                x = x.astype(self.precision)
            if txt_before_scout is not None:
                self.token_scout_probe("single_txt", index, txt_before_scout, x[:, : self.token_scout_txt_len])
            return x
        t0 = time.perf_counter()
        forecast_linear2 = self.forecast_single_linear2(index, norm_x, gate)
        if forecast_linear2 is not None:
            residual = apply_token_modulation(forecast_linear2, gate, modulation_dims=modulation_dims)
            self.record_single_residual(index, residual)
            x = x + residual
            if CAST_BLOCK_OUTPUT:
                x = x.astype(self.precision)
            self.profile_eval("single.linear2_forecast_early", t0, x)
            if index in self.shadow_single_blocks:
                self.shadow_probe(f"single{index}.out", x)
            if txt_before_scout is not None:
                self.token_scout_probe("single_txt", index, txt_before_scout, x[:, : self.token_scout_txt_len])
            return x
        t0 = time.perf_counter()
        linear1 = self.linear(norm_x, f"{base}.linear1")
        self.profile_eval("single.linear1", t0, linear1)
        q, k, v, mlp = mx.split(linear1, [HIDDEN_DIM, HIDDEN_DIM * 2, HIDDEN_DIM * 3], axis=-1)
        t0 = time.perf_counter()
        q, k = qk_norm(
            as_heads(q),
            as_heads(k),
            self.k(f"{base}.norm.query_norm.scale"),
            self.k(f"{base}.norm.key_norm.scale"),
        )
        v = as_heads(v)
        q, k = apply_rope(q, k, rotary_emb)
        self.attention_scout_probe("single_img_to_txt", index, q, k, self.token_scout_txt_len)
        forecast_attention = self.forecast_single_attention(index)
        if forecast_attention is not None:
            attn_out = forecast_attention
            self.profile_eval("single.attention_forecast", t0, attn_out)
        else:
            attn_out = from_heads(self.kontext_kv_attention("single", index, q, k, v, reference_tokens))
            if CAST_ATTENTION_OUT:
                attn_out = attn_out.astype(self.precision)
            self.profile_eval("single.attention", t0, attn_out)
            self.record_single_attention(index, attn_out)
        t0 = time.perf_counter()
        mlp = nn.gelu_approx(mlp)
        attn_mlp = mx.concatenate([attn_out, mlp], axis=2)
        if SINGLE_LINEAR2_CAST:
            attn_mlp = attn_mlp.astype(self.precision)
        if SINGLE_LINEAR2_CONTIG:
            attn_mlp = mx.contiguous(attn_mlp)
        forecast_linear2_late = self.forecast_single_linear2_late(index, int(attn_mlp.shape[1]), gate)
        if forecast_linear2_late is not None:
            split = str(self.forecast_single_linear2_late_split or "all")
            if split == "all":
                out = forecast_linear2_late
                self.profile_eval("single.linear2_forecast_late", t0, out)
            elif split == "img_fast":
                out = self.partial_single_linear2_text_real(attn_mlp, forecast_linear2_late, f"{base}.linear2")
                self.profile_eval("single.linear2_forecast_late_img_fast", t0, out)
            else:
                actual_out = self.linear(attn_mlp, f"{base}.linear2")
                out = self.blend_single_linear2_late(actual_out, forecast_linear2_late)
                self.profile_eval("single.linear2_forecast_late_split", t0, out)
        else:
            out = self.linear(attn_mlp, f"{base}.linear2")
            self.profile_eval("single.linear2", t0, out)
            self.record_single_linear2(index, out, norm_x, gate)
        residual = apply_token_modulation(out, gate, modulation_dims=modulation_dims)
        self.record_single_residual(index, residual)
        x = x + residual
        if CAST_BLOCK_OUTPUT:
            x = x.astype(self.precision)
        x = nan_to_num_fp16(x)
        if index in self.shadow_single_blocks:
            self.shadow_probe(f"single{index}.out", x)
        if txt_before_scout is not None:
            self.token_scout_probe("single_txt", index, txt_before_scout, x[:, : self.token_scout_txt_len])
        return x

    def single_block_detail(
        self,
        index: int,
        x: mx.array,
        vec: mx.array,
        rotary_emb: mx.array,
        modulation_index: int | None = None,
        reference_tokens: int = 0,
        modulation_dims: ModulationDims | None = None,
    ) -> mx.array:
        base = f"single_blocks.{index}"
        prefix = f"single{index}"
        sync_t0 = time.perf_counter()
        mx.eval(x, vec, rotary_emb)
        self.detail_record(f"{prefix}.sync_before", time.perf_counter() - sync_t0)
        total_t0 = time.perf_counter()

        t0 = time.perf_counter()
        if self.modulation_cache is not None and modulation_index is not None:
            cached = self.modulation_cache["single"][index]  # type: ignore[index]
            shift, scale, gate = (chunk[modulation_index : modulation_index + 1] for chunk in cached)
        else:
            shift, scale, gate = self.modulate_single(vec, f"{base}.modulation")
        self.detail_eval(f"{prefix}.modulation", t0, shift, scale, gate)

        t0 = time.perf_counter()
        norm_base = layer_norm(x)
        self.detail_eval(f"{prefix}.pre_norm", t0, norm_base)

        t0 = time.perf_counter()
        norm_x = apply_token_modulation(norm_base, 1 + scale, shift, modulation_dims)
        if CAST_MODULATED_NORM:
            norm_x = norm_x.astype(scale.dtype)
        self.detail_eval(f"{prefix}.apply_mod_in", t0, norm_x)
        if index in self.shadow_single_blocks:
            self.shadow_probe(f"single{index}.mod_in", norm_x)

        t0 = time.perf_counter()
        linear1 = self.linear(norm_x, f"{base}.linear1")
        self.detail_eval(f"{prefix}.linear1_qkv_mlp", t0, linear1)

        t0 = time.perf_counter()
        q, k, v, mlp = mx.split(linear1, [HIDDEN_DIM, HIDDEN_DIM * 2, HIDDEN_DIM * 3], axis=-1)
        self.detail_eval(f"{prefix}.split_qkv_mlp", t0, q, k, v, mlp)

        t0 = time.perf_counter()
        q = as_heads(q)
        k = as_heads(k)
        v = as_heads(v)
        self.detail_eval(f"{prefix}.qkv_view_permute", t0, q, k, v)

        t0 = time.perf_counter()
        q, k = qk_norm(
            q,
            k,
            self.k(f"{base}.norm.query_norm.scale"),
            self.k(f"{base}.norm.key_norm.scale"),
        )
        self.detail_eval(f"{prefix}.qk_norm", t0, q, k)

        t0 = time.perf_counter()
        q, k = apply_rope(q, k, rotary_emb)
        self.detail_eval(f"{prefix}.apply_rope", t0, q, k)

        t0 = time.perf_counter()
        attn = attention(q, k, v)
        self.detail_eval(f"{prefix}.attention", t0, attn)

        t0 = time.perf_counter()
        attn_out = from_heads(attn)
        if CAST_ATTENTION_OUT:
            attn_out = attn_out.astype(self.precision)
        self.detail_eval(f"{prefix}.attn_patch_out", t0, attn_out)

        t0 = time.perf_counter()
        mlp = nn.gelu_approx(mlp)
        self.detail_eval(f"{prefix}.mlp_act", t0, mlp)

        t0 = time.perf_counter()
        attn_mlp = mx.concatenate([attn_out, mlp], axis=2)
        self.detail_eval(f"{prefix}.cat_attn_mlp", t0, attn_mlp)

        if (
            not self._detail_dtype_printed
            and os.environ.get("SDMLX_FLUX_VERBOSE_LOGS", "").lower() in {"1", "true", "yes", "on"}
        ):
            print(
                "Flux native detail dtypes: "
                f"norm_x={norm_x.dtype}, linear1={linear1.dtype}, q={q.dtype}, "
                f"attn={attn.dtype}, attn_out={attn_out.dtype}, mlp={mlp.dtype}, "
                f"attn_mlp={attn_mlp.dtype}, "
                f"linear2_weight={self.k(f'{base}.linear2.weight').dtype}"
            )
            self._detail_dtype_printed = True

        if SINGLE_LINEAR2_CAST:
            t0 = time.perf_counter()
            attn_mlp = attn_mlp.astype(self.precision)
            self.detail_eval(f"{prefix}.linear2_cast", t0, attn_mlp)

        if SINGLE_LINEAR2_CONTIG:
            t0 = time.perf_counter()
            attn_mlp = mx.contiguous(attn_mlp)
            self.detail_eval(f"{prefix}.linear2_contiguous", t0, attn_mlp)

        t0 = time.perf_counter()
        out = self.linear(attn_mlp, f"{base}.linear2")
        self.detail_eval(f"{prefix}.linear2_proj_mlpout", t0, out)

        t0 = time.perf_counter()
        out = x + apply_token_modulation(out, gate, modulation_dims=modulation_dims)
        if CAST_BLOCK_OUTPUT:
            out = out.astype(self.precision)
        self.detail_eval(f"{prefix}.gate_residual", t0, out)
        if index in self.shadow_single_blocks:
            self.shadow_probe(f"single{index}.out", out)
        self.detail_record(f"{prefix}.total", time.perf_counter() - total_t0)
        return out

    def predict(
        self,
        *,
        step_value: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        latents: mx.array,
        height: int,
        width: int,
        guidance: float = 0.0,
        modulation_index: int | None = None,
        txt_projected: mx.array | None = None,
        pooled_projected: mx.array | None = None,
        reference_latents: mx.array | None = None,
        reference_height: int | None = None,
        reference_width: int | None = None,
        reference_latents_method: str | None = None,
        sigma_value: float | None = None,
    ) -> mx.array:
        if self.teacache_gate_active():
            feature = self.predict_spectrum_feature(
                step_value=step_value,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                latents=latents,
                height=height,
                width=width,
                guidance=guidance,
                modulation_index=modulation_index,
                txt_projected=txt_projected,
                pooled_projected=pooled_projected,
                reference_latents=reference_latents,
                reference_height=reference_height,
                reference_width=reference_width,
                reference_latents_method=reference_latents_method,
                sigma_value=sigma_value,
            )
            return self.finish_spectrum_feature(
                feature,
                step_value=step_value,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance=guidance,
                modulation_index=modulation_index,
                pooled_projected=pooled_projected,
            )
        feature = self.predict_final_feature(
            step_value=step_value,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latents=latents,
            height=height,
            width=width,
            guidance=guidance,
            modulation_index=modulation_index,
            txt_projected=txt_projected,
            pooled_projected=pooled_projected,
            reference_latents=reference_latents,
            reference_height=reference_height,
            reference_width=reference_width,
            reference_latents_method=reference_latents_method,
        )
        return self.finish_final_feature(feature)

    def predict_spectrum_feature(
        self,
        *,
        step_value: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        latents: mx.array,
        height: int,
        width: int,
        guidance: float = 0.0,
        modulation_index: int | None = None,
        txt_projected: mx.array | None = None,
        pooled_projected: mx.array | None = None,
        reference_latents: mx.array | None = None,
        reference_height: int | None = None,
        reference_width: int | None = None,
        reference_latents_method: str | None = None,
        sigma_value: float | None = None,
    ) -> mx.array:
        time_step = mx.broadcast_to(step_value, (1,)).astype(self.precision)
        guidance_arr = mx.broadcast_to(guidance, (1,)).astype(self.precision)
        target_len = int(latents.shape[1])
        reference_tokens = int(reference_latents.shape[1]) if reference_latents is not None else 0
        use_kontext_kv_cache = reference_tokens > 0 and self.kontext_kv_cache_ready()
        use_reference_zero = (
            str(reference_latents_method or "offset") == "index_timestep_zero"
            and reference_tokens > 0
            and not use_kontext_kv_cache
            and modulation_index is None
        )
        txt = txt_projected if txt_projected is not None else self.linear(prompt_embeds, "txt_in")
        if self.modulation_cache is not None and modulation_index is not None:
            vec = self.modulation_cache["vec"][modulation_index : modulation_index + 1]  # type: ignore[index]
        elif use_reference_zero:
            vec = self.time_text_embed_with_reference_zero(
                time_step,
                pooled_prompt_embeds,
                guidance_arr,
                pooled_embed=pooled_projected,
            )
        else:
            vec = self.time_text_embed(time_step, pooled_prompt_embeds, guidance_arr, pooled_embed=pooled_projected)

        if str(self.teacache_mode or "off") == "double0_txt":
            cached_feature = self.teacache_try_reuse_feature(
                txt=txt,
                vec=vec,
                modulation_index=modulation_index,
            )
            if cached_feature is not None:
                return cached_feature

        img_input = latents if use_kontext_kv_cache else (
            mx.concatenate([latents, reference_latents], axis=1) if reference_latents is not None else latents
        )
        img = self.linear(img_input, "img_in")
        if reference_latents is not None and not use_kontext_kv_cache:
            if reference_height is None or reference_width is None:
                raise RuntimeError("FLUX Kontext needs reference_height and reference_width with reference_latents.")
            rotary_emb = self.rotary_kontext(height, width, int(prompt_embeds.shape[1]), reference_height, reference_width)
        else:
            rotary_emb = self.rotary(height, width, int(prompt_embeds.shape[1]))

        img_modulation_dims: ModulationDims | None = None
        single_modulation_dims: ModulationDims | None = None
        if use_reference_zero:
            img_modulation_dims = ((0, target_len, 0), (target_len, target_len + reference_tokens, 1))
        initial_target_img = img[:, :target_len]

        if self.seacache_gate_active():
            sigma_for_sea = float(sigma_value) if sigma_value is not None else float(step_value.item()) / 1000.0
            cached_feature = self.seacache_try_reuse_feature(
                img=img,
                vec=vec,
                height=height,
                width=width,
                target_len=target_len,
                sigma=sigma_for_sea,
                modulation_index=modulation_index,
                img_modulation_dims=img_modulation_dims,
            )
            if cached_feature is not None:
                return cached_feature

        for i in range(DOUBLE_BLOCKS):
            img, txt = self.double_block(
                i,
                img,
                txt,
                vec,
                rotary_emb,
                modulation_index=modulation_index,
                reference_tokens=reference_tokens,
                img_modulation_dims=img_modulation_dims,
            )

        img = nan_to_num_flux_fp16(img)
        self.token_scout_txt_len = int(txt.shape[1])
        x = mx.concatenate([txt, img], axis=1)
        if use_reference_zero:
            txt_len = int(txt.shape[1])
            single_modulation_dims = (
                (0, txt_len + target_len, 0),
                (txt_len + target_len, txt_len + target_len + reference_tokens, 1),
            )
            self.kontext_reference_zero_calls += 1
            self.kontext_reference_zero_last = {
                "method": str(reference_latents_method or "offset"),
                "target_tokens": target_len,
                "reference_tokens": reference_tokens,
                "txt_tokens": txt_len,
                "img_modulation_dims": img_modulation_dims,
                "single_modulation_dims": single_modulation_dims,
                "kv_cache_ready": bool(use_kontext_kv_cache),
            }
        for i in range(SINGLE_BLOCKS):
            x = self.single_block(
                i,
                x,
                vec,
                rotary_emb,
                modulation_index=modulation_index,
                reference_tokens=reference_tokens,
                modulation_dims=single_modulation_dims,
            )

        feature = x[:, txt.shape[1] :, ...]
        feature = feature[:, :target_len, ...] if reference_latents is not None else feature
        self.teacache_record_real_feature(feature)
        self.seacache_record_real_feature(feature, initial_target_img)
        return feature

    def finish_spectrum_feature(
        self,
        feature: mx.array,
        *,
        step_value: mx.array,
        pooled_prompt_embeds: mx.array,
        guidance: float = 0.0,
        modulation_index: int | None = None,
        pooled_projected: mx.array | None = None,
        sanitize: bool = False,
    ) -> mx.array:
        if sanitize:
            feature = sanitize_for_final_layer(feature, self.precision)
        time_step = mx.broadcast_to(step_value, (1,)).astype(self.precision)
        guidance_arr = mx.broadcast_to(guidance, (1,)).astype(self.precision)
        if self.modulation_cache is not None and modulation_index is not None:
            cached_shift, cached_scale = self.modulation_cache["final"]  # type: ignore[misc]
            shift = cached_shift[modulation_index : modulation_index + 1]
            scale = cached_scale[modulation_index : modulation_index + 1]
        else:
            vec = self.time_text_embed(time_step, pooled_prompt_embeds, guidance_arr, pooled_embed=pooled_projected)
            final = self.linear(nn.silu(vec), "final_layer.adaLN_modulation.1")
            shift, scale = mx.split(final, 2, axis=-1)
        feature = modulated_layer_norm(feature, scale, shift)
        return self.finish_final_feature(feature)

    def predict_final_feature(
        self,
        *,
        step_value: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        latents: mx.array,
        height: int,
        width: int,
        guidance: float = 0.0,
        modulation_index: int | None = None,
        txt_projected: mx.array | None = None,
        pooled_projected: mx.array | None = None,
        reference_latents: mx.array | None = None,
        reference_height: int | None = None,
        reference_width: int | None = None,
        reference_latents_method: str | None = None,
    ) -> mx.array:
        time_step = mx.broadcast_to(step_value, (1,)).astype(self.precision)
        guidance_arr = mx.broadcast_to(guidance, (1,)).astype(self.precision)
        target_len = int(latents.shape[1])
        reference_tokens = int(reference_latents.shape[1]) if reference_latents is not None else 0
        use_kontext_kv_cache = reference_tokens > 0 and self.kontext_kv_cache_ready()
        use_reference_zero = (
            str(reference_latents_method or "offset") == "index_timestep_zero"
            and reference_tokens > 0
            and not use_kontext_kv_cache
            and modulation_index is None
        )
        img_input = latents if use_kontext_kv_cache else (
            mx.concatenate([latents, reference_latents], axis=1) if reference_latents is not None else latents
        )
        img = self.linear(img_input, "img_in")
        txt = txt_projected if txt_projected is not None else self.linear(prompt_embeds, "txt_in")
        if self.modulation_cache is not None and modulation_index is not None:
            vec = self.modulation_cache["vec"][modulation_index : modulation_index + 1]  # type: ignore[index]
        elif use_reference_zero:
            vec = self.time_text_embed_with_reference_zero(
                time_step,
                pooled_prompt_embeds,
                guidance_arr,
                pooled_embed=pooled_projected,
            )
        else:
            vec = self.time_text_embed(time_step, pooled_prompt_embeds, guidance_arr, pooled_embed=pooled_projected)
        if reference_latents is not None and not use_kontext_kv_cache:
            if reference_height is None or reference_width is None:
                raise RuntimeError("FLUX Kontext needs reference_height and reference_width with reference_latents.")
            rotary_emb = self.rotary_kontext(height, width, int(prompt_embeds.shape[1]), reference_height, reference_width)
        else:
            rotary_emb = self.rotary(height, width, int(prompt_embeds.shape[1]))

        img_modulation_dims: ModulationDims | None = None
        single_modulation_dims: ModulationDims | None = None
        if use_reference_zero:
            img_modulation_dims = ((0, target_len, 0), (target_len, target_len + reference_tokens, 1))

        for i in range(DOUBLE_BLOCKS):
            img, txt = self.double_block(
                i,
                img,
                txt,
                vec,
                rotary_emb,
                modulation_index=modulation_index,
                reference_tokens=reference_tokens,
                img_modulation_dims=img_modulation_dims,
            )

        img = nan_to_num_flux_fp16(img)
        self.token_scout_txt_len = int(txt.shape[1])
        x = mx.concatenate([txt, img], axis=1)
        if use_reference_zero:
            txt_len = int(txt.shape[1])
            single_modulation_dims = (
                (0, txt_len + target_len, 0),
                (txt_len + target_len, txt_len + target_len + reference_tokens, 1),
            )
            self.kontext_reference_zero_calls += 1
            self.kontext_reference_zero_last = {
                "method": str(reference_latents_method or "offset"),
                "target_tokens": target_len,
                "reference_tokens": reference_tokens,
                "txt_tokens": txt_len,
                "img_modulation_dims": img_modulation_dims,
                "single_modulation_dims": single_modulation_dims,
                "kv_cache_ready": bool(use_kontext_kv_cache),
            }
        for i in range(SINGLE_BLOCKS):
            x = self.single_block(
                i,
                x,
                vec,
                rotary_emb,
                modulation_index=modulation_index,
                reference_tokens=reference_tokens,
                modulation_dims=single_modulation_dims,
            )

        img = x[:, txt.shape[1] :, ...]
        if reference_latents is not None:
            img = img[:, :target_len, ...]
        if self.modulation_cache is not None and modulation_index is not None:
            cached_shift, cached_scale = self.modulation_cache["final"]  # type: ignore[misc]
            shift = cached_shift[modulation_index : modulation_index + 1]
            scale = cached_scale[modulation_index : modulation_index + 1]
        else:
            final = self.linear(nn.silu(vec), "final_layer.adaLN_modulation.1")
            shift, scale = mx.split(final, 2, axis=-1)
            if use_reference_zero:
                shift = shift[0:1]
                scale = scale[0:1]
        img = modulated_layer_norm(img, scale, shift)
        return img

    def finish_final_feature(self, feature: mx.array) -> mx.array:
        noise = self.linear(feature, "final_layer.linear")
        self.shadow_probe("noise", noise)
        return noise
