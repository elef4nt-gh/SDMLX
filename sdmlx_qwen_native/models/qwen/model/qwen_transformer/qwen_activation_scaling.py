from __future__ import annotations

from dataclasses import dataclass
import os

import mlx.core as mx
from mlx import nn


@dataclass(frozen=True)
class QwenActivationScaling:
    enabled: bool = False
    mode: str = "off"
    qk_scale: float = 1.0
    proj_scale: float = 1.0
    ffn_proj_up: float = 1.0
    ffn_proj_down: float = 1.0

    @property
    def runtime(self) -> bool:
        return self.enabled and self.mode == "runtime"

    @property
    def folded(self) -> bool:
        return self.enabled and self.mode == "folded"

    @staticmethod
    def disabled() -> "QwenActivationScaling":
        return QwenActivationScaling()

    @staticmethod
    def draw_things(
        block_idx: int,
        num_layers: int,
        *,
        mode: str = "runtime",
        profile: str = "qwen_image_edit",
    ) -> "QwenActivationScaling":
        """Draw Things-style non-BF16 scaling profile."""

        profile = str(profile or "qwen_image_edit").strip().lower()
        default_ffn_up = 8.0 if profile == "qwen_image" else 1.0
        ffn_down = _env_float("SDMLX_QWEN_FFN_DOWN_SCALE", 32.0)
        ffn_last_down = _env_float("SDMLX_QWEN_FFN_LAST_DOWN_SCALE", 512.0)
        return QwenActivationScaling(
            enabled=True,
            mode=mode,
            qk_scale=_env_float("SDMLX_QWEN_QK_SCALE", 8.0),
            proj_scale=(
                _env_float("SDMLX_QWEN_PROJ_LATE_SCALE", 32.0)
                if block_idx >= num_layers - 16
                else _env_float("SDMLX_QWEN_PROJ_SCALE", 4.0)
            ),
            ffn_proj_up=_env_float("SDMLX_QWEN_FFN_UP_SCALE", default_ffn_up),
            ffn_proj_down=ffn_last_down if block_idx >= num_layers - 1 else ffn_down,
        )


def scaled_linear(
    module: nn.Module,
    x: mx.array,
    *,
    input_scale: float,
    bias_scale: float | None = None,
    compute_dtype=None,
    output_dtype=None,
) -> mx.array:
    """Run a linear layer with scaled input and compensated bias.

    Draw Things folds these scale factors into weight mapping and AdaLN chunks.
    In the MLX lab path we do it at runtime so the original weights stay
    untouched while the matrix multiply sees smaller FP16 activations.
    """

    if bias_scale is None:
        bias_scale = input_scale

    x_in = x * (1.0 / input_scale) if input_scale != 1.0 else x
    if compute_dtype is not None and x_in.dtype != compute_dtype:
        x_in = x_in.astype(compute_dtype)

    out = module(x_in)
    bias = _find_bias(module)
    if bias is not None and bias_scale != 1.0:
        out = out + bias * ((1.0 / bias_scale) - 1.0)
    if output_dtype is not None and out.dtype != output_dtype:
        out = out.astype(output_dtype)
    return out


def scale_linear_bias(module: nn.Module, factor: float) -> bool:
    if factor == 1.0:
        return False
    owner = _find_bias_owner(module)
    if owner is None:
        return False
    owner.bias = owner["bias"] * factor
    return True


def rms_norm_with_epsilon(module: nn.Module, x: mx.array, *, eps: float) -> mx.array:
    input_dtype = x.dtype
    variance = mx.power(x.astype(mx.float32), 2).mean(axis=-1, keepdims=True)
    out = x * mx.rsqrt(variance + eps)
    weight = _find_weight(module)
    if weight is not None:
        if weight.dtype in [mx.bfloat16, mx.float16]:
            out = out.astype(weight.dtype)
        out = out * weight
        if out.dtype != input_dtype:
            out = out.astype(input_dtype)
    return out


def _find_bias(module: nn.Module) -> mx.array | None:
    owner = _find_bias_owner(module)
    return None if owner is None else owner["bias"]


def _find_bias_owner(module: nn.Module) -> nn.Module | None:
    if "bias" in module:
        return module
    for attr in ("linear", "base_linear"):
        base = getattr(module, attr, None)
        if base is not None and "bias" in base:
            return base
    return None


def _find_weight(module: nn.Module) -> mx.array | None:
    if "weight" in module:
        return module["weight"]
    return None


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0.0 else default
