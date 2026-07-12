import os

import mlx.core as mx
from mlx import nn

from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_activation_scaling import (
    QwenActivationScaling,
    scale_linear_bias,
    scaled_linear,
)


class QwenFeedForward(nn.Module):
    def __init__(self, dim: int = 3072):
        super().__init__()
        self.mlp_in = nn.Linear(dim, 4 * dim, bias=True)
        self.mlp_out = nn.Linear(4 * dim, dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        activation_scaling: QwenActivationScaling | None = None,
        block_idx: int | None = None,
        branch: str = "",
    ) -> mx.array:
        if activation_scaling is None or not activation_scaling.enabled:
            hidden_states = self.mlp_in(hidden_states)
            hidden_states = nn.gelu_approx(hidden_states)
            hidden_states = self.mlp_out(hidden_states)
            return hidden_states

        output_dtype = hidden_states.dtype
        hidden_states = scaled_linear(
            self.mlp_in,
            hidden_states,
            input_scale=activation_scaling.ffn_proj_up,
            bias_scale=activation_scaling.ffn_proj_up if activation_scaling.runtime else 1.0,
            compute_dtype=mx.float16,
        )
        _ffn_stage_print_if_needed(block_idx=block_idx, branch=branch, stage="mlp_in", tensor=hidden_states)
        if activation_scaling.ffn_proj_up != 1.0:
            hidden_states = hidden_states.astype(mx.float32) * activation_scaling.ffn_proj_up
            _ffn_stage_print_if_needed(block_idx=block_idx, branch=branch, stage="mlp_in_restored", tensor=hidden_states)
        hidden_states = nn.gelu_approx(hidden_states)
        _ffn_stage_print_if_needed(block_idx=block_idx, branch=branch, stage="gelu", tensor=hidden_states)
        if _dense_dequant_enabled(block_idx=block_idx, branch=branch):
            hidden_states = _dense_quantized_linear(
                self.mlp_out,
                hidden_states,
                input_scale=activation_scaling.ffn_proj_down,
                bias_scale=activation_scaling.ffn_proj_down if activation_scaling.runtime else 1.0,
                output_dtype=output_dtype,
            )
            return hidden_states
        mlp_out_compute_dtype = None if _env_flag("SDMLX_QWEN_FFN_OUT_FLOAT32_INPUT") else mx.float16
        hidden_states = scaled_linear(
            self.mlp_out,
            hidden_states,
            input_scale=activation_scaling.ffn_proj_down,
            bias_scale=activation_scaling.ffn_proj_down if activation_scaling.runtime else 1.0,
            compute_dtype=mlp_out_compute_dtype,
            output_dtype=output_dtype,
        )
        return hidden_states

    def apply_drawthings_folded_weight_scaling(self, activation_scaling: QwenActivationScaling) -> None:
        if getattr(self, "_drawthings_folded_weights_applied", False):
            return
        if not activation_scaling.folded:
            return
        scale_linear_bias(self.mlp_in, 1.0 / activation_scaling.ffn_proj_up)
        scale_linear_bias(self.mlp_out, 1.0 / activation_scaling.ffn_proj_down)
        self._drawthings_folded_weights_applied = True


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _dense_dequant_enabled(*, block_idx: int | None, branch: str) -> bool:
    if not _env_flag("SDMLX_QWEN_FFN_OUT_DENSE_DEQUANT"):
        return False
    target_block = _env_int("SDMLX_QWEN_FFN_OUT_DENSE_DEQUANT_BLOCK", -1)
    if target_block >= 0 and int(block_idx or -1) != target_block:
        return False
    target_branch = os.environ.get("SDMLX_QWEN_FFN_OUT_DENSE_DEQUANT_BRANCH", "").strip().lower()
    return not target_branch or str(branch).lower() == target_branch


def _dense_quantized_linear(
    module: nn.Module,
    x: mx.array,
    *,
    input_scale: float,
    bias_scale: float,
    output_dtype,
) -> mx.array:
    if not isinstance(module, nn.QuantizedLinear):
        return scaled_linear(
            module,
            x,
            input_scale=input_scale,
            bias_scale=bias_scale,
            compute_dtype=None,
            output_dtype=output_dtype,
        )

    x_in = x * (1.0 / input_scale) if input_scale != 1.0 else x
    dense = mx.dequantize(
        module.weight,
        module.scales,
        module.biases,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    ).astype(mx.float32)
    out = mx.matmul(x_in.astype(mx.float32), mx.transpose(dense))
    bias = module.get("bias")
    if bias is not None:
        out = out + bias.astype(mx.float32)
        if bias_scale != 1.0:
            out = out + bias.astype(mx.float32) * ((1.0 / bias_scale) - 1.0)
    if output_dtype is not None and out.dtype != output_dtype:
        out = out.astype(output_dtype)
    return out


def _ffn_stage_print_if_needed(*, block_idx: int | None, branch: str, stage: str, tensor: mx.array) -> None:
    if not _env_flag("SDMLX_QWEN_FFN_STAGE_FINITE_DIAGNOSTICS"):
        return
    target_block = _env_int("SDMLX_QWEN_FFN_STAGE_FINITE_BLOCK", int(block_idx or -1))
    if block_idx is None or int(block_idx) != target_block:
        return
    target_branch = os.environ.get("SDMLX_QWEN_FFN_STAGE_FINITE_BRANCH", "").strip().lower()
    if target_branch and str(branch).lower() != target_branch:
        return
    arr = tensor.astype(mx.float32)
    finite = mx.isfinite(arr)
    finite_ratio = mx.mean(finite.astype(mx.float32))
    nan_count = mx.sum(mx.isnan(arr).astype(mx.int32))
    inf_count = mx.sum(mx.isinf(arr).astype(mx.int32))
    safe = mx.where(finite, arr, mx.zeros_like(arr))
    rms = mx.sqrt(mx.mean(mx.square(safe)))
    max_abs = mx.max(mx.abs(safe))
    mx.eval(finite_ratio, nan_count, inf_count, rms, max_abs)
    print(
        "SDMLX Qwen FFN stage finite "
        f"block={block_idx} branch={branch} {stage}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite={float(finite_ratio.item()):.6f} nan={int(nan_count.item())} "
        f"inf={int(inf_count.item())} rms={float(rms.item()):.6f} "
        f"max_abs={float(max_abs.item()):.6f}"
    )
