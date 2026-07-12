from __future__ import annotations

import os

import mlx.core as mx
from mlx import nn

from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_activation_scaling import QwenActivationScaling
from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention
from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_feed_forward import QwenFeedForward


class QwenTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 3072,
        num_heads: int = 24,
        head_dim: int = 128,
        layer_idx: int = 0,
        num_layers: int = 60,
        fp16_activation_scaling: bool = False,
        activation_scaling_mode: str = "runtime",
        activation_scaling_profile: str = "qwen_image_edit",
    ):
        super().__init__()

        self.activation_scaling = (
            QwenActivationScaling.draw_things(
                layer_idx,
                num_layers,
                mode=activation_scaling_mode,
                profile=activation_scaling_profile,
            )
            if fp16_activation_scaling
            else QwenActivationScaling.disabled()
        )

        self.img_mod_silu = nn.SiLU()
        self.img_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.img_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.attn = QwenAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)
        self.img_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.img_ff = QwenFeedForward(dim=dim)

        self.txt_mod_silu = nn.SiLU()
        self.txt_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.txt_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_ff = QwenFeedForward(dim=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        text_embeddings: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        block_idx: int | None = None,
        timestep_zero_index: int | None = None,
        mod_params: tuple[mx.array, mx.array, mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, mx.array]:
        if mod_params is None:
            mod_params = self.compute_mod_params(text_embeddings, timestep_zero_index=timestep_zero_index)
        img_mod1, img_mod2, txt_mod1, txt_mod2 = mod_params

        img_normed = self.img_norm1(hidden_states)
        attn_mod_scale = self.activation_scaling.qk_scale if self.activation_scaling.folded else 1.0
        attn_gate_scale = (
            self.activation_scaling.qk_scale * self.activation_scaling.proj_scale
            if self.activation_scaling.folded
            else 1.0
        )
        img_modulated, img_gate1 = QwenTransformerBlock._modulate(
            img_normed,
            img_mod1,
            timestep_zero_index,
            activation_scale=attn_mod_scale,
            gate_scale=attn_gate_scale,
        )

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = QwenTransformerBlock._modulate(
            txt_normed,
            txt_mod1,
            activation_scale=attn_mod_scale,
            gate_scale=attn_gate_scale,
        )

        img_attn_output, txt_attn_output = self.attn(
            img_modulated=img_modulated,
            txt_modulated=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            block_idx=block_idx,
            activation_scaling=self.activation_scaling,
        )
        _stage_finite_print_if_needed(block_idx, "img_attn_output_raw", img_attn_output)
        _stage_finite_print_if_needed(block_idx, "txt_attn_output_raw", txt_attn_output)
        if self.activation_scaling.runtime:
            attn_restore = self.activation_scaling.qk_scale * self.activation_scaling.proj_scale
            img_attn_output = img_attn_output * attn_restore
            txt_attn_output = txt_attn_output * attn_restore
            _stage_finite_print_if_needed(block_idx, "img_attn_output_restored", img_attn_output)
            _stage_finite_print_if_needed(block_idx, "txt_attn_output_restored", txt_attn_output)

        hidden_states = QwenTransformerBlock._apply_gate(
            delta=img_attn_output,
            base=hidden_states,
            gate=img_gate1,
            timestep_zero_index=timestep_zero_index,
        )
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
        _stage_finite_print_if_needed(block_idx, "img_after_attn_residual", hidden_states)
        _stage_finite_print_if_needed(block_idx, "txt_after_attn_residual", encoder_hidden_states)

        img_normed2 = self.img_norm2(hidden_states)
        ffn_mod_scale = self.activation_scaling.ffn_proj_up if self.activation_scaling.folded else 1.0
        ffn_gate_scale = self.activation_scaling.ffn_proj_down if self.activation_scaling.folded else 1.0
        img_modulated2, img_gate2 = QwenTransformerBlock._modulate(
            img_normed2,
            img_mod2,
            timestep_zero_index,
            activation_scale=ffn_mod_scale,
            gate_scale=ffn_gate_scale,
        )

        img_mlp_output = self.img_ff(
            img_modulated2,
            activation_scaling=self.activation_scaling,
            block_idx=block_idx,
            branch="img",
        )
        _stage_finite_print_if_needed(block_idx, "img_mlp_output_raw", img_mlp_output)
        if self.activation_scaling.runtime:
            img_mlp_output = img_mlp_output * self.activation_scaling.ffn_proj_down
            _stage_finite_print_if_needed(block_idx, "img_mlp_output_restored", img_mlp_output)

        hidden_states = QwenTransformerBlock._apply_gate(
            delta=img_mlp_output,
            base=hidden_states,
            gate=img_gate2,
            timestep_zero_index=timestep_zero_index,
        )
        _stage_finite_print_if_needed(block_idx, "img_after_mlp_residual", hidden_states)

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = QwenTransformerBlock._modulate(
            txt_normed2,
            txt_mod2,
            activation_scale=ffn_mod_scale,
            gate_scale=ffn_gate_scale,
        )
        txt_mlp_output = self.txt_ff(
            txt_modulated2,
            activation_scaling=self.activation_scaling,
            block_idx=block_idx,
            branch="txt",
        )
        _stage_finite_print_if_needed(block_idx, "txt_mlp_output_raw", txt_mlp_output)
        if self.activation_scaling.runtime:
            txt_mlp_output = txt_mlp_output * self.activation_scaling.ffn_proj_down
            _stage_finite_print_if_needed(block_idx, "txt_mlp_output_restored", txt_mlp_output)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
        _stage_finite_print_if_needed(block_idx, "txt_after_mlp_residual", encoder_hidden_states)

        return encoder_hidden_states, hidden_states

    def compute_mod_params(
        self,
        text_embeddings: mx.array,
        timestep_zero_index: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        img_mod_params = self.img_mod_linear(self.img_mod_silu(text_embeddings))
        txt_text_embeddings = mx.split(text_embeddings, 2, axis=0)[0] if timestep_zero_index is not None else text_embeddings
        txt_mod_params = self.txt_mod_linear(self.txt_mod_silu(txt_text_embeddings))

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)
        return img_mod1, img_mod2, txt_mod1, txt_mod2

    @staticmethod
    def _modulate(
        x: mx.array,
        mod_params: mx.array,
        timestep_zero_index: int | None = None,
        *,
        activation_scale: float = 1.0,
        gate_scale: float = 1.0,
    ) -> tuple[mx.array, mx.array]:
        shift, scale, gate = mx.split(mod_params, 3, axis=-1)
        if activation_scale != 1.0:
            shift = shift / activation_scale
            scale = ((1 + scale) / activation_scale) - 1
        if gate_scale != 1.0:
            gate = gate * gate_scale
        if timestep_zero_index is None:
            return x * (1 + scale[:, None, :]) + shift[:, None, :], gate[:, None, :]

        actual_batch = shift.shape[0] // 2
        shift, shift_zero = shift[:actual_batch], shift[actual_batch:]
        scale, scale_zero = scale[:actual_batch], scale[actual_batch:]
        gate, gate_zero = gate[:actual_batch], gate[actual_batch:]

        regular = x[:, :timestep_zero_index] * (1 + scale[:, None, :]) + shift[:, None, :]
        zero = x[:, timestep_zero_index:] * (1 + scale_zero[:, None, :]) + shift_zero[:, None, :]
        return mx.concatenate([regular, zero], axis=1), (gate[:, None, :], gate_zero[:, None, :])

    def apply_drawthings_folded_weight_scaling(self) -> None:
        if getattr(self, "_drawthings_folded_weights_applied", False):
            return
        if not self.activation_scaling.folded:
            return
        self.attn.apply_drawthings_folded_weight_scaling(self.activation_scaling)
        self.img_ff.apply_drawthings_folded_weight_scaling(self.activation_scaling)
        self.txt_ff.apply_drawthings_folded_weight_scaling(self.activation_scaling)
        self._drawthings_folded_weights_applied = True

    @staticmethod
    def _apply_gate(
        delta: mx.array,
        base: mx.array,
        gate: mx.array | tuple[mx.array, mx.array],
        timestep_zero_index: int | None = None,
    ) -> mx.array:
        if timestep_zero_index is None:
            return base + gate * delta

        gate_regular, gate_zero = gate
        regular = delta[:, :timestep_zero_index] * gate_regular
        zero = delta[:, timestep_zero_index:] * gate_zero
        return base + mx.concatenate([regular, zero], axis=1)


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


def _stage_finite_print_if_needed(block_idx: int | None, stage: str, tensor: mx.array) -> None:
    if not _env_flag("SDMLX_QWEN_BLOCK_STAGE_FINITE_DIAGNOSTICS"):
        return
    if block_idx is None:
        return
    target = _env_int("SDMLX_QWEN_BLOCK_STAGE_FINITE_BLOCK", int(block_idx))
    if int(block_idx) != target:
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
        "SDMLX Qwen block stage finite "
        f"block={block_idx} {stage}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite={float(finite_ratio.item()):.6f} nan={int(nan_count.item())} "
        f"inf={int(inf_count.item())} rms={float(rms.item()):.6f} "
        f"max_abs={float(max_abs.item()):.6f}"
    )
