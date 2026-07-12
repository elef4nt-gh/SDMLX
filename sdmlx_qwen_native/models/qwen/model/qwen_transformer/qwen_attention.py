from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_activation_scaling import (
    QwenActivationScaling,
    rms_norm_with_epsilon,
    scale_linear_bias,
    scaled_linear,
)

_QWEN_PREP_KERNEL = None
_QWEN_PREP_KERNEL_ATTEMPTED = False
_QWEN_PREP_KERNEL_NOTICE_SHOWN = False
_QWEN_FP16_ATTENTION_NOTICE_SHOWN = False
_QWEN_PREP_COMPARE_COUNTS: dict[int, int] = {}
_QWEN_ATTENTION_FINITE_COUNTS: dict[int, int] = {}


def _load_qwen_prep_kernel():
    global _QWEN_PREP_KERNEL
    global _QWEN_PREP_KERNEL_ATTEMPTED
    if _env_disabled("SDMLX_QWEN_FUSED_QK_PREP"):
        return None
    if _QWEN_PREP_KERNEL_ATTEMPTED:
        return _QWEN_PREP_KERNEL
    _QWEN_PREP_KERNEL_ATTEMPTED = True

    for path in _qwen_prep_kernel_search_paths():
        if path and path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    try:
        import sdmlx_int8_sdpa_ext as ext  # type: ignore
    except Exception:
        _QWEN_PREP_KERNEL = None
    else:
        _QWEN_PREP_KERNEL = ext
    return _QWEN_PREP_KERNEL


def _qwen_prep_kernel_search_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get("SDMLX_QWEN_KERNEL_LAB_PATH", "").strip()
    if env_path:
        paths.append(Path(env_path).expanduser())

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "sdmlx_int8_sdpa_ext").exists():
            paths.append(parent)
    return paths


def _qwen_prep_kernel_weight(module: nn.Module) -> mx.array:
    weight = module["weight"]
    if weight.dtype != mx.float16:
        weight = weight.astype(mx.float16)
    return weight


def _qwen_fused_qk_prep_enabled(block_idx: int | None) -> bool:
    if _env_disabled("SDMLX_QWEN_FUSED_QK_PREP"):
        return False
    if block_idx is None:
        return False
    start_block = _env_int("SDMLX_QWEN_FUSED_QK_PREP_START_BLOCK", 0)
    end_block = _env_int("SDMLX_QWEN_FUSED_QK_PREP_END_BLOCK", 1_000_000)
    return start_block <= int(block_idx) <= end_block


def _qwen_fused_qk_prep_dtype() -> str:
    value = os.environ.get("SDMLX_QWEN_FUSED_QK_PREP_DTYPE", "").strip().lower()
    if value in {"f16", "float16", "fp16"}:
        return "f16"
    return "f32"


class QwenAttention(nn.Module):
    def __init__(self, dim: int = 3072, num_heads: int = 24, head_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.attn_to_out = [nn.Linear(dim, dim)]
        self.to_add_out = nn.Linear(dim, dim)

    def __call__(
        self,
        img_modulated: mx.array,
        txt_modulated: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        image_rotary_emb: tuple[mx.array, mx.array],
        block_idx: int | None = None,
        activation_scaling: QwenActivationScaling | None = None,
    ) -> tuple[mx.array, mx.array]:
        if activation_scaling is None:
            activation_scaling = QwenActivationScaling.disabled()
        output_dtype = img_modulated.dtype

        if activation_scaling.runtime:
            qk_scale = activation_scaling.qk_scale
            img_query = scaled_linear(
                self.to_q,
                img_modulated,
                input_scale=qk_scale,
                bias_scale=qk_scale,
                compute_dtype=mx.float16,
            )
            img_key = scaled_linear(
                self.to_k,
                img_modulated,
                input_scale=qk_scale,
                bias_scale=qk_scale,
                compute_dtype=mx.float16,
            )
            img_value = scaled_linear(
                self.to_v,
                img_modulated,
                input_scale=qk_scale,
                bias_scale=qk_scale,
                compute_dtype=mx.float16,
            )

            txt_query = scaled_linear(
                self.add_q_proj,
                txt_modulated,
                input_scale=qk_scale,
                bias_scale=qk_scale,
                compute_dtype=mx.float16,
            )
            txt_key = scaled_linear(
                self.add_k_proj,
                txt_modulated,
                input_scale=qk_scale,
                bias_scale=qk_scale,
                compute_dtype=mx.float16,
            )
            txt_value = scaled_linear(
                self.add_v_proj,
                txt_modulated,
                input_scale=qk_scale,
                bias_scale=qk_scale,
                compute_dtype=mx.float16,
            )
        elif activation_scaling.folded:
            img_query = self.to_q(img_modulated.astype(mx.float16))
            img_key = self.to_k(img_modulated.astype(mx.float16))
            img_value = self.to_v(img_modulated.astype(mx.float16))

            txt_query = self.add_q_proj(txt_modulated.astype(mx.float16))
            txt_key = self.add_k_proj(txt_modulated.astype(mx.float16))
            txt_value = self.add_v_proj(txt_modulated.astype(mx.float16))
        else:
            img_query = self.to_q(img_modulated)
            img_key = self.to_k(img_modulated)
            img_value = self.to_v(img_modulated)

            txt_query = self.add_q_proj(txt_modulated)
            txt_key = self.add_k_proj(txt_modulated)
            txt_value = self.add_v_proj(txt_modulated)

        attention_diag = _qwen_attention_finite_diag_enabled(block_idx)
        if attention_diag:
            _qwen_attention_finite_print(block_idx, "img_query_projected", img_query)
            _qwen_attention_finite_print(block_idx, "img_key_projected", img_key)
            _qwen_attention_finite_print(block_idx, "img_value_projected", img_value)
            _qwen_attention_finite_print(block_idx, "txt_query_projected", txt_query)
            _qwen_attention_finite_print(block_idx, "txt_key_projected", txt_key)
            _qwen_attention_finite_print(block_idx, "txt_value_projected", txt_value)

        img_query = mx.reshape(img_query, (img_query.shape[0], img_query.shape[1], self.num_heads, self.head_dim))
        img_key = mx.reshape(img_key, (img_key.shape[0], img_key.shape[1], self.num_heads, self.head_dim))
        img_value = mx.reshape(img_value, (img_value.shape[0], img_value.shape[1], self.num_heads, self.head_dim))

        txt_query = mx.reshape(txt_query, (txt_query.shape[0], txt_query.shape[1], self.num_heads, self.head_dim))
        txt_key = mx.reshape(txt_key, (txt_key.shape[0], txt_key.shape[1], self.num_heads, self.head_dim))
        txt_value = mx.reshape(txt_value, (txt_value.shape[0], txt_value.shape[1], self.num_heads, self.head_dim))

        seq_txt = txt_modulated.shape[1]
        image_first_attention = _env_flag("SDMLX_QWEN_IMAGE_FIRST_ATTENTION")
        fused_qk_prep = (
            _qwen_fused_qk_prep_enabled(block_idx)
            and
            not image_first_attention
            and image_rotary_emb is not None
            and self.norm_q is not None
            and self.norm_k is not None
            and self.norm_added_q is not None
            and self.norm_added_k is not None
        )
        kernel_ext = _load_qwen_prep_kernel() if fused_qk_prep else None

        if kernel_ext is not None:
            global _QWEN_PREP_KERNEL_NOTICE_SHOWN
            if not _QWEN_PREP_KERNEL_NOTICE_SHOWN and _env_flag("SDMLX_QWEN_VERBOSE"):
                start_block = _env_int("SDMLX_QWEN_FUSED_QK_PREP_START_BLOCK", 0)
                end_block = _env_int("SDMLX_QWEN_FUSED_QK_PREP_END_BLOCK", 1_000_000)
                print(
                    "SDMLX Qwen: fused Q/K prep kernel active "
                    f"(blocks {start_block}-{end_block}, dtype={_qwen_fused_qk_prep_dtype()})"
                )
                _QWEN_PREP_KERNEL_NOTICE_SHOWN = True
            (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
            attention_dtype = img_query.dtype
            norm_eps = 1e-6
            if activation_scaling.enabled:
                norm_eps = 1e-6 / (activation_scaling.qk_scale * activation_scaling.qk_scale)
            qk_kernel_dtype = _qwen_fused_qk_prep_dtype()
            qk_kernel_is_f32 = qk_kernel_dtype == "f32"
            if qk_kernel_is_f32:
                qk_kernel = getattr(
                    kernel_ext,
                    "qwen_qk_rmsnorm_rope_split_cos_sin_seq_to_bh_f32_probe",
                    None,
                )
                if qk_kernel is not None:
                    joint_query = mx.concatenate([txt_query, img_query], axis=1).astype(mx.float32)
                    joint_key = mx.concatenate([txt_key, img_key], axis=1).astype(mx.float32)
                else:
                    qk_kernel_is_f32 = False
            if not qk_kernel_is_f32:
                joint_query = mx.concatenate([txt_query, img_query], axis=1).astype(mx.float16)
                joint_key = mx.concatenate([txt_key, img_key], axis=1).astype(mx.float16)
                qk_kernel = kernel_ext.qwen_qk_rmsnorm_rope_split_cos_sin_seq_to_bh_probe
            query_bhsd, key_bhsd = qk_kernel(
                joint_query,
                joint_key,
                _qwen_prep_kernel_weight(self.norm_q),
                _qwen_prep_kernel_weight(self.norm_k),
                _qwen_prep_kernel_weight(self.norm_added_q),
                _qwen_prep_kernel_weight(self.norm_added_k),
                img_cos,
                img_sin,
                txt_cos,
                txt_sin,
                seq_txt,
                float(norm_eps),
            )
            if not qk_kernel_is_f32 and query_bhsd.dtype != attention_dtype:
                query_bhsd = query_bhsd.astype(attention_dtype)
                key_bhsd = key_bhsd.astype(attention_dtype)
            if _qwen_fused_qk_compare_enabled(block_idx):
                ref_query_bhsd, ref_key_bhsd = self._reference_qk_prep_qwen(
                    img_query=img_query,
                    img_key=img_key,
                    txt_query=txt_query,
                    txt_key=txt_key,
                    img_cos=img_cos,
                    img_sin=img_sin,
                    txt_cos=txt_cos,
                    txt_sin=txt_sin,
                    activation_scaling=activation_scaling,
                    norm_eps=norm_eps,
                )
                _qwen_print_qk_compare(
                    block_idx,
                    ref_query_bhsd,
                    query_bhsd,
                    ref_key_bhsd,
                    key_bhsd,
                )
                if _env_flag("SDMLX_QWEN_FUSED_QK_PREP_COMPARE_USE_REFERENCE"):
                    query_bhsd = ref_query_bhsd
                    key_bhsd = ref_key_bhsd
            joint_value = mx.concatenate([txt_value, img_value], axis=1)
            value_bhsd = mx.transpose(joint_value, (0, 2, 1, 3))
        else:
            if activation_scaling.enabled:
                norm_eps = 1e-6 / (activation_scaling.qk_scale * activation_scaling.qk_scale)
                if self.norm_q is not None:
                    img_query = rms_norm_with_epsilon(self.norm_q, img_query, eps=norm_eps)
                if self.norm_k is not None:
                    img_key = rms_norm_with_epsilon(self.norm_k, img_key, eps=norm_eps)
                if self.norm_added_q is not None:
                    txt_query = rms_norm_with_epsilon(self.norm_added_q, txt_query, eps=norm_eps)
                if self.norm_added_k is not None:
                    txt_key = rms_norm_with_epsilon(self.norm_added_k, txt_key, eps=norm_eps)
            else:
                if self.norm_q is not None:
                    img_query = self.norm_q(img_query)
                if self.norm_k is not None:
                    img_key = self.norm_k(img_key)
                if self.norm_added_q is not None:
                    txt_query = self.norm_added_q(txt_query)
                if self.norm_added_k is not None:
                    txt_key = self.norm_added_k(txt_key)

            if image_rotary_emb is not None:
                (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
                img_query = QwenAttention._apply_rope_qwen(img_query, img_cos, img_sin)
                img_key = QwenAttention._apply_rope_qwen(img_key, img_cos, img_sin)
                txt_query = QwenAttention._apply_rope_qwen(txt_query, txt_cos, txt_sin)
                txt_key = QwenAttention._apply_rope_qwen(txt_key, txt_cos, txt_sin)

            if image_first_attention:
                joint_query = mx.concatenate([img_query, txt_query], axis=1)
                joint_key = mx.concatenate([img_key, txt_key], axis=1)
                joint_value = mx.concatenate([img_value, txt_value], axis=1)
            else:
                joint_query = mx.concatenate([txt_query, img_query], axis=1)
                joint_key = mx.concatenate([txt_key, img_key], axis=1)
                joint_value = mx.concatenate([txt_value, img_value], axis=1)
            query_bhsd = mx.transpose(joint_query, (0, 2, 1, 3))
            key_bhsd = mx.transpose(joint_key, (0, 2, 1, 3))
            value_bhsd = mx.transpose(joint_value, (0, 2, 1, 3))

        if not _env_disabled("SDMLX_QWEN_FORCE_FP16_ATTENTION_INPUTS"):
            global _QWEN_FP16_ATTENTION_NOTICE_SHOWN
            if not _QWEN_FP16_ATTENTION_NOTICE_SHOWN and _env_flag("SDMLX_QWEN_VERBOSE"):
                print("SDMLX Qwen: fp16 attention inputs active")
                _QWEN_FP16_ATTENTION_NOTICE_SHOWN = True
            if query_bhsd.dtype != mx.float16:
                query_bhsd = query_bhsd.astype(mx.float16)
            if key_bhsd.dtype != mx.float16:
                key_bhsd = key_bhsd.astype(mx.float16)
            if value_bhsd.dtype != mx.float16:
                value_bhsd = value_bhsd.astype(mx.float16)

        if attention_diag:
            _qwen_attention_finite_print(block_idx, "query_bhsd_pre_sdpa", query_bhsd)
            _qwen_attention_finite_print(block_idx, "key_bhsd_pre_sdpa", key_bhsd)
            _qwen_attention_finite_print(block_idx, "value_bhsd_pre_sdpa", value_bhsd)

        mask = self._convert_mask_for_qwen(
            mask=encoder_hidden_states_mask,
            joint_seq_len=query_bhsd.shape[2],
            txt_seq_len=seq_txt,
            image_first=image_first_attention,
        )

        hidden_states = self._compute_attention_qwen(
            query_bhsd=query_bhsd,
            key_bhsd=key_bhsd,
            value_bhsd=value_bhsd,
            mask=mask,
            block_idx=block_idx,
        )
        if attention_diag:
            _qwen_attention_finite_print(block_idx, "sdpa_hidden_states", hidden_states)

        if image_first_attention:
            seq_img = img_modulated.shape[1]
            img_attn_output = hidden_states[:, :seq_img, :]
            txt_attn_output = hidden_states[:, seq_img:, :]
        else:
            txt_attn_output = hidden_states[:, :seq_txt, :]
            img_attn_output = hidden_states[:, seq_txt:, :]
        if activation_scaling.runtime:
            out_scale = activation_scaling.qk_scale * activation_scaling.proj_scale
            img_attn_output = scaled_linear(
                self.attn_to_out[0],
                img_attn_output,
                input_scale=activation_scaling.proj_scale,
                bias_scale=out_scale,
                compute_dtype=mx.float16,
                output_dtype=output_dtype,
            )
            txt_attn_output = scaled_linear(
                self.to_add_out,
                txt_attn_output,
                input_scale=activation_scaling.proj_scale,
                bias_scale=out_scale,
                compute_dtype=mx.float16,
                output_dtype=txt_modulated.dtype,
            )
        elif activation_scaling.folded:
            img_attn_output = scaled_linear(
                self.attn_to_out[0],
                img_attn_output,
                input_scale=activation_scaling.proj_scale,
                bias_scale=1.0,
                compute_dtype=mx.float16,
                output_dtype=output_dtype,
            )
            txt_attn_output = scaled_linear(
                self.to_add_out,
                txt_attn_output,
                input_scale=activation_scaling.proj_scale,
                bias_scale=1.0,
                compute_dtype=mx.float16,
                output_dtype=txt_modulated.dtype,
            )
        else:
            img_attn_output = self.attn_to_out[0](img_attn_output)
            txt_attn_output = self.to_add_out(txt_attn_output)

        if attention_diag:
            _qwen_attention_finite_print(block_idx, "img_attn_output", img_attn_output)
            _qwen_attention_finite_print(block_idx, "txt_attn_output", txt_attn_output)

        return img_attn_output, txt_attn_output

    def _reference_qk_prep_qwen(
        self,
        *,
        img_query: mx.array,
        img_key: mx.array,
        txt_query: mx.array,
        txt_key: mx.array,
        img_cos: mx.array,
        img_sin: mx.array,
        txt_cos: mx.array,
        txt_sin: mx.array,
        activation_scaling: QwenActivationScaling,
        norm_eps: float,
    ) -> tuple[mx.array, mx.array]:
        if activation_scaling.enabled:
            img_query_ref = rms_norm_with_epsilon(self.norm_q, img_query, eps=norm_eps)
            img_key_ref = rms_norm_with_epsilon(self.norm_k, img_key, eps=norm_eps)
            txt_query_ref = rms_norm_with_epsilon(self.norm_added_q, txt_query, eps=norm_eps)
            txt_key_ref = rms_norm_with_epsilon(self.norm_added_k, txt_key, eps=norm_eps)
        else:
            img_query_ref = self.norm_q(img_query)
            img_key_ref = self.norm_k(img_key)
            txt_query_ref = self.norm_added_q(txt_query)
            txt_key_ref = self.norm_added_k(txt_key)

        img_query_ref = QwenAttention._apply_rope_qwen(img_query_ref, img_cos, img_sin)
        img_key_ref = QwenAttention._apply_rope_qwen(img_key_ref, img_cos, img_sin)
        txt_query_ref = QwenAttention._apply_rope_qwen(txt_query_ref, txt_cos, txt_sin)
        txt_key_ref = QwenAttention._apply_rope_qwen(txt_key_ref, txt_cos, txt_sin)

        query_bhsd = mx.transpose(mx.concatenate([txt_query_ref, img_query_ref], axis=1), (0, 2, 1, 3))
        key_bhsd = mx.transpose(mx.concatenate([txt_key_ref, img_key_ref], axis=1), (0, 2, 1, 3))
        return query_bhsd, key_bhsd

    def apply_drawthings_folded_weight_scaling(self, activation_scaling: QwenActivationScaling) -> None:
        if getattr(self, "_drawthings_folded_weights_applied", False):
            return
        if not activation_scaling.folded:
            return

        qk_scale = activation_scaling.qk_scale
        out_scale = activation_scaling.qk_scale * activation_scaling.proj_scale
        for module in (self.to_q, self.to_k, self.to_v, self.add_q_proj, self.add_k_proj, self.add_v_proj):
            scale_linear_bias(module, 1.0 / qk_scale)
        scale_linear_bias(self.attn_to_out[0], 1.0 / out_scale)
        scale_linear_bias(self.to_add_out, 1.0 / out_scale)

        norm_eps = 1e-6 / (activation_scaling.qk_scale * activation_scaling.qk_scale)
        for module in (self.norm_q, self.norm_k, self.norm_added_q, self.norm_added_k):
            if module is not None:
                module.eps = norm_eps

        self._drawthings_folded_weights_applied = True

    def _compute_attention_qwen(
        self,
        query_bhsd: mx.array,
        key_bhsd: mx.array,
        value_bhsd: mx.array,
        mask: mx.array | None = None,
        block_idx: int | None = None,
    ) -> mx.array:
        head_dim = query_bhsd.shape[-1]
        scale_value = 1.0 / (head_dim**0.5)
        hidden_states_bhsd = scaled_dot_product_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            scale=scale_value,
            mask=mask,
        )
        hidden_states = mx.transpose(hidden_states_bhsd, (0, 2, 1, 3))
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = mx.reshape(hidden_states, (batch_size, seq_len, self.num_heads * self.head_dim))
        hidden_states = hidden_states.astype(query_bhsd.dtype)
        return hidden_states

    @staticmethod
    def _convert_mask_for_qwen(
        mask: mx.array | None,
        joint_seq_len: int,
        txt_seq_len: int,
        image_first: bool = False,
    ) -> mx.array | None:
        if mask is None:
            return None

        bsz = mask.shape[0]
        img_seq_len = joint_seq_len - txt_seq_len

        ones_img = mx.ones((bsz, img_seq_len), dtype=mx.float32)
        if image_first:
            joint_mask = mx.concatenate([ones_img, mask.astype(mx.float32)], axis=1)
        else:
            joint_mask = mx.concatenate([mask.astype(mx.float32), ones_img], axis=1)

        if mx.all(joint_mask >= 0.999):
            return None

        additive = (1.0 - joint_mask) * (-1e9)
        return additive.reshape((additive.shape[0], 1, 1, additive.shape[1]))

    @staticmethod
    def _apply_rope_qwen(x: mx.array, cos_vals: mx.array, sin_vals: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        x_reshaped = mx.reshape(x_float, (*x.shape[:-1], -1, 2))

        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]

        freqs_cos = cos_vals[None, :, None, :]
        freqs_sin = sin_vals[None, :, None, :]

        if freqs_cos.shape[-1] != x_real.shape[-1]:
            freqs_cos = freqs_cos[..., : x_real.shape[-1]]
            freqs_sin = freqs_sin[..., : x_real.shape[-1]]

        out_real = x_real * freqs_cos - x_imag * freqs_sin
        out_imag = x_real * freqs_sin + x_imag * freqs_cos

        out_pairs = mx.stack([out_real, out_imag], axis=-1)
        x_out = mx.reshape(out_pairs, (*x.shape[:-1], -1))

        return x_out.astype(x.dtype)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_disabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"0", "false", "no", "off", "disabled"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _qwen_attention_finite_diag_enabled(block_idx: int | None) -> bool:
    if block_idx is None:
        return False
    if not _env_flag("SDMLX_QWEN_ATTENTION_FINITE_DIAGNOSTICS"):
        return False
    target = os.environ.get("SDMLX_QWEN_ATTENTION_FINITE_BLOCK", "").strip().lower()
    if target and target not in {"all", "*"}:
        try:
            if int(target) != int(block_idx):
                return False
        except ValueError:
            return False
    limit = _env_int("SDMLX_QWEN_ATTENTION_FINITE_LIMIT", 1)
    count = _QWEN_ATTENTION_FINITE_COUNTS.get(int(block_idx), 0)
    if limit >= 0 and count >= limit:
        return False
    _QWEN_ATTENTION_FINITE_COUNTS[int(block_idx)] = count + 1
    return True


def _qwen_attention_finite_print(block_idx: int | None, name: str, tensor: mx.array) -> None:
    arr = tensor.astype(mx.float32)
    finite = mx.isfinite(arr)
    nan = mx.isnan(arr)
    inf = mx.isinf(arr)
    safe = mx.where(finite, arr, mx.zeros_like(arr))
    abs_safe = mx.abs(safe)
    finite_ratio = mx.mean(finite.astype(mx.float32))
    rms = mx.sqrt(mx.mean(mx.square(safe)))
    mean = mx.mean(safe)
    max_abs = mx.max(abs_safe)
    nan_count = mx.sum(nan.astype(mx.int32))
    inf_count = mx.sum(inf.astype(mx.int32))
    mx.eval(finite_ratio, rms, mean, max_abs, nan_count, inf_count)
    print(
        "SDMLX Qwen attention finite "
        f"block={block_idx} {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite={float(finite_ratio.item()):.6f} "
        f"nan={int(nan_count.item())} inf={int(inf_count.item())} "
        f"rms={float(rms.item()):.6f} mean={float(mean.item()):.6f} "
        f"max_abs={float(max_abs.item()):.6f}"
    )


def _qwen_fused_qk_compare_enabled(block_idx: int | None) -> bool:
    if block_idx is None:
        return False
    target = os.environ.get("SDMLX_QWEN_FUSED_QK_PREP_COMPARE_BLOCK", "").strip()
    if not target:
        return False
    try:
        return int(target) == int(block_idx)
    except ValueError:
        return False


def _qwen_tensor_metric(reference: mx.array, candidate: mx.array) -> dict[str, float | bool]:
    ref = reference.astype(mx.float32)
    cand = candidate.astype(mx.float32)
    diff = cand - ref
    mse = mx.mean(mx.square(diff))
    ref_mse = mx.mean(mx.square(ref))
    return {
        "mae": float(mx.mean(mx.abs(diff)).item()),
        "rmse": float(mx.sqrt(mse).item()),
        "rel_rmse": float(mx.sqrt(mse / mx.maximum(ref_mse, mx.array(1e-12, dtype=mx.float32))).item()),
        "max_abs": float(mx.max(mx.abs(diff)).item()),
        "ref_rms": float(mx.sqrt(ref_mse).item()),
        "cand_rms": float(mx.sqrt(mx.mean(mx.square(cand))).item()),
        "ref_finite": bool(mx.all(mx.isfinite(ref)).item()),
        "cand_finite": bool(mx.all(mx.isfinite(cand)).item()),
    }


def _qwen_print_qk_compare(
    block_idx: int | None,
    ref_q: mx.array,
    fused_q: mx.array,
    ref_k: mx.array,
    fused_k: mx.array,
) -> None:
    if block_idx is None:
        return
    idx = int(block_idx)
    limit = _env_int("SDMLX_QWEN_FUSED_QK_PREP_COMPARE_LIMIT", 4)
    count = _QWEN_PREP_COMPARE_COUNTS.get(idx, 0)
    if limit >= 0 and count >= limit:
        return
    _QWEN_PREP_COMPARE_COUNTS[idx] = count + 1
    mx.eval(ref_q, fused_q, ref_k, fused_k)
    print(
        "SDMLX Qwen Lab: fused Q/K prep compare "
        f"block={idx} sample={count + 1} "
        f"Q={_qwen_tensor_metric(ref_q, fused_q)} "
        f"K={_qwen_tensor_metric(ref_k, fused_k)}"
    )
