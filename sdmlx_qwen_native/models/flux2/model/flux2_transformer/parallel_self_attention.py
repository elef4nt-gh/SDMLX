import os

import mlx.core as mx
from mlx import nn

from sdmlx_qwen_native.models.common.config.model_config import ModelConfig
from sdmlx_qwen_native.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.feed_forward import Flux2SwiGLU
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache

try:
    import sdmlx_int8_sdpa_ext as _sdmlx_int8_sdpa_ext
except Exception:
    _sdmlx_int8_sdpa_ext = None


def _sdmlx_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _sdmlx_env_float(name: str, default: float = 1.0) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if parsed == 0.0:
        return default
    return parsed


def _sdmlx_split_to_out_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_SINGLE_TO_OUT_SPLIT", False)


def _sdmlx_to_out_fp16_pretranspose_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_SINGLE_TO_OUT_FP16_PRETRANSPOSE", False)


def _sdmlx_split_to_out_fp16_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_SINGLE_TO_OUT_SPLIT_FP16", False)


def _sdmlx_split_qkv_mlp_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_SINGLE_QKV_MLP_SPLIT", False)


def _sdmlx_qkv_fp16_mlp_bf16_enabled() -> bool:
    return (
        _sdmlx_env_bool("SDMLX_FLUX2_MIXED_FP16", False)
        or _sdmlx_env_bool("SDMLX_FLUX2_SINGLE_QKV_FP16_PROJ", False)
    )


def _sdmlx_all_fp16_proj_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_SINGLE_ALL_FP16_PROJ", False)


def _sdmlx_mlp_fp16_proj_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_SINGLE_MLP_FP16_PROJ", False)


def _sdmlx_full_mlp_split_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_SINGLE_FULL_MLP_SPLIT", False)


def _sdmlx_drawthings_fp16_split_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_SINGLE_DRAWTHINGS_FP16_SPLIT", False)


def _sdmlx_cache_qkv_fp16_weight_t_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_CACHE_QKV_FP16_WEIGHT_T", False)


def _sdmlx_cache_all_fp16_weight_t_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_CACHE_ALL_FP16_WEIGHT_T", False)


def _sdmlx_native_qkv_prep_enabled() -> bool:
    return _sdmlx_env_bool("SDMLX_FLUX2_NATIVE_QKV_PREP", False)


def _sdmlx_qkv_fp16_scale() -> float:
    return _sdmlx_env_float("SDMLX_FLUX2_QKV_FP16_SCALE", 1.0)


def _sdmlx_to_out_fp16_scale() -> float:
    return _sdmlx_env_float("SDMLX_FLUX2_TO_OUT_FP16_SCALE", 1.0)


def _sdmlx_mlp_fp16_scale() -> float:
    return _sdmlx_env_float("SDMLX_FLUX2_MLP_FP16_SCALE", 1.0)


def _sdmlx_can_use_native_qkv_prep(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    image_rotary_emb,
    heads: int,
    dim_head: int,
) -> bool:
    if not _sdmlx_native_qkv_prep_enabled() or _sdmlx_int8_sdpa_ext is None:
        return False
    if image_rotary_emb is None or dim_head != 128:
        return False
    if query.dtype != mx.float16 or key.dtype != mx.float16 or value.dtype != mx.float16:
        return False
    if q_weight.shape != (dim_head,) or k_weight.shape != (dim_head,):
        return False
    batch, seq_len, inner_dim = query.shape
    if inner_dim != heads * dim_head:
        return False
    cos, sin = image_rotary_emb
    return cos.dtype == mx.float32 and sin.dtype == mx.float32 and cos.shape == (seq_len, 64) and sin.shape == (seq_len, 64)


def _sdmlx_native_qkv_prep(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    image_rotary_emb,
    heads: int,
    dim_head: int,
):
    batch, seq_len, _ = query.shape
    query = mx.reshape(query, (batch, seq_len, heads, dim_head))
    key = mx.reshape(key, (batch, seq_len, heads, dim_head))
    value = mx.reshape(value, (batch, seq_len, heads, dim_head))
    cos, sin = image_rotary_emb
    empty_cos = cos[:0]
    empty_sin = sin[:0]
    return _sdmlx_int8_sdpa_ext.qwen_qkv_rmsnorm_rope_split_cos_sin_seq_to_bh_probe(
        query,
        key,
        value,
        q_weight.astype(mx.float16),
        k_weight.astype(mx.float16),
        q_weight.astype(mx.float16),
        k_weight.astype(mx.float16),
        cos,
        sin,
        empty_cos,
        empty_sin,
        0,
        1e-5,
        stream=mx.gpu,
    )


class Flux2ParallelSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float = 3.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.to_qkv_mlp_proj = nn.Linear(dim, self.inner_dim * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.mlp_act = Flux2SwiGLU()
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, dim, bias=False)
        self._sdmlx_to_out_weight_t_fp16 = None
        self._sdmlx_qkv_mlp_weight_t_fp16 = None
        self._sdmlx_qkv_weight_t_fp16 = None
        self._sdmlx_mlp_weight_t = None

    def _call_standard_no_hooks(self, hidden_states: mx.array, image_rotary_emb):
        mlp_hidden_is_activated = False
        if _sdmlx_drawthings_fp16_split_enabled() and hasattr(self.to_qkv_mlp_proj, "weight"):
            weight = self.to_qkv_mlp_proj.weight
            q_weight = weight[: self.inner_dim]
            k_weight = weight[self.inner_dim : self.inner_dim * 2]
            v_weight = weight[self.inner_dim * 2 : self.inner_dim * 3]
            mlp_weight = weight[self.inner_dim * 3 :]
            mlp_gate_weight = mlp_weight[: self.mlp_hidden_dim]
            mlp_value_weight = mlp_weight[self.mlp_hidden_dim :]

            qkv_scale = _sdmlx_qkv_fp16_scale()
            hidden_states_q = (hidden_states * qkv_scale).astype(mx.float16)
            query = hidden_states_q @ mx.transpose(q_weight.astype(mx.float16))
            if qkv_scale != 1.0:
                query = query / qkv_scale
            key = hidden_states @ mx.transpose(k_weight)
            value = hidden_states @ mx.transpose(v_weight)

            mlp_scale = _sdmlx_mlp_fp16_scale()
            hidden_states_mlp = (hidden_states * mlp_scale).astype(mx.float16)
            mlp_gate = hidden_states_mlp @ mx.transpose(mlp_gate_weight.astype(mx.float16))
            if mlp_scale != 1.0:
                mlp_gate = mlp_gate / mlp_scale
            mlp_value = hidden_states @ mx.transpose(mlp_value_weight)
            mlp_hidden = self.mlp_act.gate_fn(mlp_gate) * mlp_value
            mlp_hidden_is_activated = True
        elif _sdmlx_all_fp16_proj_enabled() and hasattr(self.to_qkv_mlp_proj, "weight"):
            proj_scale = _sdmlx_qkv_fp16_scale()
            hidden_states_fp16 = (hidden_states * proj_scale).astype(mx.float16)
            weight = self.to_qkv_mlp_proj.weight
            if _sdmlx_cache_all_fp16_weight_t_enabled():
                weight_t = self._sdmlx_qkv_mlp_weight_t_fp16
                if weight_t is None:
                    weight_t = mx.contiguous(mx.transpose(weight.astype(mx.float16)))
                    self._sdmlx_qkv_mlp_weight_t_fp16 = weight_t
                proj = hidden_states_fp16 @ weight_t
            else:
                proj = hidden_states_fp16 @ mx.transpose(weight.astype(mx.float16))
            if proj_scale != 1.0:
                proj = proj / proj_scale
            qkv, mlp_hidden = mx.split(proj, [self.inner_dim * 3], axis=-1)
            query, key, value = mx.split(qkv, 3, axis=-1)
        elif _sdmlx_qkv_fp16_mlp_bf16_enabled() and hasattr(self.to_qkv_mlp_proj, "weight"):
            weight = self.to_qkv_mlp_proj.weight
            qkv_weight = weight[: self.inner_dim * 3]
            mlp_weight = weight[self.inner_dim * 3 :]
            qkv_scale = _sdmlx_qkv_fp16_scale()
            hidden_states_fp16 = (hidden_states * qkv_scale).astype(mx.float16)
            if _sdmlx_cache_qkv_fp16_weight_t_enabled():
                qkv_weight_t = self._sdmlx_qkv_weight_t_fp16
                if qkv_weight_t is None:
                    qkv_weight_t = mx.contiguous(mx.transpose(qkv_weight.astype(mx.float16)))
                    self._sdmlx_qkv_weight_t_fp16 = qkv_weight_t
                qkv = hidden_states_fp16 @ qkv_weight_t
            else:
                qkv = hidden_states_fp16 @ mx.transpose(qkv_weight.astype(mx.float16))
            if qkv_scale != 1.0:
                qkv = qkv / qkv_scale
            if _sdmlx_full_mlp_split_enabled():
                mlp_gate_weight = mlp_weight[: self.mlp_hidden_dim]
                mlp_value_weight = mlp_weight[self.mlp_hidden_dim :]
                if _sdmlx_mlp_fp16_proj_enabled():
                    mlp_scale = _sdmlx_mlp_fp16_scale()
                    hidden_states_mlp = (hidden_states * mlp_scale).astype(mx.float16)
                    mlp_gate = hidden_states_mlp @ mx.transpose(mlp_gate_weight.astype(mx.float16))
                    mlp_value = hidden_states_mlp @ mx.transpose(mlp_value_weight.astype(mx.float16))
                    if mlp_scale != 1.0:
                        mlp_gate = mlp_gate / mlp_scale
                        mlp_value = mlp_value / mlp_scale
                else:
                    mlp_gate = hidden_states @ mx.transpose(mlp_gate_weight)
                    mlp_value = hidden_states @ mx.transpose(mlp_value_weight)
                mlp_hidden = self.mlp_act.gate_fn(mlp_gate) * mlp_value
                mlp_hidden_is_activated = True
            elif _sdmlx_mlp_fp16_proj_enabled():
                mlp_scale = _sdmlx_mlp_fp16_scale()
                hidden_states_mlp = (hidden_states * mlp_scale).astype(mx.float16)
                mlp_hidden = hidden_states_mlp @ mx.transpose(mlp_weight.astype(mx.float16))
                if mlp_scale != 1.0:
                    mlp_hidden = mlp_hidden / mlp_scale
            elif _sdmlx_cache_qkv_fp16_weight_t_enabled():
                mlp_weight_t = self._sdmlx_mlp_weight_t
                if mlp_weight_t is None:
                    mlp_weight_t = mx.transpose(mlp_weight)
                    self._sdmlx_mlp_weight_t = mlp_weight_t
                mlp_hidden = hidden_states @ mlp_weight_t
            else:
                mlp_hidden = hidden_states @ mx.transpose(mlp_weight)
            query, key, value = mx.split(qkv, 3, axis=-1)
        elif _sdmlx_split_qkv_mlp_enabled() and hasattr(self.to_qkv_mlp_proj, "weight"):
            weight = self.to_qkv_mlp_proj.weight
            q_weight = weight[: self.inner_dim]
            k_weight = weight[self.inner_dim : self.inner_dim * 2]
            v_weight = weight[self.inner_dim * 2 : self.inner_dim * 3]
            mlp_weight = weight[self.inner_dim * 3 :]
            query = hidden_states @ mx.transpose(q_weight)
            key = hidden_states @ mx.transpose(k_weight)
            value = hidden_states @ mx.transpose(v_weight)
            mlp_hidden = hidden_states @ mx.transpose(mlp_weight)
        else:
            proj = self.to_qkv_mlp_proj(hidden_states)
            qkv, mlp_hidden = mx.split(proj, [self.inner_dim * 3], axis=-1)
            query, key, value = mx.split(qkv, 3, axis=-1)

        batch, seq_len, _ = query.shape
        if _sdmlx_can_use_native_qkv_prep(
            query,
            key,
            value,
            self.norm_q.weight,
            self.norm_k.weight,
            image_rotary_emb,
            self.heads,
            self.dim_head,
        ):
            query, key, value = _sdmlx_native_qkv_prep(
                query,
                key,
                value,
                self.norm_q.weight,
                self.norm_k.weight,
                image_rotary_emb,
                self.heads,
                self.dim_head,
            )
        else:
            query = mx.transpose(mx.reshape(query, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3))
            key = mx.transpose(mx.reshape(key, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3))
            value = mx.transpose(mx.reshape(value, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3))

            query = self.norm_q(query.astype(mx.float32)).astype(ModelConfig.precision)
            key = self.norm_k(key.astype(mx.float32)).astype(ModelConfig.precision)

            if image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                query, key = AttentionUtils.apply_rope_bshd(query, key, cos, sin)

        hidden_states = AttentionUtils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch,
            num_heads=self.heads,
            head_dim=self.dim_head,
        )

        if not mlp_hidden_is_activated:
            mlp_hidden = self.mlp_act(mlp_hidden)
        if hidden_states.dtype != mlp_hidden.dtype:
            hidden_states = hidden_states.astype(mlp_hidden.dtype)
        if _sdmlx_split_to_out_fp16_enabled() and hasattr(self.to_out, "weight"):
            to_out_scale = _sdmlx_to_out_fp16_scale()
            weight = self.to_out.weight
            attn_weight = weight[:, : self.inner_dim]
            mlp_weight = weight[:, self.inner_dim :]
            attn_out = (hidden_states * to_out_scale).astype(mx.float16) @ mx.transpose(attn_weight.astype(mx.float16))
            mlp_out = (mlp_hidden * to_out_scale).astype(mx.float16) @ mx.transpose(mlp_weight.astype(mx.float16))
            hidden_states = attn_out + mlp_out
            if to_out_scale != 1.0:
                hidden_states = hidden_states / to_out_scale
            hidden_states = hidden_states.astype(ModelConfig.precision)
        elif _sdmlx_to_out_fp16_pretranspose_enabled() and hasattr(self.to_out, "weight"):
            to_out_scale = _sdmlx_to_out_fp16_scale()
            merged = (mx.concatenate([hidden_states, mlp_hidden], axis=-1) * to_out_scale).astype(mx.float16)
            weight_t = self._sdmlx_to_out_weight_t_fp16
            if weight_t is None:
                weight_t = mx.contiguous(mx.transpose(self.to_out.weight.astype(mx.float16)))
                self._sdmlx_to_out_weight_t_fp16 = weight_t
            hidden_states = merged @ weight_t
            if to_out_scale != 1.0:
                hidden_states = hidden_states / to_out_scale
            hidden_states = hidden_states.astype(ModelConfig.precision)
        elif _sdmlx_split_to_out_enabled() and hasattr(self.to_out, "weight"):
            weight = self.to_out.weight
            attn_weight = weight[:, : self.inner_dim]
            mlp_weight = weight[:, self.inner_dim :]
            out = hidden_states @ mx.transpose(attn_weight)
            out = out + (mlp_hidden @ mx.transpose(mlp_weight))
            bias = getattr(self.to_out, "bias", None)
            if bias is not None:
                out = out + bias
            hidden_states = out
        else:
            hidden_states = mx.concatenate([hidden_states, mlp_hidden], axis=-1)
            hidden_states = self.to_out(hidden_states)
        return hidden_states

    def __call__(
        self,
        hidden_states: mx.array,
        image_rotary_emb,
        kv_cache: Flux2KVCache | None = None,
        kv_cache_layer_idx: int | None = None,
    ):
        hooks_enabled = bool(getattr(self, "sdmlx_flux2_hooks_enabled", False))
        if not hooks_enabled and kv_cache is None:
            return self._call_standard_no_hooks(hidden_states, image_rotary_emb)
        phase_mark = getattr(self, "sdmlx_phase_mark", None) if hooks_enabled else None
        phase_prefix = getattr(self, "sdmlx_phase_prefix", "single.attn") if hooks_enabled else "single.attn"

        def mark(label: str) -> None:
            if phase_mark is not None:
                phase_mark(f"{phase_prefix}.{label}")

        mlp_hidden_is_activated = False
        mark("qkv_mlp_start")
        if _sdmlx_drawthings_fp16_split_enabled() and hasattr(self.to_qkv_mlp_proj, "weight"):
            weight = self.to_qkv_mlp_proj.weight
            q_weight = weight[: self.inner_dim]
            k_weight = weight[self.inner_dim : self.inner_dim * 2]
            v_weight = weight[self.inner_dim * 2 : self.inner_dim * 3]
            mlp_weight = weight[self.inner_dim * 3 :]
            mlp_gate_weight = mlp_weight[: self.mlp_hidden_dim]
            mlp_value_weight = mlp_weight[self.mlp_hidden_dim :]

            qkv_scale = _sdmlx_qkv_fp16_scale()
            hidden_states_q = (hidden_states * qkv_scale).astype(mx.float16)
            query = hidden_states_q @ mx.transpose(q_weight.astype(mx.float16))
            if qkv_scale != 1.0:
                query = query / qkv_scale
            key = hidden_states @ mx.transpose(k_weight)
            value = hidden_states @ mx.transpose(v_weight)

            mlp_scale = _sdmlx_mlp_fp16_scale()
            hidden_states_mlp = (hidden_states * mlp_scale).astype(mx.float16)
            mlp_gate = hidden_states_mlp @ mx.transpose(mlp_gate_weight.astype(mx.float16))
            if mlp_scale != 1.0:
                mlp_gate = mlp_gate / mlp_scale
            mlp_value = hidden_states @ mx.transpose(mlp_value_weight)
            mlp_hidden = self.mlp_act.gate_fn(mlp_gate) * mlp_value
            mlp_hidden_is_activated = True
        elif _sdmlx_all_fp16_proj_enabled() and hasattr(self.to_qkv_mlp_proj, "weight"):
            proj_scale = _sdmlx_qkv_fp16_scale()
            hidden_states_fp16 = (hidden_states * proj_scale).astype(mx.float16)
            weight = self.to_qkv_mlp_proj.weight
            if _sdmlx_cache_all_fp16_weight_t_enabled():
                weight_t = self._sdmlx_qkv_mlp_weight_t_fp16
                if weight_t is None:
                    weight_t = mx.contiguous(mx.transpose(weight.astype(mx.float16)))
                    self._sdmlx_qkv_mlp_weight_t_fp16 = weight_t
                proj = hidden_states_fp16 @ weight_t
            else:
                proj = hidden_states_fp16 @ mx.transpose(weight.astype(mx.float16))
            if proj_scale != 1.0:
                proj = proj / proj_scale
            qkv, mlp_hidden = mx.split(proj, [self.inner_dim * 3], axis=-1)
            query, key, value = mx.split(qkv, 3, axis=-1)
        elif _sdmlx_qkv_fp16_mlp_bf16_enabled() and hasattr(self.to_qkv_mlp_proj, "weight"):
            weight = self.to_qkv_mlp_proj.weight
            qkv_weight = weight[: self.inner_dim * 3]
            mlp_weight = weight[self.inner_dim * 3 :]
            qkv_scale = _sdmlx_qkv_fp16_scale()
            hidden_states_fp16 = (hidden_states * qkv_scale).astype(mx.float16)
            if _sdmlx_cache_qkv_fp16_weight_t_enabled():
                qkv_weight_t = self._sdmlx_qkv_weight_t_fp16
                if qkv_weight_t is None:
                    qkv_weight_t = mx.contiguous(mx.transpose(qkv_weight.astype(mx.float16)))
                    self._sdmlx_qkv_weight_t_fp16 = qkv_weight_t
                qkv = hidden_states_fp16 @ qkv_weight_t
            else:
                qkv = hidden_states_fp16 @ mx.transpose(qkv_weight.astype(mx.float16))
            if qkv_scale != 1.0:
                qkv = qkv / qkv_scale
            if _sdmlx_full_mlp_split_enabled():
                mlp_gate_weight = mlp_weight[: self.mlp_hidden_dim]
                mlp_value_weight = mlp_weight[self.mlp_hidden_dim :]
                if _sdmlx_mlp_fp16_proj_enabled():
                    mlp_scale = _sdmlx_mlp_fp16_scale()
                    hidden_states_mlp = (hidden_states * mlp_scale).astype(mx.float16)
                    mlp_gate = hidden_states_mlp @ mx.transpose(mlp_gate_weight.astype(mx.float16))
                    mlp_value = hidden_states_mlp @ mx.transpose(mlp_value_weight.astype(mx.float16))
                    if mlp_scale != 1.0:
                        mlp_gate = mlp_gate / mlp_scale
                        mlp_value = mlp_value / mlp_scale
                else:
                    mlp_gate = hidden_states @ mx.transpose(mlp_gate_weight)
                    mlp_value = hidden_states @ mx.transpose(mlp_value_weight)
                mlp_hidden = self.mlp_act.gate_fn(mlp_gate) * mlp_value
                mlp_hidden_is_activated = True
            elif _sdmlx_mlp_fp16_proj_enabled():
                mlp_scale = _sdmlx_mlp_fp16_scale()
                hidden_states_mlp = (hidden_states * mlp_scale).astype(mx.float16)
                mlp_hidden = hidden_states_mlp @ mx.transpose(mlp_weight.astype(mx.float16))
                if mlp_scale != 1.0:
                    mlp_hidden = mlp_hidden / mlp_scale
            elif _sdmlx_cache_qkv_fp16_weight_t_enabled():
                mlp_weight_t = self._sdmlx_mlp_weight_t
                if mlp_weight_t is None:
                    mlp_weight_t = mx.transpose(mlp_weight)
                    self._sdmlx_mlp_weight_t = mlp_weight_t
                mlp_hidden = hidden_states @ mlp_weight_t
            else:
                mlp_hidden = hidden_states @ mx.transpose(mlp_weight)
            query, key, value = mx.split(qkv, 3, axis=-1)
        elif _sdmlx_split_qkv_mlp_enabled() and hasattr(self.to_qkv_mlp_proj, "weight"):
            weight = self.to_qkv_mlp_proj.weight
            q_weight = weight[: self.inner_dim]
            k_weight = weight[self.inner_dim : self.inner_dim * 2]
            v_weight = weight[self.inner_dim * 2 : self.inner_dim * 3]
            mlp_weight = weight[self.inner_dim * 3 :]
            query = hidden_states @ mx.transpose(q_weight)
            key = hidden_states @ mx.transpose(k_weight)
            value = hidden_states @ mx.transpose(v_weight)
            mlp_hidden = hidden_states @ mx.transpose(mlp_weight)
        else:
            proj = self.to_qkv_mlp_proj(hidden_states)
            qkv, mlp_hidden = mx.split(proj, [self.inner_dim * 3], axis=-1)
            query, key, value = mx.split(qkv, 3, axis=-1)
        mark("qkv_mlp_done")

        batch, seq_len, _ = query.shape
        mark("prep_start")
        if _sdmlx_can_use_native_qkv_prep(
            query,
            key,
            value,
            self.norm_q.weight,
            self.norm_k.weight,
            image_rotary_emb,
            self.heads,
            self.dim_head,
        ):
            query, key, value = _sdmlx_native_qkv_prep(
                query,
                key,
                value,
                self.norm_q.weight,
                self.norm_k.weight,
                image_rotary_emb,
                self.heads,
                self.dim_head,
            )
        else:
            query = mx.transpose(mx.reshape(query, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3))
            key = mx.transpose(mx.reshape(key, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3))
            value = mx.transpose(mx.reshape(value, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3))

            query = self.norm_q(query.astype(mx.float32)).astype(ModelConfig.precision)
            key = self.norm_k(key.astype(mx.float32)).astype(ModelConfig.precision)

            if image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                query, key = AttentionUtils.apply_rope_bshd(query, key, cos, sin)
        mark("prep_done")

        enhance_state = getattr(self, "sdmlx_flux2_enhance_state", None) if hooks_enabled else None
        enhance_block_type = getattr(self, "sdmlx_flux2_enhance_block_type", "single") if hooks_enabled else "single"
        enhance_block_index = int(getattr(self, "sdmlx_flux2_enhance_block_index", 0)) if hooks_enabled else 0
        enhance_text_len = int(getattr(enhance_state, "text_token_count", 0)) if enhance_state is not None else 0
        enhance_total_seq = int(query.shape[2]) if enhance_state is not None else 0
        attn_mask = None
        if enhance_state is not None and kv_cache is None:
            key, value, attn_mask = enhance_state.apply_kv_controls(
                key,
                value,
                text_len=enhance_text_len,
                total_seq=enhance_total_seq,
            )
            mark("enhance_kv_done")

        if kv_cache is not None:
            kv_cache.store_reference("single", kv_cache_layer_idx, key, value)
            mark("kv_store_done")
            key, value = kv_cache.append_reference("single", kv_cache_layer_idx, key, value)
            mark("kv_append_done")
            hidden_states = kv_cache.compute_extract_attention(
                query=query,
                key=key,
                value=value,
                batch_size=batch,
                num_heads=self.heads,
                head_dim=self.dim_head,
            )
            mark("attention_done")
            kv_cache.attention_barrier(hidden_states)
            mark("barrier_checked")
        else:
            mark("full_attention_start")
            hidden_states = AttentionUtils.compute_attention(
                query=query,
                key=key,
                value=value,
                batch_size=batch,
                num_heads=self.heads,
                head_dim=self.dim_head,
                mask=attn_mask,
            )
            mark("full_attention_done")
            if enhance_state is not None:
                hidden_states = enhance_state.apply_feature_pull(
                    hidden_states,
                    text_len=enhance_text_len,
                    total_seq=enhance_total_seq,
                    block_type=enhance_block_type,
                    block_index=enhance_block_index,
                )
                mark("enhance_feature_done")

        if not mlp_hidden_is_activated:
            mark("mlp_act_start")
            mlp_hidden = self.mlp_act(mlp_hidden)
            mark("mlp_act_done")
        if hidden_states.dtype != mlp_hidden.dtype:
            hidden_states = hidden_states.astype(mlp_hidden.dtype)
        mark("to_out_start")
        if _sdmlx_split_to_out_fp16_enabled() and hasattr(self.to_out, "weight"):
            to_out_scale = _sdmlx_to_out_fp16_scale()
            weight = self.to_out.weight
            attn_weight = weight[:, : self.inner_dim]
            mlp_weight = weight[:, self.inner_dim :]
            attn_out = (hidden_states * to_out_scale).astype(mx.float16) @ mx.transpose(attn_weight.astype(mx.float16))
            mlp_out = (mlp_hidden * to_out_scale).astype(mx.float16) @ mx.transpose(mlp_weight.astype(mx.float16))
            hidden_states = attn_out + mlp_out
            if to_out_scale != 1.0:
                hidden_states = hidden_states / to_out_scale
            hidden_states = hidden_states.astype(ModelConfig.precision)
        elif _sdmlx_to_out_fp16_pretranspose_enabled() and hasattr(self.to_out, "weight"):
            to_out_scale = _sdmlx_to_out_fp16_scale()
            merged = (mx.concatenate([hidden_states, mlp_hidden], axis=-1) * to_out_scale).astype(mx.float16)
            weight_t = self._sdmlx_to_out_weight_t_fp16
            if weight_t is None:
                weight_t = mx.contiguous(mx.transpose(self.to_out.weight.astype(mx.float16)))
                self._sdmlx_to_out_weight_t_fp16 = weight_t
            hidden_states = merged @ weight_t
            if to_out_scale != 1.0:
                hidden_states = hidden_states / to_out_scale
            hidden_states = hidden_states.astype(ModelConfig.precision)
        elif _sdmlx_split_to_out_enabled() and hasattr(self.to_out, "weight"):
            weight = self.to_out.weight
            attn_weight = weight[:, : self.inner_dim]
            mlp_weight = weight[:, self.inner_dim :]
            out = hidden_states @ mx.transpose(attn_weight)
            out = out + (mlp_hidden @ mx.transpose(mlp_weight))
            bias = getattr(self.to_out, "bias", None)
            if bias is not None:
                out = out + bias
            hidden_states = out
        else:
            hidden_states = mx.concatenate([hidden_states, mlp_hidden], axis=-1)
            hidden_states = self.to_out(hidden_states)
        mark("to_out_done")
        return hidden_states
