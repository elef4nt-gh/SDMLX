import os

import mlx.core as mx
from mlx import nn

from sdmlx_qwen_native.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache


def _sdmlx_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _sdmlx_flux2_memory_probe_enabled(phase_prefix: str) -> bool:
    if not _sdmlx_env_bool("SDMLX_FLUX2_MEMORY_PHASE_PROBE", False):
        return False
    target = os.environ.get("SDMLX_FLUX2_MEMORY_PHASE_TARGET", "transformer.step_0.full.double_0.attn")
    return phase_prefix.startswith(target)


class Flux2Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, added_kv_proj_dim: int | None = None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.added_kv_proj_dim = added_kv_proj_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=1e-5)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=1e-5)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.to_add_out = nn.Linear(self.inner_dim, dim, bias=False)

    def _call_standard_no_hooks(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb,
    ):
        query, key, value = AttentionUtils.process_qkv(
            hidden_states=hidden_states,
            to_q=self.to_q,
            to_k=self.to_k,
            to_v=self.to_v,
            norm_q=self.norm_q,
            norm_k=self.norm_k,
            num_heads=self.heads,
            head_dim=self.dim_head,
        )

        enc_query = enc_key = enc_value = None
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            enc_query, enc_key, enc_value = AttentionUtils.process_qkv(
                hidden_states=encoder_hidden_states,
                to_q=self.add_q_proj,
                to_k=self.add_k_proj,
                to_v=self.add_v_proj,
                norm_q=self.norm_added_q,
                norm_k=self.norm_added_k,
                num_heads=self.heads,
                head_dim=self.dim_head,
            )
            query = mx.concatenate([enc_query, query], axis=2)
            key = mx.concatenate([enc_key, key], axis=2)
            value = mx.concatenate([enc_value, value], axis=2)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = AttentionUtils.apply_rope_bshd(query, key, cos, sin)

        hidden_states = AttentionUtils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=hidden_states.shape[0],
            num_heads=self.heads,
            head_dim=self.dim_head,
        )

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        hidden_states = self.to_out(hidden_states)
        return hidden_states, encoder_hidden_states

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb,
        kv_cache: Flux2KVCache | None = None,
        kv_cache_layer_idx: int | None = None,
    ):
        hooks_enabled = bool(getattr(self, "sdmlx_flux2_hooks_enabled", False))
        if not hooks_enabled and kv_cache is None:
            return self._call_standard_no_hooks(hidden_states, encoder_hidden_states, image_rotary_emb)
        phase_mark = getattr(self, "sdmlx_phase_mark", None) if hooks_enabled else None
        phase_prefix = getattr(self, "sdmlx_phase_prefix", "double.attn") if hooks_enabled else "double.attn"
        memory_probe = _sdmlx_flux2_memory_probe_enabled(phase_prefix) if hooks_enabled else False

        def mark(label: str) -> None:
            if phase_mark is not None:
                phase_mark(f"{phase_prefix}.{label}")

        def probe(label: str, *arrays) -> None:
            if not memory_probe or phase_mark is None:
                return
            mx.eval(*arrays)
            phase_mark(f"{phase_prefix}.{label}_probe_eval")

        mark("qkv_start")
        query, key, value = AttentionUtils.process_qkv(
            hidden_states=hidden_states,
            to_q=self.to_q,
            to_k=self.to_k,
            to_v=self.to_v,
            norm_q=self.norm_q,
            norm_k=self.norm_k,
            num_heads=self.heads,
            head_dim=self.dim_head,
        )
        mark("qkv_done")
        probe("qkv", query, key, value)

        enc_query = enc_key = enc_value = None
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            mark("encoder_qkv_start")
            enc_query, enc_key, enc_value = AttentionUtils.process_qkv(
                hidden_states=encoder_hidden_states,
                to_q=self.add_q_proj,
                to_k=self.add_k_proj,
                to_v=self.add_v_proj,
                norm_q=self.norm_added_q,
                norm_k=self.norm_added_k,
                num_heads=self.heads,
                head_dim=self.dim_head,
            )
            query = mx.concatenate([enc_query, query], axis=2)
            key = mx.concatenate([enc_key, key], axis=2)
            value = mx.concatenate([enc_value, value], axis=2)
            mark("encoder_concat_done")
            probe("encoder_concat", query, key, value)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = AttentionUtils.apply_rope_bshd(query, key, cos, sin)
            mark("rope_done")
            probe("rope", query, key, value)

        enhance_state = getattr(self, "sdmlx_flux2_enhance_state", None) if hooks_enabled else None
        enhance_block_type = getattr(self, "sdmlx_flux2_enhance_block_type", "double") if hooks_enabled else "double"
        enhance_block_index = int(getattr(self, "sdmlx_flux2_enhance_block_index", 0)) if hooks_enabled else 0
        enhance_text_len = int(encoder_hidden_states.shape[1]) if enhance_state is not None and encoder_hidden_states is not None else 0
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
            kv_cache.store_reference("double", kv_cache_layer_idx, key, value)
            mark("kv_store_done")
            key, value = kv_cache.append_reference("double", kv_cache_layer_idx, key, value)
            mark("kv_append_done")
            hidden_states = kv_cache.compute_extract_attention(
                query=query,
                key=key,
                value=value,
                batch_size=hidden_states.shape[0],
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
                batch_size=hidden_states.shape[0],
                num_heads=self.heads,
                head_dim=self.dim_head,
                mask=attn_mask,
            )
            mark("full_attention_done")
            probe("full_attention", hidden_states)
            if enhance_state is not None:
                hidden_states = enhance_state.apply_feature_pull(
                    hidden_states,
                    text_len=enhance_text_len,
                    total_seq=enhance_total_seq,
                    block_type=enhance_block_type,
                    block_index=enhance_block_index,
                )
                mark("enhance_feature_done")

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
            mark("encoder_to_out_done")
            probe("encoder_to_out", encoder_hidden_states, hidden_states)

        hidden_states = self.to_out(hidden_states)
        mark("to_out_done")
        probe("to_out", hidden_states)
        return hidden_states, encoder_hidden_states
