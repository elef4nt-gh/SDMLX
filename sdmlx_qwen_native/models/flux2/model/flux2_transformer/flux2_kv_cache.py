from __future__ import annotations

from typing import Literal

import mlx.core as mx

from sdmlx_qwen_native.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils

CacheMode = Literal["extract", "cached"]
StreamType = Literal["double", "single"]


class Flux2KVCache:
    def __init__(self, num_double_layers: int, num_single_layers: int) -> None:
        self._double: list[tuple[mx.array, mx.array] | None] = [None] * num_double_layers
        self._single: list[tuple[mx.array, mx.array] | None] = [None] * num_single_layers
        self.mode: CacheMode | None = None
        self.num_ref_tokens: int = 0
        self.sdmlx_attention_barrier: bool = False
        self.sdmlx_attention_clear_cache: bool = True
        self.sdmlx_attention_first_step_barriers: int = 0
        self._sdmlx_attention_step_idx: int = 0
        self._sdmlx_attention_step_position: int = 0
        self._sdmlx_attention_sites_per_step: int = num_double_layers + num_single_layers

    def configure(
        self,
        *,
        mode: CacheMode,
        num_ref_tokens: int,
    ) -> None:
        self.mode = mode
        self.num_ref_tokens = int(num_ref_tokens)

    def start_step(self, step_idx: int) -> None:
        self._sdmlx_attention_step_idx = int(step_idx)
        self._sdmlx_attention_step_position = 0

    @property
    def has_reference_tokens(self) -> bool:
        return self.num_ref_tokens > 0

    @property
    def is_extracting(self) -> bool:
        return self.mode == "extract" and self.has_reference_tokens

    @property
    def is_cached(self) -> bool:
        return self.mode == "cached"

    def store_reference(self, stream: StreamType, layer_idx: int | None, key: mx.array, value: mx.array) -> None:
        if not self.is_extracting:
            return
        # sdmlx_qwen_native uses [txt, target, ref], so reference tokens are the trailing slice.
        # Materialize the slice so the cache holds only compact K/V tensors, not the full lazy graph.
        ref_key = mx.contiguous(key[:, :, -self.num_ref_tokens :, :])
        ref_value = mx.contiguous(value[:, :, -self.num_ref_tokens :, :])
        mx.eval(ref_key, ref_value)
        self._slots(stream)[self._layer_idx(layer_idx)] = (
            ref_key,
            ref_value,
        )

    def append_reference(
        self,
        stream: StreamType,
        layer_idx: int | None,
        key: mx.array,
        value: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if not self.is_cached:
            return key, value
        ref_key, ref_value = self._load(stream, layer_idx)
        return mx.concatenate([key, ref_key], axis=2), mx.concatenate([value, ref_value], axis=2)

    def attention_barrier(self, *arrays: mx.array) -> None:
        if not self.sdmlx_attention_barrier or not self.has_reference_tokens:
            return
        if self._sdmlx_attention_step_idx != 0:
            return
        barriers = max(0, int(self.sdmlx_attention_first_step_barriers))
        if barriers <= 0:
            return
        self._sdmlx_attention_step_position += 1
        sites = max(1, int(self._sdmlx_attention_sites_per_step))
        if barriers < sites:
            previous_slot = ((self._sdmlx_attention_step_position - 1) * barriers) // sites
            current_slot = (self._sdmlx_attention_step_position * barriers) // sites
            if current_slot == previous_slot:
                return
        mx.eval(*arrays)
        if self.sdmlx_attention_clear_cache:
            mx.clear_cache()

    def compute_extract_attention(
        self,
        *,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> mx.array:
        if not self.is_extracting:
            return AttentionUtils.compute_attention(
                query=query,
                key=key,
                value=value,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
            )

        non_ref_count = query.shape[2] - self.num_ref_tokens
        non_ref_attn = AttentionUtils.compute_attention(
            query=query[:, :, :non_ref_count, :],
            key=key,
            value=value,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        ref_attn = AttentionUtils.compute_attention(
            query=query[:, :, non_ref_count:, :],
            key=key[:, :, non_ref_count:, :],
            value=value[:, :, non_ref_count:, :],
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        return mx.concatenate([non_ref_attn, ref_attn], axis=1)

    def _slots(self, stream: StreamType) -> list[tuple[mx.array, mx.array] | None]:
        if stream == "double":
            return self._double
        if stream == "single":
            return self._single
        raise ValueError(f"Unknown stream {stream!r}")

    @staticmethod
    def _layer_idx(layer_idx: int | None) -> int:
        if layer_idx is None:
            raise ValueError("KV cache layer index is required")
        return layer_idx

    def _load(self, stream: StreamType, layer_idx: int | None) -> tuple[mx.array, mx.array]:
        slot = self._slots(stream)[self._layer_idx(layer_idx)]
        if slot is None:
            raise RuntimeError(f"KV cache slot for {stream} layer {layer_idx} is empty")
        return slot
