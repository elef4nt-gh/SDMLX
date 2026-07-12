from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
from mlx import nn

from sdmlx_qwen_native.models.common.config.config import Config
from sdmlx_qwen_native.models.common.config.model_config import ModelConfig
from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX
from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_time_text_embed import QwenTimeTextEmbed
from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock
from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_transformer_rms_norm import QwenTransformerRMSNorm


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=False)
        self.norm = nn.LayerNorm(dims=embedding_dim, eps=1e-6, affine=False)

    def __call__(self, x: mx.array, text_embeddings: mx.array) -> mx.array:
        text_embeddings = self.linear(nn.silu(text_embeddings).astype(ModelConfig.precision))
        chunk_size = self.embedding_dim
        scale = text_embeddings[:, 0 * chunk_size : 1 * chunk_size]
        shift = text_embeddings[:, 1 * chunk_size : 2 * chunk_size]
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class QwenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        patch_size: int = 2,
        fp16_activation_scaling: bool | None = None,
        activation_scaling_mode: str | None = None,
        activation_scaling_profile: str = "qwen_image_edit",
    ) -> None:
        super().__init__()
        if activation_scaling_mode is None:
            activation_scaling_mode = _env_activation_scaling_mode()
        if fp16_activation_scaling is None:
            fp16_activation_scaling = activation_scaling_mode != "off"
        self.fp16_activation_scaling = fp16_activation_scaling
        self.activation_scaling_mode = activation_scaling_mode if self.fp16_activation_scaling else "off"
        self.activation_scaling_profile = str(activation_scaling_profile or "qwen_image_edit")
        if self.fp16_activation_scaling and _env_flag("SDMLX_QWEN_VERBOSE"):
            print(
                "SDMLX Qwen: Draw Things FP16 activation scaling enabled "
                f"({self.activation_scaling_mode}, profile={self.activation_scaling_profile})"
            )

        self.inner_dim = num_attention_heads * attention_head_dim
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_norm = QwenTransformerRMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=self.inner_dim)
        self.pos_embed = QwenEmbedRopeMLX(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        self.transformer_blocks = [
            QwenTransformerBlock(
                dim=self.inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                layer_idx=i,
                num_layers=num_layers,
                fp16_activation_scaling=self.fp16_activation_scaling,
                activation_scaling_mode=self.activation_scaling_mode,
                activation_scaling_profile=self.activation_scaling_profile,
            )
            for i in range(num_layers)
        ]  # fmt: off
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)
        self._timestep_mod_cache: dict[tuple, tuple[mx.array, list[tuple[mx.array, mx.array, mx.array, mx.array]]]] = {}
        self._timestep_mod_cache_order: list[tuple] = []

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        qwen_image_ids: mx.array | None = None,
        cond_image_grid: tuple[int, int, int] | None = None,
        timestep_zero_index: int | None = None,
        image_rotary_embeddings: tuple[mx.array, mx.array] | None = None,
        preprojected_image_tail: mx.array | None = None,
        projected_encoder_hidden_states: mx.array | None = None,
        block_attention_mask: mx.array | None = None,
    ) -> mx.array:
        timestep = QwenTransformer._compute_timestep(t, config)
        timestep_mod_cache_key = self._timestep_mod_cache_key(
            t=t,
            config=config,
            hidden_dtype=hidden_states.dtype,
            batch_size=int(hidden_states.shape[0]),
            timestep_zero_index=timestep_zero_index,
        )
        return self.forward_with_timestep(
            timestep=timestep,
            config=config,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            qwen_image_ids=qwen_image_ids,
            cond_image_grid=cond_image_grid,
            timestep_zero_index=timestep_zero_index,
            image_rotary_embeddings=image_rotary_embeddings,
            preprojected_image_tail=preprojected_image_tail,
            projected_encoder_hidden_states=projected_encoder_hidden_states,
            block_attention_mask=block_attention_mask,
            timestep_mod_cache_key=timestep_mod_cache_key,
        )

    def forward_with_timestep(
        self,
        timestep: mx.array,
        config: Config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        qwen_image_ids: mx.array | None = None,
        cond_image_grid: tuple[int, int, int] | None = None,
        timestep_zero_index: int | None = None,
        image_rotary_embeddings: tuple[mx.array, mx.array] | None = None,
        preprojected_image_tail: mx.array | None = None,
        projected_encoder_hidden_states: mx.array | None = None,
        block_attention_mask: mx.array | None = None,
        timestep_mod_cache_key: tuple | None = None,
    ) -> mx.array:
        hidden_states = self.project_image_hidden_states(hidden_states)
        if preprojected_image_tail is not None:
            hidden_states = mx.concatenate([hidden_states, preprojected_image_tail], axis=1)
        batch_size = hidden_states.shape[0]
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)
        if timestep_zero_index is not None:
            timestep = mx.concatenate([timestep, mx.zeros_like(timestep)], axis=0)
        if projected_encoder_hidden_states is None:
            encoder_hidden_states = self.project_encoder_hidden_states(encoder_hidden_states)
        else:
            encoder_hidden_states = projected_encoder_hidden_states
        cached_mods = self._get_timestep_mod_cache(timestep_mod_cache_key)
        if cached_mods is not None:
            text_embeddings, block_mod_params = cached_mods
        else:
            text_embeddings = self.time_text_embed(timestep, hidden_states)
            block_mod_params = None
            if timestep_mod_cache_key is not None:
                block_mod_params = [
                    block.compute_mod_params(text_embeddings, timestep_zero_index=timestep_zero_index)
                    for block in self.transformer_blocks
                ]
                self._set_timestep_mod_cache(timestep_mod_cache_key, text_embeddings, block_mod_params)
        if image_rotary_embeddings is None:
            image_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                pos_embed=self.pos_embed,
                config=config,
                cond_image_grid=cond_image_grid,
            )
        if block_attention_mask is None:
            block_attention_mask = QwenTransformer._block_attention_mask(encoder_hidden_states_mask)
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=block_attention_mask,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                timestep_zero_index=timestep_zero_index,
                mod_params=None if block_mod_params is None else block_mod_params[idx],
            )
            if _env_flag("SDMLX_QWEN_BLOCK_FINITE_DIAGNOSTICS"):
                QwenTransformer._print_block_finite_if_needed(idx, "img_hidden", hidden_states)
                QwenTransformer._print_block_finite_if_needed(idx, "txt_hidden", encoder_hidden_states)
        if timestep_zero_index is not None:
            text_embeddings = mx.split(text_embeddings, 2, axis=0)[0]
        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

    def _get_timestep_mod_cache(
        self,
        cache_key: tuple | None,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array, mx.array, mx.array]]] | None:
        if cache_key is None:
            return None
        return self._timestep_mod_cache.get(cache_key)

    def _set_timestep_mod_cache(
        self,
        cache_key: tuple | None,
        text_embeddings: mx.array,
        block_mod_params: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    ) -> None:
        if cache_key is None:
            return
        flat_mods = [part for params in block_mod_params for part in params]
        mx.eval(text_embeddings, *flat_mods)
        self._timestep_mod_cache[cache_key] = (text_embeddings, block_mod_params)
        self._timestep_mod_cache_order.append(cache_key)
        limit = _env_int("SDMLX_QWEN_TIMESTEP_MOD_CACHE_LIMIT", 32)
        while limit >= 0 and len(self._timestep_mod_cache_order) > limit:
            old_key = self._timestep_mod_cache_order.pop(0)
            self._timestep_mod_cache.pop(old_key, None)

    @staticmethod
    def _timestep_mod_cache_key(
        *,
        t: int | float,
        config: Config,
        hidden_dtype,
        batch_size: int,
        timestep_zero_index: int | None,
    ) -> tuple | None:
        if not _env_flag("SDMLX_QWEN_TIMESTEP_MOD_CACHE") or _env_disabled("SDMLX_QWEN_TIMESTEP_MOD_CACHE"):
            return None
        scheduler_name = getattr(config, "_scheduler_str", None)
        if scheduler_name not in {"linear", "comfy_aura_simple"}:
            return None
        try:
            timestep_index = int(t)
        except (TypeError, ValueError):
            return None
        flow_shift = getattr(config, "flow_shift", None)
        flow_shift_key = None if flow_shift is None else round(float(flow_shift), 6)
        return (
            scheduler_name,
            int(config.num_inference_steps),
            int(config.width),
            int(config.height),
            flow_shift_key,
            timestep_index,
            str(hidden_dtype),
            int(batch_size),
            timestep_zero_index is not None,
        )

    def project_image_hidden_states(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.img_in(hidden_states)
        if self.fp16_activation_scaling:
            hidden_states = hidden_states.astype(mx.float32)
        return hidden_states

    def project_encoder_hidden_states(self, encoder_hidden_states: mx.array) -> mx.array:
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        if self.fp16_activation_scaling:
            encoder_hidden_states = encoder_hidden_states.astype(mx.float32)
        return encoder_hidden_states

    def apply_drawthings_folded_weight_scaling(self) -> None:
        if getattr(self, "_drawthings_folded_weights_applied", False):
            return
        if self.activation_scaling_mode != "folded":
            return
        for block in self.transformer_blocks:
            block.apply_drawthings_folded_weight_scaling()
        self._drawthings_folded_weights_applied = True

    @staticmethod
    def _apply_transformer_block(
        idx: int,
        block: QwenTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: tuple[mx.array, mx.array],
        timestep_zero_index: int | None = None,
        mod_params: tuple[mx.array, mx.array, mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, mx.array]:
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=text_embeddings,
            image_rotary_emb=image_rotary_embeddings,
            block_idx=idx,
            timestep_zero_index=timestep_zero_index,
            mod_params=mod_params,
        )

    @staticmethod
    def _compute_timestep(
        t: int | float,
        config: Config,
    ) -> mx.array:
        if isinstance(t, int):
            if t < len(config.scheduler.sigmas):
                timestep_idx = t
                time_step = config.scheduler.sigmas[timestep_idx]
            else:
                timestep_idx = None
                for idx, ts in enumerate(config.scheduler.timesteps):
                    if abs(int(ts.item()) - t) < 1:
                        timestep_idx = idx
                        break
                if timestep_idx is None:
                    time_step = t / 1000.0
                else:
                    time_step = config.scheduler.sigmas[timestep_idx]
        else:
            timestep_idx = None
            time_step = t

        timestep = mx.array(np.full((1,), time_step, dtype=np.float32))
        return timestep

    @staticmethod
    def _compute_rotary_embeddings(
        encoder_hidden_states_mask: mx.array,
        pos_embed: QwenEmbedRopeMLX,
        config: Config,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    ) -> tuple[mx.array, mx.array]:
        latent_height = config.height // 16
        latent_width = config.width // 16

        if cond_image_grid is None:
            img_shapes = [(1, latent_height, latent_width)]
        else:
            if isinstance(cond_image_grid, list):
                img_shapes = [(1, latent_height, latent_width)] + cond_image_grid
            else:
                img_shapes = [(1, latent_height, latent_width), cond_image_grid]

        txt_seq_lens = [int(mx.sum(encoder_hidden_states_mask[i]).item()) for i in range(encoder_hidden_states_mask.shape[0])]  # fmt: off
        img_rotary_emb, txt_rotary_emb = pos_embed(video_fhw=img_shapes, txt_seq_lens=txt_seq_lens)
        return img_rotary_emb, txt_rotary_emb

    @staticmethod
    def _block_attention_mask(mask: mx.array | None) -> mx.array | None:
        if mask is None:
            return None
        if bool(mx.all(mask >= 0.999).item()):
            return None
        return mask

    @staticmethod
    def _print_block_finite_if_needed(idx: int, name: str, tensor: mx.array) -> None:
        arr = tensor.astype(mx.float32)
        finite = mx.isfinite(arr)
        finite_ratio = mx.mean(finite.astype(mx.float32))
        mx.eval(finite_ratio)
        ratio = float(finite_ratio.item())
        if ratio >= 1.0 and not _env_flag("SDMLX_QWEN_BLOCK_FINITE_ALL"):
            return
        nan_count = mx.sum(mx.isnan(arr).astype(mx.int32))
        inf_count = mx.sum(mx.isinf(arr).astype(mx.int32))
        safe = mx.where(finite, arr, mx.zeros_like(arr))
        rms = mx.sqrt(mx.mean(mx.square(safe)))
        max_abs = mx.max(mx.abs(safe))
        mx.eval(nan_count, inf_count, rms, max_abs)
        print(
            "SDMLX Qwen block finite "
            f"block={idx} {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"finite={ratio:.6f} nan={int(nan_count.item())} inf={int(inf_count.item())} "
            f"rms={float(rms.item()):.6f} max_abs={float(max_abs.item()):.6f}"
        )


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


def _env_activation_scaling_mode() -> str:
    if _env_flag("SDMLX_QWEN_DRAWTHINGS_FOLDED"):
        return "folded"
    value = os.environ.get("SDMLX_QWEN_FP16_ACTIVATION_SCALING", "").strip().lower()
    if value in {"folded", "fold"}:
        return "folded"
    if value in {"1", "true", "yes", "on", "runtime"}:
        return "runtime"
    if value in {"0", "false", "no", "off", "disabled"}:
        return "off"
    return "runtime"
