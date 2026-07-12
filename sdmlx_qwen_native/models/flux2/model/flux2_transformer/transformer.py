import mlx.core as mx
from mlx import nn

from sdmlx_qwen_native.models.common.config.model_config import ModelConfig
from sdmlx_qwen_native.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.modulation import Flux2Modulation
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.pos_embed import Flux2PosEmbed
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.single_transformer_block import Flux2SingleTransformerBlock
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.timestep_guidance_embeddings import Flux2TimestepGuidanceEmbeddings
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.transformer_block import Flux2TransformerBlock


class Flux2Transformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 5,
        num_single_layers: int = 20,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 7680,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        guidance_embeds: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            guidance_embeds=guidance_embeds,
        )
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1)

        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)
        self.transformer_blocks = [
            Flux2TransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ]
        self.single_transformer_blocks = [
            Flux2SingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_single_layers)
        ]
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

    def _call_standard_no_hooks(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array | float | int,
        img_ids: mx.array,
        txt_ids: mx.array,
        guidance: mx.array | float | int | None = None,
    ) -> mx.array:
        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep, dtype=hidden_states.dtype)
        if timestep.ndim == 0:
            timestep = mx.full((hidden_states.shape[0],), timestep, dtype=hidden_states.dtype)
        timestep = timestep.astype(hidden_states.dtype)
        timestep_scale = mx.where(mx.max(timestep) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
        timestep = timestep * timestep_scale
        if guidance is not None:
            if not isinstance(guidance, mx.array):
                guidance = mx.array(guidance, dtype=hidden_states.dtype)
            if guidance.ndim == 0:
                guidance = mx.full((hidden_states.shape[0],), guidance, dtype=hidden_states.dtype)
            guidance = guidance.astype(hidden_states.dtype)
            guidance_scale = mx.where(mx.max(guidance) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
            guidance = guidance * guidance_scale
        temb = self.time_guidance_embed(timestep, guidance)
        temb = temb.astype(ModelConfig.precision)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
            mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
        )

        temb_mod_params_img = self.double_stream_modulation_img(temb)
        temb_mod_params_txt = self.double_stream_modulation_txt(temb)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=temb_mod_params_img,
                temb_mod_params_txt=temb_mod_params_txt,
                image_rotary_emb=concat_rotary_emb,
            )

        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        temb_mod_params_single = self.single_stream_modulation(temb)[0]
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb_mod_params=temb_mod_params_single,
                image_rotary_emb=concat_rotary_emb,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array | float | int,
        img_ids: mx.array,
        txt_ids: mx.array,
        guidance: mx.array | float | int | None = None,
        kv_cache: Flux2KVCache | None = None,
    ) -> mx.array:
        phase_mark = getattr(self, "sdmlx_phase_mark", None)
        step_idx = getattr(self, "sdmlx_phase_step_idx", "?")
        step_mode = getattr(self, "sdmlx_phase_step_mode", "unknown")
        enhance_state = getattr(self, "sdmlx_flux2_enhance_state", None)
        hooks_enabled = phase_mark is not None or enhance_state is not None
        eval_clear_each_block = bool(getattr(self, "sdmlx_eval_clear_each_block", False))
        if not hooks_enabled and kv_cache is None and not eval_clear_each_block:
            return self._call_standard_no_hooks(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
            )

        def mark(label: str) -> None:
            if phase_mark is not None:
                phase_mark(f"transformer.step_{step_idx}.{step_mode}.{label}")

        mark("start")
        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep, dtype=hidden_states.dtype)
        if timestep.ndim == 0:
            timestep = mx.full((hidden_states.shape[0],), timestep, dtype=hidden_states.dtype)
        timestep = timestep.astype(hidden_states.dtype)
        timestep_scale = mx.where(mx.max(timestep) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
        timestep = timestep * timestep_scale
        if guidance is not None:
            if not isinstance(guidance, mx.array):
                guidance = mx.array(guidance, dtype=hidden_states.dtype)
            if guidance.ndim == 0:
                guidance = mx.full((hidden_states.shape[0],), guidance, dtype=hidden_states.dtype)
            guidance = guidance.astype(hidden_states.dtype)
            guidance_scale = mx.where(mx.max(guidance) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
            guidance = guidance * guidance_scale
        temb = self.time_guidance_embed(timestep, guidance)
        temb = temb.astype(ModelConfig.precision)
        ref_temb = None
        if kv_cache is not None and kv_cache.mode == "extract" and kv_cache.num_ref_tokens > 0:
            ref_temb = self.time_guidance_embed(mx.zeros_like(timestep), guidance)
            ref_temb = ref_temb.astype(ModelConfig.precision)
        mark("time_guidance_done")

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        mark("embed_done")
        target_len = int(hidden_states.shape[1])
        if kv_cache is not None and kv_cache.mode == "extract" and kv_cache.num_ref_tokens > 0:
            target_len = max(0, target_len - int(kv_cache.num_ref_tokens))
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
            mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
        )
        mark("rope_concat_done")

        temb_mod_params_img = self.double_stream_modulation_img(temb)
        temb_mod_params_txt = self.double_stream_modulation_txt(temb)
        if ref_temb is not None:
            ref_temb_mod_params_img = self.double_stream_modulation_img(ref_temb)
            temb_mod_params_img = tuple(
                self._blend_trailing_ref_mod_params(
                    mod_params=mod_params,
                    ref_mod_params=ref_mod_params,
                    seq_len=hidden_states.shape[1],
                    num_ref_tokens=kv_cache.num_ref_tokens,
                )
                for mod_params, ref_mod_params in zip(temb_mod_params_img, ref_temb_mod_params_img, strict=True)
            )
        mark("double_modulation_done")

        clear_cache_after_block_eval = bool(getattr(self, "sdmlx_clear_cache_after_block_eval", True))
        eval_clear_policy = str(getattr(self, "sdmlx_eval_clear_policy", "all") or "all").strip().lower()

        def should_eval_clear_block(kind: str, idx: int) -> bool:
            if not eval_clear_each_block:
                return False
            if eval_clear_policy in {"", "all", "true", "1"}:
                return True
            if eval_clear_policy == "double_only":
                return kind == "double"
            if eval_clear_policy == "single_only":
                return kind == "single"
            if eval_clear_policy == "first_each_stage":
                return idx == 0
            if eval_clear_policy == "interval4":
                return idx % 4 == 0
            if ":" in eval_clear_policy:
                for part in eval_clear_policy.split(";"):
                    part = part.strip()
                    if not part or ":" not in part:
                        continue
                    part_kind, values = part.split(":", 1)
                    if part_kind.strip() != kind:
                        continue
                    try:
                        selected = {int(value.strip()) for value in values.split(",") if value.strip()}
                    except ValueError:
                        return False
                    return idx in selected
            return False

        for idx, block in enumerate(self.transformer_blocks):
            if hooks_enabled:
                block.attn.sdmlx_flux2_hooks_enabled = True
                block.attn.sdmlx_phase_mark = phase_mark
                block.attn.sdmlx_phase_prefix = f"transformer.step_{step_idx}.{step_mode}.double_{idx}.attn"
                block.attn.sdmlx_flux2_enhance_state = enhance_state
                block.attn.sdmlx_flux2_enhance_block_type = "double"
                block.attn.sdmlx_flux2_enhance_block_index = idx
            elif bool(getattr(block.attn, "sdmlx_flux2_hooks_enabled", False)):
                block.attn.sdmlx_flux2_hooks_enabled = False
                block.attn.sdmlx_phase_mark = None
                block.attn.sdmlx_flux2_enhance_state = None
            mark(f"double_{idx}.start")
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=temb_mod_params_img,
                temb_mod_params_txt=temb_mod_params_txt,
                image_rotary_emb=concat_rotary_emb,
                kv_cache=kv_cache,
                kv_cache_layer_idx=idx,
            )
            if should_eval_clear_block("double", idx):
                mark(f"double_{idx}.eval_clear_start")
                mx.eval(encoder_hidden_states, hidden_states)
                if clear_cache_after_block_eval:
                    mx.clear_cache()
                mark(f"double_{idx}.eval_clear_done")
            mark(f"double_{idx}.done")

        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        mark("single_concat_done")

        temb_mod_params_single = self.single_stream_modulation(temb)[0]
        if ref_temb is not None:
            ref_temb_mod_params_single = self.single_stream_modulation(ref_temb)[0]
            temb_mod_params_single = self._blend_trailing_ref_mod_params(
                mod_params=temb_mod_params_single,
                ref_mod_params=ref_temb_mod_params_single,
                seq_len=hidden_states.shape[1],
                num_ref_tokens=kv_cache.num_ref_tokens,
            )
        mark("single_modulation_done")
        for idx, block in enumerate(self.single_transformer_blocks):
            if hooks_enabled:
                block.attn.sdmlx_flux2_hooks_enabled = True
                block.attn.sdmlx_phase_mark = phase_mark
                block.attn.sdmlx_phase_prefix = f"transformer.step_{step_idx}.{step_mode}.single_{idx}.attn"
                block.attn.sdmlx_flux2_enhance_state = enhance_state
                block.attn.sdmlx_flux2_enhance_block_type = "single"
                block.attn.sdmlx_flux2_enhance_block_index = idx
            elif bool(getattr(block.attn, "sdmlx_flux2_hooks_enabled", False)):
                block.attn.sdmlx_flux2_hooks_enabled = False
                block.attn.sdmlx_phase_mark = None
                block.attn.sdmlx_flux2_enhance_state = None
            mark(f"single_{idx}.start")
            hidden_states = block(
                hidden_states=hidden_states,
                temb_mod_params=temb_mod_params_single,
                image_rotary_emb=concat_rotary_emb,
                kv_cache=kv_cache,
                kv_cache_layer_idx=idx,
            )
            if should_eval_clear_block("single", idx):
                mark(f"single_{idx}.eval_clear_start")
                mx.eval(hidden_states)
                if clear_cache_after_block_eval:
                    mx.clear_cache()
                mark(f"single_{idx}.eval_clear_done")
            mark(f"single_{idx}.done")

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        if kv_cache is not None and kv_cache.mode == "extract" and kv_cache.num_ref_tokens > 0:
            hidden_states = hidden_states[:, : -kv_cache.num_ref_tokens, ...]
        mark("pre_output_done")
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        mark("output_done")
        return hidden_states

    @staticmethod
    def _blend_trailing_ref_mod_params(
        *,
        mod_params: tuple[mx.array, ...],
        ref_mod_params: tuple[mx.array, ...],
        seq_len: int,
        num_ref_tokens: int,
    ) -> tuple[mx.array, ...]:
        non_ref_tokens = seq_len - num_ref_tokens
        blended = []
        for mod_param, ref_mod_param in zip(mod_params, ref_mod_params, strict=True):
            mod_expanded = mx.broadcast_to(mod_param, (mod_param.shape[0], non_ref_tokens, mod_param.shape[-1]))
            ref_expanded = mx.broadcast_to(
                ref_mod_param,
                (ref_mod_param.shape[0], num_ref_tokens, ref_mod_param.shape[-1]),
            )
            blended.append(mx.concatenate([mod_expanded, ref_expanded], axis=1))
        return tuple(blended)
