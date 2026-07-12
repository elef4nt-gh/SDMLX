from pathlib import Path
from typing import Callable
import gc
import os

import mlx.core as mx
from mlx import nn

from sdmlx_qwen_native.models.common.config.config import Config
from sdmlx_qwen_native.models.common.config.model_config import ModelConfig
from sdmlx_qwen_native.models.flux2.flux2_initializer import Flux2Initializer
from sdmlx_qwen_native.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.flux2_kv_cache import CacheMode, Flux2KVCache
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from sdmlx_qwen_native.models.flux2.model.flux2_vae.vae import Flux2VAE
from sdmlx_qwen_native.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from sdmlx_qwen_native.utils.apple_silicon import AppleSiliconUtil
from sdmlx_qwen_native.utils.exceptions import StopImageGenerationException
from sdmlx_qwen_native.utils.generated_image import GeneratedImage
from sdmlx_qwen_native.utils.image_util import ImageUtil


def _uses_classical_cfg(model_config: ModelConfig) -> bool:
    name = model_config.model_name.lower()
    return "klein-base" in name or "klein_base" in name


class Flux2KleinEdit(nn.Module):
    vae: Flux2VAE
    transformer: Flux2Transformer
    text_encoder: Qwen3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
        vae_variant: str = "standard",
        vae_path: str | None = None,
    ):
        super().__init__()
        Flux2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config or ModelConfig.flux2_klein_4b(),
            vae_variant=vae_variant,
            vae_path=vae_path,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_paths: list[Path | str] | None = None,
        reference_sizes: list[tuple[int, int]] | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
        use_kv_cache: bool | None = None,
        flow_shift: float | None = None,
        interrupt_callback: Callable[[], None] | None = None,
        profile_callback: Callable[[str], None] | None = None,
    ) -> GeneratedImage:
        def mark(label: str) -> None:
            if profile_callback is not None:
                profile_callback(label)

        mark("generate.start")
        primary_image_path = None
        if image_paths:
            primary_image_path = image_paths[0]

        # 0. Create a new config based on the model type and input parameters
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=primary_image_path,
            image_strength=image_strength,
            scheduler=scheduler,
            flow_shift=flow_shift,
        )
        mark("generate.config_ready")
        # 1. Encode prompt(s)
        prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt="" if _uses_classical_cfg(self.model_config) else None,
            guidance=guidance,
        )
        mark("generate.text_encoded")
        self._release_text_encoder_after_encode(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
        )
        mark("generate.text_release_done")

        # 2. Prepare latents
        latents, latent_ids, latent_height, latent_width = _Flux2KleinEditHelpers.prepare_generation_latents(
            seed=seed,
            height=config.height,
            width=config.width,
        )
        mark("generate.target_latents_ready")

        # 3. Reference image conditioning (edit-style, concat reference tokens)
        image_latents, image_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
            vae=self.vae,
            tiling_config=self.tiling_config,
            image_paths=image_paths,
            reference_sizes=reference_sizes,
            height=config.height,
            width=config.width,
            batch_size=latents.shape[0],
            profile_callback=profile_callback,
        )
        mark("generate.reference_latents_ready")
        if image_latents is not None and image_latent_ids is not None:
            mx.eval(image_latents, image_latent_ids)
            mark("generate.reference_latents_eval")
            clear_after_reference_latents = bool(getattr(self, "sdmlx_clear_after_reference_latents", True))
            env_clear_after_reference_latents = os.environ.get("SDMLX_FLUX2_CLEAR_AFTER_REFERENCE_LATENTS")
            if env_clear_after_reference_latents is not None:
                clear_after_reference_latents = env_clear_after_reference_latents.lower() in {"1", "true", "yes", "on"}
            if clear_after_reference_latents:
                mx.clear_cache()
                mark("generate.reference_latents_clear")

        cache_enabled = (
            (use_kv_cache if use_kv_cache is not None else self.model_config.supports_kv_cache)
            and image_latents is not None
            and image_latents.shape[1] > 0
        )
        num_ref_tokens = int(image_latents.shape[1]) if cache_enabled else 0
        kv_cache, negative_kv_cache = self._create_kv_caches(
            cache_enabled=cache_enabled,
            needs_negative_cache=negative_prompt_embeds is not None,
        )
        mark("generate.kv_cache_ready")

        # 4. Denoising loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)
        cached_predict = self._cached_predict(self.transformer) if cache_enabled else None
        mark("generate.loop_ready")
        for step_idx, t in enumerate(config.time_steps):
            try:
                if interrupt_callback is not None:
                    interrupt_callback()
                mark(f"generate.step_{step_idx}.start")
                self.transformer.sdmlx_phase_step_idx = step_idx
                if cache_enabled and step_idx == 0:
                    self.transformer.sdmlx_phase_step_mode = "extract"
                    self._configure_kv_caches(
                        kv_cache=kv_cache,
                        negative_kv_cache=negative_kv_cache,
                        mode="extract",
                        num_ref_tokens=num_ref_tokens,
                    )
                    self._start_kv_cache_step(
                        kv_cache=kv_cache,
                        negative_kv_cache=negative_kv_cache,
                        step_idx=step_idx,
                    )
                    noise = predict(
                        latents=latents,
                        image_latents=image_latents,
                        latent_ids=latent_ids,
                        image_latent_ids=image_latent_ids,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        guidance=guidance,
                        timestep=config.scheduler.timesteps[t],
                        kv_cache=kv_cache,
                        negative_kv_cache=negative_kv_cache,
                    )
                    mark("generate.step_0.predict_returned")
                    mx.eval(noise)
                    mark("generate.step_0.noise_eval")
                    if bool(getattr(self, "sdmlx_release_reference_latents_after_kv_extract", True)):
                        image_latents = None
                        image_latent_ids = None
                        mark("generate.step_0.reference_latents_released")
                    if bool(getattr(self, "sdmlx_clear_cache_each_step", False)):
                        mx.clear_cache()
                        mark("generate.step_0.cache_cleared")
                elif cache_enabled:
                    self.transformer.sdmlx_phase_step_mode = "cached"
                    self._configure_kv_caches(
                        kv_cache=kv_cache,
                        negative_kv_cache=negative_kv_cache,
                        mode="cached",
                        num_ref_tokens=num_ref_tokens,
                    )
                    self._start_kv_cache_step(
                        kv_cache=kv_cache,
                        negative_kv_cache=negative_kv_cache,
                        step_idx=step_idx,
                    )
                    assert cached_predict is not None
                    noise = cached_predict(
                        latents=latents,
                        latent_ids=latent_ids,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        guidance=guidance,
                        timestep=config.scheduler.timesteps[t],
                        kv_cache=kv_cache,
                        negative_kv_cache=negative_kv_cache,
                    )
                    mark(f"generate.step_{step_idx}.cached_predict_returned")
                else:
                    self.transformer.sdmlx_phase_step_mode = "full"
                    noise = predict(
                        latents=latents,
                        image_latents=image_latents,
                        latent_ids=latent_ids,
                        image_latent_ids=image_latent_ids,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        guidance=guidance,
                        timestep=config.scheduler.timesteps[t],
                    )
                    mark(f"generate.step_{step_idx}.predict_returned")

                # 5.t Take one denoise step
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas
                )
                mark(f"generate.step_{step_idx}.scheduler_returned")

                ctx.in_loop(t, latents)
                mx.eval(latents)
                mark(f"generate.step_{step_idx}.latents_eval")
                if bool(getattr(self, "sdmlx_clear_cache_each_step", False)):
                    mx.clear_cache()
                    mark(f"generate.step_{step_idx}.post_step_cache_cleared")
                if interrupt_callback is not None:
                    interrupt_callback()
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)
        if interrupt_callback is not None:
            interrupt_callback()

        # 6. Decode latents
        mark("generate.before_decode")
        packed_latents = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)  # fmt: off
        decoded = self.vae.decode_packed_latents(packed_latents, tiling_config=self.tiling_config)
        mark("generate.decode_returned")
        mx.eval(decoded)
        mark("generate.decode_eval")
        mx.clear_cache()
        mark("generate.decode_clear")
        image = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=None,
            quantization=self.bits,
            image_paths=image_paths,
            image_path=config.image_path,
            generation_time=config.time_steps.format_dict["elapsed"],
        )
        mark("generate.to_image_returned")
        return image

    def _create_kv_caches(
        self,
        *,
        cache_enabled: bool,
        needs_negative_cache: bool,
    ) -> tuple[Flux2KVCache | None, Flux2KVCache | None]:
        if not cache_enabled:
            return None, None

        kv_cache = self._new_kv_cache()
        negative_kv_cache = self._new_kv_cache() if needs_negative_cache else None
        return kv_cache, negative_kv_cache

    def _new_kv_cache(self) -> Flux2KVCache:
        kv_cache = Flux2KVCache(
            num_double_layers=len(self.transformer.transformer_blocks),
            num_single_layers=len(self.transformer.single_transformer_blocks),
        )
        kv_cache.sdmlx_attention_barrier = bool(getattr(self, "sdmlx_kv_attention_barrier", False))
        kv_cache.sdmlx_attention_first_step_barriers = int(
            getattr(self, "sdmlx_kv_attention_first_step_barriers", 0) or 0
        )
        kv_cache.sdmlx_phase_mark = getattr(self.transformer, "sdmlx_phase_mark", None)
        return kv_cache

    @staticmethod
    def _start_kv_cache_step(
        *,
        kv_cache: Flux2KVCache | None,
        negative_kv_cache: Flux2KVCache | None,
        step_idx: int,
    ) -> None:
        if kv_cache is not None:
            kv_cache.start_step(step_idx)
        if negative_kv_cache is not None:
            negative_kv_cache.start_step(step_idx)

    @staticmethod
    def _configure_kv_caches(
        *,
        kv_cache: Flux2KVCache | None,
        negative_kv_cache: Flux2KVCache | None,
        mode: CacheMode,
        num_ref_tokens: int,
    ) -> None:
        assert kv_cache is not None
        kv_cache.configure(mode=mode, num_ref_tokens=num_ref_tokens)
        if negative_kv_cache is not None:
            negative_kv_cache.configure(mode=mode, num_ref_tokens=num_ref_tokens)

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        prompt_embeds, text_ids = _Flux2KleinEditHelpers.encode_text(
            prompt,
            tokenizer=self.tokenizers["qwen3"],
            text_encoder=self.text_encoder,
        )
        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            negative_prompt_embeds, negative_text_ids = _Flux2KleinEditHelpers.encode_text(
                negative_prompt,
                tokenizer=self.tokenizers["qwen3"],
                text_encoder=self.text_encoder,
            )
        return prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids

    def _release_text_encoder_after_encode(
        self,
        *,
        prompt_embeds: mx.array,
        text_ids: mx.array,
        negative_prompt_embeds: mx.array | None,
        negative_text_ids: mx.array | None,
    ) -> None:
        if not bool(getattr(self, "sdmlx_release_text_encoder_after_encode", False)):
            return
        arrays = [prompt_embeds, text_ids]
        if negative_prompt_embeds is not None:
            arrays.append(negative_prompt_embeds)
        if negative_text_ids is not None:
            arrays.append(negative_text_ids)
        mx.eval(*arrays)
        if "text_encoder" in self:
            del self.text_encoder
            gc.collect()
            mx.clear_cache()

    def _predict(self, transformer):
        def predict(
            latents: mx.array,
            image_latents: mx.array,
            latent_ids: mx.array,
            image_latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
            kv_cache: Flux2KVCache | None = None,
            negative_kv_cache: Flux2KVCache | None = None,
        ) -> mx.array:
            hidden_states = mx.concatenate([latents, image_latents], axis=1).astype(ModelConfig.precision)
            img_ids = mx.concatenate([latent_ids, image_latent_ids], axis=1)

            noise = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=None,
                kv_cache=kv_cache,
            )
            noise = noise[:, : latents.shape[1]]
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                    kv_cache=negative_kv_cache or kv_cache,
                )
                negative_noise = negative_noise[:, : latents.shape[1]]
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        if (
            AppleSiliconUtil.is_m1_or_m2()
            or self.model_config.supports_kv_cache
            or bool(getattr(transformer, "sdmlx_eval_clear_each_block", False))
        ):
            return predict
        return mx.compile(predict)

    @staticmethod
    def _cached_predict(transformer):
        def predict(
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
            kv_cache: Flux2KVCache,
            negative_kv_cache: Flux2KVCache | None = None,
        ) -> mx.array:
            model_latents = latents.astype(ModelConfig.precision)
            noise = transformer(
                hidden_states=model_latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=None,
                kv_cache=kv_cache,
            )
            noise = noise[:, : latents.shape[1]]
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=model_latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=latent_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                    kv_cache=negative_kv_cache or kv_cache,
                )
                negative_noise = negative_noise[:, : latents.shape[1]]
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        return predict
