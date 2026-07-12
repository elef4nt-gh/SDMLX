import math
import os
import time
import hashlib
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx import nn
from PIL import Image
from tqdm import tqdm

from sdmlx_qwen_native.models.common.config.config import Config
from sdmlx_qwen_native.models.common.config.model_config import ModelConfig
from sdmlx_qwen_native.models.common.vae.vae_util import VAEUtil
from sdmlx_qwen_native.models.common.weights.saving.model_saver import ModelSaver
from sdmlx_qwen_native.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from sdmlx_qwen_native.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from sdmlx_qwen_native.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from sdmlx_qwen_native.models.qwen.qwen_initializer import QwenImageInitializer
from sdmlx_qwen_native.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
from sdmlx_qwen_native.models.qwen.variants.txt2img.qwen_image import (
    QwenImage,
    _raise_if_nonfinite,
    _sanitize_decoded_nonfinite,
)
from sdmlx_qwen_native.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from sdmlx_qwen_native.utils.exceptions import StopImageGenerationException
from sdmlx_qwen_native.utils.generated_image import GeneratedImage
from sdmlx_qwen_native.utils.image_util import ImageUtil


class QwenImageEdit(nn.Module):
    vae: QwenVAE
    transformer: QwenTransformer
    text_encoder: QwenTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_mod_scales: list[float] | None = None,
        mod_lora_scale: float = 0.0,
        model_config: ModelConfig = ModelConfig.qwen_image_edit(),
    ):
        super().__init__()
        self.ref_method = self._detect_ref_method(model_path=model_path, model_config=model_config)
        model_config = self._resolve_model_identity(model_path=model_path, model_config=model_config)
        QwenImageInitializer.init_edit(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_mod_scales=lora_mod_scales,
            mod_lora_scale=mod_lora_scale,
            model_config=model_config,
        )
        self._vlm_conditioning_cache: dict[tuple, tuple[mx.array, mx.array]] = {}
        self._vlm_conditioning_cache_order: list[tuple] = []
        self._vlm_conditioning_cache_max = 6

    @staticmethod
    def _detect_ref_method(model_path: str | None, model_config: ModelConfig) -> str:
        model_id = f"{model_path or ''} {model_config.model_name or ''}".lower()
        return "index_timestep_zero" if "2511" in model_id else "index"

    @staticmethod
    def _resolve_model_identity(model_path: str | None, model_config: ModelConfig) -> ModelConfig:
        model_id = f"{model_path or ''} {model_config.model_name or ''}".lower()
        if "2511" not in model_id:
            return model_config
        resolved = model_config.copy_with_model_identity(
            model_name="Qwen/Qwen-Image-Edit-2511",
            base_model=str(model_path) if model_path else model_config.base_model,
            aliases=["qwen-image-edit-2511", "qwen-edit-2511", "qwen-edit-plus-2511"],
        )
        resolved.supports_guidance = True
        return resolved

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=QwenWeightDefinition,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: list[str] | None = None,
        num_inference_steps: int = 4,
        height: int | None = None,
        width: int | None = None,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        scheduler: str = "linear",
        flow_shift: float | None = None,
        negative_prompt: str | None = None,
        use_reference_latents: bool = True,
        use_picture_prefix: bool | None = None,
        vl_target_area: int | None = None,
        reference_target_area: int | None = None,
        reference_target_areas: list[int] | None = None,
        reference_target_multiple: int | None = None,
        prompt_template: str | None = None,
        image_slots: list[int] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        images: list[Image.Image] | None = None,
    ) -> GeneratedImage:
        image_inputs = list(images or image_paths or [])
        if not image_inputs:
            raise ValueError("Qwen Image Edit requires at least one input image.")
        timing = {}
        t0_total = time.perf_counter()
        t0 = time.perf_counter()
        config, _vl_width, _vl_height, _vae_width, _vae_height = self._compute_dimensions(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            flow_shift=flow_shift,
            image_path=image_path,
            image_paths=image_inputs,
            num_inference_steps=num_inference_steps,
            vl_target_area=vl_target_area,
        )
        timing["dimensions"] = time.perf_counter() - t0
        timesteps = config.scheduler.timesteps
        time_steps = tqdm(
            range(len(timesteps)),
            disable=_qwen_env_flag("SDMLX_QWEN_DISABLE_TERMINAL_PROGRESS"),
        )

        # 1. Create initial latents
        t0 = time.perf_counter()
        latents = QwenLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
        )
        timing["noise"] = time.perf_counter() - t0

        # 2. Encode the prompt
        t0 = time.perf_counter()
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self._encode_prompts_with_images(
            prompt=prompt,
            config=config,
            image_paths=image_inputs,
            negative_prompt=negative_prompt,
            encode_negative=abs(config.guidance - 1.0) >= 1e-6,
            vl_width=None,
            vl_height=None,
            vl_target_area=vl_target_area,
            use_picture_prefix=use_picture_prefix,
            prompt_template=prompt_template,
            image_slots=image_slots,
            cache_scope=(
                bool(use_reference_latents),
                int(reference_target_area) if reference_target_area is not None else None,
                tuple(int(area) for area in reference_target_areas) if reference_target_areas is not None else None,
                int(reference_target_multiple) if reference_target_multiple is not None else None,
            ),
        )
        timing["prompt_image_encode"] = time.perf_counter() - t0

        # 3. Generate image conditioning latents
        t0 = time.perf_counter()
        active_tokens = int(latents.shape[1])
        text_tokens = int(mx.sum(prompt_mask[0]).item()) if prompt_mask is not None else int(prompt_embeds.shape[1])
        if use_reference_latents:
            if reference_target_areas is not None:
                resolved_reference_target_area = [
                    max(16 * 16, int(area))
                    for area in list(reference_target_areas)
                ]
            elif reference_target_area is None:
                resolved_reference_target_area = max(
                    16 * 16,
                    (int(config.width) * int(config.height)) // max(1, len(image_inputs)),
                )
            else:
                resolved_reference_target_area = max(16 * 16, int(reference_target_area))
            static_image_latents, qwen_image_ids, cond_image_grids, num_images = (
                QwenEditUtil.create_image_conditioning_latents(
                    vae=self.vae,
                    width=None,
                    height=None,
                    image_paths=image_inputs,
                    target_area=resolved_reference_target_area,
                    target_multiple=reference_target_multiple,
                    tiling_config=self.tiling_config,
                )
            )
            reference_tokens = int(static_image_latents.shape[1])
            ref_sizes = ", ".join(f"{grid[2] * 16}x{grid[1] * 16}" for grid in cond_image_grids)
            if _qwen_env_flag("SDMLX_QWEN_VERBOSE"):
                print(f"SDMLX Qwen: reference-latents: {ref_sizes}")
        else:
            resolved_reference_target_area = 0
            static_image_latents = None
            qwen_image_ids = None
            cond_image_grids = []
            num_images = 0
            reference_tokens = 0
            if _qwen_env_flag("SDMLX_QWEN_VERBOSE"):
                print("SDMLX Qwen: reference-latents: off")
        timing["reference_vae_encode"] = time.perf_counter() - t0
        if _qwen_env_flag("SDMLX_QWEN_VERBOSE"):
            print(
                "SDMLX Qwen token layout: "
                f"active={active_tokens}, reference={reference_tokens}, "
                f"text={text_tokens}, joint={active_tokens + reference_tokens + text_tokens}, "
                + (
                    f"reference_target_areas={resolved_reference_target_area}"
                    if isinstance(resolved_reference_target_area, list)
                    else f"reference_target_area={resolved_reference_target_area}"
                )
            )

        if static_image_latents is None:
            cond_image_grid = None
        elif num_images > 1:
            cond_image_grid = cond_image_grids
        else:
            cond_image_grid = cond_image_grids[0]

        t0 = time.perf_counter()
        prompt_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(
            encoder_hidden_states_mask=prompt_mask,
            pos_embed=self.transformer.pos_embed,
            config=config,
            cond_image_grid=cond_image_grid,
        )
        negative_rotary_embeddings = None
        if abs(config.guidance - 1.0) >= 1e-6:
            negative_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(
                encoder_hidden_states_mask=negative_prompt_mask,
                pos_embed=self.transformer.pos_embed,
                config=config,
                cond_image_grid=cond_image_grid,
            )
        timing["rotary"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        projected_static_image_latents = (
            self.transformer.project_image_hidden_states(static_image_latents)
            if static_image_latents is not None
            else None
        )
        projected_prompt_embeds = self.transformer.project_encoder_hidden_states(prompt_embeds)
        projected_negative_prompt_embeds = None
        if abs(config.guidance - 1.0) >= 1e-6:
            projected_negative_prompt_embeds = self.transformer.project_encoder_hidden_states(negative_prompt_embeds)
        prompt_block_mask = QwenTransformer._block_attention_mask(prompt_mask)
        negative_block_mask = (
            QwenTransformer._block_attention_mask(negative_prompt_mask)
            if abs(config.guidance - 1.0) >= 1e-6
            else None
        )
        eval_cache = [projected_prompt_embeds, prompt_rotary_embeddings[0][0], prompt_rotary_embeddings[0][1]]
        if projected_static_image_latents is not None:
            eval_cache.append(projected_static_image_latents)
        if projected_negative_prompt_embeds is not None:
            eval_cache.append(projected_negative_prompt_embeds)
        if negative_rotary_embeddings is not None:
            eval_cache.extend([negative_rotary_embeddings[0][0], negative_rotary_embeddings[0][1]])
        mx.eval(*eval_cache)
        timing["fixed_cache"] = time.perf_counter() - t0

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        step_eval_interval = _qwen_step_eval_interval()
        step_timings_enabled = _qwen_env_flag("SDMLX_QWEN_STEP_TIMINGS") or _qwen_env_flag(
            "SDMLX_QWEN_SAMPLING_TIMINGS"
        )
        finite_diagnostics = _qwen_env_flag("SDMLX_QWEN_FINITE_DIAGNOSTICS")
        step_timings = []
        active_progress_callback = progress_callback
        if finite_diagnostics:
            _qwen_finite_stats("initial_latents", latents)
            _qwen_finite_stats("prompt_embeds", prompt_embeds)
            if projected_static_image_latents is not None:
                _qwen_finite_stats("projected_static_image_latents", projected_static_image_latents)
            _qwen_finite_stats("projected_prompt_embeds", projected_prompt_embeds)

        t0 = time.perf_counter()
        for t in time_steps:
            try:
                t_step = time.perf_counter()
                t_pos = time.perf_counter()
                # 5.t Concatenate the updated latents with the static image latents
                if static_image_latents is not None:
                    hidden_states = latents
                    hidden_states_neg = latents
                else:
                    hidden_states = latents
                    hidden_states_neg = latents

                # 6.t Predict the noise
                timestep_zero_index = (
                    latents.shape[1]
                    if static_image_latents is not None and self.ref_method == "index_timestep_zero"
                    else None
                )
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_block_mask,
                    qwen_image_ids=qwen_image_ids,
                    cond_image_grid=cond_image_grid,
                    timestep_zero_index=timestep_zero_index,
                    image_rotary_embeddings=prompt_rotary_embeddings,
                    preprojected_image_tail=projected_static_image_latents,
                    projected_encoder_hidden_states=projected_prompt_embeds,
                    block_attention_mask=prompt_block_mask,
                )[:, : latents.shape[1]]
                _raise_if_nonfinite(f"step_{int(t)}_noise", noise)
                if step_timings_enabled:
                    mx.eval(noise)
                if finite_diagnostics:
                    _qwen_finite_stats(f"step_{int(t)}_noise", noise)
                positive_seconds = time.perf_counter() - t_pos
                if abs(config.guidance - 1.0) < 1e-6:
                    guided_noise = noise
                    negative_seconds = 0.0
                else:
                    t_neg = time.perf_counter()
                    noise_negative = self.transformer(
                        t=t,
                        config=config,
                        hidden_states=hidden_states_neg,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_block_mask,
                        qwen_image_ids=qwen_image_ids,
                        cond_image_grid=cond_image_grid,
                        timestep_zero_index=timestep_zero_index,
                        image_rotary_embeddings=negative_rotary_embeddings,
                        preprojected_image_tail=projected_static_image_latents,
                        projected_encoder_hidden_states=projected_negative_prompt_embeds,
                        block_attention_mask=negative_block_mask,
                    )[:, : latents.shape[1]]
                    _raise_if_nonfinite(f"step_{int(t)}_negative_noise", noise_negative)
                    if step_timings_enabled:
                        mx.eval(noise_negative)
                    negative_seconds = time.perf_counter() - t_neg
                    guided_noise = QwenImage.compute_guided_noise(noise, noise_negative, config.guidance)
                    _raise_if_nonfinite(f"step_{int(t)}_guided_noise", guided_noise)

                # 7.t Take one denoise step
                t_scheduler = time.perf_counter()
                latents = config.scheduler.step(noise=guided_noise, timestep=t, latents=latents)
                _raise_if_nonfinite(f"step_{int(t)}_latents", latents)
                scheduler_seconds = time.perf_counter() - t_scheduler
                if finite_diagnostics:
                    _qwen_finite_stats(f"step_{int(t)}_latents", latents)

                # 8.t Call subscribers in-loop
                ctx.in_loop(t, latents, time_steps=time_steps)

                # (Optional) Evaluate to enable progress tracking
                eval_seconds = 0.0
                if step_eval_interval > 0 and (int(t) + 1) % step_eval_interval == 0:
                    t_eval = time.perf_counter()
                    mx.eval(latents)
                    eval_seconds = time.perf_counter() - t_eval
                if active_progress_callback is not None:
                    try:
                        active_progress_callback(int(t) + 1, len(timesteps))
                    except Exception:
                        active_progress_callback = None
                if step_timings_enabled:
                    step_timings.append(
                        {
                            "step": int(t),
                            "positive": positive_seconds,
                            "negative": negative_seconds,
                            "scheduler": scheduler_seconds,
                            "eval": eval_seconds,
                            "total": time.perf_counter() - t_step,
                        }
                    )

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents, time_steps=time_steps)
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(timesteps)}")
        timing["sampling"] = time.perf_counter() - t0
        if step_timings_enabled and step_timings:
            _print_qwen_step_timings(step_timings)

        # 9. Call subscribers after loop
        ctx.after_loop(latents)

        # 10. Decode the latent array and return the image
        t0 = time.perf_counter()
        t_decode_part = time.perf_counter()
        latents = QwenLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        _raise_if_nonfinite("unpacked_latents", latents)
        timing["decode_unpack"] = time.perf_counter() - t_decode_part
        t_decode_part = time.perf_counter()
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        decoded = _sanitize_decoded_nonfinite("decoded_latents", decoded)
        timing["decode_vae_call"] = time.perf_counter() - t_decode_part
        t_decode_part = time.perf_counter()
        mx.eval(decoded)
        if finite_diagnostics:
            _qwen_finite_stats("decoded", decoded)
        timing["decode_eval"] = time.perf_counter() - t_decode_part
        timing["decode"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        generated = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_paths=image_paths,
            generation_time=time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
            latent=latents,
        )
        timing["to_image"] = time.perf_counter() - t0
        timing["total"] = time.perf_counter() - t0_total
        if _qwen_env_flag("SDMLX_QWEN_TIMINGS") or _qwen_env_flag("SDMLX_QWEN_VERBOSE"):
            print(
                "SDMLX Qwen Native timings: "
                + ", ".join(f"{key}={value:.2f}s" for key, value in timing.items())
            )
        return generated

    def _encode_prompts_with_images(
        self,
        prompt: str,
        negative_prompt: str,
        image_paths: list[str],
        config,
        vl_width: int | None = None,
        vl_height: int | None = None,
        vl_target_area: int | None = None,
        encode_negative: bool = True,
        use_picture_prefix: bool | None = None,
        prompt_template: str | None = None,
        image_slots: list[int] | None = None,
        cache_scope: tuple | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        tokenizer = self.tokenizers["qwen_vl"]
        image_digests = []
        for image in image_paths:
            pil_image = ImageUtil.load_image(image).convert("RGB")
            digest = hashlib.sha256()
            digest.update(str(pil_image.size).encode("utf-8"))
            digest.update(pil_image.mode.encode("utf-8"))
            digest.update(pil_image.tobytes())
            image_digests.append(digest.hexdigest())

        def encode_one(text: str) -> tuple[mx.array, mx.array]:
            cache_key = (
                id(self.qwen_vl_encoder),
                id(tokenizer),
                str(getattr(self.model_config, "model_name", "qwen-image-edit")),
                str(text),
                tuple(image_digests),
                int(vl_width) if vl_width is not None else None,
                int(vl_height) if vl_height is not None else None,
                int(vl_target_area) if vl_target_area is not None else None,
                bool(tokenizer.use_picture_prefix if use_picture_prefix is None else use_picture_prefix),
                str(prompt_template) if prompt_template is not None else None,
                tuple(int(slot) for slot in image_slots) if image_slots is not None else None,
                cache_scope,
            )
            cached = self._vlm_conditioning_cache.get(cache_key)
            if cached is not None:
                if cache_key in self._vlm_conditioning_cache_order:
                    self._vlm_conditioning_cache_order.remove(cache_key)
                self._vlm_conditioning_cache_order.append(cache_key)
                return cached

            input_ids, attention_mask, pixel_values, image_grid_thw = tokenizer.tokenize_with_image(
                text,
                image_paths,
                vl_width=vl_width,
                vl_height=vl_height,
                vl_target_area=vl_target_area,
                use_picture_prefix=use_picture_prefix,
                prompt_template=prompt_template,
                image_slots=image_slots,
            )
            template_drop_idx = getattr(tokenizer, "last_template_drop_idx", None)
            hidden_states = self.qwen_vl_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                template_drop_idx=template_drop_idx,
            )
            encoded = (hidden_states[0].astype(mx.float16), hidden_states[1].astype(mx.float16))
            mx.eval(*encoded)
            self._vlm_conditioning_cache[cache_key] = encoded
            self._vlm_conditioning_cache_order.append(cache_key)
            while len(self._vlm_conditioning_cache_order) > self._vlm_conditioning_cache_max:
                old_key = self._vlm_conditioning_cache_order.pop(0)
                self._vlm_conditioning_cache.pop(old_key, None)
            return encoded

        final_prompt_embeds, final_prompt_mask = encode_one(prompt)

        if encode_negative:
            neg_prompt = negative_prompt if negative_prompt is not None else ""
            negative_prompt_embeds, negative_prompt_mask = encode_one(neg_prompt)
        else:
            negative_prompt_embeds = final_prompt_embeds
            negative_prompt_mask = final_prompt_mask

        return (
            final_prompt_embeds,  # prompt_embeds
            final_prompt_mask,  # prompt_mask
            negative_prompt_embeds,  # negative_prompt_embeds
            negative_prompt_mask,  # negative_prompt_mask
        )

    def _compute_dimensions(
        self,
        image_paths: list[str],
        num_inference_steps: int,
        height: int | None,
        width: int | None,
        guidance: float,
        image_path: Path | str | None,
        scheduler: str,
        flow_shift: float | None,
        vl_target_area: int | None = None,
    ) -> tuple[Config, int, int, int, int]:
        last_image = ImageUtil.load_image(image_paths[-1]).convert("RGB")
        image_size = last_image.size

        target_area = 1024 * 1024
        ratio = image_size[0] / image_size[1]
        calculated_width = math.sqrt(target_area * ratio)
        calculated_height = calculated_width / ratio
        calculated_width = round(calculated_width / 32) * 32
        calculated_height = round(calculated_height / 32) * 32

        use_height = height or int(calculated_height)
        use_width = width or int(calculated_width)

        vae_scale_factor = 8
        multiple_of = vae_scale_factor * 2
        use_width = use_width // multiple_of * multiple_of
        use_height = use_height // multiple_of * multiple_of

        config = Config(
            width=use_width,
            height=use_height,
            guidance=guidance,
            scheduler=scheduler,
            flow_shift=flow_shift,
            image_path=image_path,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        def area_dimensions(target_area: int, multiple_of: int) -> tuple[int, int]:
            ratio = image_size[0] / image_size[1]
            area_width = math.sqrt(target_area * ratio)
            area_height = area_width / ratio
            area_width = max(multiple_of, round(area_width / multiple_of) * multiple_of)
            area_height = max(multiple_of, round(area_height / multiple_of) * multiple_of)
            return int(area_width), int(area_height)

        vl_width, vl_height = area_dimensions(int(vl_target_area or 384 * 384), 28)

        vae_width, vae_height = area_dimensions(1024 * 1024, 16)

        return config, int(vl_width), int(vl_height), int(vae_width), int(vae_height)


def _qwen_step_eval_interval() -> int:
    value = os.getenv("SDMLX_QWEN_STEP_EVAL_INTERVAL", "1").strip()
    try:
        return max(0, int(value))
    except ValueError:
        return 1


def _qwen_env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _qwen_finite_stats(label: str, array: mx.array) -> None:
    mx.eval(array)
    items = int(array.size)
    nan_count = int(mx.sum(mx.isnan(array)).item())
    inf_count = int(mx.sum(mx.isinf(array)).item())
    finite = mx.where(mx.isfinite(array), array, mx.zeros_like(array))
    abs_max = float(mx.max(mx.abs(finite)).item()) if items else 0.0
    print(
        f"SDMLX Qwen finite {label}: shape={array.shape}, dtype={array.dtype}, "
        f"nan={nan_count}, inf={inf_count}, items={items}, abs_max={abs_max:.6g}"
    )


def _print_qwen_step_timings(step_timings: list[dict[str, float]]) -> None:
    count = len(step_timings)
    totals = {
        "total": sum(item["total"] for item in step_timings),
        "positive": sum(item["positive"] for item in step_timings),
        "negative": sum(item["negative"] for item in step_timings),
        "scheduler": sum(item["scheduler"] for item in step_timings),
        "eval": sum(item["eval"] for item in step_timings),
    }
    print(
        "SDMLX Qwen sampling timing avg: "
        + ", ".join(f"{key}={value / count:.2f}s" for key, value in totals.items())
    )
    print(
        "SDMLX Qwen sampling timing total: "
        + ", ".join(f"{key}={value:.2f}s" for key, value in totals.items())
    )
    if count <= 8:
        print(
            "SDMLX Qwen sampling timing steps: "
            + "; ".join(
                "s{step}: total={total:.2f}s, pos={positive:.2f}s, neg={negative:.2f}s, "
                "sched={scheduler:.2f}s, eval={eval:.2f}s".format(**item)
                for item in step_timings
            )
        )
