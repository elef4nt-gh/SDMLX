import math
import os
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx import nn

from sdmlx_qwen_native.models.common.config import ModelConfig
from sdmlx_qwen_native.models.common.config.config import Config
from sdmlx_qwen_native.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from sdmlx_qwen_native.models.common.vae.vae_util import VAEUtil
from sdmlx_qwen_native.models.common.weights.saving.model_saver import ModelSaver
from sdmlx_qwen_native.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from sdmlx_qwen_native.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from sdmlx_qwen_native.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from sdmlx_qwen_native.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from sdmlx_qwen_native.models.qwen.qwen_initializer import QwenImageInitializer
from sdmlx_qwen_native.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from sdmlx_qwen_native.utils.exceptions import StopImageGenerationException
from sdmlx_qwen_native.utils.generated_image import GeneratedImage
from sdmlx_qwen_native.utils.image_util import ImageUtil


def _uses_wuli_qwen_2512_two_step_lora(lora_paths: list[str] | None) -> bool:
    for path in lora_paths or []:
        name = Path(str(path)).name.lower()
        if "wuli-qwen-image-2512-turbo-lora-2steps" in name:
            return True
    return False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _qwen_2512_two_step_model_config(model_config: ModelConfig) -> ModelConfig:
    resolved = model_config.copy_with_model_identity(
        model_name=model_config.model_name,
        base_model=model_config.base_model,
        aliases=list(model_config.aliases),
    )
    # Wuli's 2-step Turbo LoRA documents:
    # exponential_shift_mu=log(2.5), use_dynamic_shifting=True, shift_terminal=0.7155.
    # Our LinearScheduler derives mu from base/max shifts, so pin both to log(2.5).
    resolved.sigma_base_shift = math.log(2.5)
    resolved.sigma_max_shift = math.log(2.5)
    resolved.sigma_shift_terminal = 0.7155
    return resolved


def _finite_guard_enabled() -> bool:
    value = os.environ.get("SDMLX_QWEN_FINITE_GUARD", "").strip().lower()
    if value:
        return value in {"1", "true", "yes", "on"}
    # Full-tensor finite checks force MLX evaluation/synchronization inside the
    # denoise loop. Keep them available for NaN diagnostics, but never on the
    # product speed path by default.
    return False


def _raise_if_nonfinite(stage: str, value: mx.array) -> None:
    if not _finite_guard_enabled():
        return
    finite = mx.isfinite(value)
    finite_count = mx.sum(finite.astype(mx.int32))
    mx.eval(finite_count)
    total_count = int(value.size)
    finite_count_value = int(finite_count.item())
    bad_count = total_count - finite_count_value
    if bad_count <= 0:
        return

    ratio = finite_count_value / total_count if total_count else 1.0
    if finite_count_value > 0:
        finite_min = mx.min(mx.where(finite, value, mx.full(value.shape, mx.inf, dtype=value.dtype)))
        finite_max = mx.max(mx.where(finite, value, mx.full(value.shape, -mx.inf, dtype=value.dtype)))
        mx.eval(finite_min, finite_max)
        min_value = float(finite_min.item())
        max_value = float(finite_max.item())
    else:
        min_value = float("nan")
        max_value = float("nan")
    raise RuntimeError(
        "SDMLX Qwen Image: non-finite tensor detected "
        f"at {stage} (bad_count={bad_count}/{total_count}, finite_ratio={ratio:.9f}, "
        f"min={min_value:.6g}, max={max_value:.6g})"
    )


def _sanitize_decoded_nonfinite(stage: str, value: mx.array) -> mx.array:
    if not _finite_guard_enabled():
        return value
    finite = mx.isfinite(value)
    finite_count = mx.sum(finite.astype(mx.int32))
    mx.eval(finite_count)
    total_count = int(value.size)
    finite_count_value = int(finite_count.item())
    bad_count = total_count - finite_count_value
    if bad_count <= 0:
        return value

    ratio = finite_count_value / total_count if total_count else 1.0
    # A handful of VAE decode outliers should not turn a valid image into a failed run.
    # Larger non-finite regions still indicate a real numerical failure.
    allowed_bad_count = max(16, int(total_count * 1e-6))
    if bad_count <= allowed_bad_count and ratio >= 0.999999:
        return mx.where(finite, value, mx.zeros_like(value))

    _raise_if_nonfinite(stage, value)
    return value


class QwenImage(nn.Module):
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
        model_config: ModelConfig = ModelConfig.qwen_image(),
    ):
        super().__init__()
        self._sdmlx_wuli_qwen_2512_two_step_lora = _uses_wuli_qwen_2512_two_step_lora(lora_paths)
        QwenImageInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_mod_scales=lora_mod_scales,
            mod_lora_scale=mod_lora_scale,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
        flow_shift: float | None = None,
        negative_prompt: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> GeneratedImage:
        # 0. Create a new config based on the model type and input parameters
        runtime_model_config = self.model_config
        force_wuli_two_step_schedule = _env_flag("SDMLX_QWEN_FORCE_WULI_2512_TWO_STEP_SCHEDULE")
        if self._sdmlx_wuli_qwen_2512_two_step_lora and (
            int(num_inference_steps) == 2 or force_wuli_two_step_schedule
        ):
            runtime_model_config = _qwen_2512_two_step_model_config(self.model_config)
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_strength=image_strength,
            model_config=runtime_model_config,
            num_inference_steps=num_inference_steps,
            flow_shift=flow_shift,
        )

        # 1. Create the initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=QwenLatentCreator,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                image_path=config.image_path,
                tiling_config=self.tiling_config,
            ),
        )

        # 2. Encode the prompt (using native MLX encoding)
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = QwenPromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_cache=self.prompt_cache,
            qwen_tokenizer=self.tokenizers["qwen"],
            qwen_text_encoder=self.text_encoder,
        )

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        guidance_value = float(config.guidance)
        skip_negative_pass = abs(guidance_value - 1.0) < 1e-6
        active_progress_callback = progress_callback
        loop_start = int(config.init_time_step)
        loop_total = max(1, int(config.num_inference_steps) - loop_start)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 4. Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                )
                _raise_if_nonfinite(f"step_{int(t)}_noise", noise)
                if skip_negative_pass:
                    guided_noise = noise
                else:
                    noise_negative = self.transformer(
                        t=t,
                        config=config,
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_mask,
                    )
                    _raise_if_nonfinite(f"step_{int(t)}_negative_noise", noise_negative)
                    guided_noise = QwenImage.compute_guided_noise(noise, noise_negative, config.guidance)
                    _raise_if_nonfinite(f"step_{int(t)}_guided_noise", guided_noise)

                # 5.t Take one denoise step
                latents = config.scheduler.step(noise=guided_noise, timestep=t, latents=latents)
                _raise_if_nonfinite(f"step_{int(t)}_latents", latents)

                # 6.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)
                if active_progress_callback is not None:
                    try:
                        active_progress_callback(int(t) - loop_start + 1, loop_total)
                    except Exception:
                        active_progress_callback = None

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 7. Call subscribers after loop
        ctx.after_loop(latents)

        # 8. Decode the latent array and return the image
        latents = QwenLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        _raise_if_nonfinite("unpacked_latents", latents)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        decoded = _sanitize_decoded_nonfinite("decoded_latents", decoded)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
            latent=latents,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=QwenWeightDefinition,
        )

    @staticmethod
    def compute_guided_noise(
        noise: mx.array,
        noise_negative: mx.array,
        guidance: float,
    ) -> mx.array:
        combined = noise_negative + guidance * (noise - noise_negative)
        cond_norm = mx.sqrt(mx.sum(noise * noise, axis=-1, keepdims=True) + 1e-12)
        noise_norm = mx.sqrt(mx.sum(combined * combined, axis=-1, keepdims=True) + 1e-12)
        noise = combined * (cond_norm / noise_norm)
        return noise
