from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import mlx.core as mx
from mlx import nn

from sdmlx_qwen_native.models.common.vae.vae_util import VAEUtil
from sdmlx_qwen_native.utils.image_util import ImageUtil

if TYPE_CHECKING:
    from sdmlx_qwen_native.models.common.vae.tiling_config import TilingConfig
    from sdmlx_qwen_native.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
    from sdmlx_qwen_native.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from sdmlx_qwen_native.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
    from sdmlx_qwen_native.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator

    LatentCreatorType: TypeAlias = type[FiboLatentCreator | FluxLatentCreator | QwenLatentCreator | ZImageLatentCreator]


class Img2Img:
    def __init__(
        self,
        vae: nn.Module,
        latent_creator: "LatentCreatorType",
        sigmas: mx.array,
        init_time_step: int,
        image_path: str | Path | None,
        tiling_config: "TilingConfig" | None = None,
    ):
        self.vae = vae
        self.sigmas = sigmas
        self.init_time_step = init_time_step
        self.image_path = image_path
        self.latent_creator = latent_creator
        self.tiling_config = tiling_config


class LatentCreator:
    @staticmethod
    def _round_to_multiple(value: float, multiple: int = 16) -> int:
        return max(multiple, int(round(float(value) / multiple)) * multiple)

    @staticmethod
    def _aspect_preserving_dimensions(
        *,
        image_width: int,
        image_height: int,
        target_width: int,
        target_height: int,
        multiple: int = 16,
    ) -> tuple[int, int]:
        if image_width <= 0 or image_height <= 0:
            return target_width, target_height

        target_area = max(multiple * multiple, int(target_width) * int(target_height))
        aspect = float(image_width) / float(image_height)
        fitted_height = (target_area / aspect) ** 0.5
        fitted_width = fitted_height * aspect
        return (
            LatentCreator._round_to_multiple(fitted_width, multiple),
            LatentCreator._round_to_multiple(fitted_height, multiple),
        )

    @staticmethod
    def create_for_txt2img_or_img2img(
        seed: int,
        height: int,
        width: int,
        img2img: Img2Img,
    ) -> mx.array:
        latent_creator = img2img.latent_creator

        if img2img.image_path is None:
            # txt2img: just create noise
            return latent_creator.create_noise(seed, height, width)
        else:
            # img2img: blend encoded image with noise
            pure_noise = latent_creator.create_noise(seed, height, width)
            encoded = LatentCreator.encode_image(
                width=width,
                height=height,
                vae=img2img.vae,
                image_path=img2img.image_path,
                tiling_config=img2img.tiling_config,
            )
            latents = latent_creator.pack_latents(encoded, height, width)
            sigma = img2img.sigmas[img2img.init_time_step]
            return LatentCreator.add_noise_by_interpolation(clean=latents, noise=pure_noise, sigma=sigma)

    @staticmethod
    def encode_image(
        vae: nn.Module,
        image_path: str | Path,
        height: int,
        width: int,
        tiling_config: "TilingConfig" | None = None,
        preserve_aspect: bool = False,
    ) -> mx.array:
        image = ImageUtil.load_image(image_path).convert("RGB")
        if preserve_aspect:
            width, height = LatentCreator._aspect_preserving_dimensions(
                image_width=image.width,
                image_height=image.height,
                target_width=width,
                target_height=height,
            )
        scaled_user_image = ImageUtil.scale_to_dimensions(
            image=image,
            target_width=width,
            target_height=height,
        )
        image_array = ImageUtil.to_array(scaled_user_image)
        return VAEUtil.encode(vae=vae, image=image_array, tiling_config=tiling_config)

    @staticmethod
    def add_noise_by_interpolation(clean: mx.array, noise: mx.array, sigma: float) -> mx.array:
        return (1 - sigma) * clean + sigma * noise
