import hashlib
import os

import mlx.core as mx
from PIL import Image

from sdmlx_qwen_native.models.common.latent_creator.latent_creator import LatentCreator
from sdmlx_qwen_native.models.common.vae.tiling_config import TilingConfig
from sdmlx_qwen_native.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator


_QWEN_IMAGE_CONDITIONING_LATENT_CACHE: dict[tuple, tuple[mx.array, mx.array, tuple[int, int, int]]] = {}
_QWEN_IMAGE_CONDITIONING_LATENT_CACHE_ORDER: list[tuple] = []
_QWEN_IMAGE_CONDITIONING_LATENT_CACHE_MAX = 8


class QwenEditUtil:
    @staticmethod
    def _image_content_digest(image) -> str:
        hasher = hashlib.sha256()
        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
            hasher.update(str(rgb.size).encode("utf-8"))
            hasher.update(rgb.mode.encode("utf-8"))
            hasher.update(rgb.tobytes())
        else:
            with open(image, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def clear_image_conditioning_cache() -> None:
        _QWEN_IMAGE_CONDITIONING_LATENT_CACHE.clear()
        _QWEN_IMAGE_CONDITIONING_LATENT_CACHE_ORDER.clear()

    @staticmethod
    def _cached_image_conditioning(
        vae,
        image,
        width: int,
        height: int,
        tiling_config: TilingConfig | None,
    ) -> tuple[mx.array, mx.array, tuple[int, int, int]]:
        cache_key = (
            id(vae),
            QwenEditUtil._image_content_digest(image),
            int(width),
            int(height),
            repr(tiling_config),
            str(getattr(vae, "dtype", "model-defined")),
        )
        cached = _QWEN_IMAGE_CONDITIONING_LATENT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        input_image = LatentCreator.encode_image(
            vae=vae,
            image_path=image,
            height=height,
            width=width,
            tiling_config=tiling_config,
        )

        image_latents = QwenLatentCreator.pack_latents(
            latents=input_image,
            height=height,
            width=width,
            num_channels_latents=16,
        )
        image_grid = (1, height // 16, width // 16)
        image_ids = QwenEditUtil._create_image_ids(
            height=image_grid[1] * 16,
            width=image_grid[2] * 16,
        )
        mx.eval(image_latents, image_ids)

        _QWEN_IMAGE_CONDITIONING_LATENT_CACHE[cache_key] = (image_latents, image_ids, image_grid)
        _QWEN_IMAGE_CONDITIONING_LATENT_CACHE_ORDER.append(cache_key)
        while len(_QWEN_IMAGE_CONDITIONING_LATENT_CACHE_ORDER) > _QWEN_IMAGE_CONDITIONING_LATENT_CACHE_MAX:
            old_key = _QWEN_IMAGE_CONDITIONING_LATENT_CACHE_ORDER.pop(0)
            _QWEN_IMAGE_CONDITIONING_LATENT_CACHE.pop(old_key, None)
        return image_latents, image_ids, image_grid

    @staticmethod
    def create_image_conditioning_latents(
        vae,
        height: int | None,
        width: int | None,
        image_paths,
        vl_width: int | None = None,
        vl_height: int | None = None,
        target_area: int | list[int] | tuple[int, ...] | None = None,
        target_multiple: int | None = None,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[mx.array, mx.array, list[tuple[int, int, int]], int]:
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        def target_area_for_image(index: int) -> int:
            target_area_env = os.getenv("MFLUX_QWEN_REF_TARGET_AREA")
            if target_area_env is not None:
                return max(16 * 16, int(target_area_env))
            if isinstance(target_area, (list, tuple)):
                if index < len(target_area):
                    return max(16 * 16, int(target_area[index]))
                if target_area:
                    return max(16 * 16, int(target_area[-1]))
                return 1024 * 1024
            return max(16 * 16, int(target_area or 1024 * 1024))

        def image_reference_size(image, index: int) -> tuple[int, int]:
            if width is not None and height is not None:
                return int(width), int(height)
            if vl_width is not None and vl_height is not None:
                return int(vl_width), int(vl_height)

            from sdmlx_qwen_native.utils.image_util import ImageUtil

            pil_image = ImageUtil.load_image(image).convert("RGB")
            img_w, img_h = pil_image.size
            use_target_area = target_area_for_image(index)
            multiple = max(16, int(target_multiple or 16))
            ratio = img_w / img_h
            calc_w = int(round(((use_target_area * ratio) ** 0.5) / multiple) * multiple)
            calc_h = int(round((calc_w / ratio) / multiple) * multiple)
            return max(multiple, calc_w), max(multiple, calc_h)

        all_image_latents = []
        all_image_ids = []
        all_image_grids = []
        for index, image in enumerate(image_paths):
            calc_w, calc_h = image_reference_size(image, index)
            image_latents, image_ids, image_grid = QwenEditUtil._cached_image_conditioning(
                vae=vae,
                image=image,
                width=calc_w,
                height=calc_h,
                tiling_config=tiling_config,
            )
            all_image_latents.append(image_latents)
            all_image_ids.append(image_ids)
            all_image_grids.append(image_grid)

        image_latents = mx.concatenate(all_image_latents, axis=1)
        image_ids = mx.concatenate(all_image_ids, axis=1)

        num_images = len(image_paths)
        return image_latents, image_ids, all_image_grids, num_images

    @staticmethod
    def _create_image_ids(
        height: int,
        width: int,
    ) -> mx.array:
        latent_height = height // 16
        latent_width = width // 16

        image_ids = mx.zeros((latent_height, latent_width, 3))

        row_coords = mx.arange(0, latent_height)[:, None]
        row_coords = mx.broadcast_to(row_coords, (latent_height, latent_width))
        image_ids = mx.concatenate(
            [
                image_ids[:, :, :1],
                row_coords[:, :, None],
                image_ids[:, :, 2:],
            ],
            axis=2,
        )

        col_coords = mx.arange(0, latent_width)[None, :]
        col_coords = mx.broadcast_to(col_coords, (latent_height, latent_width))
        image_ids = mx.concatenate(
            [
                image_ids[:, :, :2],
                col_coords[:, :, None],
            ],
            axis=2,
        )

        image_ids = mx.reshape(image_ids, (latent_height * latent_width, 3))

        first_dim = mx.ones((image_ids.shape[0], 1))
        image_ids = mx.concatenate([first_dim, image_ids[:, 1:]], axis=1)

        image_ids = mx.expand_dims(image_ids, axis=0)

        return image_ids
