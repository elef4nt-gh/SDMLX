import math
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np
from PIL import Image

from sdmlx_qwen_native.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor


class QwenVisionLanguageTokenizer:
    def __init__(
        self,
        processor: QwenVisionLanguageProcessor,
        max_length: int = 1024,
        use_picture_prefix: bool = True,
    ):
        self.processor = processor
        self.max_length = max_length
        self.use_picture_prefix = use_picture_prefix

        self.picture_prefix_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency "
            "with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n"
            "{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.single_image_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency "
            "with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.edit_template = self.picture_prefix_template if use_picture_prefix else self.single_image_template
        self.edit_template_start_idx = 64
        self.last_template_drop_idx: int | None = None

    @staticmethod
    def _compute_template_drop_idx(input_ids) -> int | None:
        ids = np.asarray(input_ids)
        if ids.ndim == 0:
            return None
        row = ids[0] if ids.ndim > 1 else ids
        template_end = -1
        im_start_count = 0
        for index, token in enumerate(row.tolist()):
            if int(token) == 151644 and im_start_count < 2:
                template_end = index
                im_start_count += 1
        if template_end < 0:
            return None
        if len(row) > template_end + 2 and int(row[template_end + 1]) == 872 and int(row[template_end + 2]) == 198:
            template_end += 3
        return max(0, int(template_end))

    def tokenize_with_image(
        self,
        prompt: str,
        image: Union[Image.Image, np.ndarray, str, list],
        vl_width: int | None = None,
        vl_height: int | None = None,
        vl_target_area: int | None = None,
        use_picture_prefix: bool | None = None,
        prompt_template: str | None = None,
        image_slots: list[int] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # Normalize image to list format
        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        # Format prompt based on tokenizer mode. The SDMLX nodes can switch
        # between Comfy's normal and Plus Qwen edit templates per invocation.
        active_use_picture_prefix = self.use_picture_prefix if use_picture_prefix is None else bool(use_picture_prefix)
        if active_use_picture_prefix:
            # Edit format: Add "Picture N:" prefix for each image
            # For multiple images: "Picture 1: ... Picture 2: ... Picture N: ..."
            slots = list(image_slots) if image_slots is not None else list(range(1, len(images) + 1))
            if len(slots) != len(images):
                raise ValueError("Qwen image slot count must match image count.")
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            base_img_prompt = ""
            for slot in slots:
                base_img_prompt += img_prompt_template.format(int(slot))
            active_template = prompt_template or self.picture_prefix_template
            formatted_text = active_template.format(base_img_prompt + prompt)
        else:
            # Regular Edit format: Vision tokens already in template
            # Just format with user prompt directly
            if len(images) > 1:
                raise ValueError("Qwen normal image edit conditioning accepts only one image.")
            active_template = prompt_template or self.single_image_template
            formatted_text = active_template.format(prompt)

        # Process images: convert to PIL Images and resize to CONDITION_IMAGE_SIZE
        condition_image_size = int(vl_target_area or 384 * 384)

        processed_images = []
        for img in images:
            # Convert to PIL Image
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unsupported image type: {type(img)}")

            if vl_width is not None and vl_height is not None:
                condition_width = int(vl_width)
                condition_height = int(vl_height)
            else:
                # Comfy's Qwen Image Edit Plus node first rescales each image
                # independently to the target area while preserving aspect ratio.
                # The MLX image processor performs its own patch-size alignment
                # after this pre-scale.
                img_w, img_h = img.size
                ratio = img_w / img_h
                condition_width = math.sqrt(condition_image_size * ratio)
                condition_height = condition_width / ratio
                condition_width = round(condition_width)
                condition_height = round(condition_height)

            area_resample = Image.Resampling.BOX if hasattr(Image, "Resampling") else Image.BOX
            img = img.resize((int(condition_width), int(condition_height)), area_resample)
            processed_images.append(img)

        # Use our MLX processor for both text and images
        model_inputs = self.processor(
            text=[formatted_text],
            images=processed_images,
            padding=True,
            return_tensors=None,  # Return numpy/MLX arrays, not PyTorch
        )
        self.last_template_drop_idx = self._compute_template_drop_idx(model_inputs["input_ids"]) if prompt_template else None

        grid_thw = model_inputs["image_grid_thw"][0]
        factor = 14 * 2
        self._vl_image_width = int(grid_thw[2]) * factor
        self._vl_image_height = int(grid_thw[1]) * factor

        # Convert to MLX arrays if needed
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = mx.array(model_inputs["pixel_values"])
        image_grid_thw = mx.array(model_inputs["image_grid_thw"])

        return input_ids, attention_mask, pixel_values, image_grid_thw

    def tokenize_text_only(self, prompt: str) -> tuple[mx.array, mx.array]:
        # Use the regular text-only template
        text_template = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        formatted_text = text_template.format(prompt)
        tokens = self.processor.tokenizer(
            formatted_text,
            max_length=self.max_length + 34,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Convert PyTorch tensors to MLX arrays
        input_ids = mx.array(tokens["input_ids"].numpy())
        attention_mask = mx.array(tokens["attention_mask"].numpy())

        return input_ids, attention_mask
