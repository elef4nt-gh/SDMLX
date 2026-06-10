from __future__ import annotations

from pathlib import Path

import mlx.core as mx


class Config:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
        self,
        num_inference_steps: int = 4,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 4.0,
        init_image_path: Path | None = None,
        init_image_strength: float | None = None,
    ):
        self.width = 16 * (int(width) // 16)
        self.height = 16 * (int(height) // 16)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance = float(guidance)
        self.init_image_path = init_image_path
        self.init_image_strength = init_image_strength
