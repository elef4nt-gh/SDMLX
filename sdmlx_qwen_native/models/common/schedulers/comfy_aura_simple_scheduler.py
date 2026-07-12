from __future__ import annotations

import mlx.core as mx

from sdmlx_qwen_native.models.common.schedulers.base_scheduler import BaseScheduler


class ComfyAuraSimpleScheduler(BaseScheduler):
    """ComfyUI ModelSamplingAuraFlow(shift=3.1) + simple scheduler."""

    def __init__(self, config):
        self.config = config
        self.shift = float(getattr(config, "flow_shift", None) or 3.1)
        self._sigmas = self._get_sigmas()
        self._timesteps = mx.arange(config.num_inference_steps, dtype=mx.float32)

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def _get_sigmas(self) -> mx.array:
        steps = self.config.num_inference_steps
        # Comfy's simple scheduler samples from a 1000-entry model_sampling table:
        # indices 1000, 1000 - floor(1000/steps), ...
        values = []
        for x in range(steps):
            t = (1000 - int(x * 1000 / steps)) / 1000.0
            sigma = self.shift * t / (1 + (self.shift - 1) * t)
            values.append(sigma)
        values.append(0.0)
        return mx.array(values, dtype=mx.float32)

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
        return latents + noise.astype(latents.dtype) * dt
