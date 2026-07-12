import os

import mlx.core as mx
from mlx import nn


_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}


def _as_triple(value: int | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            return int(value[0]), int(value[1]), int(value[2])
        if len(value) == 1:
            item = int(value[0])
            return item, item, item
        raise ValueError(f"Expected int or length-3 tuple/list, got {value!r}")
    item = int(value)
    return item, item, item


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _DISABLED_VALUES


class QwenImageCausalConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self._single_frame_2d = _env_enabled("SDMLX_QWEN_VAE_SINGLE_FRAME_2D", default=False)

    def __call__(self, x: mx.array) -> mx.array:
        kernel_t, _, _ = _as_triple(self.kernel_size)
        stride_t, stride_h, stride_w = _as_triple(self.stride)
        pad_t, pad_h, pad_w = _as_triple(self.padding)

        if self._single_frame_2d and self._can_use_single_frame_2d(x, kernel_t, stride_t, pad_t):
            return self._single_frame_2d_conv(x, stride_h, stride_w, pad_t, pad_h, pad_w)

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            pad_spec = [
                (0, 0),
                (0, 0),
                (2 * pad_t, 0),
                (pad_h, pad_h),
                (pad_w, pad_w),
            ]
            x = mx.pad(x, pad_spec)

        x = mx.transpose(x, (0, 2, 3, 4, 1))
        x = self.conv3d(x)
        x = mx.transpose(x, (0, 4, 1, 2, 3))
        return x

    @staticmethod
    def _can_use_single_frame_2d(x: mx.array, kernel_t: int, stride_t: int, pad_t: int) -> bool:
        if len(x.shape) != 5 or x.shape[2] != 1 or stride_t != 1:
            return False
        padded_t = 1 + 2 * pad_t
        if padded_t < kernel_t:
            return False
        out_t = (padded_t - kernel_t) // stride_t + 1
        slice_t = 2 * pad_t
        return out_t == 1 and 0 <= slice_t < kernel_t

    def _single_frame_2d_conv(
        self,
        x: mx.array,
        stride_h: int,
        stride_w: int,
        pad_t: int,
        pad_h: int,
        pad_w: int,
    ) -> mx.array:
        # For a single frame with causal left-padding, only one temporal kernel
        # slice can ever see non-zero input. This is equivalent to a 2D conv.
        slice_t = 2 * pad_t
        weight_2d = self.conv3d.weight[:, slice_t, :, :, :]
        x = x[:, :, 0, :, :]
        x = mx.transpose(x, (0, 2, 3, 1))
        x = mx.conv2d(x, weight_2d, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
        if self.conv3d.bias is not None:
            x = x + self.conv3d.bias.reshape(1, 1, 1, -1)
        x = mx.transpose(x, (0, 3, 1, 2))
        return x[:, :, None, :, :]
