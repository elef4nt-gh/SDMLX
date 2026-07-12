import mlx.core as mx
from mlx import nn


class LoKrLinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear | nn.QuantizedLinear,
        *,
        lokr_w1=None,
        lokr_w2=None,
        lokr_w1_a=None,
        lokr_w1_b=None,
        lokr_w2_a=None,
        lokr_w2_b=None,
        lokr_t2=None,
        alpha=None,
        scale: float = 1.0,
        output_slice_index: int | None = None,
        output_slice_splits: int | None = None,
    ):
        layer = LoKrLinear(
            lokr_w1=lokr_w1,
            lokr_w2=lokr_w2,
            lokr_w1_a=lokr_w1_a,
            lokr_w1_b=lokr_w1_b,
            lokr_w2_a=lokr_w2_a,
            lokr_w2_b=lokr_w2_b,
            lokr_t2=lokr_t2,
            alpha=alpha,
            scale=scale,
            output_slice_index=output_slice_index,
            output_slice_splits=output_slice_splits,
        )
        layer.linear = linear
        return layer

    def __init__(
        self,
        *,
        lokr_w1=None,
        lokr_w2=None,
        lokr_w1_a=None,
        lokr_w1_b=None,
        lokr_w2_a=None,
        lokr_w2_b=None,
        lokr_t2=None,
        alpha=None,
        scale: float = 1.0,
        output_slice_index: int | None = None,
        output_slice_splits: int | None = None,
    ):
        super().__init__()
        self.linear = None
        self.lokr_w1 = lokr_w1
        self.lokr_w2 = lokr_w2
        self.lokr_w1_a = lokr_w1_a
        self.lokr_w1_b = lokr_w1_b
        self.lokr_w2_a = lokr_w2_a
        self.lokr_w2_b = lokr_w2_b
        self.lokr_t2 = lokr_t2
        self.alpha = alpha
        self.scale = scale
        self.output_slice_index = output_slice_index
        self.output_slice_splits = output_slice_splits

    def _alpha_scale(self):
        if self.alpha is None:
            return 1.0
        if self.lokr_w1 is None and self.lokr_w1_b is not None:
            return float(self.alpha) / float(self.lokr_w1_b.shape[0])
        if self.lokr_w2 is None and self.lokr_w2_b is not None:
            return float(self.alpha) / float(self.lokr_w2_b.shape[0])
        return 1.0

    def _w1(self, dtype):
        if self.lokr_w1 is not None:
            return self.lokr_w1.astype(dtype)
        return mx.matmul(self.lokr_w1_a.astype(dtype), self.lokr_w1_b.astype(dtype))

    def _w2(self, dtype):
        if self.lokr_w2 is not None:
            return self.lokr_w2.astype(dtype)
        if self.lokr_t2 is not None:
            raise NotImplementedError("LoKr Tucker tensors are not supported yet")
        return mx.matmul(self.lokr_w2_a.astype(dtype), self.lokr_w2_b.astype(dtype))

    def __call__(self, x):
        base_out = self.linear(x)
        return base_out + self.delta(x)

    def delta(self, x):
        w1 = self._w1(x.dtype)
        w2 = self._w2(x.dtype)
        if self.output_slice_index is not None:
            return (self.scale * self._alpha_scale()) * self._delta_output_slice(
                x,
                w1,
                w2,
                self.output_slice_index,
                self.output_slice_splits or 1,
            )

        return (self.scale * self._alpha_scale()) * self._delta_full(x, w1, w2)

    def _delta_full(self, x, w1, w2):
        uq = w1.shape[1]
        in_features = x.shape[-1]
        if in_features % uq != 0:
            raise ValueError(f"LoKr input dim {in_features} is not divisible by {uq}")

        grouped = mx.reshape(x, (*x.shape[:-1], uq, in_features // uq))
        h2 = mx.matmul(grouped, mx.swapaxes(w2, -1, -2))
        h2 = mx.swapaxes(h2, -1, -2)
        h1 = mx.matmul(h2, mx.swapaxes(w1, -1, -2))
        out = mx.reshape(mx.swapaxes(h1, -1, -2), (*x.shape[:-1], -1))
        return out

    def _delta_output_slice(self, x, w1, w2, slice_index: int, slice_splits: int):
        uq = w1.shape[1]
        in_features = x.shape[-1]
        if in_features % uq != 0:
            raise ValueError(f"LoKr input dim {in_features} is not divisible by {uq}")

        out_l = int(w1.shape[0])
        out_k = int(w2.shape[0])
        full_out = out_l * out_k
        if slice_splits <= 0 or full_out % slice_splits != 0:
            raise ValueError(f"LoKr output dim {full_out} is not divisible by slice count {slice_splits}")
        if slice_index < 0 or slice_index >= slice_splits:
            raise ValueError(f"LoKr output slice index {slice_index} is outside 0..{slice_splits - 1}")

        slice_size = full_out // slice_splits
        start = slice_index * slice_size
        end = start + slice_size
        grouped = mx.reshape(x, (*x.shape[:-1], uq, in_features // uq))
        segments = []
        for row_idx in range(out_l):
            row_start = row_idx * out_k
            row_end = row_start + out_k
            seg_start = max(start, row_start)
            seg_end = min(end, row_end)
            if seg_start >= seg_end:
                continue
            w2_start = seg_start - row_start
            w2_end = seg_end - row_start
            w2_part = w2[w2_start:w2_end]
            h2 = mx.matmul(grouped, mx.swapaxes(w2_part, -1, -2))
            w1_row = mx.reshape(w1[row_idx], (1,) * (h2.ndim - 2) + (uq, 1))
            segments.append(mx.sum(h2 * w1_row, axis=-2))
        if not segments:
            raise ValueError(f"LoKr output slice {slice_index}/{slice_splits} produced no segments")
        if len(segments) == 1:
            return segments[0]
        return mx.concatenate(segments, axis=-1)
