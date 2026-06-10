from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from native_flux_core import normalize_flux_weight_key


def convert(source: Path, output: Path, *, quiet: bool = False) -> None:
    t0 = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if temp.exists():
        temp.unlink()

    weights: dict[str, torch.Tensor] = {}
    skipped = 0
    renamed = 0
    with safe_open(str(source), framework="pt", device="cpu") as f:
        for key in f.keys():
            normalized = normalize_flux_weight_key(key)
            if normalized is None:
                skipped += 1
                continue
            if normalized != key:
                renamed += 1
            tensor = f.get_tensor(key)
            if tensor.is_floating_point():
                tensor = tensor.to(torch.float16)
            weights[normalized] = tensor.contiguous()

    save_file(
        weights,
        str(temp),
        metadata={
            "sdmlx_conversion": "fp8_or_prefixed_flux_to_fp16_transformer",
            "source": str(source),
        },
    )
    temp.replace(output)
    if not quiet:
        print(
            "SDMLX FLUX FP8 cache: "
            f"wrote {output}, tensors={len(weights)}, skipped={skipped}, "
            f"renamed={renamed}, time={time.perf_counter() - t0:.2f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert/normalize a FLUX FP8 or prefixed safetensor to fp16 transformer cache.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    convert(args.source, args.output, quiet=args.quiet)


if __name__ == "__main__":
    main()
