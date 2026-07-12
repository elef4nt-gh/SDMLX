import time

import mlx.core as mx
import mlx.nn as nn

from sdmlx_qwen_native.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from sdmlx_qwen_native.models.common.lora.layer.linear_lora_layer import LoRALinear
from sdmlx_qwen_native.models.common.lora.layer.lokr_linear_layer import LoKrLinear


class LoRASaver:
    @staticmethod
    def bake_quantized_lora_with_stats(
        module: nn.Module,
        *,
        keep_touched_dense: bool = False,
    ) -> dict[str, int | float]:
        stats: dict[str, int | float] = {
            "lora_wrappers": 0,
            "fused_wrappers": 0,
            "baked_modules": 0,
            "baked_loras": 0,
            "dense_modules": 0,
            "requantized_modules": 0,
            "passthrough": 0,
            "skipped": 0,
            "seconds": 0.0,
        }

        def _assign(parent, attr_name, idx, new_child):
            if parent is None:
                return
            if isinstance(parent, list) and idx is not None:
                parent[idx] = new_child
            elif isinstance(parent, dict) and attr_name is not None:
                parent[attr_name] = new_child
            elif attr_name is not None:
                setattr(parent, attr_name, new_child)

        def _bake_one(base_linear: nn.Module, loras: list[nn.Module]) -> nn.Module:
            if not isinstance(base_linear, nn.QuantizedLinear):
                stats["skipped"] = int(stats["skipped"]) + len(loras)
                return FusedLoRALinear(base_linear=base_linear, loras=loras) if loras else base_linear

            bakeable_loras: list[LoRALinear] = []
            passthrough_loras: list[nn.Module] = []
            for lora in loras:
                if isinstance(lora, LoRALinear) and hasattr(lora, "lora_A") and hasattr(lora, "lora_B"):
                    bakeable_loras.append(lora)
                else:
                    passthrough_loras.append(lora)

            if not bakeable_loras:
                stats["passthrough"] = int(stats["passthrough"]) + len(passthrough_loras)
                return FusedLoRALinear(base_linear=base_linear, loras=passthrough_loras) if passthrough_loras else base_linear

            dense = mx.dequantize(
                base_linear.weight,
                base_linear.scales,
                base_linear.biases,
                group_size=base_linear.group_size,
                bits=base_linear.bits,
                mode=base_linear.mode,
            ).astype(mx.float32)

            for lora in bakeable_loras:
                delta = mx.matmul(lora.lora_A.astype(mx.float32), lora.lora_B.astype(mx.float32))
                delta = mx.transpose(delta) * float(lora.scale)
                if tuple(dense.shape) != tuple(delta.shape):
                    print(
                        "⚠️  Skipping quantized LoRA bake due to shape mismatch: "
                        f"weight {dense.shape} vs delta {delta.shape}"
                    )
                    stats["skipped"] = int(stats["skipped"]) + len(loras)
                    return FusedLoRALinear(base_linear=base_linear, loras=loras)
                dense = dense + delta

            if keep_touched_dense:
                out_features, in_features = dense.shape
                bias = base_linear.get("bias")
                dense_linear = nn.Linear(in_features, out_features, bias=bias is not None)
                dense_linear.weight = dense.astype(mx.float16)
                if bias is not None:
                    dense_linear.bias = bias.astype(mx.float16)
                mx.eval(dense_linear.weight, *(x for x in (dense_linear.get("bias"),) if x is not None))
                stats["baked_modules"] = int(stats["baked_modules"]) + 1
                stats["baked_loras"] = int(stats["baked_loras"]) + len(bakeable_loras)
                stats["dense_modules"] = int(stats["dense_modules"]) + 1
                if passthrough_loras:
                    stats["passthrough"] = int(stats["passthrough"]) + len(passthrough_loras)
                    return FusedLoRALinear(base_linear=dense_linear, loras=passthrough_loras)
                return dense_linear

            weight, scales, *biases = mx.quantize(
                dense.astype(mx.bfloat16),
                base_linear.group_size,
                base_linear.bits,
                mode=base_linear.mode,
            )
            base_linear.weight = weight
            base_linear.scales = scales
            base_linear.biases = biases[0] if biases else None
            mx.eval(base_linear.weight, base_linear.scales, *(x for x in (base_linear.biases,) if x is not None))
            stats["baked_modules"] = int(stats["baked_modules"]) + 1
            stats["baked_loras"] = int(stats["baked_loras"]) + len(bakeable_loras)
            stats["requantized_modules"] = int(stats["requantized_modules"]) + 1

            if passthrough_loras:
                stats["passthrough"] = int(stats["passthrough"]) + len(passthrough_loras)
                return FusedLoRALinear(base_linear=base_linear, loras=passthrough_loras)
            return base_linear

        def _walk(obj, parent=None, attr_name=None, idx=None):
            if isinstance(obj, FusedLoRALinear):
                stats["fused_wrappers"] = int(stats["fused_wrappers"]) + 1
                new_child = _bake_one(obj.base_linear, list(obj.loras))
                _assign(parent, attr_name, idx, new_child)
                return
            if isinstance(obj, LoRALinear):
                stats["lora_wrappers"] = int(stats["lora_wrappers"]) + 1
                new_child = _bake_one(obj.linear, [obj])
                _assign(parent, attr_name, idx, new_child)
                return
            if isinstance(obj, LoKrLinear):
                stats["passthrough"] = int(stats["passthrough"]) + 1
                return

            if isinstance(obj, list):
                for i, child in enumerate(list(obj)):
                    _walk(child, obj, None, i)
            elif isinstance(obj, tuple):
                temp_list = list(obj)
                for i, child in enumerate(temp_list):
                    _walk(child, temp_list, None, i)
                if parent is not None:
                    _assign(parent, attr_name, idx, type(obj)(temp_list))
            elif isinstance(obj, dict):
                for key, child in list(obj.items()):
                    _walk(child, obj, key, None)
            elif isinstance(obj, nn.Module):
                for name, child in vars(obj).items():
                    if isinstance(child, (nn.Module, list, tuple, dict)):
                        _walk(child, obj, name, None)

        t0 = time.perf_counter()
        _walk(module, None, None, None)
        mx.clear_cache()
        stats["seconds"] = time.perf_counter() - t0
        return stats

    @staticmethod
    def bake_and_strip_lora(module: nn.Module) -> nn.Module:
        def _assign(parent, attr_name, idx, new_child):
            if parent is None:
                return
            if isinstance(parent, list) and idx is not None:
                parent[idx] = new_child
            elif isinstance(parent, dict) and attr_name is not None:
                parent[attr_name] = new_child
            elif attr_name is not None:
                setattr(parent, attr_name, new_child)

        def _bake_single(lora_layer: LoRALinear) -> nn.Module:
            base_linear = lora_layer.linear
            LoRASaver._apply_lora_delta(base_linear, lora_layer)
            return base_linear

        def _bake_fused(fused_layer: FusedLoRALinear) -> nn.Module:
            base_linear = fused_layer.base_linear
            for lora in fused_layer.loras:
                if isinstance(lora, LoRALinear):
                    LoRASaver._apply_lora_delta(base_linear, lora)
            return base_linear

        def _walk(obj, parent=None, attr_name=None, idx=None):
            # Replace wrappers first
            if isinstance(obj, FusedLoRALinear):
                new_child = _bake_fused(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child
            elif isinstance(obj, LoRALinear):
                new_child = _bake_single(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child

            # Recurse into containers/modules
            if isinstance(obj, list):
                for i, child in enumerate(list(obj)):
                    _walk(child, obj, None, i)
            elif isinstance(obj, tuple):
                temp_list = list(obj)
                for i, child in enumerate(temp_list):
                    _walk(child, temp_list, None, i)
                if parent is not None:
                    _assign(parent, attr_name, idx, type(obj)(temp_list))
            elif isinstance(obj, dict):
                for key, child in list(obj.items()):
                    _walk(child, obj, key, None)
            elif isinstance(obj, nn.Module):
                for name, child in vars(obj).items():
                    if isinstance(child, (nn.Module, list, tuple, dict)):
                        _walk(child, obj, name, None)

        _walk(module, None, None, None)
        return module

    @staticmethod
    def _apply_lora_delta(base_linear: nn.Module, lora_layer: LoRALinear) -> None:
        if not hasattr(base_linear, "weight"):
            return

        weight = base_linear.weight
        delta = mx.matmul(lora_layer.lora_A, lora_layer.lora_B)  # shape: [in, out]
        delta = mx.transpose(delta)  # shape: [out, in]
        delta = lora_layer.scale * delta

        if weight.shape != delta.shape:
            print(f"⚠️  Skipping LoRA bake due to shape mismatch: weight {weight.shape} vs delta {delta.shape}")
            return

        try:
            base_linear.weight = weight + delta.astype(weight.dtype)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Failed to bake LoRA into base layer: {e}")
