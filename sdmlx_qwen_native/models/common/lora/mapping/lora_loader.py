import os
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from sdmlx_qwen_native.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from sdmlx_qwen_native.models.common.lora.layer.linear_lora_layer import LoRALinear
from sdmlx_qwen_native.models.common.lora.layer.lokr_linear_layer import LoKrLinear
from sdmlx_qwen_native.models.common.lora.mapping.lora_mapping import LoRATarget
from sdmlx_qwen_native.models.common.resolution.lora_resolution import LoraResolution


@dataclass
class PatternMatch:
    source_pattern: str
    target_path: str
    matrix_name: str  # "lora_A", "lora_B", "alpha", "diff", or "diff_b"
    transpose: bool
    transform: Callable[[mx.array], mx.array] | None = None
    lokr_output_index: int | None = None
    lokr_output_splits: int | None = None


_LORA_MAPPING_PLAN_CACHE_MAX = 8
_LORA_MAPPING_PLAN_CACHE: OrderedDict[tuple, tuple[dict[str, dict], set[str], set[str], int]] = OrderedDict()


def _qwen_lora_verbose() -> bool:
    value = str(os.environ.get("SDMLX_QWEN_VERBOSE") or os.environ.get("SDMLX_QWEN_DEBUG") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _qwen_lora_log(message: str) -> None:
    if _qwen_lora_verbose():
        print(message)


def _flux2_lora_policy() -> str:
    return str(os.environ.get("SDMLX_FLUX2_LORA_POLICY") or "").strip().lower().replace("-", "_")


def _flux2_lora_verbose() -> bool:
    value = str(os.environ.get("SDMLX_FLUX2_LORA_VERBOSE") or os.environ.get("SDMLX_FLUX2_LORA_DEBUG") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _flux2_character_safe_multiplier(target_path: str) -> float:
    if target_path.startswith("single_transformer_blocks."):
        if target_path.endswith(".attn.to_qkv_mlp_proj"):
            return 0.45
        if target_path.endswith(".attn.to_out"):
            return 0.65
        return 0.55
    if target_path.startswith("transformer_blocks."):
        if ".ff." in target_path or ".ff_context." in target_path:
            return 0.75
        if ".attn." in target_path:
            return 1.0
        return 0.8
    return 1.0


def _flux2_character_soft_multiplier(target_path: str) -> float:
    if target_path.startswith("single_transformer_blocks."):
        if target_path.endswith(".attn.to_qkv_mlp_proj"):
            return 0.75
        if target_path.endswith(".attn.to_out"):
            return 0.9
        return 0.8
    if target_path.startswith("transformer_blocks."):
        if ".ff." in target_path or ".ff_context." in target_path:
            return 0.9
        return 1.0
    return 1.0


def _lora_effective_scale_for_target(scale: float, target_path: str, role: str | None) -> float:
    if role != "flux2":
        return scale
    policy = _flux2_lora_policy()
    if policy in {"", "off", "none", "default", "full"}:
        return scale
    if policy == "character_safe":
        multiplier = _flux2_character_safe_multiplier(target_path)
    elif policy in {"character_soft", "character_mild"}:
        multiplier = _flux2_character_soft_multiplier(target_path)
    else:
        if _flux2_lora_verbose():
            print(f"SDMLX FLUX.2 LoRA: unknown policy {policy!r}; using full LoRA scale")
        return scale
    if _flux2_lora_verbose() and multiplier != 1.0:
        print(
            f"SDMLX FLUX.2 LoRA policy {policy}: "
            f"{target_path} scale {scale:g} -> {scale * multiplier:g}"
        )
    return scale * multiplier


class LoRALoader:
    @staticmethod
    def load_and_apply_lora(
        lora_mapping: list[LoRATarget],
        transformer: nn.Module,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        role: str | None = None,
    ) -> tuple[list[str], list[float]]:
        resolved_paths = LoraResolution.resolve_paths(lora_paths)
        if not resolved_paths:
            return resolved_paths, []

        resolved_scales = LoraResolution.resolve_scales(lora_scales, len(resolved_paths))
        if len(resolved_scales) != len(resolved_paths):
            raise ValueError(
                f"Number of LoRA scales ({len(resolved_scales)}) must match number of LoRA files ({len(resolved_paths)})"
            )

        _qwen_lora_log(f"Loading {len(resolved_paths)} LoRA file(s)...")

        for lora_file, scale in zip(resolved_paths, resolved_scales):
            LoRALoader._apply_single_lora(transformer, lora_file, scale, lora_mapping, role=role)

        _qwen_lora_log("Supported LoRA weights applied successfully")

        return resolved_paths, resolved_scales

    @staticmethod
    def _apply_single_lora(
        transformer: nn.Module,
        lora_file: str,
        scale: float,
        lora_mapping: list[LoRATarget],
        *,
        role: str | None,
    ) -> None:
        # Load the LoRA weights
        if not Path(lora_file).exists():
            print(f"❌ LoRA file not found: {lora_file}")
            return

        role_suffix = f", role={role}" if role else ""
        _qwen_lora_log(f"Applying LoRA: {Path(lora_file).name} (scale={scale}{role_suffix})")

        # Build pattern mappings from LoRATargets
        pattern_mappings = LoRALoader._build_pattern_mappings(lora_mapping)
        cache_key = LoRALoader._mapping_plan_cache_key(lora_file, pattern_mappings, role=role)
        cached_plan = _LORA_MAPPING_PLAN_CACHE.get(cache_key)
        if cached_plan is not None:
            _LORA_MAPPING_PLAN_CACHE.move_to_end(cache_key)
            lora_data_by_target, matched_keys, unmatched_keys, total_keys = cached_plan
            _qwen_lora_log("   Reusing cached LoRA mapping plan")
        else:
            try:
                weights = dict(mx.load(lora_file, return_metadata=True)[0].items())
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                print(f"❌ Failed to load LoRA file: {e}")
                return

            total_keys = len(weights)
            lora_data_by_target, matched_keys = LoRALoader._build_lora_application_plan(
                weights,
                pattern_mappings,
            )
            unmatched_keys = set(weights.keys()) - matched_keys
            LoRALoader._store_mapping_plan_cache(
                cache_key,
                lora_data_by_target,
                matched_keys,
                unmatched_keys,
                total_keys,
            )

        applied_count = LoRALoader._apply_lora_plan(transformer, lora_data_by_target, scale, role=role)

        _qwen_lora_log(f"   Applied to {applied_count} layers ({len(matched_keys)}/{total_keys} keys matched)")

        if role == "modulation":
            if not matched_keys:
                _qwen_lora_log("   No Qwen modulation keys found in this LoRA file")
            elif unmatched_keys:
                _qwen_lora_log(
                    f"   {len(unmatched_keys)} non-modulation keys were already "
                    "handled by the stable pass"
                )
            return

        if unmatched_keys:
            all_weight_keys = matched_keys | unmatched_keys
            modulation_keys = {key for key in unmatched_keys if LoRALoader._is_qwen_modulation_key(key)}
            lokr_keys = {key for key in unmatched_keys if LoRALoader._is_lokr_key(key)}
            lokr_alpha_keys = {
                key
                for key in unmatched_keys
                if LoRALoader._is_lokr_alpha_key(key, all_weight_keys)
            }
            lokr_keys.update(lokr_alpha_keys)
            remaining_unmatched = unmatched_keys - modulation_keys - lokr_keys
            if modulation_keys and not remaining_unmatched and role == "stable":
                _qwen_lora_log(
                    f"   {len(modulation_keys)} Qwen modulation keys are outside "
                    "the stable pass (handled only when Qwen LoRA modulation is enabled)"
                )
            else:
                if modulation_keys and role == "stable":
                    _qwen_lora_log(
                        f"   {len(modulation_keys)} Qwen modulation keys are outside "
                        "the stable pass (handled only when Qwen LoRA modulation is enabled)"
                    )
                if lokr_keys:
                    _qwen_lora_log(
                        f"   Skipped {len(lokr_keys)} LoKr/LyCORIS keys "
                        "(unmapped target or experimental modulation target)"
                    )
                if remaining_unmatched:
                    _qwen_lora_log(f"   {len(remaining_unmatched)} unmatched keys in LoRA file:")
                    for key in sorted(remaining_unmatched)[:5]:
                        _qwen_lora_log(f"      - {key}")
                    if len(remaining_unmatched) > 5:
                        _qwen_lora_log(f"      ... and {len(remaining_unmatched) - 5} more")

    @staticmethod
    def _is_qwen_modulation_key(key: str) -> bool:
        has_mod_target = (
            ".img_mod.1." in key
            or ".txt_mod.1." in key
            or "_img_mod_1." in key
            or "_txt_mod_1." in key
        )
        has_adapter_suffix = (
            ".lora_A" in key
            or ".lora_B" in key
            or ".lora_up" in key
            or ".lora_down" in key
            or key.endswith(".diff")
            or key.endswith(".diff_b")
            or ".lokr_w1" in key
            or ".lokr_w2" in key
            or ".lokr_t2" in key
            or key.endswith(".alpha")
        )
        return has_mod_target and has_adapter_suffix

    @staticmethod
    def _is_lokr_key(key: str) -> bool:
        return (
            key.endswith(".lokr_w1")
            or key.endswith(".lokr_w2")
            or key.endswith(".lokr_w1_a")
            or key.endswith(".lokr_w1_b")
            or key.endswith(".lokr_w2_a")
            or key.endswith(".lokr_w2_b")
            or key.endswith(".lokr_t2")
        )

    @staticmethod
    def _is_lokr_alpha_key(key: str, weights: dict) -> bool:
        if not key.endswith(".alpha"):
            return False
        prefix = key[: -len(".alpha")]
        return (
            f"{prefix}.lokr_w1" in weights
            or f"{prefix}.lokr_w2" in weights
            or f"{prefix}.lokr_w1_a" in weights
            or f"{prefix}.lokr_w2_a" in weights
            or f"{prefix}.lokr_t2" in weights
        )

    @staticmethod
    def _build_pattern_mappings(targets: list[LoRATarget]) -> list[PatternMatch]:
        mappings = []

        for target in targets:
            lokr_output_index = LoRALoader._lokr_output_index_for_transform(target.up_transform)
            lokr_w2_transform = None if lokr_output_index is not None else target.up_transform

            # Add up weight patterns (lora_B)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="lora_B",
                    transpose=True,
                    transform=target.up_transform,
                )
                for pattern in target.possible_up_patterns
            )

            # Add down weight patterns (lora_A)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="lora_A",
                    transpose=True,
                    transform=target.down_transform,
                )
                for pattern in target.possible_down_patterns
            )

            # Add alpha patterns and matching LoKr/LyCORIS tensors.
            for pattern in target.possible_alpha_patterns:
                mappings.append(
                    PatternMatch(
                        source_pattern=pattern,
                        target_path=target.model_path,
                        matrix_name="alpha",
                        transpose=False,
                        transform=None,
                    )
                )
                if pattern.endswith(".alpha"):
                    base_pattern = pattern[: -len(".alpha")]
                    mappings.extend(
                        [
                            PatternMatch(
                                source_pattern=f"{base_pattern}.lokr_w1",
                                target_path=target.model_path,
                                matrix_name="lokr_w1",
                                transpose=False,
                                transform=None,
                            ),
                            PatternMatch(
                                source_pattern=f"{base_pattern}.lokr_w2",
                                target_path=target.model_path,
                                matrix_name="lokr_w2",
                                transpose=False,
                                transform=lokr_w2_transform,
                                lokr_output_index=lokr_output_index,
                                lokr_output_splits=3 if lokr_output_index is not None else None,
                            ),
                            PatternMatch(
                                source_pattern=f"{base_pattern}.lokr_w1_a",
                                target_path=target.model_path,
                                matrix_name="lokr_w1_a",
                                transpose=False,
                                transform=None,
                            ),
                            PatternMatch(
                                source_pattern=f"{base_pattern}.lokr_w1_b",
                                target_path=target.model_path,
                                matrix_name="lokr_w1_b",
                                transpose=False,
                                transform=None,
                            ),
                            PatternMatch(
                                source_pattern=f"{base_pattern}.lokr_w2_a",
                                target_path=target.model_path,
                                matrix_name="lokr_w2_a",
                                transpose=False,
                                transform=lokr_w2_transform,
                                lokr_output_index=lokr_output_index,
                                lokr_output_splits=3 if lokr_output_index is not None else None,
                            ),
                            PatternMatch(
                                source_pattern=f"{base_pattern}.lokr_w2_b",
                                target_path=target.model_path,
                                matrix_name="lokr_w2_b",
                                transpose=False,
                                transform=None,
                            ),
                            PatternMatch(
                                source_pattern=f"{base_pattern}.lokr_t2",
                                target_path=target.model_path,
                                matrix_name="lokr_t2",
                                transpose=False,
                                transform=None,
                            ),
                        ]
                    )

            # Add direct Comfy-style model deltas.
            for pattern in target.possible_diff_patterns:
                mappings.append(
                    PatternMatch(
                        source_pattern=pattern,
                        target_path=target.model_path,
                        matrix_name="diff",
                        transpose=False,
                        transform=None,
                    )
                )
            for pattern in target.possible_diff_b_patterns:
                mappings.append(
                    PatternMatch(
                        source_pattern=pattern,
                        target_path=target.model_path,
                        matrix_name="diff_b",
                        transpose=False,
                        transform=None,
                    )
                )

        return mappings

    @staticmethod
    def _lokr_output_index_for_transform(transform: Callable[[mx.array], mx.array] | None) -> int | None:
        if transform is None:
            return None
        return {
            "split_q_up": 0,
            "split_k_up": 1,
            "split_v_up": 2,
        }.get(getattr(transform, "__name__", ""))

    @staticmethod
    def _apply_lora_with_mapping(
        transformer: nn.Module,
        weights: dict,
        scale: float,
        pattern_mappings: list[PatternMatch],
        *,
        role: str | None,
    ) -> tuple[int, set]:
        lora_data_by_target, matched_keys = LoRALoader._build_lora_application_plan(weights, pattern_mappings)
        applied_count = LoRALoader._apply_lora_plan(transformer, lora_data_by_target, scale, role=role)
        return applied_count, matched_keys

    @staticmethod
    def _build_lora_application_plan(
        weights: dict,
        pattern_mappings: list[PatternMatch],
    ) -> tuple[dict[str, dict], set[str]]:
        lora_data_by_target: dict[str, dict] = {}
        matched_keys: set[str] = set()

        # For each weight key, find ALL matching patterns (not just first)
        # This allows multiple targets to use the same source (e.g., QKV split)
        for weight_key, weight_value in weights.items():
            for mapping in pattern_mappings:
                match_result = LoRALoader._match_pattern(weight_key, mapping.source_pattern)
                if match_result is None:
                    continue

                matched_keys.add(weight_key)
                block_idx = match_result

                # Resolve target path with block index if needed
                target_path = mapping.target_path
                if block_idx is not None and "{block}" in target_path:
                    target_path = target_path.format(block=block_idx)

                # Apply transform if specified
                transformed_value = weight_value
                if mapping.transform is not None:
                    transformed_value = mapping.transform(weight_value)

                # Apply transpose if needed
                if mapping.transpose:
                    transformed_value = transformed_value.T

                # Store for this target
                if target_path not in lora_data_by_target:
                    lora_data_by_target[target_path] = {}

                lora_data_by_target[target_path][mapping.matrix_name] = transformed_value
                if mapping.lokr_output_index is not None:
                    lora_data_by_target[target_path]["_lokr_output_index"] = mapping.lokr_output_index
                    lora_data_by_target[target_path]["_lokr_output_splits"] = mapping.lokr_output_splits or 1

        return lora_data_by_target, matched_keys

    @staticmethod
    def _apply_lora_plan(
        transformer: nn.Module,
        lora_data_by_target: dict[str, dict],
        scale: float,
        *,
        role: str | None,
    ) -> int:
        # Apply LoRA to each target
        applied_count = 0
        skipped_alpha_only = 0
        for target_path, lora_data in lora_data_by_target.items():
            if set(lora_data) == {"alpha"}:
                skipped_alpha_only += 1
                continue
            if LoRALoader._apply_lora_matrices_to_target(transformer, target_path, lora_data, scale, role=role):
                applied_count += 1

        if skipped_alpha_only:
            _qwen_lora_log(
                f"   Skipped {skipped_alpha_only} alpha-only LoRA targets "
                "(no lora_A/lora_B matrices found)"
            )

        return applied_count

    @staticmethod
    def _mapping_plan_cache_key(
        lora_file: str,
        pattern_mappings: list[PatternMatch],
        *,
        role: str | None,
    ) -> tuple:
        path = Path(lora_file).expanduser()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        try:
            stat = resolved.stat()
            identity = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
        except Exception:
            identity = (str(resolved), None, None)
        mapping_signature = tuple(
            (
                mapping.source_pattern,
                mapping.target_path,
                mapping.matrix_name,
                bool(mapping.transpose),
                id(mapping.transform) if mapping.transform is not None else None,
                mapping.lokr_output_index,
                mapping.lokr_output_splits,
            )
            for mapping in pattern_mappings
        )
        return (identity, role, mapping_signature)

    @staticmethod
    def _store_mapping_plan_cache(
        cache_key: tuple,
        lora_data_by_target: dict[str, dict],
        matched_keys: set[str],
        unmatched_keys: set[str],
        total_keys: int,
    ) -> None:
        _LORA_MAPPING_PLAN_CACHE[cache_key] = (
            lora_data_by_target,
            set(matched_keys),
            set(unmatched_keys),
            int(total_keys),
        )
        _LORA_MAPPING_PLAN_CACHE.move_to_end(cache_key)
        while len(_LORA_MAPPING_PLAN_CACHE) > _LORA_MAPPING_PLAN_CACHE_MAX:
            _LORA_MAPPING_PLAN_CACHE.popitem(last=False)

    @staticmethod
    def _match_pattern(weight_key: str, pattern: str) -> int | None:
        if "{block}" in pattern:
            # Find all numbers in the weight key
            numbers_in_key = re.findall(r"\d+", weight_key)
            for num_str in numbers_in_key:
                test_block_idx = int(num_str)
                concrete_pattern = pattern.replace("{block}", str(test_block_idx))
                if weight_key == concrete_pattern:
                    return test_block_idx
            return None
        else:
            if weight_key == pattern:
                return 0  # Return 0 to indicate match (no block)
            return None

    @staticmethod
    def _apply_lora_matrices_to_target(
        transformer: nn.Module, target_path: str, lora_data: dict, scale: float, *, role: str | None
    ) -> bool:
        # Navigate to the target layer
        current_module = transformer
        path_parts = target_path.split(".")

        try:
            for part in path_parts:
                if part.isdigit():
                    current_module = current_module[int(part)]
                elif isinstance(current_module, dict) and part in current_module:
                    current_module = current_module[part]
                else:
                    current_module = getattr(current_module, part)
        except (AttributeError, IndexError, KeyError):
            print(f"❌ Could not find target path: {target_path}")
            return False

        if LoRALoader._has_lokr_data(lora_data):
            return LoRALoader._apply_lokr_matrices_to_target(
                transformer, current_module, path_parts, target_path, lora_data, scale, role=role
            )

        effective_scale = _lora_effective_scale_for_target(scale, target_path, role)
        direct_delta_applied = LoRALoader._apply_direct_deltas(current_module, target_path, lora_data, effective_scale)

        # Check if we have the required matrices
        if "lora_A" not in lora_data or "lora_B" not in lora_data:
            if direct_delta_applied:
                return True
            print(f"❌ Missing required LoRA matrices for {target_path}")
            return False

        # Values are already transformed and transposed
        lora_A = lora_data["lora_A"]
        lora_B = lora_data["lora_B"]

        # Handle alpha scaling
        alpha_scale = 1.0
        if "alpha" in lora_data:
            alpha_value = lora_data["alpha"]
            rank = lora_A.shape[1]
            alpha_scale = float(alpha_value) / rank

        # Create new LoRA layer. Support stacking normal A/B LoRAs on LoKr
        # wrappers too; mixed Qwen LoRA chains commonly combine both formats.
        is_linear = hasattr(current_module, "weight")
        is_lora_linear = isinstance(current_module, LoRALinear)
        is_lokr_linear = isinstance(current_module, LoKrLinear)
        is_fused_linear = isinstance(current_module, FusedLoRALinear)

        if is_linear or is_lora_linear or is_lokr_linear or is_fused_linear:
            # Handle fusion: if the current module is already a LoRA layer, fuse them
            if is_lora_linear or is_lokr_linear:
                lora_layer = LoRALinear.from_linear(current_module.linear, r=lora_A.shape[1], scale=effective_scale)
                lora_layer._mflux_lora_role = role
                lora_layer.lora_A = lora_A
                lora_layer.lora_B = lora_B
                if "alpha" in lora_data:
                    lora_layer.lora_B = lora_layer.lora_B * alpha_scale
                fused_layer = FusedLoRALinear(base_linear=current_module.linear, loras=[current_module, lora_layer])
                replacement_layer = fused_layer
            elif is_fused_linear:
                lora_layer = LoRALinear.from_linear(
                    current_module.base_linear, r=lora_A.shape[1], scale=effective_scale
                )
                lora_layer._mflux_lora_role = role
                lora_layer.lora_A = lora_A
                lora_layer.lora_B = lora_B
                if "alpha" in lora_data:
                    lora_layer.lora_B = lora_layer.lora_B * alpha_scale
                fused_layer = FusedLoRALinear(
                    base_linear=current_module.base_linear, loras=current_module.loras + [lora_layer]
                )
                replacement_layer = fused_layer
            else:
                # First LoRA on this layer
                lora_layer = LoRALinear.from_linear(current_module, r=lora_A.shape[1], scale=effective_scale)
                lora_layer._mflux_lora_role = role
                lora_layer.lora_A = lora_A
                lora_layer.lora_B = lora_B
                if "alpha" in lora_data:
                    lora_layer.lora_B = lora_layer.lora_B * alpha_scale
                replacement_layer = lora_layer

            # Replace the layer in the parent module
            parent_module = transformer
            for part in path_parts[:-1]:
                if part.isdigit():
                    parent_module = parent_module[int(part)]
                elif isinstance(parent_module, dict) and part in parent_module:
                    parent_module = parent_module[part]
                else:
                    parent_module = getattr(parent_module, part)

            final_attr = path_parts[-1]
            if final_attr.isdigit():
                parent_module[int(final_attr)] = replacement_layer
            elif isinstance(parent_module, dict) and final_attr in parent_module:
                parent_module[final_attr] = replacement_layer
            else:
                setattr(parent_module, final_attr, replacement_layer)

            return True
        else:
            print(f"❌ Target layer {target_path} is not a linear layer")
            return False

    @staticmethod
    def _apply_direct_deltas(current_module, target_path: str, lora_data: dict, scale: float) -> bool:
        applied = False

        if isinstance(current_module, FusedLoRALinear):
            base_module = current_module.base_linear
        elif isinstance(current_module, (LoRALinear, LoKrLinear)):
            base_module = current_module.linear
        else:
            base_module = current_module

        if "diff" in lora_data:
            if not hasattr(base_module, "weight"):
                print(f"❌ Direct LoRA diff target has no weight: {target_path}")
            elif isinstance(base_module, nn.QuantizedLinear):
                diff = lora_data["diff"]
                dense = mx.dequantize(
                    base_module.weight,
                    base_module.scales,
                    base_module.biases,
                    group_size=base_module.group_size,
                    bits=base_module.bits,
                    mode=base_module.mode,
                ).astype(mx.float32)
                if tuple(diff.shape) != tuple(dense.shape):
                    print(
                        f"❌ Direct LoRA diff shape mismatch for quantized {target_path}: "
                        f"{tuple(diff.shape)} != {tuple(dense.shape)}"
                    )
                else:
                    dense = dense + (diff * scale).astype(mx.float32)
                    weight, scales, *biases = mx.quantize(
                        dense.astype(mx.bfloat16),
                        base_module.group_size,
                        base_module.bits,
                        mode=base_module.mode,
                    )
                    base_module.weight = weight
                    base_module.scales = scales
                    base_module.biases = biases[0] if biases else None
                    applied = True
            else:
                diff = lora_data["diff"]
                weight = base_module.weight
                if tuple(diff.shape) != tuple(weight.shape):
                    print(
                        f"❌ Direct LoRA diff shape mismatch for {target_path}: "
                        f"{tuple(diff.shape)} != {tuple(weight.shape)}"
                    )
                else:
                    base_module.weight = weight + (diff * scale).astype(weight.dtype)
                    applied = True

        if "diff_b" in lora_data:
            if not hasattr(base_module, "bias") or base_module.bias is None:
                _qwen_lora_log(f"   Skipped direct LoRA bias delta for biasless target: {target_path}")
            else:
                diff_b = lora_data["diff_b"]
                bias = base_module.bias
                if tuple(diff_b.shape) != tuple(bias.shape):
                    print(
                        f"❌ Direct LoRA bias delta shape mismatch for {target_path}: "
                        f"{tuple(diff_b.shape)} != {tuple(bias.shape)}"
                    )
                else:
                    base_module.bias = bias + (diff_b * scale).astype(bias.dtype)
                    applied = True

        return applied

    @staticmethod
    def _has_lokr_data(lora_data: dict) -> bool:
        return any(
            key in lora_data
            for key in (
                "lokr_w1",
                "lokr_w2",
                "lokr_w1_a",
                "lokr_w1_b",
                "lokr_w2_a",
                "lokr_w2_b",
                "lokr_t2",
            )
        )

    @staticmethod
    def _replace_target_module(transformer: nn.Module, path_parts: list[str], replacement_layer) -> None:
        parent_module = transformer
        for part in path_parts[:-1]:
            if part.isdigit():
                parent_module = parent_module[int(part)]
            elif isinstance(parent_module, dict) and part in parent_module:
                parent_module = parent_module[part]
            else:
                parent_module = getattr(parent_module, part)

        final_attr = path_parts[-1]
        if final_attr.isdigit():
            parent_module[int(final_attr)] = replacement_layer
        elif isinstance(parent_module, dict) and final_attr in parent_module:
            parent_module[final_attr] = replacement_layer
        else:
            setattr(parent_module, final_attr, replacement_layer)

    @staticmethod
    def _apply_lokr_matrices_to_target(
        transformer: nn.Module,
        current_module,
        path_parts: list[str],
        target_path: str,
        lora_data: dict,
        scale: float,
        *,
        role: str | None,
    ) -> bool:
        if "lokr_t2" in lora_data:
            print(f"❌ LoKr Tucker tensors are not supported yet for {target_path}")
            return False

        has_w1 = "lokr_w1" in lora_data or ("lokr_w1_a" in lora_data and "lokr_w1_b" in lora_data)
        has_w2 = "lokr_w2" in lora_data or ("lokr_w2_a" in lora_data and "lokr_w2_b" in lora_data)
        if not has_w1 or not has_w2:
            print(f"❌ Missing required LoKr matrices for {target_path}")
            return False

        is_linear = hasattr(current_module, "weight")
        is_lora_linear = isinstance(current_module, LoRALinear)
        is_lokr_linear = isinstance(current_module, LoKrLinear)
        is_fused_linear = isinstance(current_module, FusedLoRALinear)
        if not (is_linear or is_lora_linear or is_lokr_linear or is_fused_linear):
            print(f"❌ Target layer {target_path} is not a linear layer")
            return False

        if is_fused_linear:
            base_linear = current_module.base_linear
        elif is_lora_linear or is_lokr_linear:
            base_linear = current_module.linear
        else:
            base_linear = current_module

        LoRALoader._validate_lokr_shape(base_linear, target_path, lora_data)

        lokr_layer = LoKrLinear.from_linear(
            base_linear,
            lokr_w1=lora_data.get("lokr_w1"),
            lokr_w2=lora_data.get("lokr_w2"),
            lokr_w1_a=lora_data.get("lokr_w1_a"),
            lokr_w1_b=lora_data.get("lokr_w1_b"),
            lokr_w2_a=lora_data.get("lokr_w2_a"),
            lokr_w2_b=lora_data.get("lokr_w2_b"),
            lokr_t2=lora_data.get("lokr_t2"),
            alpha=lora_data.get("alpha"),
            scale=scale,
            output_slice_index=lora_data.get("_lokr_output_index"),
            output_slice_splits=lora_data.get("_lokr_output_splits"),
        )
        lokr_layer._mflux_lora_role = role

        if is_fused_linear:
            replacement_layer = FusedLoRALinear(
                base_linear=current_module.base_linear, loras=current_module.loras + [lokr_layer]
            )
        elif is_lora_linear or is_lokr_linear:
            replacement_layer = FusedLoRALinear(base_linear=base_linear, loras=[current_module, lokr_layer])
        else:
            replacement_layer = lokr_layer

        LoRALoader._replace_target_module(transformer, path_parts, replacement_layer)
        return True

    @staticmethod
    def _lokr_matrix_shape(lora_data: dict, direct_key: str, a_key: str, b_key: str) -> tuple[int, int] | None:
        if direct_key in lora_data:
            shape = tuple(int(dim) for dim in lora_data[direct_key].shape)
            if len(shape) == 2:
                return shape
            return None
        if a_key in lora_data and b_key in lora_data:
            a_shape = tuple(int(dim) for dim in lora_data[a_key].shape)
            b_shape = tuple(int(dim) for dim in lora_data[b_key].shape)
            if len(a_shape) == 2 and len(b_shape) == 2 and a_shape[1] == b_shape[0]:
                return (a_shape[0], b_shape[1])
        return None

    @staticmethod
    def _linear_weight_shape(linear: nn.Module) -> tuple[int, int] | None:
        if not hasattr(linear, "weight"):
            return None
        shape = tuple(int(dim) for dim in linear.weight.shape)
        if len(shape) != 2:
            return None
        out_features, in_features = shape
        if isinstance(linear, nn.QuantizedLinear):
            in_features *= 32 // linear.bits
        return (out_features, in_features)

    @staticmethod
    def _validate_lokr_shape(base_linear: nn.Module, target_path: str, lora_data: dict) -> None:
        linear_shape = LoRALoader._linear_weight_shape(base_linear)
        w1_shape = LoRALoader._lokr_matrix_shape(lora_data, "lokr_w1", "lokr_w1_a", "lokr_w1_b")
        w2_shape = LoRALoader._lokr_matrix_shape(lora_data, "lokr_w2", "lokr_w2_a", "lokr_w2_b")
        if linear_shape is None or w1_shape is None or w2_shape is None:
            return

        expected_out, expected_in = linear_shape
        full_lokr_out = w1_shape[0] * w2_shape[0]
        lokr_out = full_lokr_out
        output_index = lora_data.get("_lokr_output_index")
        output_splits = int(lora_data.get("_lokr_output_splits") or 1)
        if output_index is not None:
            if output_splits <= 0 or full_lokr_out % output_splits != 0:
                raise ValueError(
                    "SDMLX LoKr shape mismatch for "
                    f"{target_path}: full LoKr output {full_lokr_out} "
                    f"is not divisible by slice count {output_splits}."
                )
            output_index = int(output_index)
            if output_index < 0 or output_index >= output_splits:
                raise ValueError(
                    "SDMLX LoKr shape mismatch for "
                    f"{target_path}: output slice index {output_index} "
                    f"is outside 0..{output_splits - 1}."
                )
            lokr_out = full_lokr_out // output_splits
        lokr_in = w1_shape[1] * w2_shape[1]
        if expected_out != lokr_out or expected_in != lokr_in:
            raise ValueError(
                "SDMLX LoKr shape mismatch for "
                f"{target_path}: base linear expects in/out ({expected_in}, {expected_out}), "
                f"but LoKr expands to ({lokr_in}, {lokr_out}) "
                f"from w1={w1_shape}, w2={w2_shape}. "
                "This usually means a fused LoKr target was not split to the runtime projection."
            )
