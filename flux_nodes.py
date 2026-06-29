from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open


SUITE_ROOT = Path(__file__).resolve().parent
ROOT = SUITE_ROOT.parent
NATIVE_ROOT = SUITE_ROOT / "flux_native"
COMFY_ROOT = Path(os.environ.get("SDMLX_COMFY_ROOT", ROOT / "ComfyUI")).expanduser()

for path in (SUITE_ROOT, NATIVE_ROOT):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

import folder_paths  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.sd1_clip  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.latent_formats  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.text_encoders.t5  # noqa: E402
import comfy.text_encoders.flux  # noqa: E402
from sdmlx_flux_vae.config import Config  # noqa: E402
from sdmlx_flux_vae.vae import VAE  # noqa: E402
from transformers import T5TokenizerFast  # noqa: E402

import native_flux_core  # noqa: E402
from native_flux_core import FluxNativeTransformer  # noqa: E402
from lua_adapter import load_model as load_lua_model, upscale_latent as lua_upscale_latent  # noqa: E402


MODEL_TYPE = "sdmlx_model"
MODEL_INPUT_TYPE = MODEL_TYPE
SEACACHE_ADVANCED_TYPE = "sdmlx_flux_seacache_advanced"
VAE_CACHE: dict[tuple[str, str], VAE] = {}
FLUX_PREVIEWER_CACHE: dict[str, Any] = {}
FLUX_LUA_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}
FLUX_TEXT_CONDITIONING_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
POST_DECODE_CACHE_LIMIT_GB = 2.0
FLUX_KONTEXT_MAX_ENCODE_PIXELS = 2048 * 2048
DEFAULT_FLUX_GUIDANCE = 3.5
VAE_DTYPE = "float16"
FLUX_ACCEL_CACHE_VERSION = "v1"
FLUX_ACCEL_NONE = "None"
FLUX_ACCEL_PATCH_PACKAGE_DIR = "AccelerationPatches"
FLUX_ACCEL_PATCH_REPO_ID = "elef4nt/sdmlx-acceleration-patches"
KNOWN_FLUX_ACCEL_PATCHES = [
    "Hyper-Flux.1-Dev-4-step-Lora.sdmlxpatch",
]
FLUX_ACCEL_PATCH_LABELS = {
    "Hyper-Flux.1-Dev-4-step-Lora.sdmlxpatch": "Hyper FLUX Dev 4-step",
}
FLUX_ACCEL_PATCH_BY_LABEL = {label: name for name, label in FLUX_ACCEL_PATCH_LABELS.items()}
FLUX_ACCEL_PATCH_SOURCE_ALIASES = {
    "Hyper-Flux.1-Dev-4-step-Lora.sdmlxpatch": [
        "Hyper-Flux.1-Dev-4-step-Lora.safetensors",
        "schnell_v1.0.safetensors",
    ],
}
FLUX_ACCEL_BAKED_CACHE_BASES = {
    "Hyper-Flux.1-Dev-4-step-Lora.sdmlxpatch": {"flux1-dev.safetensors"},
}
FLUX_VERBOSE_LOGS = os.environ.get("SDMLX_FLUX_VERBOSE_LOGS", "").lower() in {"1", "true", "yes", "on"}
VAE_CACHE_LIMIT_POLICY = os.environ.get("SDMLX_FLUX_VAE_CACHE_LIMITS", "auto").lower()
FLUX_SEACACHE_DETAIL_THRESHOLD = 0.1
FLUX_SEACACHE_DETAIL_THRESHOLD_END = 0.4
FLUX_SEACACHE_DETAIL_START_AT = 6
FLUX_SEACACHE_GENERAL_THRESHOLD = 0.2
FLUX_SEACACHE_GENERAL_THRESHOLD_END = 0.6
FLUX_SEACACHE_GENERAL_START_AT = 3
FLUX_SEACACHE_DEFAULT_END_FROM_TAIL_STEPS = 3
FLUX_SEACACHE_DEFAULT_FINAL_GUARD = "last1"
FLUX_SCHNELL_SEACACHE_STEP3_ENABLED = True
KONTEXT_OFFSET_CACHE_FILL_STEP = 3
FLUX_KONTEXT_BASE_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]
FLUX_KONTEXT_SCALE_PROFILES = {
    "kontext": 1.0,
    "balanced": 0.75,
    "preview": 0.5,
}


def _round_to_multiple(value: float, multiple: int = 16) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _flux_kontext_dimensions_for_profile(profile: str) -> list[tuple[int, int]]:
    scale = FLUX_KONTEXT_SCALE_PROFILES.get(profile, 1.0)
    if scale == 1.0:
        return list(FLUX_KONTEXT_BASE_RESOLUTIONS)
    factor = scale ** 0.5
    dims: list[tuple[int, int]] = []
    for width, height in FLUX_KONTEXT_BASE_RESOLUTIONS:
        dims.append((_round_to_multiple(width * factor), _round_to_multiple(height * factor)))
    return dims


def _flux_dimension_options() -> list[str]:
    options = ["custom"]
    for width, height in FLUX_KONTEXT_BASE_RESOLUTIONS:
        options.append(f"{width} x {height}")
    return options


def _parse_flux_dimension_option(option: str, width: int, height: int) -> tuple[int, int]:
    if not option or option == "custom":
        return int(width), int(height)
    parts = option.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        return int(width), int(height)
    return int(parts[0]), int(parts[1])


def _closest_flux_dimensions(width: int, height: int, profile: str) -> tuple[int, int]:
    aspect_ratio = float(width) / max(float(height), 1.0)
    return min(
        _flux_kontext_dimensions_for_profile(profile),
        key=lambda item: abs(aspect_ratio - (item[0] / item[1])),
    )


class ModelConfig(Enum):
    FLUX1_DEV = ("black-forest-labs/FLUX.1-dev", "dev", 1000, 512)
    FLUX1_SCHNELL = ("black-forest-labs/FLUX.1-schnell", "schnell", 1000, 256)

    def __init__(
        self,
        model_name: str,
        alias: str,
        num_train_steps: int,
        max_sequence_length: int,
    ):
        self.model_name = model_name
        self.alias = alias
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length


class RuntimeConfig:
    def __init__(self, config: Config, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.sigmas = self._create_sigmas(config, model_config)

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def guidance(self) -> float:
        return self.config.guidance

    @property
    def num_inference_steps(self) -> int:
        return self.config.num_inference_steps

    @property
    def num_train_steps(self) -> int:
        return self.model_config.num_train_steps

    @property
    def init_time_step(self) -> int:
        if self.config.init_image_path is None:
            return 0
        strength = max(0.0, min(1.0, float(self.config.init_image_strength or 0.0)))
        return max(1, int(self.num_inference_steps * strength))

    @staticmethod
    def _create_sigmas(config: Config, model_config: ModelConfig) -> mx.array:
        sigmas = RuntimeConfig._create_sigmas_values(config.num_inference_steps)
        if model_config == ModelConfig.FLUX1_DEV:
            sigmas = RuntimeConfig._shift_sigmas(sigmas=sigmas, width=config.width, height=config.height)
        return sigmas

    @staticmethod
    def _create_sigmas_values(num_inference_steps: int) -> mx.array:
        sigmas = np.linspace(1.0, 1 / int(num_inference_steps), int(num_inference_steps))
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _shift_sigmas(sigmas: mx.array, width: int, height: int) -> mx.array:
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        mu = m * int(width) * int(height) / 256 + b
        mu = mx.array(mu)
        shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
        shifted_sigmas[-1] = 0
        return shifted_sigmas


@dataclass(frozen=True)
class _FluxVAESource:
    key: str
    conv: bool = False
    squeeze_1x1: bool = False


def _flux_log(message: str, *, debug: bool = False) -> None:
    if debug or FLUX_VERBOSE_LOGS:
        print(message)


def _flux_notice(message: str) -> None:
    print(message)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _flux_kontext_cache_fill_step(max_steps: int) -> int:
    return max(1, min(int(max_steps), int(KONTEXT_OFFSET_CACHE_FILL_STEP)))


def _parse_flux_profile_steps(raw: str | None, *, default_step: int, max_steps: int) -> set[int]:
    if not raw:
        return {max(1, min(int(max_steps), int(default_step)))}
    steps: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        try:
            step = int(item)
        except ValueError:
            continue
        if 1 <= step <= int(max_steps):
            steps.add(step)
    return steps or {max(1, min(int(max_steps), int(default_step)))}


def _flux_profile_output_dir() -> Path:
    configured = os.environ.get("SDMLX_FLUX_PROFILE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    try:
        return Path(folder_paths.get_output_directory()) / "sdmlx_profiles"
    except Exception:
        return ROOT / "tmp" / "sdmlx_profiles"


def _write_flux_profile_report(
    transformer: FluxNativeTransformer,
    *,
    model_name: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    profile_steps: set[int],
    reference_tokens: int,
    target_tokens: int,
) -> Path | None:
    report = transformer.profile_report()
    if not report:
        return None
    out_dir = _flux_profile_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(model_name))[:80]
    step_part = "-".join(str(step) for step in sorted(profile_steps))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / (
        f"flux1_kontext_profile_{stamp}_{safe_model}_{width}x{height}_"
        f"{steps}steps_seed{int(seed)}_profile_steps{step_part}.tsv"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# SDMLX FLUX.1 Kontext profile\n")
        handle.write(f"# model\t{model_name}\n")
        handle.write(f"# size\t{width}x{height}\n")
        handle.write(f"# steps\t{steps}\n")
        handle.write(f"# seed\t{seed}\n")
        handle.write(f"# profiled_steps\t{','.join(str(step) for step in sorted(profile_steps))}\n")
        handle.write(f"# reference_tokens\t{reference_tokens}\n")
        handle.write(f"# target_tokens\t{target_tokens}\n")
        handle.write("bucket\tcount\ttotal_s\tmean_s\n")
        for label, count, total, mean in report:
            handle.write(f"{label}\t{count}\t{total:.6f}\t{mean:.6f}\n")
    return path


def _seacache_final_guard_steps(value: Any) -> int:
    guard = str(value or FLUX_SEACACHE_DEFAULT_FINAL_GUARD).strip().lower()
    if guard == "last1":
        return 1
    if guard == "last2":
        return 2
    return FLUX_SEACACHE_DEFAULT_END_FROM_TAIL_STEPS


def _flux_model_size_from_latent(latent: dict[str, Any]) -> tuple[int, int]:
    try:
        samples = latent.get("samples") if isinstance(latent, dict) else None
        latent_h = int(samples.shape[-2])
        latent_w = int(samples.shape[-1])
    except Exception:
        return 0, 0
    latent_h += latent_h % 2
    latent_w += latent_w % 2
    return latent_w * 8, latent_h * 8


def _flux_seacache_profile_for_size(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "general"
    latent_w = max(1, int(width) // 16)
    latent_h = max(1, int(height) // 16)
    tokens = latent_w * latent_h
    aspect = max(width, height) / max(1, min(width, height))
    if tokens >= 4000 and aspect < 1.2:
        return "detail"
    return "general"


def _apply_vae_cache_limit(phase: str, model_family: str = "unknown") -> None:
    policy = VAE_CACHE_LIMIT_POLICY
    if policy in {"0", "false", "no", "off"}:
        return
    if policy in {"1", "true", "yes", "on"}:
        mx.set_cache_limit(int(POST_DECODE_CACHE_LIMIT_GB * 1024**3))
        return
    mx.set_cache_limit(int(POST_DECODE_CACHE_LIMIT_GB * 1024**3))


def _make_flux_terminal_progress_bar(total_steps: int):
    try:
        from tqdm.auto import tqdm

        kwargs = {
            "total": int(total_steps),
            "desc": "SDMLX FLUX",
            "unit": "step",
            "dynamic_ncols": True,
            "leave": True,
            "smoothing": 0.3,
        }
        try:
            return tqdm(**kwargs, colour="green")
        except TypeError:
            return tqdm(**kwargs)
    except Exception:
        return None


def _mlx_memory_line() -> str:
    active = mx.get_active_memory() / 1024**3
    cached = mx.get_cache_memory() / 1024**3
    peak = mx.get_peak_memory() / 1024**3
    return f"active={active:.2f}GB, cache={cached:.2f}GB, peak={peak:.2f}GB"


def _array_gib(value: mx.array) -> float:
    return float(value.nbytes) / 1024**3


def _transformer_memory_line(transformer: FluxNativeTransformer) -> str:
    w_gib = sum(_array_gib(value) for value in transformer.w.values())
    wt_gib = sum(_array_gib(value) for value in transformer.wt.values())
    q_gib = 0.0
    for q_weight, scales, quant_biases, _group_size, _bits in transformer.q.values():
        q_gib += _array_gib(q_weight) + _array_gib(scales)
        if quant_biases is not None:
            q_gib += _array_gib(quant_biases)
    fp8_mxfp8_gib = 0.0
    for q_weight, scales in transformer.fp8_mxfp8.values():
        fp8_mxfp8_gib += _array_gib(q_weight) + _array_gib(scales)
    gguf_affine_gib = 0.0
    for q_weight, scales, quant_biases, _shape in getattr(transformer, "gguf_q8", {}).values():
        gguf_affine_gib += _array_gib(q_weight) + _array_gib(scales) + _array_gib(quant_biases)
    fp8_gib = sum(_array_gib(transformer.w[key]) for key in transformer.fp8_weight_keys if key in transformer.w)
    lora_gib = 0.0
    for adapters in getattr(transformer, "lora_adapters", {}).values():
        for down_t, up_t, _start, _length in adapters:
            lora_gib += _array_gib(down_t) + _array_gib(up_t)
    return (
        f"w={w_gib:.2f}GB, wt={wt_gib:.2f}GB, q={q_gib:.2f}GB, packed_mxfp8={fp8_mxfp8_gib:.2f}GB, "
        f"gguf_affine={gguf_affine_gib:.2f}GB, "
        f"lora={lora_gib:.2f}GB, "
        f"native_fp8={fp8_gib:.2f}GB, keys(w/wt/q/fp8)="
        f"{len(transformer.w)}/{len(transformer.wt)}/{len(transformer.q)}/{len(transformer.fp8_weight_keys)}, "
        f"gguf_affine={len(getattr(transformer, 'gguf_q8', {}))}, "
        f"packed={len(transformer.fp8_mxfp8)}, lora_targets={len(getattr(transformer, 'lora_adapters', {}))}"
    )


@dataclass
class SDMLXFluxNativeModel:
    name: str
    path: Path
    transformer: FluxNativeTransformer
    precision: mx.Dtype
    model_config: ModelConfig
    pretransposed_linears: int
    casted_weights: int
    acceleration_patch: str | None = None
    acceleration_strength: float = 0.0
    model_family: str = "unknown"
    parent_transformer: FluxNativeTransformer | None = None


def _diffusion_model_names() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add_name(name: str) -> None:
        if name not in seen:
            seen.add(name)
            names.append(name)

    for folder_name in ("diffusion_models", "unet", "unet_gguf"):
        try:
            for name in folder_paths.get_filename_list(folder_name):
                add_name(name)
        except Exception:
            pass

    for name in _scan_model_folder_extensions("diffusion_models", {".gguf"}):
        add_name(name)

    for name in _scan_qwen_model_roots("diffusion_models"):
        add_name(name)

    preferred = "flux1-schnell-fp16.safetensors"
    if preferred in names:
        names.remove(preferred)
        return [preferred] + names
    return names or [preferred]


def _scan_model_folder_extensions(folder_name: str, extensions: set[str]) -> list[str]:
    names: list[str] = []
    folder_info = getattr(folder_paths, "folder_names_and_paths", {}).get(folder_name)
    if not folder_info:
        return names
    roots = folder_info[0]
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                names.append(str(path.relative_to(root_path)))
            except ValueError:
                names.append(path.name)
    return sorted(names, key=str.lower)


def _scan_qwen_model_roots(folder_name: str) -> list[str]:
    names: list[str] = []
    folder_info = getattr(folder_paths, "folder_names_and_paths", {}).get(folder_name)
    if not folder_info:
        return names
    try:
        from .qwen_nodes import is_qwen_model_root
    except Exception:
        return names

    roots = folder_info[0]
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if not path.is_dir() or not is_qwen_model_root(path):
                continue
            try:
                names.append(str(path.relative_to(root_path)))
            except ValueError:
                names.append(path.name)
    return sorted(set(names), key=str.lower)


def _model_config_from_family(model_family: str) -> ModelConfig:
    if str(model_family).strip().lower() == "dev":
        return ModelConfig.FLUX1_DEV
    return ModelConfig.FLUX1_SCHNELL


def _model_family_from_name(model_name: str) -> str:
    lower = model_name.lower()
    if "dev" in lower:
        return "dev"
    if "schnell" in lower or "sch" in lower:
        return "schnell"
    return "unknown"


def _model_family_from_structure(path: Path) -> str:
    try:
        normalized_keys: list[str] = []
        if path.suffix.lower() == ".gguf":
            import gguf

            reader = gguf.GGUFReader(str(path))
            normalized_keys = [
                normalized
                for tensor in reader.tensors
                if (normalized := native_flux_core.normalize_flux_weight_key(tensor.name)) is not None
            ]
        else:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                normalized_keys = [
                    normalized
                    for key in f.keys()
                    if (normalized := native_flux_core.normalize_flux_weight_key(key)) is not None
                ]
    except Exception as exc:
        _flux_log(f"SDMLX FLUX Loader: model-family structure check failed for {path.name}: {exc}")
        return "unknown"

    has_guidance = any(key.startswith("guidance_in.") for key in normalized_keys)
    has_txt_in = "txt_in.weight" in normalized_keys
    has_final = any(key.startswith("final_layer.") for key in normalized_keys)
    if has_guidance:
        return "dev"
    if has_txt_in and has_final:
        return "schnell"
    return "unknown"


def _model_family_from_path(path: Path, model_name: str) -> str:
    name_family = _model_family_from_name(model_name)
    structure_family = _model_family_from_structure(path)
    if structure_family == "unknown":
        return name_family
    if name_family not in {"unknown", structure_family}:
        _flux_log(
            "SDMLX FLUX Loader: "
            f"model-family name/structure mismatch for {model_name}: "
            f"name={name_family}, structure={structure_family}; using structure."
        )
    return structure_family


def _vae_names() -> list[str]:
    names = list(folder_paths.get_filename_list("vae"))
    preferred = "ae.safetensors"
    if preferred in names:
        names.remove(preferred)
        return [preferred] + names
    return names or [preferred]


def _lora_names() -> list[str]:
    names = list(folder_paths.get_filename_list("loras"))
    return names or ["select_lora.safetensors"]


def _flux_acceleration_patch_options() -> list[str]:
    names = set(KNOWN_FLUX_ACCEL_PATCHES)
    for package_dir in _sdmlx_acceleration_patch_package_dirs(create=False):
        for path in package_dir.iterdir():
            if path.name in KNOWN_FLUX_ACCEL_PATCHES and path.is_dir() and path.name.endswith(".sdmlxpatch") and _is_flux_acceleration_patch_package(path):
                names.add(path.name)
    return [FLUX_ACCEL_NONE] + sorted((_flux_acceleration_patch_label(name) for name in names), key=str.lower)


def _normalized_flux_acceleration_patch_name(selection: str) -> str | None:
    if not selection or selection == FLUX_ACCEL_NONE:
        return None
    name = FLUX_ACCEL_PATCH_BY_LABEL.get(str(selection), str(selection))
    name = Path(name).name
    if not name.endswith(".sdmlxpatch"):
        name += ".sdmlxpatch"
    return name


def _flux_acceleration_patch_label(package_name: str) -> str:
    return FLUX_ACCEL_PATCH_LABELS.get(package_name, package_name.removesuffix(".sdmlxpatch"))


def _flux_acceleration_patch_uses_baked_cache(source: Path, selection: str) -> bool:
    package_name = _normalized_flux_acceleration_patch_name(selection)
    if package_name is None:
        return False
    allowed_bases = FLUX_ACCEL_BAKED_CACHE_BASES.get(package_name, set())
    return source.name.lower() in {name.lower() for name in allowed_bases}


def _name_stem(value: str | Path | None) -> str:
    if value is None:
        return ""
    return Path(str(value)).stem.strip().lower()


def _flux_acceleration_patch_source_stems(selection: str) -> set[str]:
    package_name = _normalized_flux_acceleration_patch_name(selection)
    if package_name is None:
        return set()
    stems = {_name_stem(package_name)}
    for alias in FLUX_ACCEL_PATCH_SOURCE_ALIASES.get(package_name, []):
        stems.add(_name_stem(alias))

    package_dir = _find_flux_acceleration_patch_package_dir(package_name)
    if package_dir is None:
        return {stem for stem in stems if stem}
    for metadata_name in ("manifest.json", "source_metadata.json"):
        metadata_path = package_dir / metadata_name
        if not metadata_path.exists():
            continue
        try:
            metadata = _read_json_file(metadata_path)
        except Exception:
            continue
        for key in ("source_file", "source_lora_name", "source_patch_name"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                stems.add(_name_stem(value))
    return {stem for stem in stems if stem}


def _runtime_lora_source_stems(transformer: FluxNativeTransformer) -> set[str]:
    return {
        _name_stem(source)
        for source in getattr(transformer, "lora_sources", [])
        if _name_stem(source)
    }


def _assert_no_duplicate_flux_acceleration_patch_lora(model: SDMLXFluxNativeModel, selection: str) -> None:
    patch_stems = _flux_acceleration_patch_source_stems(selection)
    runtime_stems = _runtime_lora_source_stems(model.transformer)
    duplicates = sorted(patch_stems & runtime_stems)
    if not duplicates:
        return
    label = _flux_acceleration_patch_label(_normalized_flux_acceleration_patch_name(selection) or str(selection))
    duplicate_label = ", ".join(duplicates)
    raise RuntimeError(
        "SDMLX FLUX: acceleration-patch is already baked from the selected LoRA "
        f"({label}; duplicate source: {duplicate_label}). "
        "Remove the duplicate LoRA Loader or set acceleration_patch to None."
    )


def _sdmlx_acceleration_patch_package_dir(*, create: bool = True) -> Path:
    path = _sdmlx_models_dir() / FLUX_ACCEL_PATCH_PACKAGE_DIR
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _sdmlx_acceleration_patch_package_dirs(*, create: bool = False) -> list[Path]:
    dirs: list[Path] = []
    for root in _sdmlx_model_roots(existing_only=not create):
        dirs.append(root / FLUX_ACCEL_PATCH_PACKAGE_DIR)
    if not dirs:
        dirs.append(_sdmlx_models_dir() / FLUX_ACCEL_PATCH_PACKAGE_DIR)
    if create:
        dirs[0].mkdir(parents=True, exist_ok=True)
    return [path for path in dirs if path.exists() or create]


def _find_flux_acceleration_patch_package_dir(package_name: str) -> Path | None:
    for package_root in _sdmlx_acceleration_patch_package_dirs(create=False):
        package_dir = package_root / package_name
        if package_dir.is_dir() and _is_flux_acceleration_patch_package(package_dir):
            factors_path = package_dir / "patch.safetensors"
            if factors_path.exists() and factors_path.stat().st_size > 0:
                return package_dir
    return None


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise RuntimeError(f"SDMLX FLUX: expected JSON object in {path}")
    return value


def _is_flux_acceleration_patch_package(package_dir: Path) -> bool:
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = _read_json_file(manifest_path)
    except Exception:
        return False
    if str(manifest.get("format", "")).strip() != "sdmlx-acceleration-patch-v1":
        return False
    family = str(manifest.get("base_model_family", "")).lower()
    return "flux" in family


def _ensure_flux_acceleration_patch_package(selection: str, *, debug: bool = False) -> tuple[Path, str]:
    package_name = _normalized_flux_acceleration_patch_name(selection)
    if package_name is None:
        raise RuntimeError("SDMLX FLUX: acceleration patch is None.")
    existing_package_dir = _find_flux_acceleration_patch_package_dir(package_name)
    if existing_package_dir is not None:
        return existing_package_dir / "patch.safetensors", package_name

    package_dir = _sdmlx_acceleration_patch_package_dirs(create=True)[0] / package_name
    manifest_path = package_dir / "manifest.json"
    factors_path = package_dir / "patch.safetensors"

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "SDMLX FLUX: huggingface_hub is not available; "
            f"Acceleration Patch {package_name} cannot be downloaded."
        ) from exc

    package_dir.mkdir(parents=True, exist_ok=True)
    _flux_log(f"SDMLX FLUX: Downloading Acceleration Patch {package_name} from Hugging Face...", debug=debug)
    for filename in ("manifest.json", "patch.safetensors", "source_metadata.json"):
        try:
            hf_hub_download(
                repo_id=FLUX_ACCEL_PATCH_REPO_ID,
                repo_type="model",
                filename=f"{package_name}/{filename}",
                local_dir=str(package_dir.parent),
            )
        except Exception:
            if filename != "source_metadata.json":
                raise
    if not manifest_path.exists() or not factors_path.exists() or not _is_flux_acceleration_patch_package(package_dir):
        raise RuntimeError(f"SDMLX FLUX: Acceleration Patch {package_name} could not be downloaded completely.")
    return factors_path, package_name


def _full_diffusion_model_path(model_name: str) -> Path:
    for folder_name in ("diffusion_models", "unet", "unet_gguf"):
        try:
            path = folder_paths.get_full_path(folder_name, model_name)
        except Exception:
            path = None
        if path is not None:
            return Path(path)

    folder_info = getattr(folder_paths, "folder_names_and_paths", {}).get("diffusion_models")
    if folder_info:
        for root in folder_info[0]:
            candidate = Path(root) / model_name
            if candidate.exists():
                return candidate
    raise RuntimeError(f"SDMLX FLUX: model not found in diffusion_models/unet_gguf: {model_name}")


def _full_lora_path(lora_name: str) -> Path:
    if hasattr(folder_paths, "get_full_path_or_raise"):
        return Path(folder_paths.get_full_path_or_raise("loras", lora_name))
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise RuntimeError(f"SDMLX FLUX: LoRA not found in loras: {lora_name}")
    return Path(path)


def _sdmlx_model_roots(*, existing_only: bool = True) -> list[Path]:
    roots: list[Path] = []
    try:
        folder_map = getattr(folder_paths, "folder_names_and_paths", {})
        for key in ("sdmlx", "SDMLX"):
            if key in folder_map:
                roots.extend(Path(p).expanduser() for p in folder_paths.get_folder_paths(key))
    except Exception:
        pass

    models_dir = Path(getattr(folder_paths, "models_dir", COMFY_ROOT / "models"))
    roots.append(models_dir / "SDMLX")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if root.exists() or not existing_only:
            unique.append(root)
    return unique


def _sdmlx_models_dir() -> Path:
    roots = _sdmlx_model_roots(existing_only=True)
    if roots:
        roots[0].mkdir(parents=True, exist_ok=True)
        return roots[0]
    models_dir = Path(getattr(folder_paths, "models_dir", COMFY_ROOT / "models"))
    path = models_dir / "SDMLX"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _comfy_text_encoders_dir() -> Path:
    return Path(comfy.text_encoders.t5.__file__).resolve().parent


def _comfy_t5xxl_config_path() -> Path:
    return _comfy_text_encoders_dir() / "t5_config_xxl.json"


def _comfy_t5xxl_tokenizer_dir() -> Path:
    return _comfy_text_encoders_dir() / "t5_tokenizer"


def _sdmlx_acceleration_cache_dir() -> Path:
    path = _sdmlx_models_dir() / "cache" / "acceleration-patches"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sdmlx_acceleration_cache_dirs(*, create_primary: bool = False) -> list[Path]:
    dirs = [root / "cache" / "acceleration-patches" for root in _sdmlx_model_roots(existing_only=True)]
    if not dirs:
        dirs.append(_sdmlx_acceleration_cache_dir())
    if create_primary:
        dirs[0].mkdir(parents=True, exist_ok=True)
    return [path for path in dirs if path.exists() or create_primary]


def _sdmlx_acceleration_cache_dir_for_patch(patch_path: Path) -> Path:
    try:
        if patch_path.parent.name.endswith(".sdmlxpatch") and patch_path.parent.parent.name == FLUX_ACCEL_PATCH_PACKAGE_DIR:
            root = patch_path.parent.parent.parent
            path = root / "cache" / "acceleration-patches"
            path.mkdir(parents=True, exist_ok=True)
            return path
    except Exception:
        pass
    return _sdmlx_acceleration_cache_dir()


def _fp8_cache_path(source: Path) -> Path:
    stat = source.stat()
    cache_key = f"{source.name}:{stat.st_size}:{int(stat.st_mtime)}"
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    return NATIVE_ROOT / "model_cache" / f"{source.stem}-sdmlx-fp16-{digest}.safetensors"


def _safe_cache_stem(value: str) -> str:
    keep = []
    for char in Path(value).stem:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("._") or "cache"


def _flux_acceleration_cache_path(source: Path, patch_path: Path, strength: float, package_name: str | None = None) -> Path:
    source_stat = source.stat()
    patch_stat = patch_path.stat()
    strength_key = f"{float(strength):.6g}"
    cache_key = "|".join(
        (
            FLUX_ACCEL_CACHE_VERSION,
            str(source.resolve()),
            str(source_stat.st_size),
            str(source_stat.st_mtime_ns),
            str(patch_path.resolve()),
            str(patch_stat.st_size),
            str(patch_stat.st_mtime_ns),
            strength_key,
            "fp16",
        )
    )
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    source_stem = _safe_cache_stem(source.name)
    patch_stem = _safe_cache_stem(package_name or patch_path.name).removesuffix(".sdmlxpatch")
    return _sdmlx_acceleration_cache_dir_for_patch(patch_path) / f"{source_stem}-accel-{patch_stem}-s{strength_key}-{digest}.safetensors"


def _find_existing_flux_acceleration_cache(source: Path, package_name: str, strength: float) -> Path | None:
    source_stem = _safe_cache_stem(source.name)
    patch_stem = _safe_cache_stem(package_name).removesuffix(".sdmlxpatch")
    strength_key = f"{float(strength):.6g}"
    pattern = f"{source_stem}-accel-{patch_stem}-s{strength_key}-*.safetensors"
    candidates = []
    for cache_dir in _sdmlx_acceleration_cache_dirs(create_primary=False):
        candidates.extend(
            path
            for path in cache_dir.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _ensure_flux_acceleration_cache(source: Path, lora_name: str, strength: float, *, debug: bool = False) -> Path:
    if not lora_name or lora_name == FLUX_ACCEL_NONE or float(strength) == 0.0:
        return source
    patch_path, package_name = _ensure_flux_acceleration_patch_package(lora_name, debug=debug)
    cache_path = _flux_acceleration_cache_path(source, patch_path, strength, package_name)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        _flux_log(f"SDMLX FLUX Acceleration Cache: using {cache_path.name}", debug=debug)
        return cache_path
    compatible_cache = _find_existing_flux_acceleration_cache(source, package_name, strength)
    if compatible_cache is not None:
        _flux_log(f"SDMLX FLUX Acceleration Cache: using compatible existing cache {compatible_cache.name}", debug=debug)
        return compatible_cache

    _flux_log(
        "SDMLX FLUX Acceleration Cache: building "
        f"model={source.name}, patch={package_name}, strength={float(strength):g}",
        debug=debug,
    )
    script = NATIVE_ROOT / "build_flux_lora_acceleration_cache.py"
    metadata = {
        "cache_version": FLUX_ACCEL_CACHE_VERSION,
        "source_model_name": source.name,
        "source_patch_name": package_name,
    }
    cmd = [
        sys.executable,
        str(script),
        "--base-model",
        str(source),
        "--lora",
        str(patch_path),
        "--output",
        str(cache_path),
        "--strength",
        str(float(strength)),
        "--metadata-json",
        json.dumps(metadata),
    ]
    if not (debug or FLUX_VERBOSE_LOGS):
        cmd.append("--quiet")
    subprocess.run(cmd, check=True)
    return cache_path


def _ensure_fp8_transformer_cache(source: Path) -> Path:
    if not native_flux_core._has_fp8_weights(source):
        return source

    cache_path = _fp8_cache_path(source)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        _flux_log(f"SDMLX FLUX Cache: using cached fp16 transformer for FP8 model: {cache_path.name}")
        return cache_path

    _flux_log(
        "SDMLX FLUX Cache: FP8 model detected; "
        f"building fp16 transformer cache for {source.name}"
    )
    script = NATIVE_ROOT / "convert_flux_fp8_to_fp16_cache.py"
    cmd = [sys.executable, str(script), str(source), str(cache_path)]
    if not FLUX_VERBOSE_LOGS:
        cmd.append("--quiet")
    subprocess.run(cmd, check=True)
    return cache_path


def _load_prepared_flux_transformer(
    path: Path,
    *,
    precision: mx.Dtype,
    fp8_mode: str,
    fp8_dequant: bool = False,
    drop_raw: str = "off",
) -> tuple[FluxNativeTransformer, int, int, float]:
    t0 = time.perf_counter()
    transformer = FluxNativeTransformer.load(
        path=path,
        precision=precision,
        fp8_mode=fp8_mode,
    )
    casted = transformer.cast_weights(precision)
    pretransposed = transformer.prepare_transposed_linears(scope="all", fp8_dequant=fp8_dequant, drop_raw=drop_raw)
    mx.eval(transformer.w, transformer.wt, transformer.fp8_mxfp8, getattr(transformer, "gguf_q8", {}))
    return transformer, casted, pretransposed, time.perf_counter() - t0


def _transformer_has_model_weights(transformer: FluxNativeTransformer) -> bool:
    return bool(
        transformer.w
        or transformer.wt
        or transformer.q
        or transformer.fp8_mxfp8
        or getattr(transformer, "gguf_q8", {})
        or transformer.fp8_weight_keys
    )


def _ensure_flux_model_weights_loaded(model: SDMLXFluxNativeModel) -> None:
    if _transformer_has_model_weights(model.transformer):
        return
    load_path = Path(model.path)
    is_gguf = load_path.suffix.lower() == ".gguf"
    use_native_fp8 = False if is_gguf else native_flux_core._has_fp8_weights(load_path)
    transformer, casted, pretransposed, _load_s = _load_prepared_flux_transformer(
        load_path,
        precision=model.precision,
        fp8_mode="native" if use_native_fp8 else "dequant",
        drop_raw="all" if is_gguf else "off",
    )
    model.transformer = transformer
    model.casted_weights = casted
    model.pretransposed_linears = pretransposed
    model.parent_transformer = None


def _full_vae_path(vae_name: str) -> Path:
    if hasattr(folder_paths, "get_full_path_or_raise"):
        return Path(folder_paths.get_full_path_or_raise("vae", vae_name))
    path = folder_paths.get_full_path("vae", vae_name)
    if path is None:
        raise RuntimeError(f"SDMLX FLUX: VAE not found in vae folder: {vae_name}")
    return Path(path)


def _dtype_from_name(name: str) -> mx.Dtype:
    if name == "float16":
        return mx.float16
    if name == "bfloat16":
        return mx.bfloat16
    if name == "float32":
        return mx.float32
    raise RuntimeError(f"SDMLX FLUX: unsupported dtype: {name}")


def _configure_native_core() -> None:
    native_flux_core.USE_FAST_SDPA = True
    native_flux_core.SDPA_MEMORY_EFFICIENT_THRESHOLD = None
    native_flux_core.FUSED_ADALN = False
    native_flux_core.CAST_MODULATED_NORM = True
    native_flux_core.CAST_ATTENTION_OUT = False
    native_flux_core.CAST_BLOCK_OUTPUT = True
    native_flux_core.NAN_TO_NUM_FP16 = True
    native_flux_core.ROPE_FLOAT32 = False
    native_flux_core.QK_NORM_MODE = "fast"
    native_flux_core.SINGLE_LINEAR1_FLAT = False
    native_flux_core.SINGLE_LINEAR2_FLAT = False
    native_flux_core.SINGLE_LINEAR2_CONTIG = False
    native_flux_core.SINGLE_LINEAR2_CAST = True
    split_linear2 = os.environ.get("SDMLX_FLUX_SINGLE_LINEAR2_SPLIT_PROJ", "1").strip().lower()
    native_flux_core.SINGLE_LINEAR2_SPLIT_PROJ = split_linear2 not in {"0", "false", "no", "off"}


def _reset_transformer_runtime_state(transformer: FluxNativeTransformer) -> None:
    transformer.sdmlx_eval_clear_each_block = (
        os.environ.get("SDMLX_FLUX_EVAL_CLEAR_EACH_BLOCK", "").lower() in {"1", "true", "yes", "on"}
    )
    transformer.profile_enabled = False
    transformer.profile = {}
    transformer.profile_steps = set()
    transformer.profile_by_step = False
    transformer.detail_profile = {}
    transformer.shadow_enabled = False
    transformer.shadow_previous = {}
    transformer.shadow_metrics = []
    transformer.forecast_step_index = 0
    transformer.forecast_single_history = {}
    transformer.forecast_single_attention_history = {}
    transformer.forecast_single_linear2_history = {}
    transformer.forecast_single_linear2_norm_history = {}
    transformer.forecast_single_linear2_gate_history = {}
    transformer.forecast_single_linear2_scout_metrics = []
    transformer.forecast_single_weather_latest = {}
    transformer.forecast_single_weather_metrics = []
    transformer.forecast_single_linear2_late_split = "all"
    transformer.forecast_single_linear2_partial_hits = 0
    transformer.forecast_single_hits = []
    transformer.forecast_single_current_step = 0
    transformer.forecast_single_locked_blocks = set()
    transformer.forecast_single_next_locked_blocks = set()
    transformer.forecast_single_storm_ref = None
    transformer.forecast_single_storm_metrics = []
    transformer.forecast_double_img_mlp_history = {}
    transformer.forecast_double_img_mlp_hits = []
    transformer.forecast_double_txt_history = {}
    transformer.forecast_double_txt_hits = []
    transformer.token_scout_enabled = False
    transformer.token_scout_txt_len = 0
    transformer.token_scout_previous = {}
    transformer.token_scout_metrics = []
    transformer.attention_scout_enabled = False
    transformer.attention_scout_metrics = []
    transformer.teacache_mode = "off"
    transformer.teacache_threshold = 0.08
    transformer.teacache_threshold_end = None
    transformer.teacache_warmup_steps = 5
    transformer.teacache_final_steps = 3
    transformer.teacache_total_steps = 0
    transformer.reset_teacache_gate()
    if hasattr(transformer, "set_kontext_kv_cache"):
        transformer.set_kontext_kv_cache(False, 0)


def _apply_flux_mlx_mode(
    transformer: FluxNativeTransformer,
    steps: int,
    model_family: str,
    *,
    use_schnell_seacache_profile: bool = False,
) -> str:
    _reset_transformer_runtime_state(transformer)
    transformer.forecast_single_mode = "off"
    transformer.forecast_single_scope = "residual"
    transformer.forecast_single_steps = set()
    transformer.forecast_single_blocks = set()
    transformer.forecast_single_plan = {}
    transformer.forecast_single_attention_plan = {}
    transformer.forecast_single_linear2_plan = {}
    transformer.forecast_single_linear2_late_plan = {}
    transformer.forecast_single_linear2_shape = "plain"
    transformer.forecast_single_linear2_shape_blocks = None
    transformer.forecast_single_linear2_shape_clamp = 0.0
    transformer.forecast_single_linear2_block_gain = {}
    transformer.forecast_single_adaptive = False
    transformer.forecast_single_adaptive_blocks = set()
    transformer.forecast_single_adaptive_step2_blocks = set()
    transformer.forecast_single_adaptive_step_sensitivity = {}
    transformer.forecast_single_adaptive_step_lowest_count = {}
    transformer.forecast_single_adaptive_low_pool_split = False
    transformer.forecast_single_adaptive_low_pool_source_step = 0
    transformer.forecast_single_adaptive_low_pool_steps = {}
    transformer.forecast_single_adaptive_lowest_cache = {}
    transformer.forecast_single_gain = 1.0
    transformer.forecast_single_step_gain = {}
    transformer.forecast_single_delta_clamp = 0.0
    transformer.forecast_single_step_delta_clamp = {}
    transformer.forecast_single_total_steps = steps
    transformer.forecast_single_last_weather_step = 0
    transformer.forecast_single_weather_debug = False
    transformer.forecast_double_img_mlp_mode = "off"
    transformer.forecast_double_img_mlp_steps = set()
    transformer.forecast_double_img_mlp_blocks = set()
    transformer.forecast_double_img_mlp_plan = {}
    transformer.forecast_double_img_mlp_gain = 1.0
    transformer.forecast_double_img_mlp_step_gain = {}
    transformer.forecast_double_txt_mode = "off"
    transformer.forecast_double_txt_steps = set()
    transformer.forecast_double_txt_blocks = set()
    transformer.forecast_double_txt_plan = {}
    transformer.forecast_double_txt_gain = 1.0
    transformer.forecast_double_txt_step_gain = {}

    family = str(model_family or "unknown").strip().lower()
    if family != "schnell":
        return family if family in {"dev", "unknown"} else "unknown"

    if use_schnell_seacache_profile:
        return "schnell"

    return "schnell"


def _unwrap_flux1_conditioning(conditioning: Any) -> Any:
    if isinstance(conditioning, dict) and conditioning.get("type") == "flux1":
        return conditioning.get("conditioning")
    return conditioning


def _conditioning_to_mlx(conditioning: Any, precision: mx.Dtype) -> tuple[mx.array, mx.array, float | None]:
    wrapper_guidance = None
    if isinstance(conditioning, dict) and conditioning.get("type") == "flux1":
        try:
            wrapper_guidance = float(conditioning["guidance"]) if conditioning.get("guidance") is not None else None
        except Exception:
            wrapper_guidance = None
        conditioning = conditioning.get("conditioning")
    if isinstance(conditioning, dict) and "cond" in conditioning:
        cond = conditioning.get("cond")
        pooled = conditioning.get("pooled_output", conditioning.get("pooled"))
        if pooled is None:
            raise RuntimeError("SDMLX FLUX: positive SDMLX conditioning has no pooled output.")
        guidance = conditioning.get("guidance")
        try:
            guidance = float(guidance) if guidance is not None else None
        except Exception:
            guidance = None
        if guidance is None:
            guidance = wrapper_guidance
        cond_np = _tensor_to_numpy(cond)
        pooled_np = _tensor_to_numpy(pooled)
        if cond_np.ndim != 3 or cond_np.shape[-1] != 4096:
            raise RuntimeError(f"SDMLX FLUX: expected T5 embeddings with shape [B,T,4096], got {cond_np.shape}.")
        if pooled_np.ndim != 2 or pooled_np.shape[-1] != 768:
            raise RuntimeError(f"SDMLX FLUX: expected pooled CLIP output with shape [B,768], got {pooled_np.shape}.")
        if cond_np.shape[0] != 1 or pooled_np.shape[0] != 1:
            raise RuntimeError("SDMLX FLUX prototype currently supports batch_size=1.")
        prompt_embeds = mx.array(cond_np).astype(precision)
        pooled_prompt_embeds = mx.array(pooled_np).astype(precision)
        mx.eval(prompt_embeds, pooled_prompt_embeds)
        return prompt_embeds, pooled_prompt_embeds, guidance

    if not conditioning:
        raise RuntimeError("SDMLX FLUX: positive conditioning is empty.")
    if len(conditioning) != 1:
        _flux_log(f"SDMLX FLUX: conditioning has {len(conditioning)} entries; using entry 0.")
    cond = conditioning[0][0]
    meta = conditioning[0][1] if len(conditioning[0]) > 1 else {}
    pooled = meta.get("pooled_output")
    if pooled is None:
        raise RuntimeError("SDMLX FLUX: positive conditioning has no pooled_output.")

    cond_np = _tensor_to_numpy(cond)
    pooled_np = _tensor_to_numpy(pooled)
    if cond_np.ndim != 3 or cond_np.shape[-1] != 4096:
        raise RuntimeError(f"SDMLX FLUX: expected T5 embeddings with shape [B,T,4096], got {cond_np.shape}.")
    if pooled_np.ndim != 2 or pooled_np.shape[-1] != 768:
        raise RuntimeError(f"SDMLX FLUX: expected pooled CLIP output with shape [B,768], got {pooled_np.shape}.")
    if cond_np.shape[0] != 1 or pooled_np.shape[0] != 1:
        raise RuntimeError("SDMLX FLUX prototype currently supports batch_size=1.")

    guidance = None
    if "guidance" in meta:
        try:
            guidance = float(meta["guidance"])
        except Exception:
            guidance = None
    if guidance is None:
        guidance = wrapper_guidance
    prompt_embeds = mx.array(cond_np).astype(precision)
    pooled_prompt_embeds = mx.array(pooled_np).astype(precision)
    mx.eval(prompt_embeds, pooled_prompt_embeds)
    return prompt_embeds, pooled_prompt_embeds, guidance


def _detect_first_pad_from_embeds(prompt_embeds: mx.array) -> int | None:
    total_tokens = int(prompt_embeds.shape[1])
    if total_tokens < 16:
        return None
    embeds = np.array(prompt_embeds[0].astype(mx.float32), dtype=np.float32)
    diffs = np.mean(np.abs(np.diff(embeds, axis=0)), axis=1)
    for index in range(8, total_tokens - 10):
        previous_jump = diffs[index - 1] > 0.08
        following_median = float(np.median(diffs[index + 1 : index + 9]))
        if previous_jump and following_median < 0.04:
            return index
    return None


def _pad_pool_token_context(prompt_embeds: mx.array, *, budget: int = 128) -> tuple[mx.array, str]:
    total_tokens = int(prompt_embeds.shape[1])
    budget = max(1, min(int(budget), total_tokens))
    label = f"pad_pool{budget}"
    first_pad = _detect_first_pad_from_embeds(prompt_embeds)
    if first_pad is None:
        return prompt_embeds, f"{label} skipped: first_pad=unknown, tokens={total_tokens}->{total_tokens}"
    first_pad = max(0, min(first_pad, total_tokens))
    if budget >= total_tokens or first_pad >= total_tokens:
        return prompt_embeds, f"{label} skipped: first_pad={first_pad}, tokens={total_tokens}->{total_tokens}"
    if budget <= first_pad:
        out = prompt_embeds[:, :budget, :]
        mx.eval(out)
        return out, f"{label} budget_before_pad: first_pad={first_pad}, tokens={total_tokens}->{int(out.shape[1])}"

    dtype = prompt_embeds.dtype
    pad_budget = budget - first_pad
    chunks = np.array_split(np.arange(first_pad, total_tokens), pad_budget)
    pooled_chunks = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        start = int(chunk[0])
        stop = int(chunk[-1]) + 1
        pooled_chunks.append(mx.mean(prompt_embeds[:, start:stop, :].astype(mx.float32), axis=1, keepdims=True))
    out = mx.concatenate([prompt_embeds[:, :first_pad, :].astype(mx.float32)] + pooled_chunks, axis=1).astype(dtype)
    mx.eval(out)
    return out, f"{label}: first_pad={first_pad}, pad_pools={len(pooled_chunks)}, tokens={total_tokens}->{int(out.shape[1])}"


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _pack_flux_latents_np(latents: np.ndarray) -> np.ndarray:
    if latents.ndim != 4:
        raise RuntimeError(f"SDMLX FLUX: expected 4D latent tensor, got shape {latents.shape}.")
    batch, channels, latent_h, latent_w = latents.shape
    if batch != 1:
        raise RuntimeError("SDMLX FLUX prototype currently supports batch_size=1.")
    if channels != 16:
        raise RuntimeError(
            "SDMLX FLUX needs a 16-channel FLUX/SD3 latent. "
            f"Got {channels} channels; use Comfy's EmptySD3LatentImage/FLUX latent node."
        )
    if latent_h % 2 != 0 or latent_w % 2 != 0:
        raise RuntimeError(f"SDMLX FLUX: latent size must be even, got {latent_w}x{latent_h}.")
    packed = latents.reshape(batch, 16, latent_h // 2, 2, latent_w // 2, 2)
    packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
    return packed.reshape(batch, (latent_h // 2) * (latent_w // 2), 64)


def _pad_flux_latents_np_to_patch_size(latents: np.ndarray) -> np.ndarray:
    if latents.ndim != 4:
        raise RuntimeError(f"SDMLX FLUX: expected 4D latent tensor, got shape {latents.shape}.")
    _batch, _channels, latent_h, latent_w = latents.shape
    pad_h = latent_h % 2
    pad_w = latent_w % 2
    if pad_h == 0 and pad_w == 0:
        return latents
    return np.pad(latents, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant")


def _prepare_noise_from_latent(
    latent: dict[str, Any],
    seed: int,
    precision: mx.Dtype,
) -> tuple[mx.array, int, int, tuple[int, int]]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise RuntimeError("SDMLX FLUX: latent_image must be a Comfy LATENT with samples.")
    samples = latent["samples"]
    if tuple(samples.shape)[1] != 16:
        raise RuntimeError(
            "SDMLX FLUX needs a 16-channel FLUX/SD3 latent. "
            f"Got shape {tuple(samples.shape)}."
        )
    output_latent_shape = (int(samples.shape[-2]), int(samples.shape[-1]))
    batch_inds = latent.get("batch_index") if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(samples, int(seed), batch_inds)
    noise_np = noise.detach().cpu().float().numpy().astype(np.float32, copy=False)
    noise_np = _pad_flux_latents_np_to_patch_size(noise_np)
    height = int(noise_np.shape[-2]) * 8
    width = int(noise_np.shape[-1]) * 8
    packed = _pack_flux_latents_np(noise_np)
    latents = mx.array(packed).astype(precision)
    mx.eval(latents)
    return latents, height, width, output_latent_shape


def _prepare_kontext_image_from_samples(samples: Any, precision: mx.Dtype) -> tuple[mx.array, int, int]:
    if not hasattr(samples, "detach"):
        raise RuntimeError("SDMLX FLUX Kontext: reference latent must be a torch tensor.")
    shape = tuple(samples.shape)
    if len(shape) != 4 or shape[1] != 16:
        raise RuntimeError(
            "SDMLX FLUX Kontext needs a 16-channel encoded FLUX image latent. "
            f"Got shape {shape}; encode the source image with Comfy's FLUX VAE first."
        )
    if shape[0] != 1:
        raise RuntimeError("SDMLX FLUX Kontext prototype currently supports batch_size=1.")
    model_space = comfy.latent_formats.Flux().process_in(samples.detach().cpu().float())
    model_space_np = model_space.numpy().astype(np.float32, copy=False)
    model_space_np = _pad_flux_latents_np_to_patch_size(model_space_np)
    height = int(model_space_np.shape[-2]) * 8
    width = int(model_space_np.shape[-1]) * 8
    packed = _pack_flux_latents_np(model_space_np)
    latents = mx.array(packed).astype(precision)
    mx.eval(latents)
    return latents, height, width


def _reference_latents_from_conditioning(conditioning: Any) -> tuple[list[Any], str | None]:
    if isinstance(conditioning, dict) and conditioning.get("type") == "flux1":
        refs = conditioning.get("reference_latents") or []
        if hasattr(refs, "detach"):
            refs = [refs]
        elif not isinstance(refs, (list, tuple)):
            refs = list(refs)
        method = conditioning.get("reference_latents_method")
        if refs:
            return list(refs), str(method) if method is not None else None
        conditioning = conditioning.get("conditioning")
    if not conditioning:
        return [], None
    ref_latents: list[Any] = []
    seen_refs: set[tuple[Any, ...]] = set()
    method = None
    for entry in conditioning:
        meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
        entry_refs = meta.get("reference_latents") or []
        if entry_refs is None:
            entry_refs = []
        if hasattr(entry_refs, "detach"):
            entry_refs = [entry_refs]
        elif not isinstance(entry_refs, (list, tuple)):
            entry_refs = list(entry_refs)
        for ref in entry_refs:
            if hasattr(ref, "data_ptr"):
                ref_key = ("tensor", int(ref.data_ptr()), tuple(ref.shape))
            else:
                ref_key = ("object", id(ref))
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            ref_latents.append(ref)
        if method is None and meta.get("reference_latents_method") is not None:
            method = meta.get("reference_latents_method")
    if method is not None:
        method = str(method)
    return ref_latents, method


def _latent_to_comfy(
    latents: mx.array,
    height: int,
    width: int,
    template: dict[str, Any],
    crop_latent_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    samples = _flux_latents_to_comfy_tensor(latents, height, width, crop_latent_shape=crop_latent_shape)
    template_samples = template.get("samples")
    if hasattr(template_samples, "device"):
        samples = samples.to(device=template_samples.device)
    out = dict(template)
    out["samples"] = samples
    out["downscale_ratio_spacial"] = 8
    return out


def _flux_latents_to_comfy_tensor(
    latents: mx.array,
    height: int,
    width: int,
    device: Any = None,
    crop_latent_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    unpacked = mx.reshape(latents, (1, height // 16, width // 16, 16, 2, 2))
    unpacked = mx.transpose(unpacked, (0, 3, 1, 4, 2, 5))
    unpacked = mx.reshape(unpacked, (1, 16, height // 16 * 2, width // 16 * 2)).astype(mx.float32)
    mx.eval(unpacked)
    samples_np = np.array(unpacked, dtype=np.float32)
    samples = torch.from_numpy(samples_np)
    if crop_latent_shape is not None:
        crop_h, crop_w = crop_latent_shape
        samples = samples[:, :, : int(crop_h), : int(crop_w)]
    samples = comfy.latent_formats.Flux().process_out(samples)
    if device is not None:
        samples = samples.to(device=device)
    return samples


def _get_flux_system_previewer():
    try:
        import comfy.model_management
        import latent_preview
        from comfy.cli_args import LatentPreviewMethod

        device = comfy.model_management.get_torch_device()
        preview_method = latent_preview.args.preview_method
        cache_key = f"flux_system:{preview_method}:{device}"
        if cache_key in FLUX_PREVIEWER_CACHE:
            return FLUX_PREVIEWER_CACHE[cache_key]

        latent_format = comfy.latent_formats.Flux()
        if preview_method == LatentPreviewMethod.NoPreviews:
            FLUX_PREVIEWER_CACHE[cache_key] = (None, None)
            return FLUX_PREVIEWER_CACHE[cache_key]
        else:
            previewer = latent_preview.get_previewer(device, latent_format)

        FLUX_PREVIEWER_CACHE[cache_key] = (previewer, device)
        return FLUX_PREVIEWER_CACHE[cache_key]
    except Exception as exc:
        if not FLUX_PREVIEWER_CACHE.get("previewer_error_logged"):
            _flux_log(f"SDMLX FLUX: Comfy preview not available: {exc}")
            FLUX_PREVIEWER_CACHE["previewer_error_logged"] = True
        return (None, None)


def _decode_flux_preview_bytes(
    latents: mx.array,
    height: int,
    width: int,
    previewer: Any,
    device: Any,
    crop_latent_shape: tuple[int, int] | None = None,
):
    if previewer is None or device is None:
        return None
    try:
        preview_latents = _flux_latents_to_comfy_tensor(
            latents,
            height,
            width,
            device=device,
            crop_latent_shape=crop_latent_shape,
        )
        return previewer.decode_latent_to_preview_image("JPEG", preview_latents)
    except Exception as exc:
        if not FLUX_PREVIEWER_CACHE.get("decode_error_logged"):
            _flux_log(f"SDMLX FLUX: Comfy preview could not be decoded: {exc}")
            FLUX_PREVIEWER_CACHE["decode_error_logged"] = True
        return None


def _release_flux_preview_resources() -> None:
    if not FLUX_PREVIEWER_CACHE:
        return
    error_flags = {key: value for key, value in FLUX_PREVIEWER_CACHE.items() if str(key).endswith("_logged")}
    FLUX_PREVIEWER_CACHE.clear()
    FLUX_PREVIEWER_CACHE.update(error_flags)
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _flux_vae_source(key: str) -> _FluxVAESource | None:
    key = key.replace("conv_shortcut.", "nin_shortcut.")

    for side in ("decoder", "encoder"):
        if key.startswith(f"{side}.conv_in.conv2d."):
            return _FluxVAESource(key.replace(f"{side}.conv_in.conv2d.", f"{side}.conv_in."), conv=True)
        if key.startswith(f"{side}.conv_out.conv2d."):
            return _FluxVAESource(key.replace(f"{side}.conv_out.conv2d.", f"{side}.conv_out."), conv=True)
        if key.startswith(f"{side}.conv_norm_out.norm."):
            return _FluxVAESource(key.replace(f"{side}.conv_norm_out.norm.", f"{side}.norm_out."))

    for side in ("decoder", "encoder"):
        if key.startswith(f"{side}.mid_block.resnets."):
            parts = key.split(".")
            block_index = int(parts[3])
            rest = ".".join(parts[4:])
            return _FluxVAESource(
                f"{side}.mid.block_{block_index + 1}.{rest}",
                conv=rest.startswith("conv") and rest.endswith(".weight"),
            )

        if key.startswith(f"{side}.mid_block.attentions.0."):
            rest = key.split("attentions.0.", 1)[1]
            rest = rest.replace("group_norm.", "norm.")
            rest = rest.replace("to_q.", "q.")
            rest = rest.replace("to_k.", "k.")
            rest = rest.replace("to_v.", "v.")
            rest = rest.replace("to_out.0.", "proj_out.")
            return _FluxVAESource(
                f"{side}.mid.attn_1.{rest}",
                squeeze_1x1=rest.endswith(".weight") and not rest.startswith("norm."),
            )

    if key.startswith("decoder.up_blocks."):
        parts = key.split(".")
        block_index = int(parts[2])
        comfy_index = 3 - block_index
        rest = ".".join(parts[3:]).replace("resnets.", "block.").replace("upsamplers.0.", "upsample.")
        return _FluxVAESource(
            f"decoder.up.{comfy_index}.{rest}",
            conv=rest.endswith(".weight") and (".conv" in rest or "upsample." in rest or "nin_shortcut." in rest),
        )

    if key.startswith("encoder.down_blocks."):
        parts = key.split(".")
        block_index = int(parts[2])
        rest = ".".join(parts[3:]).replace("resnets.", "block.").replace("downsamplers.0.", "downsample.")
        return _FluxVAESource(
            f"encoder.down.{block_index}.{rest}",
            conv=rest.endswith(".weight") and (".conv" in rest or "downsample." in rest or "nin_shortcut." in rest),
        )

    return None


def _mapped_flux_vae_value(value: mx.array, source: _FluxVAESource) -> mx.array:
    if source.squeeze_1x1:
        value = value[:, :, 0, 0]
    elif source.conv and len(value.shape) == 4:
        value = mx.transpose(value, (0, 2, 3, 1))
    return value.astype(Config.precision)


def _load_comfy_flux_vae_weights(vae: VAE, vae_path: Path) -> None:
    raw = mx.load(str(vae_path))
    mapped = []
    from mlx.utils import tree_flatten, tree_unflatten

    for key, current_value in tree_flatten(vae.parameters()):
        source = _flux_vae_source(key)
        if source is None:
            raise KeyError(f"No FLUX VAE source mapping for {key}")
        value = _mapped_flux_vae_value(raw[source.key], source)
        if value.shape != current_value.shape:
            raise ValueError(f"FLUX VAE shape mismatch for {key}: {value.shape} != {current_value.shape}")
        mapped.append((key, value))
    vae.update(tree_unflatten(mapped))
    mx.eval(vae.parameters())


def _load_flux_vae(vae_name: str, dtype_name: str) -> VAE:
    vae_path = _full_vae_path(vae_name)
    cache_key = (str(vae_path), dtype_name)
    cached = VAE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dtype = _dtype_from_name(dtype_name)
    Config.precision = dtype
    t0 = time.perf_counter()
    vae = VAE()
    _load_comfy_flux_vae_weights(vae, vae_path)
    mx.eval(vae.parameters())
    VAE_CACHE[cache_key] = vae
    _flux_log(
        "SDMLX VAE Loader: "
        f"loaded FLUX VAE {vae_name}, dtype={dtype_name}, load={time.perf_counter() - t0:.2f}s"
    )
    return vae


def _resolve_flux_vae(mlx_vae: Any, op_name: str) -> tuple[VAE, str]:
    if isinstance(mlx_vae, dict):
        if mlx_vae.get("type") == "flux" and "flux_vae" in mlx_vae:
            return mlx_vae["flux_vae"], str(mlx_vae.get("name") or "flux_vae")
        if "weights" in mlx_vae:
            raise RuntimeError(
                f"{op_name} needs a FLUX VAE such as ae.safetensors. "
                "The connected VAE looks like an SDXL VAE."
            )
    if isinstance(mlx_vae, str):
        return _load_flux_vae(mlx_vae, VAE_DTYPE), mlx_vae
    raise RuntimeError(f"{op_name} expects mlx_vae from 🍏 SDMLX VAE Loader.")


def decode_flux_latent_with_vae(samples: dict[str, Any], mlx_vae: Any) -> torch.Tensor:
    model_family = str(samples.get("sdmlx_flux_model_family", "unknown")).lower() if isinstance(samples, dict) else "unknown"
    debug = bool(samples.get("sdmlx_flux_debug", False)) if isinstance(samples, dict) else False
    t0 = time.perf_counter()
    _apply_vae_cache_limit("pre_decode", model_family)
    pre_limit_s = time.perf_counter() - t0
    dtype = _dtype_from_name(VAE_DTYPE)
    vae, _vae_name = _resolve_flux_vae(mlx_vae, "SDMLX VAE Decode")
    latents = _comfy_flux_latent_to_mx(samples, dtype)
    mx.reset_peak_memory()
    raw_t0 = time.perf_counter()
    decoded = vae.decode(latents)
    mx.eval(decoded)
    raw_s = time.perf_counter() - raw_t0
    export_t0 = time.perf_counter()
    image = _mlx_decoded_to_comfy_image(decoded)
    export_s = time.perf_counter() - export_t0
    del decoded, latents
    post_t0 = time.perf_counter()
    _apply_vae_cache_limit("post_decode", model_family)
    mx.eval(mx.zeros((1,), dtype=mx.float16))
    post_limit_s = time.perf_counter() - post_t0
    if debug:
        print(
            "SDMLX VAE Decode: "
            f"family={model_family}, pre_limit={pre_limit_s:.3f}s, raw={raw_s:.3f}s, "
            f"export={export_s:.3f}s, post_limit={post_limit_s:.3f}s, {_mlx_memory_line()}"
        )
    return image


def encode_flux_image_with_vae(pixels: Any, mlx_vae: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    _apply_vae_cache_limit("pre_encode", "kontext")
    dtype = _dtype_from_name(VAE_DTYPE)
    vae, vae_name = _resolve_flux_vae(mlx_vae, "SDMLX VAE Encode")
    image = _comfy_image_to_mx_vae_image(pixels, dtype)
    raw_t0 = time.perf_counter()
    latents = vae.encode(image)
    mx.eval(latents)
    raw_s = time.perf_counter() - raw_t0
    export_t0 = time.perf_counter()
    samples = _mx_flux_model_latent_to_comfy_latent(latents)
    export_s = time.perf_counter() - export_t0
    del image, latents
    _apply_vae_cache_limit("post_encode", "kontext")
    _flux_log(
        "SDMLX VAE Encode: "
        f"vae={vae_name}, raw={raw_s:.3f}s, export={export_s:.3f}s, total={time.perf_counter() - t0:.3f}s"
    )
    return {"samples": samples}


def _comfy_flux_latent_to_mx(samples: dict[str, Any], dtype: mx.Dtype) -> mx.array:
    if not isinstance(samples, dict) or "samples" not in samples:
        raise RuntimeError("SDMLX VAE Decode: expected a Comfy LATENT input.")
    tensor = samples["samples"]
    if tuple(tensor.shape)[1] != 16:
        raise RuntimeError(
            "SDMLX VAE Decode needs a 16-channel FLUX latent. "
            f"Got shape {tuple(tensor.shape)}."
        )
    model_space = comfy.latent_formats.Flux().process_in(tensor.detach().cpu().float())
    latents = mx.array(model_space.numpy().astype(np.float32, copy=False)).astype(dtype)
    mx.eval(latents)
    return latents


def _mlx_decoded_to_comfy_image(decoded: mx.array) -> torch.Tensor:
    image = mx.clip((decoded.astype(mx.float32) / 2.0) + 0.5, 0.0, 1.0)
    image = mx.transpose(image, (0, 2, 3, 1))
    mx.eval(image)
    return torch.from_numpy(np.array(image, dtype=np.float32))


def _comfy_image_to_mx_vae_image(image: Any, dtype: mx.Dtype) -> mx.array:
    if not hasattr(image, "detach"):
        raise RuntimeError("SDMLX VAE Encode: expected a Comfy IMAGE tensor.")
    tensor = image.detach().cpu().float()
    if tensor.ndim != 4:
        raise RuntimeError(f"SDMLX VAE Encode: expected BHWC image tensor, got shape {tuple(tensor.shape)}.")
    if tensor.shape[0] != 1:
        raise RuntimeError("SDMLX VAE Encode currently supports batch_size=1.")
    if tensor.shape[-1] < 3:
        raise RuntimeError(f"SDMLX VAE Encode: expected RGB image, got shape {tuple(tensor.shape)}.")
    height = int(tensor.shape[1])
    width = int(tensor.shape[2])
    pixels = height * width
    if pixels > FLUX_KONTEXT_MAX_ENCODE_PIXELS:
        raise RuntimeError(
            "SDMLX VAE Encode: input image is too large for direct FLUX Kontext encoding "
            f"({width}x{height}, {pixels / 1_000_000:.2f} MP). "
            "Scale the reference image before VAE Encode, e.g. to a 1024px or 768px long edge."
        )
    tensor = tensor[:, :, :, :3]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    image_np = tensor.numpy().astype(np.float32, copy=False)
    image_np = np.transpose(image_np, (0, 3, 1, 2))
    image_np = (image_np * 2.0) - 1.0
    encoded = mx.array(image_np).astype(dtype)
    mx.eval(encoded)
    return encoded


def _mx_flux_model_latent_to_comfy_latent(latents: mx.array) -> torch.Tensor:
    comfy_latents = mx.array(np.array(latents.astype(mx.float32), dtype=np.float32))
    mx.eval(comfy_latents)
    torch_latents = torch.from_numpy(np.array(comfy_latents, dtype=np.float32))
    torch_latents = comfy.latent_formats.Flux().process_out(torch_latents)
    return torch_latents


def _torch_device_from_name(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _torch_sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _cached_flux_lua_model(weights_path: str, device: torch.device, dtype_name: str) -> torch.nn.Module:
    clean_path = str(weights_path or "").strip()
    key = (clean_path or "__auto__", str(device), dtype_name)
    cached = FLUX_LUA_CACHE.get(key)
    if cached is not None:
        return cached
    dtype = torch.float32 if dtype_name == "float32" else torch.float16
    model = load_lua_model(weights_path=clean_path or None, device=device, dtype=dtype)
    FLUX_LUA_CACHE[key] = model
    return model


def _torch_tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if getattr(tensor, "is_floating_point", lambda: False)():
            tensor = tensor.to(torch.float16)
        return tensor.numpy()
    return np.asarray(value, dtype=np.float16)


def _lora_factor_array(value: Any) -> np.ndarray | None:
    array = _torch_tensor_to_numpy(value)
    if array.ndim == 4 and array.shape[2:] == (1, 1):
        array = array[:, :, 0, 0]
    if array.ndim != 2:
        return None
    return array.astype(np.float16, copy=False)


def _lora_alpha(state: dict[str, Any], prefix: str, rank: int) -> float:
    for key in (f"{prefix}.alpha", f"{prefix}.lora_alpha", f"{prefix}.network_alpha", f"{prefix}.scale"):
        value = state.get(key)
        if value is None:
            continue
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    return float(rank)


LORA_WEIGHT_SUFFIXES = (
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_down.default.weight", ".lora_up.default.weight"),
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_A.default.weight", ".lora_B.default.weight"),
    (".down.weight", ".up.weight"),
    (".down.default.weight", ".up.default.weight"),
)


def _iter_lora_pairs(state: dict[str, Any]):
    seen: set[tuple[str, str]] = set()
    for key in sorted(state):
        for down_suffix, up_suffix in LORA_WEIGHT_SUFFIXES:
            if not key.endswith(down_suffix):
                continue
            prefix = key[: -len(down_suffix)]
            up_key = f"{prefix}{up_suffix}"
            if up_key not in state:
                continue
            pair_id = (prefix, down_suffix)
            if pair_id in seen:
                continue
            seen.add(pair_id)
            down = _lora_factor_array(state[key])
            up = _lora_factor_array(state[up_key])
            if down is None or up is None:
                yield prefix, None, None, 0.0
                continue
            rank = int(down.shape[0])
            alpha = _lora_alpha(state, prefix, rank)
            yield prefix, up, down, alpha


def _lora_alpha_from_safetensors(handle: safe_open, prefix: str, rank: int, key_set: set[str]) -> float:
    for key in (f"{prefix}.alpha", f"{prefix}.lora_alpha", f"{prefix}.network_alpha", f"{prefix}.scale"):
        if key not in key_set:
            continue
        try:
            return float(handle.get_tensor(key).float().reshape(()).item())
        except Exception:
            return float(rank)
    return float(rank)


def _iter_lora_pairs_from_safetensors(path: Path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        key_set = set(keys)
        seen: set[tuple[str, str]] = set()
        for key in keys:
            for down_suffix, up_suffix in LORA_WEIGHT_SUFFIXES:
                if not key.endswith(down_suffix):
                    continue
                prefix = key[: -len(down_suffix)]
                up_key = f"{prefix}{up_suffix}"
                if up_key not in key_set:
                    continue
                pair_id = (prefix, down_suffix)
                if pair_id in seen:
                    continue
                seen.add(pair_id)
                down = _lora_factor_array(handle.get_tensor(key))
                up = _lora_factor_array(handle.get_tensor(up_key))
                if down is None or up is None:
                    yield prefix, None, None, 0.0
                    continue
                rank = int(down.shape[0])
                alpha = _lora_alpha_from_safetensors(handle, prefix, rank, key_set)
                yield prefix, up, down, alpha


def _cleanup_after_lora_load() -> None:
    gc.collect()
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _flux_key_from_underscores(value: str) -> str:
    key = value.replace("_", ".")
    fixes = (
        ("single.transformer.blocks", "single_transformer_blocks"),
        ("transformer.blocks", "transformer_blocks"),
        ("double.blocks", "double_blocks"),
        ("single.blocks", "single_blocks"),
        ("img.attn", "img_attn"),
        ("txt.attn", "txt_attn"),
        ("img.mlp", "img_mlp"),
        ("txt.mlp", "txt_mlp"),
        ("to.q", "to_q"),
        ("to.k", "to_k"),
        ("to.v", "to_v"),
        ("to.out.0", "to_out.0"),
        ("to.add.out", "to_add_out"),
        ("add.q.proj", "add_q_proj"),
        ("add.k.proj", "add_k_proj"),
        ("add.v.proj", "add_v_proj"),
        ("proj.mlp", "proj_mlp"),
        ("norm1.context", "norm1_context"),
        ("norm.query.norm", "norm.query_norm"),
        ("norm.key.norm", "norm.key_norm"),
        ("query.norm", "query_norm"),
        ("key.norm", "key_norm"),
        ("x.embedder", "x_embedder"),
        ("context.embedder", "context_embedder"),
    )
    for source, target in fixes:
        key = key.replace(source, target)
    return key


def _direct_flux_weight_key(base: str) -> str | None:
    key = base
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    if not key.endswith(".weight"):
        key = f"{key}.weight"
    key = native_flux_core.normalize_flux_weight_key(key)
    if key is None:
        return None
    native_prefixes = (
        "img_in.",
        "txt_in.",
        "time_in.",
        "vector_in.",
        "guidance_in.",
        "final_layer.",
        "double_blocks.",
        "single_blocks.",
    )
    if key.startswith(native_prefixes):
        return key
    return None


def _diffusers_flux_target(base: str) -> tuple[str, int | None, int | None] | None:
    parts = base.split(".")
    if len(parts) < 2:
        return None

    if parts[0] == "transformer_blocks" and len(parts) >= 4:
        try:
            index = int(parts[1])
        except ValueError:
            return None
        tail = ".".join(parts[2:])
        qkv = {
            "attn.to_q": ("img_attn.qkv", 0, native_flux_core.HIDDEN_DIM),
            "attn.to_k": ("img_attn.qkv", native_flux_core.HIDDEN_DIM, native_flux_core.HIDDEN_DIM),
            "attn.to_v": ("img_attn.qkv", native_flux_core.HIDDEN_DIM * 2, native_flux_core.HIDDEN_DIM),
            "attn.add_q_proj": ("txt_attn.qkv", 0, native_flux_core.HIDDEN_DIM),
            "attn.add_k_proj": ("txt_attn.qkv", native_flux_core.HIDDEN_DIM, native_flux_core.HIDDEN_DIM),
            "attn.add_v_proj": ("txt_attn.qkv", native_flux_core.HIDDEN_DIM * 2, native_flux_core.HIDDEN_DIM),
        }
        if tail in qkv:
            target, start, length = qkv[tail]
            return f"double_blocks.{index}.{target}.weight", start, length
        block_map = {
            "attn.to_out.0": "img_attn.proj",
            "norm1.linear": "img_mod.lin",
            "norm1_context.linear": "txt_mod.lin",
            "attn.to_add_out": "txt_attn.proj",
            "ff.net.0.proj": "img_mlp.0",
            "ff.linear_in": "img_mlp.0",
            "ff.net.2": "img_mlp.2",
            "ff.linear_out": "img_mlp.2",
            "ff_context.net.0.proj": "txt_mlp.0",
            "ff_context.linear_in": "txt_mlp.0",
            "ff_context.net.2": "txt_mlp.2",
            "ff_context.linear_out": "txt_mlp.2",
        }
        if tail in block_map:
            return f"double_blocks.{index}.{block_map[tail]}.weight", None, None

    if parts[0] == "single_transformer_blocks" and len(parts) >= 3:
        try:
            index = int(parts[1])
        except ValueError:
            return None
        tail = ".".join(parts[2:])
        qkv_mlp = {
            "attn.to_q": (0, native_flux_core.HIDDEN_DIM),
            "attn.to_k": (native_flux_core.HIDDEN_DIM, native_flux_core.HIDDEN_DIM),
            "attn.to_v": (native_flux_core.HIDDEN_DIM * 2, native_flux_core.HIDDEN_DIM),
            "proj_mlp": (native_flux_core.HIDDEN_DIM * 3, native_flux_core.MLP_DIM),
        }
        if tail in qkv_mlp:
            start, length = qkv_mlp[tail]
            return f"single_blocks.{index}.linear1.weight", start, length
        block_map = {
            "norm.linear": "modulation.lin",
            "proj_out": "linear2",
            "attn.to_qkv_mlp_proj": "linear1",
            "attn.to_out": "linear2",
        }
        if tail in block_map:
            return f"single_blocks.{index}.{block_map[tail]}.weight", None, None

    basic_map = {
        "x_embedder": "img_in",
        "context_embedder": "txt_in",
        "time_text_embed.timestep_embedder.linear_1": "time_in.in_layer",
        "time_text_embed.timestep_embedder.linear_2": "time_in.out_layer",
        "time_text_embed.text_embedder.linear_1": "vector_in.in_layer",
        "time_text_embed.text_embedder.linear_2": "vector_in.out_layer",
        "time_text_embed.guidance_embedder.linear_1": "guidance_in.in_layer",
        "time_text_embed.guidance_embedder.linear_2": "guidance_in.out_layer",
        "proj_out": "final_layer.linear",
    }
    if base in basic_map:
        return f"{basic_map[base]}.weight", None, None
    return None


def _flux_lora_targets(prefix: str) -> list[tuple[str, int | None, int | None]]:
    bases = [prefix]
    if prefix.startswith("lora_unet_"):
        bases.append(_flux_key_from_underscores(prefix[len("lora_unet_") :]))
    if prefix.startswith("lora_transformer_"):
        bases.append(_flux_key_from_underscores(prefix[len("lora_transformer_") :]))
    if prefix.startswith("lycoris_"):
        bases.append(_flux_key_from_underscores(prefix[len("lycoris_") :]))

    expanded: list[str] = []
    for base in bases:
        for strip in ("base_model.model.", "unet.", "transformer."):
            if base.startswith(strip):
                expanded.append(base[len(strip) :])
        expanded.append(base)

    targets: list[tuple[str, int | None, int | None]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for base in expanded:
        direct = _direct_flux_weight_key(base)
        if direct is not None:
            item = (direct, None, None)
            if item not in seen:
                targets.append(item)
                seen.add(item)
        diffusers = _diffusers_flux_target(base)
        if diffusers is not None and diffusers not in seen:
            targets.append(diffusers)
            seen.add(diffusers)
    return targets


def _clone_flux_model_for_patch(model: SDMLXFluxNativeModel) -> SDMLXFluxNativeModel:
    transformer = copy.copy(model.transformer)
    transformer.w = dict(model.transformer.w)
    transformer.wt = dict(model.transformer.wt)
    transformer.q = dict(model.transformer.q)
    transformer.fp8_weight_keys = set(model.transformer.fp8_weight_keys)
    transformer.fp8_mxfp8 = dict(model.transformer.fp8_mxfp8)
    transformer.gguf_q8 = dict(getattr(model.transformer, "gguf_q8", {}))
    transformer.lora_adapters = {
        key: list(value) for key, value in getattr(model.transformer, "lora_adapters", {}).items()
    }
    transformer.lora_sources = list(getattr(model.transformer, "lora_sources", []))
    _reset_transformer_runtime_state(transformer)
    return SDMLXFluxNativeModel(
        model.name,
        model.path,
        transformer,
        model.precision,
        model.model_config,
        model.pretransposed_linears,
        model.casted_weights,
        model.acceleration_patch,
        model.acceleration_strength,
        model.model_family,
        model.transformer,
    )


def _target_weight_shape(transformer: FluxNativeTransformer, target_key: str) -> tuple[int, int] | None:
    if target_key in transformer.w:
        shape = tuple(int(dim) for dim in transformer.w[target_key].shape)
        if len(shape) == 2:
            return shape
    if target_key in transformer.wt:
        shape = tuple(int(dim) for dim in transformer.wt[target_key].shape)
        if len(shape) == 2:
            return shape[1], shape[0]
    if target_key in transformer.q:
        weight, _scales, _biases, _group_size, _bits = transformer.q[target_key]
        shape = tuple(int(dim) for dim in weight.shape)
        if len(shape) == 2:
            return shape
    if target_key in getattr(transformer, "gguf_q8", {}):
        _q_weight, _scales, _biases, shape = transformer.gguf_q8[target_key]
        return int(shape[0]), int(shape[1])
    return None


def _add_runtime_lora_to_flux_target(
    transformer: FluxNativeTransformer,
    target_key: str,
    start: int | None,
    length: int | None,
    up: np.ndarray,
    down: np.ndarray,
    alpha: float,
    strength: float,
) -> bool:
    shape = _target_weight_shape(transformer, target_key)
    if shape is None:
        return False
    out_dim, in_dim = shape
    rank = max(1, int(down.shape[0]))
    out_slice = int(up.shape[0])
    if int(down.shape[1]) != in_dim or int(up.shape[1]) != rank:
        return False
    if start is not None:
        end = start + (length or out_slice)
        if end > out_dim or out_slice != end - start:
            return False
    elif out_slice != out_dim:
        return False

    scale = float(strength) * float(alpha) / float(rank)
    down_t = mx.array(down.T).astype(transformer.precision)
    up_t = mx.array(up.T * scale).astype(transformer.precision)
    transformer.lora_adapters.setdefault(target_key, []).append((down_t, up_t, start, length or out_slice))
    mx.eval(down_t, up_t)
    return True


def _apply_flux_runtime_lora_pairs(
    model: SDMLXFluxNativeModel,
    pairs,
    *,
    label: str,
    strength: float,
    log_prefix: str,
    acceleration_patch: str | None = None,
    lora_source: str | None = None,
) -> SDMLXFluxNativeModel:
    patched_model = _clone_flux_model_for_patch(model)
    matched = 0
    skipped = 0
    unsupported = 0
    for prefix, up, down, alpha in pairs:
        if up is None or down is None:
            unsupported += 1
            continue
        applied = False
        for target_key, start, length in _flux_lora_targets(prefix):
            if _add_runtime_lora_to_flux_target(
                patched_model.transformer,
                target_key,
                start,
                length,
                up,
                down,
                alpha,
                strength,
            ):
                matched += 1
                applied = True
                break
        if not applied:
            skipped += 1
    _flux_log(
        f"{log_prefix}: "
        f"{label}, strength={float(strength):g}, mode=runtime_lowrank, matched={matched}, "
        f"skipped={skipped}, unsupported={unsupported}"
    )
    if matched == 0:
        _flux_log(f"{log_prefix}: no FLUX transformer targets matched; model is unchanged.")
        return model
    if acceleration_patch:
        patched_model.acceleration_patch = acceleration_patch
        patched_model.acceleration_strength = float(strength)
    if lora_source:
        sources = list(getattr(patched_model.transformer, "lora_sources", []))
        sources.append(str(lora_source))
        patched_model.transformer.lora_sources = sources
    _flux_log(f"{log_prefix} memory: {_transformer_memory_line(patched_model.transformer)}, {_mlx_memory_line()}")
    return patched_model


def _apply_flux_runtime_lora_state(
    model: SDMLXFluxNativeModel,
    state: dict[str, Any],
    *,
    label: str,
    strength: float,
    log_prefix: str,
    acceleration_patch: str | None = None,
    lora_source: str | None = None,
) -> SDMLXFluxNativeModel:
    return _apply_flux_runtime_lora_pairs(
        model,
        _iter_lora_pairs(state),
        label=label,
        strength=strength,
        log_prefix=log_prefix,
        acceleration_patch=acceleration_patch,
        lora_source=lora_source,
    )


def _apply_flux_runtime_lora_file(
    model: SDMLXFluxNativeModel,
    lora_path: Path,
    *,
    label: str,
    strength: float,
    log_prefix: str,
    acceleration_patch: str | None = None,
    lora_source: str | None = None,
) -> SDMLXFluxNativeModel:
    if lora_path.suffix.lower() in {".safetensors", ".sft"}:
        try:
            return _apply_flux_runtime_lora_pairs(
                model,
                _iter_lora_pairs_from_safetensors(lora_path),
                label=label,
                strength=strength,
                log_prefix=log_prefix,
                acceleration_patch=acceleration_patch,
                lora_source=lora_source,
            )
        finally:
            _cleanup_after_lora_load()
    state = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
    try:
        return _apply_flux_runtime_lora_state(
            model,
            state,
            label=label,
            strength=strength,
            log_prefix=log_prefix,
            acceleration_patch=acceleration_patch,
            lora_source=lora_source,
        )
    finally:
        del state
        _cleanup_after_lora_load()


def _apply_flux_lora(model: SDMLXFluxNativeModel, lora_name: str, strength: float) -> SDMLXFluxNativeModel:
    if float(strength) == 0.0:
        _flux_log(f"SDMLX LoRA Loader: {lora_name} skipped because strength=0")
        return model
    _ensure_flux_model_weights_loaded(model)
    if model.acceleration_patch and Path(model.acceleration_patch).stem.lower() == Path(lora_name).stem.lower():
        raise RuntimeError(
            "SDMLX FLUX: this LoRA is already baked into the selected acceleration cache. "
            "Remove the duplicate LoRA loader or set the sampler acceleration_patch to None."
        )
    lora_path = _full_lora_path(lora_name)
    return _apply_flux_runtime_lora_file(
        model,
        lora_path,
        label=f"lora={lora_name}",
        strength=strength,
        log_prefix="SDMLX LoRA Loader",
        lora_source=lora_name,
    )


def _apply_flux_acceleration_patch_runtime(
    model: SDMLXFluxNativeModel,
    selection: str,
    strength: float,
    *,
    debug: bool = False,
) -> SDMLXFluxNativeModel:
    _ensure_flux_model_weights_loaded(model)
    patch_path, package_name = _ensure_flux_acceleration_patch_package(selection, debug=debug)
    return _apply_flux_runtime_lora_file(
        model,
        patch_path,
        label=f"acceleration_patch={package_name}",
        strength=strength,
        log_prefix="SDMLX FLUX acceleration-patch",
        acceleration_patch=package_name,
    )


def _replace_flux_model_transformer(
    model: SDMLXFluxNativeModel,
    load_path: Path,
    *,
    fp8_mode: str,
    acceleration_patch: str | None = None,
    acceleration_strength: float = 0.0,
) -> float:
    preserved_lora_adapters = {
        key: list(value) for key, value in getattr(model.transformer, "lora_adapters", {}).items()
    }
    preserved_lora_sources = list(getattr(model.transformer, "lora_sources", []))
    parent_transformer = getattr(model, "parent_transformer", None)
    if parent_transformer is not None and parent_transformer is not model.transformer:
        try:
            parent_transformer.release_model_weights()
        except Exception:
            pass
    try:
        model.transformer.release_model_weights()
    except Exception:
        pass
    mx.clear_cache()
    gc.collect()
    transformer, casted, pretransposed, load_s = _load_prepared_flux_transformer(
        load_path,
        precision=model.precision,
        fp8_mode=fp8_mode,
        drop_raw="all" if load_path.suffix.lower() == ".gguf" else "off",
    )
    transformer.lora_adapters = preserved_lora_adapters
    transformer.lora_sources = preserved_lora_sources
    model.path = load_path
    model.transformer = transformer
    model.parent_transformer = None
    model.casted_weights = casted
    model.pretransposed_linears = pretransposed
    model.acceleration_patch = acceleration_patch
    model.acceleration_strength = float(acceleration_strength)
    return load_s


class SDMLXFluxNativeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_diffusion_model_names(),),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("sdmlx_model",)
    FUNCTION = "load"
    CATEGORY = "SDMLX/Loaders"

    def load(self, model_name: str):
        path = _full_diffusion_model_path(model_name)
        try:
            from .flux2_nodes import flux2_model_from_checkpoint, is_flux2_checkpoint_file

            if path.is_file() and is_flux2_checkpoint_file(path):
                model = flux2_model_from_checkpoint(path, name=model_name)
                _flux_notice(
                    "SDMLX Diffusion Model Loader: "
                    f"model={model_name}, family={model.get('model_family')}, config={model.get('model_config')}"
                )
                return (model,)
        except RuntimeError:
            raise
        except Exception as exc:
            lowered = f"{model_name} {path}".lower()
            if "flux2" in lowered or "flux.2" in lowered or "klein" in lowered:
                raise RuntimeError(f"SDMLX FLUX.2 Klein: could not inspect selected diffusion model: {model_name}") from exc

        try:
            from .qwen_nodes import is_qwen_checkpoint_file, is_qwen_model_root, qwen_model_from_checkpoint, qwen_model_from_root

            if is_qwen_model_root(path):
                model = qwen_model_from_root(path, name=model_name)
                _flux_notice(
                    "SDMLX Diffusion Model Loader: "
                    f"model={model_name}, family={model.get('qwen_variant') or 'qwen'}, source={path.name}"
                )
                return (model,)
            if path.is_file() and is_qwen_checkpoint_file(path):
                model = qwen_model_from_checkpoint(path, name=model_name)
                _flux_notice(
                    "SDMLX Diffusion Model Loader: "
                    f"model={model_name}, family={model.get('qwen_variant') or 'qwen'}, source={path.name}"
                )
                return (model,)
        except RuntimeError:
            raise
        except Exception as exc:
            if "qwen" in str(model_name).lower() or "qwen" in str(path).lower():
                raise RuntimeError(f"SDMLX Qwen: could not inspect selected diffusion model: {model_name}") from exc

        if "qwen" in str(model_name).lower() or "qwen" in str(path).lower():
            raise RuntimeError(
                "SDMLX Qwen: selected diffusion model is not a complete Qwen root. "
                "Use a Qwen root with transformer/, text_encoder/, vae/, and tokenizer/, "
                "or a supported Qwen diffusion-model checkpoint."
            )

        Config.precision = mx.float16
        _configure_native_core()
        model_family = _model_family_from_path(path, model_name)
        model_config = _model_config_from_family(model_family)
        is_gguf = path.suffix.lower() == ".gguf"
        use_native_fp8 = False if is_gguf else native_flux_core._has_fp8_weights(path)
        use_scaled_fp8 = bool(use_native_fp8 and native_flux_core._has_scaled_fp8_weights(path))
        load_path = path if (is_gguf or use_native_fp8) else _ensure_fp8_transformer_cache(path)
        # Scaled-FP8 is a Comfy quant contract, not the raw native-FP8 contract.
        # On MPS Comfy disables native FP8 compute and materializes dense weights;
        # doing the same here keeps FLUX.1 Kontext correctness ahead of speed.
        effective_fp8_mode = "dequant" if use_scaled_fp8 else ("native" if use_native_fp8 else "dequant")
        transformer, casted, pretransposed, load_s = _load_prepared_flux_transformer(
            load_path,
            precision=Config.precision,
            fp8_mode=effective_fp8_mode,
            drop_raw="all" if is_gguf else "off",
        )
        if is_gguf:
            mx.clear_cache()
            gc.collect()
        _flux_log(
            "SDMLX FLUX Loader: "
            f"model={model_name}, family={model_family}, config={model_config.alias}, source={load_path.name}, "
            f"mode={transformer.load_mode}, fp8={'scaled_dequant' if use_scaled_fp8 else ('native' if use_native_fp8 else 'dequant')}, "
            f"precision=fp16, casted={casted}, "
            f"pretransposed={pretransposed}, fp8_weights={len(transformer.fp8_weight_keys)}, "
            f"load={load_s:.2f}s"
        )
        _flux_log(f"SDMLX FLUX Loader memory: {_transformer_memory_line(transformer)}, {_mlx_memory_line()}")
        return (
            SDMLXFluxNativeModel(
                model_name,
                load_path,
                transformer,
                Config.precision,
                model_config,
                pretransposed,
                casted,
                None,
                0.0,
                model_family,
            ),
        )


class SDMLXT5XXLCompatModel(comfy.sd1_clip.SDClipModel):
    def __init__(
        self,
        device="cpu",
        layer="last",
        layer_idx=None,
        dtype=None,
        model_options=None,
        textmodel_json_config: Path | None = None,
    ):
        model_options = dict(model_options or {})
        t5xxl_quantization_metadata = model_options.get("t5xxl_quantization_metadata")
        if t5xxl_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = t5xxl_quantization_metadata
        model_options = {**model_options, "model_name": "t5xxl"}
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=str(textmodel_json_config or _comfy_t5xxl_config_path()),
            dtype=dtype,
            special_tokens={"end": 1, "pad": 0},
            model_class=comfy.text_encoders.t5.T5,
            enable_attention_masks=False,
            return_attention_masks=False,
            model_options=model_options,
        )


class SDMLXT5XXLCompatTokenizer(comfy.sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None, tokenizer_dir: Path | None = None):
        super().__init__(
            str(tokenizer_dir or _comfy_t5xxl_tokenizer_dir()),
            embedding_directory=embedding_directory,
            pad_with_end=False,
            embedding_size=4096,
            embedding_key="t5xxl",
            tokenizer_class=T5TokenizerFast,
            has_start_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=256,
            tokenizer_data=tokenizer_data or {},
        )


class SDMLXFluxCompatTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data=None, t5_tokenizer_dir: Path | None = None):
        tokenizer_data = tokenizer_data or {}
        self.clip_l = comfy.sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = SDMLXT5XXLCompatTokenizer(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            tokenizer_dir=t5_tokenizer_dir,
        )

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        return {
            "l": self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs),
            "t5xxl": self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs),
        }

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class SDMLXFluxCompatClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None, model_options=None, t5_config_path: Path | None = None):
        super().__init__()
        model_options = model_options or {}
        dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
        self.clip_l = comfy.sd1_clip.SDClipModel(
            device=device,
            dtype=dtype,
            return_projected_pooled=False,
            model_options=model_options,
        )
        self.t5xxl = SDMLXT5XXLCompatModel(
            device=device,
            dtype=dtype_t5,
            model_options=model_options,
            textmodel_json_config=t5_config_path,
        )
        self.dtypes = {dtype, dtype_t5}

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        t5_out, _ = self.t5xxl.encode_token_weights(token_weight_pairs["t5xxl"])
        _, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs["l"])
        return t5_out, l_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        return self.t5xxl.load_sd(sd)


def sdmlx_flux_compat_clip(dtype_t5=None, t5_quantization_metadata=None, t5_config_path: Path | None = None):
    class SDMLXFluxCompatClipModel_(SDMLXFluxCompatClipModel):
        def __init__(self, device="cpu", dtype=None, model_options=None):
            model_options = dict(model_options or {})
            if t5_quantization_metadata is not None:
                model_options["t5xxl_quantization_metadata"] = t5_quantization_metadata
            super().__init__(
                dtype_t5=dtype_t5,
                device=device,
                dtype=dtype,
                model_options=model_options,
                t5_config_path=t5_config_path,
            )

    return SDMLXFluxCompatClipModel_


class _SDMLXFluxDualCLIPLoaderImpl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "t5xxl_name": (folder_paths.get_filename_list("text_encoders"),),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("mlx_clip",)
    RETURN_NAMES = ("mlx_clip",)
    FUNCTION = "load_clip"
    CATEGORY = "SDMLX/Loaders"

    @staticmethod
    def _mlx_clip_handle(clip: Any, clip_name: str, t5xxl_name: str, mode: str) -> dict[str, Any]:
        return {
            "type": "flux1",
            "clip": clip,
            "clip_name": str(clip_name),
            "t5xxl_name": str(t5xxl_name),
            "cache_key": f"flux1:{clip_name}:{t5xxl_name}:{mode}",
        }

    def load_clip(self, clip_name: str, t5xxl_name: str, device: str = "default"):
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        t5_path = folder_paths.get_full_path_or_raise("text_encoders", t5xxl_name)

        model_options: dict[str, Any] = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path, t5_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.FLUX,
                model_options=model_options,
            )
            _flux_log(
                "SDMLX Dual CLIP Loader (flux): "
                f"clip={clip_name}, t5={t5xxl_name}, tokenizer=standard"
            )
            return (self._mlx_clip_handle(clip, clip_name, t5xxl_name, "standard"),)
        except Exception as standard_exc:
            _flux_log(
                "SDMLX Dual CLIP Loader (flux): "
                f"standard tokenizer path failed for t5={t5xxl_name}; trying compatibility path. "
                f"error={standard_exc}"
            )

        clip_data = [
            comfy.utils.load_torch_file(clip_path, safe_load=True),
            comfy.utils.load_torch_file(t5_path, safe_load=True),
        ]

        class EmptyClipTarget:
            pass

        clip_target = EmptyClipTarget()
        clip_target.params = {}
        clip_target.clip = sdmlx_flux_compat_clip(**comfy.sd.t5xxl_detect(clip_data))
        clip_target.tokenizer = SDMLXFluxCompatTokenizer
        parameters = sum(comfy.utils.calculate_parameters(sd) for sd in clip_data)
        clip = comfy.sd.CLIP(
            clip_target,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            parameters=parameters,
            tokenizer_data={},
            state_dict=clip_data,
            model_options=model_options,
        )
        _flux_log(
            "SDMLX Dual CLIP Loader (flux): "
            f"clip={clip_name}, t5={t5xxl_name}, tokenizer=compat"
        )
        return (self._mlx_clip_handle(clip, clip_name, t5xxl_name, "compat"),)


class SDMLX_CLIPTextEncodeFlux:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mlx_clip": ("mlx_clip",),
                "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.001}),
            },
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "SDMLX/Conditioning"

    def encode(self, mlx_clip, clip_l: str, t5xxl: str, guidance: float):
        if not isinstance(mlx_clip, dict) or mlx_clip.get("type") != "flux1":
            raise RuntimeError(
                "SDMLX FLUX.1 CLIP Text Encode Flux: connect mlx_clip from SDMLX Dual CLIP Loader with type=flux."
            )
        clip = mlx_clip.get("clip")
        if clip is None:
            raise RuntimeError("SDMLX FLUX.1 CLIP Text Encode Flux: missing FLUX CLIP runtime.")

        try:
            guidance_value = float(guidance)
        except Exception:
            guidance_value = 3.5

        cache_key = (
            "flux1_clip_text_encode_flux_v1",
            mlx_clip.get("cache_key"),
            str(clip_l),
            str(t5xxl),
            guidance_value,
        )
        if cache_key in FLUX_TEXT_CONDITIONING_CACHE:
            return (FLUX_TEXT_CONDITIONING_CACHE[cache_key],)

        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        conditioning = clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance_value})
        out = {
            "type": "flux1",
            "conditioning": conditioning,
            "clip_l": str(clip_l),
            "t5xxl": str(t5xxl),
            "guidance": guidance_value,
        }
        FLUX_TEXT_CONDITIONING_CACHE[cache_key] = out
        return (out,)


class SDMLXFluxSeaCacheAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "threshold_start": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "threshold_end": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "start_at": ("INT", {"default": 5, "min": 0, "max": 64}),
                "final_guard": (["last1", "last2", "last3"], {"default": "last1"}),
            }
        }

    RETURN_TYPES = (SEACACHE_ADVANCED_TYPE,)
    RETURN_NAMES = ("seacache_advanced",)
    FUNCTION = "configure"
    CATEGORY = "SDMLX/Advanced"

    def configure(self, threshold_start: float, threshold_end: float, start_at: int, final_guard: str):
        return ({
            "threshold_start": float(threshold_start),
            "threshold_end": float(threshold_end),
            "start_at": int(start_at),
            "final_guard": str(final_guard),
        },)


class SDMLXFluxNativeSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdmlx_model": (MODEL_INPUT_TYPE,),
                "positive": ("CONDITIONING,mlx_conditioning",),
                "negative": ("CONDITIONING,mlx_conditioning",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 64}),
                "acceleration_patch": (_flux_acceleration_patch_options(), {"default": FLUX_ACCEL_NONE}),
                "patch_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05, "round": 0.001}),
                "seacache_acceleration": ("BOOLEAN", {"default": False}),
                "preview": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seacache_advanced": (SEACACHE_ADVANCED_TYPE,),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "SDMLX/Sampling"

    def sample(
        self,
        sdmlx_model: SDMLXFluxNativeModel,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        acceleration_patch,
        patch_strength,
        seacache_acceleration,
        preview,
        guidance=None,
        seacache_advanced=None,
        **legacy_inputs,
    ):
        if legacy_inputs and set(legacy_inputs) - {"sdmlx_flux_model"}:
            _flux_log(
                "SDMLX FLUX Sampler: ignored stale workflow input(s): "
                + ", ".join(sorted(legacy_inputs))
            )
        sdmlx_flux_model = sdmlx_model
        if not isinstance(sdmlx_flux_model, SDMLXFluxNativeModel):
            raise RuntimeError("SDMLX FLUX Sampler needs a FLUX model from the SDMLX Checkpoint Loader.")
        del negative
        debug = False
        Config.precision = sdmlx_flux_model.precision
        _configure_native_core()
        _ensure_flux_model_weights_loaded(sdmlx_flux_model)

        model_config = sdmlx_flux_model.model_config
        model_family = sdmlx_flux_model.model_family or _model_family_from_name(sdmlx_flux_model.name)
        acceleration_patch = str(acceleration_patch or FLUX_ACCEL_NONE)
        patch_strength = float(patch_strength)
        seacache_advanced_active = isinstance(seacache_advanced, dict)
        seacache_acceleration = bool(seacache_acceleration) or seacache_advanced_active
        base_path = _full_diffusion_model_path(sdmlx_flux_model.name)
        base_is_gguf = base_path.suffix.lower() == ".gguf"
        conditioning_reference_latents, _conditioning_reference_method = _reference_latents_from_conditioning(positive)
        kontext_active = bool(conditioning_reference_latents)
        if kontext_active and "kontext" not in sdmlx_flux_model.name.lower():
            raise RuntimeError("SDMLX FLUX Kontext image input requires a FLUX.1-Kontext model checkpoint.")
        if conditioning_reference_latents and len(conditioning_reference_latents) > 1:
            raise RuntimeError("SDMLX FLUX Kontext prototype currently supports one reference latent.")
        reference_latents_method = "offset"
        if kontext_active and acceleration_patch != FLUX_ACCEL_NONE and patch_strength != 0.0:
            _flux_notice("SDMLX FLUX acceleration-patch: off (Kontext image conditioning active)")
            acceleration_patch = FLUX_ACCEL_NONE
        if acceleration_patch != FLUX_ACCEL_NONE and patch_strength != 0.0 and model_family != "dev":
            _flux_notice("SDMLX FLUX acceleration-patch: off (not applicable)")
            acceleration_patch = FLUX_ACCEL_NONE
        acceleration_patch_active = acceleration_patch != FLUX_ACCEL_NONE and patch_strength != 0.0
        if acceleration_patch_active:
            _assert_no_duplicate_flux_acceleration_patch_lora(sdmlx_flux_model, acceleration_patch)
            _flux_notice(f"SDMLX FLUX acceleration-patch: {acceleration_patch}")
            use_baked_cache = _flux_acceleration_patch_uses_baked_cache(base_path, acceleration_patch)
            if base_is_gguf or not use_baked_cache:
                if debug:
                    reason = "gguf" if base_is_gguf else "noncanonical"
                    print(
                        "SDMLX FLUX Sampler: "
                        f"acceleration_patch={acceleration_patch}, strength={patch_strength:g}, "
                        f"mode=runtime_lowrank, reason={reason}"
                    )
                sdmlx_flux_model = _apply_flux_acceleration_patch_runtime(
                    sdmlx_flux_model,
                    acceleration_patch,
                    patch_strength,
                    debug=bool(debug),
                )
            else:
                load_path = _ensure_flux_acceleration_cache(base_path, acceleration_patch, patch_strength, debug=bool(debug))
                if Path(sdmlx_flux_model.path).resolve() != load_path.resolve():
                    load_s = _replace_flux_model_transformer(
                        sdmlx_flux_model,
                        load_path,
                        fp8_mode="dequant",
                        acceleration_patch=acceleration_patch,
                        acceleration_strength=patch_strength,
                    )
                    if debug:
                        print(
                            "SDMLX FLUX Sampler: "
                            f"acceleration_patch={acceleration_patch}, strength={patch_strength:g}, "
                            f"cache={load_path.name}, load={load_s:.2f}s"
                        )
        else:
            restore_native_fp8 = native_flux_core._has_fp8_weights(base_path)
            load_path = base_path if restore_native_fp8 else _ensure_fp8_transformer_cache(base_path)
            if sdmlx_flux_model.acceleration_patch and Path(sdmlx_flux_model.path).resolve() != load_path.resolve():
                load_s = _replace_flux_model_transformer(
                    sdmlx_flux_model,
                    load_path,
                    fp8_mode="native" if restore_native_fp8 else "dequant",
                    acceleration_patch=None,
                    acceleration_strength=0.0,
                )
                if debug:
                    print(f"SDMLX FLUX Sampler: acceleration_patch=None, restored_base={load_path.name}, load={load_s:.2f}s")
            sdmlx_flux_model.acceleration_patch = None
            sdmlx_flux_model.acceleration_strength = 0.0

        schnell_seacache_step3_enabled = (
            model_family == "schnell"
            and seacache_acceleration
            and FLUX_SCHNELL_SEACACHE_STEP3_ENABLED
        )
        dev_patch_seacache_step3_enabled = (
            model_family == "dev"
            and acceleration_patch_active
            and seacache_acceleration
            and int(steps) <= 4
        )
        step3_seacache_acceleration = schnell_seacache_step3_enabled or dev_patch_seacache_step3_enabled
        schnell_pool = model_family == "schnell" and seacache_acceleration
        transformer = sdmlx_flux_model.transformer
        runtime_mode = _apply_flux_mlx_mode(
            transformer,
            int(steps),
            model_family,
            use_schnell_seacache_profile=schnell_seacache_step3_enabled,
        )
        profile_width, profile_height = _flux_model_size_from_latent(latent_image)
        seacache_profile = _flux_seacache_profile_for_size(profile_width, profile_height)
        if seacache_profile == "detail":
            seacache_threshold_start = FLUX_SEACACHE_DETAIL_THRESHOLD
            seacache_threshold_end = FLUX_SEACACHE_DETAIL_THRESHOLD_END
            seacache_start_at = FLUX_SEACACHE_DETAIL_START_AT
        else:
            seacache_threshold_start = FLUX_SEACACHE_GENERAL_THRESHOLD
            seacache_threshold_end = FLUX_SEACACHE_GENERAL_THRESHOLD_END
            seacache_start_at = FLUX_SEACACHE_GENERAL_START_AT
        seacache_final_guard = FLUX_SEACACHE_DEFAULT_FINAL_GUARD
        if seacache_advanced_active:
            seacache_profile = "advanced"
            seacache_threshold_start = float(seacache_advanced.get("threshold_start", seacache_threshold_start))
            seacache_threshold_end = float(seacache_advanced.get("threshold_end", seacache_threshold_end))
            seacache_start_at = int(seacache_advanced.get("start_at", seacache_start_at))
            seacache_final_guard = str(seacache_advanced.get("final_guard", seacache_final_guard))
        if schnell_seacache_step3_enabled:
            seacache_profile = "schnell_step3"
        elif dev_patch_seacache_step3_enabled:
            seacache_profile = "dev_patch_step3"

        teacache_threshold = 0.0
        cache_method = "off"
        if seacache_acceleration:
            cache_method = "seacache"
            teacache_threshold = 999.0 if step3_seacache_acceleration else seacache_threshold_start
        seacache_curve_active = (
            cache_method == "seacache"
            and (
                step3_seacache_acceleration
                or seacache_threshold_start > 0.0
                or seacache_threshold_end > 0.0
            )
        )
        teacache_mode = "off"
        if seacache_curve_active:
            teacache_mode = "sea_img" if cache_method == "seacache" else "double0_txt"
        transformer.teacache_mode = teacache_mode
        transformer.teacache_threshold = teacache_threshold
        transformer.teacache_threshold_end = (
            seacache_threshold_end
            if seacache_curve_active and not step3_seacache_acceleration
            else None
        )
        if cache_method == "seacache":
            if step3_seacache_acceleration:
                transformer.teacache_warmup_steps = min(int(steps), 2)
                transformer.teacache_final_steps = max(0, int(steps) - 3)
                if schnell_seacache_step3_enabled:
                    _flux_log("SDMLX FLUX: FLUX-schnell SeaCache acceleration active", debug=debug)
                else:
                    _flux_log("SDMLX FLUX: Dev acceleration-patch SeaCache acceleration active", debug=debug)
            else:
                seacache_start = max(0, seacache_start_at)
                seacache_end = max(1, int(steps) + 1 - _seacache_final_guard_steps(seacache_final_guard))
                transformer.teacache_warmup_steps = min(int(steps), seacache_start)
                transformer.teacache_final_steps = max(0, min(int(steps), int(steps) + 1 - seacache_end))
        else:
            transformer.teacache_warmup_steps = 0
            transformer.teacache_final_steps = 0
        transformer.teacache_total_steps = int(steps)
        transformer.reset_teacache_gate()
        context_budget = None
        step_token_steps: set[int] = set()
        context_profile = "off"
        context_selector = "off"
        if schnell_pool:
            context_budget = 128
            step_token_steps = {step for step in (3, 4) if step <= int(steps)}
            context_profile = "pad_pool128"
            context_selector = "schnell_pool"

        prompt_embeds, pooled_prompt_embeds, conditioning_guidance = _conditioning_to_mlx(
            positive,
            sdmlx_flux_model.precision,
        )
        if guidance is None:
            using_guidance_fallback = conditioning_guidance is None
            effective_guidance = float(DEFAULT_FLUX_GUIDANCE if using_guidance_fallback else conditioning_guidance)
        else:
            try:
                effective_guidance = float(guidance)
                using_guidance_fallback = False
            except Exception:
                using_guidance_fallback = conditioning_guidance is None
                effective_guidance = float(DEFAULT_FLUX_GUIDANCE if using_guidance_fallback else conditioning_guidance)

        reference_latents = None
        reference_height = None
        reference_width = None
        if kontext_active:
            reference_latents, reference_height, reference_width = _prepare_kontext_image_from_samples(
                conditioning_reference_latents[0],
                sdmlx_flux_model.precision,
            )
        reference_tokens = int(reference_latents.shape[1]) if reference_latents is not None else 0
        kontext_cache_mode = "off"
        kontext_cache_fill_step = _flux_kontext_cache_fill_step(int(steps))
        if kontext_active and reference_tokens > 0:
            if reference_latents_method == "index_timestep_zero":
                kontext_cache_mode = "index_timestep_zero_immediate"
            elif reference_latents_method == "offset":
                kontext_cache_mode = f"offset_delayed_step{kontext_cache_fill_step}"
        # The cache is armed per real sampling step below. This lets the offset
        # experiment keep early reference passes fully real before freezing K/V.
        transformer.set_kontext_kv_cache(False, reference_tokens)
        latents, height, width, output_latent_shape = _prepare_noise_from_latent(
            latent_image,
            int(seed),
            sdmlx_flux_model.precision,
        )
        output_height = int(output_latent_shape[0]) * 8
        output_width = int(output_latent_shape[1]) * 8
        target_tokens = int(latents.shape[1])
        config = RuntimeConfig(
            Config(num_inference_steps=int(steps), width=width, height=height, guidance=effective_guidance),
            model_config,
        )
        kontext_profile_enabled = _env_flag("SDMLX_FLUX_KONTEXT_PROFILE")
        kontext_profile_steps: set[int] = set()
        if kontext_profile_enabled:
            if kontext_active and reference_tokens > 0:
                if kontext_cache_mode.startswith("offset_delayed_step"):
                    default_profile_step = min(int(steps), kontext_cache_fill_step + 1)
                else:
                    default_profile_step = 1
                kontext_profile_steps = _parse_flux_profile_steps(
                    os.environ.get("SDMLX_FLUX_KONTEXT_PROFILE_STEPS"),
                    default_step=default_profile_step,
                    max_steps=int(steps),
                )
                transformer.profile_enabled = True
                transformer.profile = {}
                transformer.profile_steps = set(kontext_profile_steps)
                transformer.profile_by_step = len(kontext_profile_steps) > 1
                _flux_notice(
                    "SDMLX FLUX Kontext profile: "
                    f"active steps={sorted(kontext_profile_steps)}, "
                    f"ref_tokens={reference_tokens}, target_tokens={target_tokens}"
                )
            else:
                _flux_notice("SDMLX FLUX Kontext profile: requested but no Kontext reference tokens are active")

        if debug:
            print(
                "SDMLX FLUX Sampler: "
                f"mode={runtime_mode}, model={sdmlx_flux_model.name}, config={model_config.alias}, "
                f"acceleration_patch={sdmlx_flux_model.acceleration_patch or 'None'}, "
                f"lora_targets={len(getattr(sdmlx_flux_model.transformer, 'lora_adapters', {}))}, "
                f"lora_sources={getattr(sdmlx_flux_model.transformer, 'lora_sources', []) or 'none'}, "
                f"kontext={'on' if kontext_active else 'off'}, "
                f"kontext_source={'conditioning' if kontext_active else 'none'}, "
                f"kontext_image={f'{reference_width}x{reference_height}' if kontext_active else 'none'}, "
                f"kontext_ref_method={reference_latents_method if kontext_active else 'none'}, "
                f"kontext_kv_cache={kontext_cache_mode}, "
                f"kontext_ref_tokens={reference_tokens}, target_tokens={target_tokens}, "
                f"ref_target_ratio={(reference_tokens / target_tokens):.2f}, "
                f"context={context_profile}, context_selector={context_selector}, "
                f"context_applies={sorted(step_token_steps) if step_token_steps else context_selector}, "
                f"seacache_profile={seacache_profile}, "
                f"teacache={teacache_mode}, teacache_threshold={teacache_threshold:.4f}, "
                f"teacache_threshold_end={getattr(transformer, 'teacache_threshold_end', None)}, "
                f"{width}x{height}, steps={steps}, seed={seed}, guidance={effective_guidance:.2f}"
            )
            if output_width != width or output_height != height:
                print(
                    "SDMLX FLUX Sampler: "
                    f"target latent padded for FLUX patches: output={output_width}x{output_height}, model={width}x{height}."
                )
            if using_guidance_fallback:
                print(
                    "SDMLX FLUX Sampler: conditioning has no guidance metadata; "
                    f"using fallback guidance={DEFAULT_FLUX_GUIDANCE:.2f}."
                )
            print(f"SDMLX FLUX Sampler: negative conditioning accepted for Comfy compatibility; ignored for FLUX-{model_config.alias}.")
        if debug:
            print(f"SDMLX FLUX Sampler memory before constants: {_mlx_memory_line()}")

        constants_t0 = time.perf_counter()
        txt_projected = transformer.linear(prompt_embeds, "txt_in")
        step_prompt_embeds = None
        step_txt_projected = None
        step_token_note = "off"
        if context_budget is not None and step_token_steps:
            step_prompt_embeds, step_token_note = _pad_pool_token_context(prompt_embeds, budget=context_budget)
            if step_prompt_embeds is prompt_embeds:
                step_prompt_embeds = None
                step_token_steps = set()
                context_selector = "off"
            else:
                step_txt_projected = transformer.linear(step_prompt_embeds, "txt_in")
        pooled_projected = transformer.pooled_text_embed(pooled_prompt_embeds)
        if step_txt_projected is not None:
            mx.eval(txt_projected, step_txt_projected, pooled_projected)
        else:
            mx.eval(txt_projected, pooled_projected)
        if debug:
            print(f"SDMLX FLUX Sampler: step constants={time.perf_counter() - constants_t0:.2f}s")
            context_step_label = "spectrum_real" if context_selector == "spectrum_real" and step_prompt_embeds is not None else (sorted(step_token_steps) or "none")
            print(f"SDMLX FLUX Sampler: token_context={step_token_note}, steps={context_step_label}")
            print(f"SDMLX FLUX Sampler memory after constants: {_mlx_memory_line()}")

        pbar = comfy.utils.ProgressBar(int(steps))
        terminal_pbar = _make_flux_terminal_progress_bar(int(steps))
        previewer, preview_device = _get_flux_system_previewer() if bool(preview) else (None, None)
        if debug:
            print(f"SDMLX FLUX Sampler: preview={'on' if previewer is not None else 'off'}")
        step_times: list[float] = []
        mx.reset_peak_memory()
        sample_t0 = time.perf_counter()
        teacache_reused_steps: list[int] = []
        context_applied_steps: list[int] = []

        def arm_kontext_cache_for_real_step(sampling_step: int) -> None:
            if not kontext_active or reference_tokens <= 0:
                return
            if kontext_cache_mode == "index_timestep_zero_immediate":
                if not getattr(transformer, "kontext_kv_cache_enabled", False):
                    transformer.set_kontext_kv_cache(True, reference_tokens)
                return
            if kontext_cache_mode.startswith("offset_delayed_step") and sampling_step >= kontext_cache_fill_step:
                if not getattr(transformer, "kontext_kv_cache_enabled", False):
                    transformer.set_kontext_kv_cache(True, reference_tokens)

        try:
            for t in range(config.init_time_step, config.num_inference_steps):
                step_t0 = time.perf_counter()
                step_value = config.sigmas[t] * config.num_train_steps
                sampling_step = t - config.init_time_step + 1
                transformer.set_forecast_step(sampling_step)
                transformer.shadow_step_index = sampling_step
                used_spectrum_forecast = False
                teacache_hits_before = int(getattr(transformer, "teacache_hits", 0))
                use_step_context = step_prompt_embeds is not None and sampling_step in step_token_steps
                active_prompt_embeds = step_prompt_embeds if use_step_context else prompt_embeds
                active_txt_projected = step_txt_projected if use_step_context else txt_projected
                if use_step_context:
                    context_applied_steps.append(sampling_step)
                arm_kontext_cache_for_real_step(sampling_step)
                noise = transformer.predict(
                    step_value=step_value,
                    prompt_embeds=active_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    latents=latents,
                    height=height,
                    width=width,
                    guidance=config.guidance * config.num_train_steps,
                    txt_projected=active_txt_projected,
                    pooled_projected=pooled_projected,
                    reference_latents=reference_latents,
                    reference_height=reference_height,
                    reference_width=reference_width,
                    reference_latents_method=reference_latents_method,
                    sigma_value=float(config.sigmas[t]),
                )
                used_teacache_reuse = int(getattr(transformer, "teacache_hits", 0)) > teacache_hits_before
                if used_teacache_reuse:
                    teacache_reused_steps.append(sampling_step)
                preview_latents = latents - noise * config.sigmas[t] if previewer is not None else None
                dt = config.sigmas[t + 1] - config.sigmas[t]
                latents = latents + noise * dt
                mx.eval(latents)
                step_s = time.perf_counter() - step_t0
                step_times.append(step_s)
                if debug:
                    reuse_label = "seacache" if teacache_mode == "sea_img" else "teacache"
                    step_kind = "spectrum" if used_spectrum_forecast else (reuse_label if used_teacache_reuse else "real")
                    print(f"SDMLX FLUX Sampler step {sampling_step}/{steps}: {step_s:.2f}s ({step_kind})")
                preview_bytes = (
                    _decode_flux_preview_bytes(
                        preview_latents,
                        height,
                        width,
                        previewer,
                        preview_device,
                        crop_latent_shape=output_latent_shape,
                    )
                    if preview_latents is not None
                    else None
                )
                if preview_bytes is not None:
                    pbar.update_absolute(sampling_step, int(steps), preview_bytes)
                else:
                    pbar.update(1)
                if terminal_pbar is not None:
                    terminal_pbar.update(1)
        finally:
            if terminal_pbar is not None:
                terminal_pbar.close()
        if previewer is not None:
            previewer = None
            preview_device = None
            _release_flux_preview_resources()

        sample_s = time.perf_counter() - sample_t0
        if debug and step3_seacache_acceleration:
            _flux_notice(f"SDMLX FLUX: SeaCache reused_steps={teacache_reused_steps or 'none'}")
        if teacache_mode == "sea_img":
            _flux_log(
                "SDMLX FLUX SeaCache: "
                f"profile={seacache_profile}, "
                f"hits={getattr(transformer, 'teacache_hits', 0)}, "
                f"real_steps={getattr(transformer, 'teacache_real_steps', 0)}, "
                f"reused_steps={teacache_reused_steps or 'none'}"
            )
        if kontext_active:
            _flux_log(
                "SDMLX FLUX Kontext: "
                f"kv_cache={kontext_cache_mode}, "
                f"ref_tokens={reference_tokens}, target_tokens={target_tokens}, "
                f"stores={getattr(transformer, 'kontext_kv_cache_stores', 0)}, "
                f"hits={getattr(transformer, 'kontext_kv_cache_hits', 0)}"
            )
        if kontext_profile_enabled and kontext_profile_steps:
            profile_path = _write_flux_profile_report(
                transformer,
                model_name=sdmlx_flux_model.name,
                width=output_width,
                height=output_height,
                steps=int(steps),
                seed=int(seed),
                profile_steps=kontext_profile_steps,
                reference_tokens=reference_tokens,
                target_tokens=target_tokens,
            )
            if profile_path is None:
                _flux_notice(
                    "SDMLX FLUX Kontext profile: no buckets recorded "
                    "(profiled step may have been reused by SeaCache)"
                )
            else:
                top = transformer.profile_report()[:4]
                top_summary = ", ".join(f"{label}={total:.2f}s" for label, _count, total, _mean in top)
                _flux_notice(f"SDMLX FLUX Kontext profile: wrote {profile_path}")
                if top_summary:
                    _flux_notice(f"SDMLX FLUX Kontext profile top: {top_summary}")
        if debug:
            print(f"SDMLX FLUX Sampler memory before latent export: {_mlx_memory_line()}")
        out_latent = _latent_to_comfy(latents, height, width, latent_image, crop_latent_shape=output_latent_shape)
        out_latent["sdmlx_flux_model_family"] = model_family
        out_latent["sdmlx_flux_model_name"] = sdmlx_flux_model.name
        out_latent["sdmlx_flux_debug"] = bool(debug)
        out_latent["sdmlx_flux_teacache"] = teacache_mode
        out_latent["sdmlx_flux_teacache_reused_steps"] = list(teacache_reused_steps)
        out_latent["sdmlx_flux_seacache_profile"] = seacache_profile
        out_latent["sdmlx_flux_context"] = context_profile
        out_latent["sdmlx_flux_context_selector"] = context_selector
        out_latent["sdmlx_flux_context_steps"] = sorted(set(context_applied_steps or step_token_steps))
        out_latent["sdmlx_flux_kontext"] = bool(kontext_active)
        if kontext_active:
            out_latent["sdmlx_flux_kontext_image_size"] = (int(reference_width or 0), int(reference_height or 0))
            out_latent["sdmlx_flux_kontext_ref_method"] = reference_latents_method
            out_latent["sdmlx_flux_kontext_kv_cache"] = bool(getattr(transformer, "kontext_kv_cache_enabled", False))
            out_latent["sdmlx_flux_kontext_ref_tokens"] = int(reference_tokens)
        if debug:
            print(f"SDMLX FLUX Sampler memory after latent export: {_mlx_memory_line()}")
        mx.clear_cache()
        if debug:
            print(f"SDMLX FLUX Sampler: cleared MLX cache after latent export ({_mlx_memory_line()})")

        single_attention_hits = sum(1 for hit in transformer.forecast_single_hits if str(hit.get("scope", "")) == "attention")
        single_linear2_hits = sum(1 for hit in transformer.forecast_single_hits if str(hit.get("scope", "")) == "linear2")
        single_linear2_late_hits = sum(1 for hit in transformer.forecast_single_hits if str(hit.get("scope", "")) == "linear2_late")
        single_other_hits = (
            len(transformer.forecast_single_hits)
            - single_attention_hits
            - single_linear2_hits
            - single_linear2_late_hits
        )
        double_hits = len(transformer.forecast_double_img_mlp_hits)
        double_txt_hits = len(transformer.forecast_double_txt_hits)
        partial_linear2_hits = int(getattr(transformer, "forecast_single_linear2_partial_hits", 0))
        if debug:
            loop_overhead_s = max(0.0, sample_s - sum(step_times))
            print(
                "SDMLX FLUX Sampler done: "
                f"sampling={sample_s:.2f}s, steps={', '.join(f'{s:.2f}' for s in step_times)}, "
                f"loop_overhead={loop_overhead_s:.2f}s, "
                f"{'seacache' if teacache_mode == 'sea_img' else 'teacache'}_reused_steps={teacache_reused_steps or 'none'}, "
                f"{'seacache' if teacache_mode == 'sea_img' else 'teacache'}_hits={getattr(transformer, 'teacache_hits', 0)}, "
                f"{'seacache' if teacache_mode == 'sea_img' else 'teacache'}_real_steps={getattr(transformer, 'teacache_real_steps', 0)}, "
                f"context_applied_steps={context_applied_steps or 'none'}, "
                f"single_attention_forecasts={single_attention_hits}, single_linear2_forecasts={single_linear2_hits}, "
                f"single_linear2_late_forecasts={single_linear2_late_hits}, "
                f"single_linear2_partial_text_real={partial_linear2_hits}, "
                f"single_other_forecasts={single_other_hits}, double_img_mlp_forecasts={double_hits}, "
                f"double_txt_forecasts={double_txt_hits}, "
                f"kontext_kv_cache_stores={getattr(transformer, 'kontext_kv_cache_stores', 0)}, "
                f"kontext_kv_cache_hits={getattr(transformer, 'kontext_kv_cache_hits', 0)}, "
                f"kontext_reference_zero_calls={getattr(transformer, 'kontext_reference_zero_calls', 0)}, "
                f"kontext_reference_zero_last={getattr(transformer, 'kontext_reference_zero_last', {}) or 'none'}"
            )
        return (out_latent,)


class SDMLXFluxLUAAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "mlx_vae": ("mlx_vae",),
                "scale": (["2x", "4x"], {"default": "2x"}),
                "device": (["auto", "mps", "cpu", "cuda"], {"default": "auto"}),
                "dtype": (["float32", "float16"], {"default": "float32"}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "weights_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional local lua_flux.pth path. Leave empty to use the official HF weights.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "upscaled_latent")
    FUNCTION = "upscale"
    CATEGORY = "SDMLX/Upscale"

    def upscale(self, samples, mlx_vae, scale, device, dtype, enabled=True, weights_path=""):
        if not isinstance(samples, dict) or "samples" not in samples:
            raise RuntimeError("SDMLX FLUX LUA Adapter expects a Comfy FLUX LATENT input.")
        head = "x4" if scale == "4x" else "x2"

        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        if not hasattr(latent, "detach"):
            raise RuntimeError("SDMLX FLUX LUA Adapter expects a torch-backed FLUX LATENT.")
        if latent.ndim != 4:
            raise RuntimeError(f"SDMLX FLUX LUA Adapter expects a 4D latent tensor, got shape {tuple(latent.shape)}.")
        if int(latent.shape[1]) != 16:
            raise RuntimeError(
                "SDMLX FLUX LUA Adapter needs a 16-channel FLUX/SD3 latent. "
                f"Got shape {tuple(latent.shape)}. SDXL MLX_LATENT/4-channel latents are not supported by lua_flux.pth."
            )

        if enabled:
            torch_device = _torch_device_from_name(device)
            model = _cached_flux_lua_model(weights_path or "", torch_device, dtype)

            upscaled = lua_upscale_latent(model, latent.detach().float(), head=head).detach().cpu().float()
            _torch_sync(torch_device)

            out_latent = dict(samples)
            out_latent["samples"] = upscaled
            out_latent["sdmlx_lua_scale"] = head
        else:
            out_latent = dict(samples)
            out_latent.pop("sdmlx_lua_scale", None)

        model_family = str(samples.get("sdmlx_flux_model_family", "unknown")).lower()
        _apply_vae_cache_limit("pre_decode", model_family)
        vae, _vae_name = _resolve_flux_vae(mlx_vae, "SDMLX FLUX LUA Adapter")
        latents = _comfy_flux_latent_to_mx(out_latent, _dtype_from_name(VAE_DTYPE))
        decoded = vae.decode(latents)
        mx.eval(decoded)
        image = _mlx_decoded_to_comfy_image(decoded)
        del decoded, latents
        _apply_vae_cache_limit("post_decode", model_family)
        mx.eval(mx.zeros((1,), dtype=mx.float16))

        if enabled:
            print(f"SDMLX FLUX LUA Adapter: {head}")
        return (image, out_latent)


class SDMLXFluxKontextImageScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "profile": (["kontext", "balanced", "preview"], {"default": "kontext"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "scale"
    CATEGORY = "SDMLX/Image"

    def scale(self, image, profile):
        width = int(image.shape[2])
        height = int(image.shape[1])
        target_width, target_height = _closest_flux_dimensions(width, height, profile)
        scaled = comfy.utils.common_upscale(
            image.movedim(-1, 1),
            target_width,
            target_height,
            "lanczos",
            "center",
        ).movedim(1, -1)
        return (scaled,)


class SDMLXFluxEmptyLatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "flux_dimensions": (_flux_dimension_options(), {"default": "1024 x 1024"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "SDMLX/Latent"

    def generate(self, width, height, flux_dimensions, batch_size=1):
        width, height = _parse_flux_dimension_option(flux_dimensions, width, height)
        latent = torch.zeros(
            [batch_size, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        return ({"samples": latent, "downscale_ratio_spacial": 8},)


NODE_CLASS_MAPPINGS = {
    "SDMLXFluxNativeLoader": SDMLXFluxNativeLoader,
    "SDMLX_CLIPTextEncodeFlux": SDMLX_CLIPTextEncodeFlux,
    "SDMLXFluxNativeSampler": SDMLXFluxNativeSampler,
    "SDMLXFluxSeaCacheAdvanced": SDMLXFluxSeaCacheAdvanced,
    "SDMLXFluxLUAAdapter": SDMLXFluxLUAAdapter,
    "SDMLXFluxKontextImageScale": SDMLXFluxKontextImageScale,
    "SDMLXFluxEmptyLatentImage": SDMLXFluxEmptyLatentImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDMLXFluxNativeLoader": "🍏 SDMLX Load Diffusion Model",
    "SDMLX_CLIPTextEncodeFlux": "🍏 SDMLX CLIP Text Encode Flux",
    "SDMLXFluxNativeSampler": "🍏 SDMLX KSampler (FLUX.1)",
    "SDMLXFluxSeaCacheAdvanced": "🍏 SDMLX FLUX SeaCache Advanced",
    "SDMLXFluxLUAAdapter": "🍏 SDMLX FLUX LUA Adapter",
    "SDMLXFluxKontextImageScale": "🍏 SDMLX FLUX Kontext Scale",
    "SDMLXFluxEmptyLatentImage": "🍏 SDMLX Empty Latent Image FLUX.1",
}
