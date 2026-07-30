from __future__ import annotations

import os
import gc
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import math
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from mlx import nn
from mlx.utils import tree_unflatten
from safetensors import safe_open

import comfy.utils


SUITE_ROOT = Path(__file__).resolve().parent
QWEN_NATIVE_ROOT = SUITE_ROOT / "sdmlx_qwen_native"

def _ensure_suite_qwen_native_runtime() -> None:
    suite_native_root = str(QWEN_NATIVE_ROOT.resolve())
    loaded_from_elsewhere = False
    for name, module in list(sys.modules.items()):
        if name != "sdmlx_qwen_native" and not name.startswith("sdmlx_qwen_native."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            module_path = str(Path(module_file).resolve())
        except Exception:
            module_path = str(module_file)
        if not module_path.startswith(suite_native_root):
            loaded_from_elsewhere = True
            break

    if loaded_from_elsewhere:
        for name in list(sys.modules):
            if name == "sdmlx_qwen_native" or name.startswith("sdmlx_qwen_native."):
                sys.modules.pop(name, None)

    # Keep the suite root ahead of older lab-native paths that may have been
    # inserted by separate lab nodes during Comfy startup.
    for path in (QWEN_NATIVE_ROOT, SUITE_ROOT):
        path_s = str(path)
        if path_s in sys.path:
            sys.path.remove(path_s)
        sys.path.insert(0, path_s)


_ensure_suite_qwen_native_runtime()

import folder_paths  # noqa: E402


MODEL_TYPE = "sdmlx_model"
FLUX2_MODEL_FAMILY = "flux2-klein"
FLUX2_PACKAGE_FORMAT = "sdmlx-flux2-klein-package-v1"
FLUX2_DEFAULT_ROOT: Path | None = None
_FLUX2_ROOT_DISPLAY_TO_PATH: dict[str, Path] = {}
_FLUX2_MODEL_CACHE: dict[tuple, Any] = {}
_FLUX2_VAE_CACHE: dict[tuple, Any] = {}
_FLUX2_TEXT_CONDITIONING_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_FLUX2_DIRECT_REFERENCE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_FLUX2_QUANT_CONTRACT_LOGGED: set[str] = set()
_FLUX2_VAE_STANDARD = "standard"
_FLUX2_VAE_SMALL_DECODER = "small_decoder"
_FLUX2_TEXT_ENCODER_CACHE_FORMAT = "sdmlx-flux2-text-encoder-cache-v1"
_FLUX2_TEXT_ENCODER_CACHE_MANIFEST_FORMAT = "sdmlx-flux2-text-encoder-cache-v1"

_FLUX2_QUANT_DENSE_BF16 = "dense_bf16"
_FLUX2_QUANT_SCALED_FP8 = "scaled_fp8"
_FLUX2_QUANT_COMFY = "comfy_quant"
_FLUX2_QUANT_COMFY_MXFP8 = "comfy_quant_mxfp8"
_FLUX2_QUANT_RAW_FP8 = "raw_fp8_unscaled"
_FLUX2_QUANT_MIXED_DENSE = "mixed_dense"
_FLUX2_QUANT_UNKNOWN = "unknown"
_FLUX2_LORA_QUANT_BAKE_REQUANTIZE = "requantize"
_FLUX2_LORA_QUANT_BAKE_DENSE_TOUCHED = "dense_touched"
_FLUX2_LORA_QUANT_BAKE_OFF = "off"
_FLUX2_LAB_RUNTIME_PLAN_TYPE = "flux2_lab_runtime_plan"
_FLUX2_LAB_PRODUCT_CURRENT = "product_current"
_FLUX2_LAB_AUTO_COMFY_LAYERED = "auto_comfy_layered"
_FLUX2_LAB_RUNTIME_REBIND = "runtime_rebind"
_FLUX2_LAB_DENSE_WEIGHT_PATCH = "dense_weight_patch"
_FLUX2_LAB_QUANTIZED_REQUANTIZE = "quantized_requantize"
_FLUX2_LAB_QUANTIZED_DENSE_TOUCHED = "quantized_dense_touched"
_FLUX2_LAB_NO_LORA = "no_lora"
_FLUX2_DEFAULT_SCHEDULER = "flow_match_euler_discrete"
FLUX2_NATIVE_RESOLUTIONS = [
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


def _flux2_dimension_options() -> list[str]:
    options = ["custom"]
    for width, height in FLUX2_NATIVE_RESOLUTIONS:
        options.append(f"{width} x {height}")
    return options


def _parse_flux2_dimension_option(option: str, width: int, height: int) -> tuple[int, int]:
    if not option or option == "custom":
        return int(width), int(height)
    parts = str(option).lower().replace(" ", "").split("x")
    if len(parts) != 2:
        return int(width), int(height)
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return int(width), int(height)


@dataclass(frozen=True)
class _Flux2MaterializedComponents:
    modes: dict[str, str]
    entries: dict[str, dict[str, Any]]


class _Flux2PhaseProfiler:
    def __init__(self, name: str):
        self.name = name
        self.enabled = os.environ.get("SDMLX_FLUX2_PHASE_PROFILE", "0").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        self.rows: list[
            tuple[
                str,
                float,
                float,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
            ]
        ] = []
        self._start = time.perf_counter()
        self._last = self._start
        self._process = None
        self._psutil = None
        if self.enabled:
            try:
                import psutil  # type: ignore

                self._process = psutil.Process(os.getpid())
                self._psutil = psutil
            except Exception:
                self._process = None
                self._psutil = None

    def mark(self, label: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        rss_gb = None
        sys_used_gb = None
        sys_free_gb = None
        mlx_active_gb = None
        mlx_cache_gb = None
        mlx_peak_gb = None
        if self._process is not None:
            try:
                rss_gb = float(self._process.memory_info().rss) / (1024**3)
            except Exception:
                rss_gb = None
        if self._psutil is not None:
            try:
                vm = self._psutil.virtual_memory()
                sys_used_gb = float(vm.total - vm.available) / (1024**3)
                sys_free_gb = float(vm.available) / (1024**3)
            except Exception:
                sys_used_gb = None
                sys_free_gb = None
        try:
            mlx_active_gb = float(mx.get_active_memory()) / (1024**3)
            mlx_cache_gb = float(mx.get_cache_memory()) / (1024**3)
            mlx_peak_gb = float(mx.get_peak_memory()) / (1024**3)
        except Exception:
            mlx_active_gb = None
            mlx_cache_gb = None
            mlx_peak_gb = None
        self.rows.append(
            (
                label,
                now - self._start,
                now - self._last,
                rss_gb,
                sys_used_gb,
                sys_free_gb,
                mlx_active_gb,
                mlx_cache_gb,
                mlx_peak_gb,
            )
        )
        self._last = now

    def emit(self) -> None:
        if not self.enabled or not self.rows:
            return
        print(f"SDMLX FLUX.2 phase profile: {self.name}")
        previous_rss = self.rows[0][3]
        previous_sys_used = self.rows[0][4]
        previous_mlx_active = self.rows[0][6]
        previous_mlx_cache = self.rows[0][7]
        for (
            label,
            total_s,
            delta_s,
            rss_gb,
            sys_used_gb,
            sys_free_gb,
            mlx_active_gb,
            mlx_cache_gb,
            mlx_peak_gb,
        ) in self.rows:
            if rss_gb is None:
                rss_text = "rss=n/a"
            else:
                delta_rss = 0.0 if previous_rss is None else rss_gb - previous_rss
                rss_text = f"rss={rss_gb:.2f}GB delta={delta_rss:+.2f}GB"
                previous_rss = rss_gb
            if sys_used_gb is None or sys_free_gb is None:
                sys_text = "sys=n/a"
            else:
                delta_sys = 0.0 if previous_sys_used is None else sys_used_gb - previous_sys_used
                sys_text = f"sys_used={sys_used_gb:.2f}GB sys_delta={delta_sys:+.2f}GB free={sys_free_gb:.2f}GB"
                previous_sys_used = sys_used_gb
            if mlx_active_gb is None or mlx_cache_gb is None or mlx_peak_gb is None:
                mlx_text = "mlx=n/a"
            else:
                delta_active = 0.0 if previous_mlx_active is None else mlx_active_gb - previous_mlx_active
                delta_cache = 0.0 if previous_mlx_cache is None else mlx_cache_gb - previous_mlx_cache
                mlx_text = (
                    f"mlx_active={mlx_active_gb:.2f}GB active_delta={delta_active:+.2f}GB "
                    f"mlx_cache={mlx_cache_gb:.2f}GB cache_delta={delta_cache:+.2f}GB "
                    f"mlx_peak={mlx_peak_gb:.2f}GB"
                )
                previous_mlx_active = mlx_active_gb
                previous_mlx_cache = mlx_cache_gb
            print(f"  {total_s:8.2f}s +{delta_s:7.2f}s {rss_text} {sys_text} {mlx_text} {label}")


def _clear_flux2_model_cache(*, keep_key: tuple | None = None) -> None:
    """Keep FLUX2 warm runs warm without retaining stale 4B/9B model stacks."""
    stale_keys = [key for key in _FLUX2_MODEL_CACHE if key != keep_key]
    if not stale_keys:
        return
    for key in stale_keys:
        _FLUX2_MODEL_CACHE.pop(key, None)
    gc.collect()
    mx.clear_cache()


def _flux2_env_enabled(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "on", "yes"}


def _flux2_text_conditioning_cache_enabled() -> bool:
    return _flux2_env_enabled("SDMLX_FLUX2_TEXT_CONDITIONING_CACHE", "1")


def _flux2_text_conditioning_cache_limit() -> int:
    raw = str(os.environ.get("SDMLX_FLUX2_TEXT_CONDITIONING_CACHE_LIMIT") or "16").strip()
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 16


def _flux2_cache_path_identity(path_value: Any) -> tuple[str, int, int]:
    if not path_value:
        return ("", 0, 0)
    try:
        path = Path(str(path_value)).expanduser().resolve()
        stat = path.stat()
        return (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    except Exception:
        return (str(path_value), 0, 0)


def _flux2_text_conditioning_cache_key(sdmlx_model: dict[str, Any], text: str) -> tuple[Any, ...] | None:
    if not _flux2_text_conditioning_cache_enabled():
        return None
    return (
        FLUX2_MODEL_FAMILY,
        str(sdmlx_model.get("model_config") or ""),
        _flux2_cache_path_identity(sdmlx_model.get("clip_path")),
        _flux2_cache_path_identity(sdmlx_model.get("tokenizer_path")),
        str(text or ""),
    )


def _flux2_get_cached_text_conditioning(sdmlx_model: dict[str, Any], text: str) -> tuple[Any, Any] | None:
    key = _flux2_text_conditioning_cache_key(sdmlx_model, text)
    if key is None:
        return None
    cached = _FLUX2_TEXT_CONDITIONING_CACHE.get(key)
    if cached is not None:
        # Keep normal dict insertion order useful for simple LRU pruning.
        _FLUX2_TEXT_CONDITIONING_CACHE.pop(key, None)
        _FLUX2_TEXT_CONDITIONING_CACHE[key] = cached
    return cached


def _flux2_store_text_conditioning(sdmlx_model: dict[str, Any], text: str, prompt_embeds: Any, text_ids: Any) -> None:
    key = _flux2_text_conditioning_cache_key(sdmlx_model, text)
    limit = _flux2_text_conditioning_cache_limit()
    if key is None or limit <= 0:
        return
    try:
        mx.eval(prompt_embeds, text_ids)
    except Exception:
        return
    _FLUX2_TEXT_CONDITIONING_CACHE[key] = (prompt_embeds, text_ids)
    while len(_FLUX2_TEXT_CONDITIONING_CACHE) > limit:
        oldest = next(iter(_FLUX2_TEXT_CONDITIONING_CACHE), None)
        if oldest is None:
            break
        _FLUX2_TEXT_CONDITIONING_CACHE.pop(oldest, None)


def _flux2_direct_reference_cache_enabled() -> bool:
    return _flux2_env_enabled("SDMLX_FLUX2_DIRECT_REFERENCE_CACHE", "1")


def _flux2_direct_reference_cache_limit() -> int:
    raw = str(os.environ.get("SDMLX_FLUX2_DIRECT_REFERENCE_CACHE_LIMIT") or "16").strip()
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 16


def _flux2_vae_identity_for_cache(mlx_vae: dict[str, Any]) -> tuple[Any, ...]:
    vae_variant = _normalize_flux2_vae_variant(mlx_vae.get("vae_variant") or _FLUX2_VAE_STANDARD)
    return (
        vae_variant,
        _flux2_cache_path_identity(mlx_vae.get("vae_path")),
        _flux2_cache_path_identity(mlx_vae.get("model_path")),
    )


def _flux2_image_content_key(image: torch.Tensor) -> tuple[Any, ...] | None:
    try:
        tensor = image.detach().cpu() if isinstance(image, torch.Tensor) else image
        array = np.asarray(tensor, dtype=np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4 or array.shape[-1] != 3:
            return None
        array = np.ascontiguousarray(np.clip(array, 0.0, 1.0))
        digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
        return (tuple(int(v) for v in array.shape), str(array.dtype), digest)
    except Exception:
        return None


def _flux2_direct_reference_cache_key(image: torch.Tensor, mlx_vae: dict[str, Any]) -> tuple[Any, ...] | None:
    if not _flux2_direct_reference_cache_enabled():
        return None
    image_key = _flux2_image_content_key(image)
    if image_key is None:
        return None
    return (FLUX2_MODEL_FAMILY, "enhanced-direct-reference-v1", _flux2_vae_identity_for_cache(mlx_vae), image_key)


def _flux2_get_cached_direct_reference(image: torch.Tensor, mlx_vae: dict[str, Any]) -> dict[str, Any] | None:
    key = _flux2_direct_reference_cache_key(image, mlx_vae)
    if key is None:
        return None
    cached = _FLUX2_DIRECT_REFERENCE_CACHE.get(key)
    if cached is not None:
        _FLUX2_DIRECT_REFERENCE_CACHE.pop(key, None)
        _FLUX2_DIRECT_REFERENCE_CACHE[key] = cached
        return dict(cached)
    return None


def _flux2_store_direct_reference(image: torch.Tensor, mlx_vae: dict[str, Any], latent: dict[str, Any]) -> None:
    key = _flux2_direct_reference_cache_key(image, mlx_vae)
    limit = _flux2_direct_reference_cache_limit()
    if key is None or limit <= 0:
        return
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None:
        return
    try:
        mx.eval(samples)
    except Exception:
        return
    _FLUX2_DIRECT_REFERENCE_CACHE[key] = dict(latent)
    while len(_FLUX2_DIRECT_REFERENCE_CACHE) > limit:
        oldest = next(iter(_FLUX2_DIRECT_REFERENCE_CACHE), None)
        if oldest is None:
            break
        _FLUX2_DIRECT_REFERENCE_CACHE.pop(oldest, None)


def _flux2_verbose_logs_enabled() -> bool:
    return _flux2_env_enabled("SDMLX_FLUX2_VERBOSE") or _flux2_env_enabled("SDMLX_FLUX2_DEBUG")


def _flux2_log(message: str, *, verbose: bool = False, debug: bool = False) -> None:
    if debug and not _flux2_env_enabled("SDMLX_FLUX2_DEBUG"):
        return
    if verbose and not _flux2_verbose_logs_enabled():
        return
    print(message)


def _flux2_lora_quant_bake_mode() -> str:
    if _flux2_env_enabled("SDMLX_FLUX2_LORA_DENSE_TOUCHED"):
        return _FLUX2_LORA_QUANT_BAKE_DENSE_TOUCHED
    value = str(os.environ.get("SDMLX_FLUX2_LORA_QUANT_BAKE") or "").strip().lower().replace("-", "_")
    if value in {"0", "false", "no", "none", "disable", "disabled", "off"}:
        return _FLUX2_LORA_QUANT_BAKE_OFF
    if value in {"dense", "dense_touched", "touched_dense", "keep_dense"}:
        return _FLUX2_LORA_QUANT_BAKE_DENSE_TOUCHED
    return _FLUX2_LORA_QUANT_BAKE_REQUANTIZE


def _flux2_raw_fp8_dense_dtype(quant_contract: dict[str, Any] | None) -> str | None:
    if _flux2_quant_contract_kind(quant_contract) != _FLUX2_QUANT_RAW_FP8:
        return None
    value = os.environ.get("SDMLX_FLUX2_RAW_FP8_DENSE_MODE")
    if value is None:
        value = os.environ.get("SDMLX_FLUX2_RAW_FP8_DENSE_DIAG")
    value = str(value or "").strip().lower().replace("-", "_")
    if value in {"0", "false", "no", "none", "disable", "disabled", "off", "q8", "mlx_q8", "quantized"}:
        return None
    if value in {"1", "true", "yes", "on", "bf16", "bfloat16"}:
        return "bf16"
    if value in {"fp16", "float16"}:
        return "fp16"
    return "bf16"


def _flux2_runtime_lora_rebind_enabled(
    quant_contract: dict[str, Any] | None,
    model_config: Any | None = None,
) -> bool:
    kind = _flux2_quant_contract_kind(quant_contract)
    if kind == _FLUX2_QUANT_DENSE_BF16 and model_config is not None:
        model_name = str(getattr(model_config, "model_name", "") or "").lower()
        if _flux2_is_4b_model_config(model_config):
            return _flux2_env_enabled("SDMLX_FLUX2_4B_RUNTIME_LORA_REBIND", "1")
        if _flux2_is_9b_model_config(model_config) and "kv" not in model_name:
            return _flux2_env_enabled("SDMLX_FLUX2_9B_DENSE_RUNTIME_LORA_REBIND", "0")
    return kind == _FLUX2_QUANT_RAW_FP8 and (
        _flux2_raw_fp8_dense_dtype(quant_contract) is not None
    )


def _flux2_dense_lora_weight_patch_enabled(
    quant_contract: dict[str, Any] | None,
    model_config: Any | None = None,
) -> bool:
    if _flux2_quant_contract_kind(quant_contract) != _FLUX2_QUANT_DENSE_BF16 or model_config is None:
        return False
    model_name = str(getattr(model_config, "model_name", "") or "").lower()
    if not _flux2_is_9b_model_config(model_config) or "kv" in model_name:
        return False
    return _flux2_env_enabled("SDMLX_FLUX2_9B_DENSE_LORA_WEIGHT_PATCH", "1")


def _normalize_flux2_kv_first_step_barriers(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    text = str(value or "0").strip().lower()
    if text in {"", "off", "false", "none", "no"}:
        return 0
    if text in {"on", "true", "yes", "strict", "soft"}:
        return 1
    try:
        barriers = int(float(text))
    except ValueError:
        return 0
    return max(0, barriers)


def _apply_flux2_kv_first_step_barriers(model: Any, barriers: Any, *, supports_kv: bool | None = None) -> int:
    """Set optional FLUX2 first-step KV-attention barriers for controlled A/B tests."""
    barrier_count = _normalize_flux2_kv_first_step_barriers(barriers)
    enabled = barrier_count > 0
    model.sdmlx_release_text_encoder_after_encode = enabled
    model.sdmlx_clear_cache_each_step = False
    model.sdmlx_release_reference_latents_after_kv_extract = True
    model.sdmlx_kv_attention_barrier = enabled
    model.sdmlx_kv_attention_first_step_barriers = barrier_count
    model.sdmlx_memory_guard_mode = "off" if not enabled else f"kv_first_step_{barrier_count}"
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        transformer.sdmlx_eval_clear_each_block = False
    return barrier_count


def _flux2_is_large_9b_model(model: Any) -> bool:
    config = getattr(model, "model_config", None)
    name = str(getattr(config, "model_name", "") or "").lower()
    overrides = getattr(config, "text_encoder_overrides", {}) or {}
    hidden_size = int(overrides.get("hidden_size") or 0)
    return hidden_size >= 4096 or "9b" in name


def _flux2_is_4b_model_config(model_config: Any) -> bool:
    name = str(getattr(model_config, "model_name", "") or "").lower()
    overrides = getattr(model_config, "text_encoder_overrides", {}) or {}
    hidden_size = int(overrides.get("hidden_size") or 0)
    return "9b" not in name and ("4b" in name or 0 < hidden_size < 4096)


def _flux2_is_9b_model_config(model_config: Any) -> bool:
    name = str(getattr(model_config, "model_name", "") or "").lower()
    overrides = getattr(model_config, "text_encoder_overrides", {}) or {}
    hidden_size = int(overrides.get("hidden_size") or 0)
    return hidden_size >= 4096 or "9b" in name


def _flux2_preserve_packed_bf16_weights(model_config: Any, quant_contract: dict[str, Any] | None) -> bool:
    if _flux2_env_enabled("SDMLX_FLUX2_PACKED_BF16_WEIGHTS"):
        return True
    if _flux2_quant_contract_kind(quant_contract) != _FLUX2_QUANT_DENSE_BF16:
        return False
    return _flux2_is_4b_model_config(model_config) or _flux2_is_9b_model_config(model_config)


def _flux2_model_quant_kind(model: Any) -> str:
    return _flux2_quant_contract_kind(getattr(model, "transformer_quant_contract", None))


def _flux2_should_keep_text_encoder_standard(model: Any, references: list[dict[str, Any]]) -> bool:
    if _flux2_is_4b_model_config(model.model_config) and not references:
        return True

    # The validated BFL 9B contract keeps the large-model memory guard intact.
    # Raw-FP8 community finetunes lost the former prepared-CLIP warm path when
    # the shared text-encoder cache was retired, so allow a narrow in-process
    # residency lane for small raw-FP8 standard workflows only.
    if not _flux2_env_enabled("SDMLX_FLUX2_RAW_FP8_KEEP_TEXT_ENCODER", "1"):
        return False
    if _flux2_model_quant_kind(model) != _FLUX2_QUANT_RAW_FP8:
        return False
    if not _flux2_is_9b_model_config(model.model_config):
        return False
    return len(references) <= 1


def _apply_flux2_pre_encode_memory_policy(model: Any, *, keep_text_encoder: bool = False) -> None:
    # The transformer never needs the Qwen3 text encoder after prompt embeds
    # and text IDs have been materialized. Keeping dense/large encoders resident
    # competes with the denoising graph for MLX memory.
    model.sdmlx_release_text_encoder_after_encode = not keep_text_encoder
    model.sdmlx_clear_after_reference_latents = True


def _flux2_should_clear_cache_before_text_encode(model: Any) -> bool:
    if not _flux2_env_enabled("SDMLX_FLUX2_PRE_TEXT_ENCODE_CACHE_HYGIENE", "1"):
        return False
    if _flux2_model_quant_kind(model) != _FLUX2_QUANT_RAW_FP8:
        return False
    if not _flux2_is_9b_model_config(model.model_config):
        return False
    return not bool(getattr(model, "sdmlx_release_text_encoder_after_encode", True))


def _flux2_clear_cache_before_text_encode(model: Any, profiler: _Flux2PhaseProfiler, label: str) -> None:
    if not _flux2_should_clear_cache_before_text_encode(model):
        return
    try:
        has_cache = mx.get_cache_memory() > 0
    except Exception:
        has_cache = True
    if not has_cache:
        return
    mx.clear_cache()
    profiler.mark(label)


def _apply_flux2_sampling_memory_policy(model: Any, *, cache_enabled: bool) -> None:
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        return
    if _flux2_is_large_9b_model(model) and not bool(cache_enabled):
        # Historical A/B result for 9B KV-checkpoint with kv_cache=off:
        # strict block/step materialization brought sustained memory from the
        # ~50 GB range to the ~22 GB range while staying timing-neutral on warm
        # runs. This is a 9B full-reference memory-control policy, not a speed
        # feature.
        model.sdmlx_clear_cache_each_step = True
        model.sdmlx_release_reference_latents_after_kv_extract = True
        model.sdmlx_memory_guard_mode = "auto_9b_full"
        transformer.sdmlx_eval_clear_each_block = True
        transformer.sdmlx_eval_clear_policy = "all"
        transformer.sdmlx_clear_cache_after_block_eval = True
        return
    model.sdmlx_clear_cache_each_step = False
    transformer.sdmlx_eval_clear_each_block = False


_FLUX2_HEAD_DIM = 128
_FLUX2_ACCEL_ENV_KEYS = (
    "SDMLX_FLUX2_SINGLE_TO_OUT_SPLIT",
    "SDMLX_FLUX2_SINGLE_TO_OUT_FP16_PRETRANSPOSE",
    "SDMLX_FLUX2_SINGLE_TO_OUT_SPLIT_FP16",
    "SDMLX_FLUX2_SINGLE_QKV_MLP_SPLIT",
    "SDMLX_FLUX2_MIXED_FP16",
    "SDMLX_FLUX2_SINGLE_QKV_FP16_PROJ",
    "SDMLX_FLUX2_SINGLE_ALL_FP16_PROJ",
    "SDMLX_FLUX2_NATIVE_QKV_PREP",
    "SDMLX_FLUX2_QKV_FP16_SCALE",
    "SDMLX_FLUX2_TO_OUT_FP16_SCALE",
    "SDMLX_FLUX2_MLP_FP16_SCALE",
    "SDMLX_FLUX2_CACHE_QKV_FP16_WEIGHT_T",
    "SDMLX_FLUX2_CACHE_ALL_FP16_WEIGHT_T",
    "SDMLX_FLUX2_SINGLE_MLP_FP16_PROJ",
    "SDMLX_FLUX2_SINGLE_FULL_MLP_SPLIT",
    "SDMLX_FLUX2_SINGLE_DRAWTHINGS_FP16_SPLIT",
)


def _throw_if_comfy_interrupted() -> None:
    try:
        import comfy.model_management as model_management  # type: ignore
    except Exception:
        return
    model_management.throw_exception_if_processing_interrupted()


@dataclass(frozen=True)
class _Flux2PackedParam:
    target: str
    source: str
    expected_shape: tuple[int, ...]
    source_slice: tuple[int, int] | None = None


@dataclass(frozen=True)
class _Flux2PackedWeights:
    flat: list[tuple[str, mx.array]]
    quantized_modules: frozenset[str]


def _sdmlx_model_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        folder_map = getattr(folder_paths, "folder_names_and_paths", {})
        for key in ("sdmlx", "SDMLX"):
            if key in folder_map:
                roots.extend(Path(p).expanduser() for p in folder_paths.get_folder_paths(key))
    except Exception:
        pass

    models_dir = Path(getattr(folder_paths, "models_dir", SUITE_ROOT.parent / "models"))
    roots.append(models_dir / "SDMLX")
    if FLUX2_DEFAULT_ROOT is not None:
        roots.append(FLUX2_DEFAULT_ROOT.parent.parent if FLUX2_DEFAULT_ROOT.exists() else FLUX2_DEFAULT_ROOT.parent)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if root.exists():
            unique.append(root)
    return unique


def _flux2_primary_sdmlx_root() -> Path:
    roots = _sdmlx_model_roots()
    if roots:
        roots[0].mkdir(parents=True, exist_ok=True)
        return roots[0]
    root = Path(getattr(folder_paths, "models_dir", SUITE_ROOT.parent / "models")) / "SDMLX"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _flux2_preferred_cache_sdmlx_root() -> Path:
    roots = _sdmlx_model_roots()
    for root in roots:
        try:
            if any(root.rglob("*flux*klein*.sdmlx")):
                root.mkdir(parents=True, exist_ok=True)
                return root
        except Exception:
            continue
    return _flux2_primary_sdmlx_root()


def _flux2_text_encoder_prepared_cache_dir() -> Path:
    path = _flux2_preferred_cache_sdmlx_root() / "cache" / "text_encoders" / "flux2-klein"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _flux2_prepared_text_encoder_cache_enabled() -> bool:
    value = os.environ.get("SDMLX_FLUX2_TEXT_ENCODER_PREPARED_CACHE")
    if value is None or str(value).strip() == "":
        return True
    value = str(value).strip().lower()
    return value in {"1", "true", "on", "yes"}


def _flux2_prepared_text_encoder_cache_allowed(model_config, transformer_quant_contract: dict[str, Any] | None) -> bool:
    if not _flux2_prepared_text_encoder_cache_enabled():
        return False
    overrides = getattr(model_config, "text_encoder_overrides", None) or {}
    if int(overrides.get("hidden_size") or 0) != 4096:
        return False
    contract = _flux2_quant_contract_kind(transformer_quant_contract)
    return contract in {_FLUX2_QUANT_DENSE_BF16, _FLUX2_QUANT_SCALED_FP8}


def _flux2_safe_package_name(path: str | os.PathLike[str]) -> str:
    base = Path(path).stem
    name = re.sub(r"[^A-Za-z0-9._ +()-]+", "_", base).strip(" .")
    return name or "flux2-klein"


def _flux2_file_digest(path: Path) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("utf-8"))
    chunk_size = 4 * 1024 * 1024
    offsets = [0]
    if stat.st_size > chunk_size:
        offsets.append(max(0, stat.st_size // 2 - chunk_size // 2))
        offsets.append(max(0, stat.st_size - chunk_size))
    seen: set[int] = set()
    with path.open("rb") as handle:
        for offset in offsets:
            if offset in seen:
                continue
            seen.add(offset)
            handle.seek(offset)
            digest.update(offset.to_bytes(8, "little", signed=False))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()[:24]


def _flux2_file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = _flux2_file_digest(path)
    return {
        "source_name": path.name,
        "source_path": str(path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_content_digest": digest,
        "source_digest": digest[:16],
    }


def _flux2_clone_or_copy(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if sys.platform == "darwin" and source.is_file():
        try:
            result = subprocess.run(  # type: ignore[name-defined]
                ["cp", "-c", str(source), str(destination)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0 and destination.exists():
                return "apfs_clone"
        except Exception:
            pass
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return "copy"


def _flux2_link_or_copy_dir(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            if destination.is_symlink() and Path(os.readlink(destination)).resolve() == source:
                return "symlink"
        except Exception:
            pass
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    try:
        os.symlink(str(source), str(destination), target_is_directory=True)
        return "symlink"
    except OSError:
        shutil.copytree(source, destination)
        return "copy"


def _flux2_link_or_copy_path(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            if destination.is_symlink() and Path(os.readlink(destination)).resolve() == source:
                return "symlink"
        except Exception:
            pass
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    try:
        os.symlink(str(source), str(destination), target_is_directory=source.is_dir())
        return "symlink"
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
            return "copy"
        return _flux2_clone_or_copy(source, destination)


def _flux2_copy_dir_into_package(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.copytree(source, destination)
    return "copy"


def _flux2_remove_package_component_path(package_path: Path, component: str) -> None:
    component_path = package_path / component
    if not component_path.exists() and not component_path.is_symlink():
        return
    if component_path.is_dir() and not component_path.is_symlink():
        shutil.rmtree(component_path)
    else:
        component_path.unlink()


def _flux2_text_encoder_hidden_size(path: str | os.PathLike[str]) -> int | None:
    candidate = Path(path).expanduser()
    config_path = candidate / "config.json" if candidate.is_dir() else candidate.parent / "config.json"
    try:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            hidden_size = int(config.get("hidden_size") or 0)
            if hidden_size:
                return hidden_size
    except Exception:
        pass
    if candidate.is_file():
        info = _flux2_qwen3_text_encoder_info(candidate)
        if info is not None:
            hidden_size = int(info.get("hidden_size") or 0)
            if hidden_size:
                return hidden_size
    try:
        for shard in sorted(candidate.glob("*.safetensors")):
            info = _flux2_qwen3_text_encoder_info(shard)
            if info is not None:
                hidden_size = int(info.get("hidden_size") or 0)
                if hidden_size:
                    return hidden_size
    except Exception:
            pass
    return None


def _flux2_text_encoder_projection_input_size(path: str | os.PathLike[str]) -> int | None:
    """Return the real first-layer projection input dim from weights, if present."""
    candidate = Path(path).expanduser()

    def inspect_file(file_path: Path) -> int | None:
        try:
            with safe_open(str(file_path), framework="pt") as handle:
                keys = set(handle.keys())
                for key in (
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.v_proj.weight",
                    "model.layers.0.mlp.gate_proj.weight",
                ):
                    if key not in keys:
                        continue
                    shape = handle.get_slice(key).get_shape()
                    shape = _flux2_comfy_quant_effective_shape(handle, key, shape)
                    if len(shape) == 2:
                        return int(shape[1])
        except Exception:
            return None
        return None

    if candidate.is_file():
        return inspect_file(candidate)
    try:
        for shard in sorted(candidate.glob("*.safetensors")):
            value = inspect_file(shard)
            if value:
                return value
    except Exception:
        pass
    return None


def _flux2_text_encoder_projection_mismatches(
    path: str | os.PathLike[str],
    expected_hidden: int,
) -> list[tuple[str, tuple[int, ...]]]:
    if not expected_hidden:
        return []
    candidate = Path(path).expanduser()
    mismatches: list[tuple[str, tuple[int, ...]]] = []
    checked: set[str] = set()
    patterns = (
        re.compile(r"model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj)\.weight$"),
        re.compile(r"model\.layers\.\d+\.mlp\.gate_proj\.weight$"),
    )

    def inspect_file(file_path: Path) -> None:
        try:
            with safe_open(str(file_path), framework="pt") as handle:
                for key in handle.keys():
                    if key in checked or not any(pattern.match(key) for pattern in patterns):
                        continue
                    checked.add(key)
                    shape = tuple(int(value) for value in handle.get_slice(key).get_shape())
                    shape = _flux2_comfy_quant_effective_shape(handle, key, shape)
                    if len(shape) == 2 and int(shape[1]) != expected_hidden:
                        mismatches.append((key, shape))
        except Exception:
            return

    if candidate.is_file():
        inspect_file(candidate)
    else:
        try:
            for shard in sorted(candidate.glob("*.safetensors")):
                inspect_file(shard)
        except Exception:
            pass
    return mismatches


def _flux2_text_encoder_matches_config(path: str | os.PathLike[str], model_config: Any) -> bool:
    expected_hidden = int((model_config.text_encoder_overrides or {}).get("hidden_size") or 0)
    actual_hidden = _flux2_text_encoder_hidden_size(path)
    projection_input = _flux2_text_encoder_projection_input_size(path)
    if expected_hidden and actual_hidden and expected_hidden != actual_hidden:
        return False
    if expected_hidden and projection_input and projection_input != expected_hidden:
        return False
    if _flux2_text_encoder_projection_mismatches(path, expected_hidden):
        return False
    return True


def _flux2_tokenizer_has_chat_template(path: str | os.PathLike[str]) -> bool:
    tokenizer_path = Path(path).expanduser()
    if not tokenizer_path.exists():
        return False
    if (tokenizer_path / "chat_template.jinja").is_file():
        return True
    config_path = tokenizer_path / "tokenizer_config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return False
    return bool(config.get("chat_template"))


def _flux2_read_package_manifest(package_path: Path) -> dict[str, Any] | None:
    manifest_path = package_path / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return manifest if is_flux2_manifest(manifest) else None
    except Exception:
        return None


def _flux2_manifest_component_entry(manifest: dict[str, Any] | None, component: str) -> dict[str, Any] | None:
    if not isinstance(manifest, dict):
        return None
    components = manifest.get("components")
    if not isinstance(components, dict):
        return None
    entry = components.get(component)
    return entry if isinstance(entry, dict) else None


def _flux2_manifest_component_is_external(manifest: dict[str, Any] | None, component: str) -> bool:
    entry = _flux2_manifest_component_entry(manifest, component)
    return bool(entry and str(entry.get("storage") or "").lower() == "external_model_path")


def _flux2_component_path_from_manifest(
    package_path: Path,
    manifest: dict[str, Any] | None,
    component: str,
) -> Path | None:
    entry = _flux2_manifest_component_entry(manifest, component)
    if not entry:
        return None
    storage = str(entry.get("storage") or "").lower()
    if storage == "external_model_path":
        raw = entry.get("source") or entry.get("path")
    else:
        raw = entry.get("path") or component
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = package_path / path
    try:
        return path.resolve()
    except Exception:
        return path


def _flux2_resolved_package_component_path(
    package_path: Path,
    component: str,
    manifest: dict[str, Any] | None = None,
) -> Path | None:
    if manifest is None:
        manifest = _flux2_read_package_manifest(package_path)
    manifest_path = _flux2_component_path_from_manifest(package_path, manifest, component)
    if manifest_path is not None and manifest_path.exists():
        return manifest_path
    package_component = package_path / component
    if package_component.exists() or package_component.is_symlink():
        try:
            return package_component.resolve()
        except Exception:
            return package_component
    return None


def _flux2_package_components_match_config(
    package_path: Path,
    model_config: Any,
    manifest: dict[str, Any] | None = None,
) -> bool:
    if manifest is None:
        manifest = _flux2_read_package_manifest(package_path)
    text_encoder = _flux2_resolved_package_component_path(package_path, "text_encoder", manifest)
    if text_encoder is None or not text_encoder.exists():
        return False
    tokenizer = package_path / "tokenizer"
    if not _flux2_tokenizer_has_chat_template(tokenizer):
        return False
    vae = _flux2_resolved_package_component_path(package_path, "vae", manifest)
    if vae is None or not vae.exists():
        return False
    if vae.is_file() and not (_is_flux2_small_decoder_vae_file(vae) or _is_flux2_vae_weight_file(vae)):
        return False
    return _flux2_text_encoder_matches_config(text_encoder, model_config)


def _flux2_link_shared_components(package_path: Path, donor: Path | None) -> dict[str, str]:
    component_modes: dict[str, str] = {}
    if donor is None:
        return component_modes
    for component in ("text_encoder", "tokenizer", "vae"):
        source = donor / component
        if source.exists():
            if component == "text_encoder":
                component_modes[component] = _flux2_link_or_copy_path(source, package_path / component)
            else:
                component_modes[component] = _flux2_clone_or_copy(source, package_path / component)
    return component_modes


def _flux2_component_manifest_entry(
    package_path: Path,
    component: str,
    mode: str,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    component_path = package_path / component
    external = mode.startswith("external") or mode.startswith("legacy_external")
    entry: dict[str, Any] = {
        "path": _flux2_manifest_path(source_path) if external and source_path is not None else component,
        "storage": "external_model_path" if external or component_path.is_symlink() else "package",
        "copy_mode": mode,
    }
    try:
        entry["source"] = _flux2_manifest_path(source_path or component_path.resolve())
    except Exception:
        pass
    return entry


def _flux2_legacy_component_fallback_enabled() -> bool:
    value = str(os.environ.get("SDMLX_FLUX2_LEGACY_COMPONENT_FALLBACK", "0")).strip().lower()
    return value not in {"0", "false", "off", "no"}


def _flux2_candidate_registered_paths(folder_name: str, preferred_names: tuple[str, ...] = ()) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: str | os.PathLike[str] | None) -> None:
        if not path:
            return
        candidate = Path(path).expanduser()
        try:
            key = str(candidate.resolve()).lower()
        except Exception:
            key = str(candidate).lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    try:
        roots = [Path(path).expanduser() for path in folder_paths.get_folder_paths(folder_name)]
    except Exception:
        roots = []

    for name in preferred_names:
        try:
            add(folder_paths.get_full_path(folder_name, name))
        except Exception:
            pass
        for root in roots:
            add(root / name)

    try:
        for name in folder_paths.get_filename_list(folder_name):
            try:
                add(folder_paths.get_full_path(folder_name, name))
            except Exception:
                pass
    except Exception:
        pass

    for root in roots:
        try:
            for child in root.iterdir():
                add(child)
        except OSError:
            continue

    return candidates


def _flux2_find_external_text_encoder(config_name: str, model_config: Any) -> Path | None:
    try:
        from .sdmlx_asset_registry import flux2_asset_entry

        preferred = flux2_asset_entry(config_name).preferred_text_encoders
    except Exception:
        preferred = ()
    for candidate in _flux2_candidate_registered_paths("text_encoders", preferred):
        if not candidate.exists():
            continue
        if not is_flux2_text_encoder_file(candidate):
            continue
        if not _flux2_text_encoder_matches_config(candidate, model_config):
            continue
        try:
            return candidate.resolve()
        except Exception:
            return candidate
    return None


def _flux2_find_external_vae_file() -> Path | None:
    preferred = (
        "full_encoder_small_decoder.safetensors",
        "flux2-vae.safetensors",
        "diffusion_pytorch_model.safetensors",
    )
    for candidate in _flux2_candidate_registered_paths("vae", preferred):
        if not candidate.exists() or not candidate.is_file():
            continue
        if not is_flux2_vae_file(candidate):
            continue
        try:
            return candidate.resolve()
        except Exception:
            return candidate
    return None


def _flux2_same_resolved_path(left: Path, right: Path | None) -> bool:
    if right is None:
        return False
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except Exception:
        return left.expanduser() == right.expanduser()


def _flux2_package_needs_component_layout_refresh(
    package_path: Path,
    config_name: str,
    model_config: Any,
) -> bool:
    manifest = _flux2_read_package_manifest(package_path)
    text_encoder = _flux2_find_external_text_encoder(config_name, model_config)
    if text_encoder is not None:
        current = _flux2_resolved_package_component_path(package_path, "text_encoder", manifest)
        if not _flux2_manifest_component_is_external(manifest, "text_encoder"):
            return True
        if not current or not _flux2_same_resolved_path(current, text_encoder):
            return True
        if (package_path / "text_encoder").exists() or (package_path / "text_encoder").is_symlink():
            return True

    vae = _flux2_find_external_vae_file()
    if vae is not None:
        current = _flux2_resolved_package_component_path(package_path, "vae", manifest)
        if not _flux2_manifest_component_is_external(manifest, "vae"):
            return True
        if not current or not _flux2_same_resolved_path(current, vae):
            return True
        if (package_path / "vae").exists() or (package_path / "vae").is_symlink():
            return True

    return False


def _flux2_find_tokenizer_donor(config_name: str, *, exclude: Path | None = None) -> Path | None:
    wanted = _flux2_config_key(config_name)
    excluded: Path | None = None
    if exclude is not None:
        try:
            excluded = exclude.expanduser().resolve()
        except Exception:
            excluded = exclude.expanduser()
    candidates = _find_flux2_roots()
    if _flux2_legacy_component_fallback_enabled():
        try:
            from .sdmlx_asset_registry import find_flux2_component_root

            legacy_root = find_flux2_component_root(config_name, exclude=excluded)
            if legacy_root is not None:
                candidates.append(legacy_root)
        except Exception:
            pass

    exact: list[Path] = []
    fallback: list[Path] = []
    for root in candidates:
        try:
            resolved = root.expanduser().resolve()
        except Exception:
            resolved = root.expanduser()
        if excluded is not None and resolved == excluded:
            continue
        tokenizer = root / "tokenizer"
        if not _flux2_tokenizer_has_chat_template(tokenizer):
            continue
        try:
            root_key = _flux2_config_key(_model_config_for_root(root).model_name)
        except Exception:
            root_key = ""
        if root_key == wanted:
            exact.append(root)
        else:
            fallback.append(root)
    return (exact or fallback or [None])[0]


def _flux2_download_tokenizer_assets(package_path: Path, config_name: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
        from .sdmlx_asset_registry import flux2_asset_entry
    except Exception as exc:
        raise RuntimeError(
            "SDMLX FLUX.2 Klein conversion: tokenizer assets are missing and "
            "huggingface_hub is not available to fetch them."
        ) from exc

    entry = flux2_asset_entry(config_name)
    package_path.mkdir(parents=True, exist_ok=True)
    print(
        "SDMLX FLUX.2 Klein conversion: "
        f"downloading tokenizer/scheduler assets from {entry.repo_id} into {package_path.name}",
        flush=True,
    )
    allow_patterns = [
        "model_index.json",
        "tokenizer/*",
        "scheduler/*",
        "README*",
        "LICENSE*",
    ]
    try:
        try:
            snapshot_download(
                repo_id=entry.repo_id,
                local_dir=str(package_path),
                allow_patterns=allow_patterns,
                ignore_patterns=[
                    ".git/*",
                    "text_encoder/*",
                    "transformer/*",
                    "vae/*",
                ],
            )
        except TypeError:
            snapshot_download(
                repo_id=entry.repo_id,
                local_dir=str(package_path),
                allow_patterns=allow_patterns,
            )
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(exc, GatedRepoError) or status_code in {401, 403}:
            raise RuntimeError(
                "SDMLX FLUX.2 Klein conversion: Hugging Face access is required "
                f"to download tokenizer/scheduler assets from {entry.repo_id}. "
                "Accept the repository terms and authenticate Hugging Face in "
                "the same Python environment that runs ComfyUI, then retry. "
                "Git credentials and Xcode are not required."
            ) from None
        raise
    shutil.rmtree(package_path / ".cache", ignore_errors=True)
    tokenizer = package_path / "tokenizer"
    if not _flux2_tokenizer_has_chat_template(tokenizer):
        raise RuntimeError(
            "SDMLX FLUX.2 Klein conversion: downloaded tokenizer is incomplete "
            f"for {entry.repo_id}."
        )
    return tokenizer


def _flux2_materialize_package_components(
    package_path: Path,
    config_name: str,
    model_config: Any,
    *,
    exclude: Path | None = None,
) -> _Flux2MaterializedComponents:
    modes: dict[str, str] = {}
    entries: dict[str, dict[str, Any]] = {}

    text_encoder = _flux2_find_external_text_encoder(config_name, model_config)
    if text_encoder is not None:
        _flux2_remove_package_component_path(package_path, "text_encoder")
        mode = "external_manifest"
        modes["text_encoder"] = mode
        entries["text_encoder"] = _flux2_component_manifest_entry(
            package_path,
            "text_encoder",
            mode,
            source_path=text_encoder,
        )
        print(
            "SDMLX FLUX.2 Klein conversion: "
            f"reusing text_encoder from models/text_encoders: {text_encoder.name}",
            flush=True,
        )
    else:
        existing = package_path / "text_encoder"
        if existing.exists() and is_flux2_text_encoder_file(existing) and _flux2_text_encoder_matches_config(existing, model_config):
            modes["text_encoder"] = "package"
            entries["text_encoder"] = _flux2_component_manifest_entry(package_path, "text_encoder", "package")

    vae = _flux2_find_external_vae_file()
    if vae is not None:
        _flux2_remove_package_component_path(package_path, "vae")
        mode = "external_manifest"
        modes["vae"] = mode
        entries["vae"] = _flux2_component_manifest_entry(
            package_path,
            "vae",
            mode,
            source_path=vae,
        )
        print(
            "SDMLX FLUX.2 Klein conversion: "
            f"reusing VAE from models/vae: {vae.name}",
            flush=True,
        )
    else:
        existing = package_path / "vae"
        if existing.exists() and (existing.is_dir() or is_flux2_vae_file(existing)):
            modes["vae"] = "package"
            entries["vae"] = _flux2_component_manifest_entry(package_path, "vae", "package")

    tokenizer = package_path / "tokenizer"
    if _flux2_tokenizer_has_chat_template(tokenizer):
        modes["tokenizer"] = "package"
        entries["tokenizer"] = _flux2_component_manifest_entry(package_path, "tokenizer", "package")
    else:
        donor = _flux2_find_tokenizer_donor(config_name, exclude=exclude)
        if donor is not None and (donor / "tokenizer").exists():
            mode = _flux2_copy_dir_into_package(donor / "tokenizer", tokenizer)
            modes["tokenizer"] = mode
            entries["tokenizer"] = _flux2_component_manifest_entry(
                package_path,
                "tokenizer",
                mode,
                source_path=donor / "tokenizer",
            )
            print(
                "SDMLX FLUX.2 Klein conversion: "
                f"copied tokenizer into package from {donor.name}",
                flush=True,
            )
        else:
            _flux2_download_tokenizer_assets(package_path, config_name)
            modes["tokenizer"] = "download"
            entries["tokenizer"] = _flux2_component_manifest_entry(package_path, "tokenizer", "download")

    donor: Path | None = None
    if ("text_encoder" not in modes or "vae" not in modes) and _flux2_legacy_component_fallback_enabled():
        donor = _flux2_find_tokenizer_donor(config_name, exclude=exclude)

    if "text_encoder" not in modes and donor is not None and (donor / "text_encoder").exists():
        _flux2_remove_package_component_path(package_path, "text_encoder")
        mode = "external_manifest"
        modes["text_encoder"] = f"legacy_{mode}"
        entries["text_encoder"] = _flux2_component_manifest_entry(
            package_path,
            "text_encoder",
            modes["text_encoder"],
            source_path=donor / "text_encoder",
        )
        print(
            "SDMLX FLUX.2 Klein conversion: "
            f"legacy text_encoder fallback from {donor.name}",
            flush=True,
        )

    if "vae" not in modes and donor is not None and (donor / "vae").exists():
        _flux2_remove_package_component_path(package_path, "vae")
        mode = "external_manifest"
        modes["vae"] = f"legacy_{mode}"
        entries["vae"] = _flux2_component_manifest_entry(
            package_path,
            "vae",
            modes["vae"],
            source_path=donor / "vae",
        )
        print(
            "SDMLX FLUX.2 Klein conversion: "
            f"legacy VAE fallback from {donor.name}",
            flush=True,
        )

    missing = [name for name in ("text_encoder", "tokenizer", "vae") if name not in modes]
    if missing:
        raise RuntimeError(
            "SDMLX FLUX.2 Klein conversion: missing required component(s): "
            f"{', '.join(missing)}. Put matching Qwen3 text encoders in models/text_encoders "
            "and a FLUX.2 VAE in models/vae, or enable the legacy component fallback during testing."
        )

    for optional in ("scheduler", "model_index.json", "README.md", "LICENSE.md"):
        source = None
        donor = _flux2_find_tokenizer_donor(config_name, exclude=exclude)
        if donor is not None:
            candidate = donor / optional
            if candidate.exists():
                source = candidate
        if source is None:
            continue
        destination = package_path / optional
        if destination.exists() or destination.is_symlink():
            continue
        if source.is_dir():
            _flux2_copy_dir_into_package(source, destination)
        else:
            _flux2_clone_or_copy(source, destination)

    return _Flux2MaterializedComponents(modes=modes, entries=entries)


def _flux2_manifest_path(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except Exception:
        return str(path)


def _flux2_is_cache_local_root(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        resolved = path.expanduser()
    parts = tuple(part.lower() for part in resolved.parts)
    name = resolved.name.lower()
    return "cache" in parts and name.startswith("flux2-klein") and name.endswith("-local")


def _is_flux2_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_transformer = (path / "transformer").exists() or any(path.glob("*.safetensors"))
    if not has_transformer:
        return False
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
                if is_flux2_manifest(manifest):
                    if not _flux2_tokenizer_has_chat_template(path / "tokenizer"):
                        return False
                    config_name = str(manifest.get("model_config") or _flux2_config_name_from_checkpoint(_flux2_transformer_file(path) or path))
                    model_config = _flux2_model_config_by_name(config_name)
                    if _flux2_package_components_match_config(path, model_config, manifest):
                        return True
                    return False
        except Exception:
            pass
    required = ("text_encoder", "vae", "tokenizer")
    if not all((path / name).exists() for name in required):
        return False
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                if is_flux2_manifest(json.load(handle)):
                    return True
        except Exception:
            pass
    text = str(path).lower()
    return "flux2" in text or "flux.2" in text or "klein" in text


def is_flux2_manifest(manifest: dict[str, Any] | None) -> bool:
    if not isinstance(manifest, dict):
        return False
    family = str(manifest.get("model_family") or manifest.get("base_model_family") or "").strip().lower()
    package_format = str(manifest.get("package_format") or manifest.get("format") or "").strip().lower()
    return family == FLUX2_MODEL_FAMILY or package_format == FLUX2_PACKAGE_FORMAT


def is_flux2_sdmlx_package(path: str | os.PathLike[str]) -> bool:
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        return False
    return is_flux2_manifest(manifest)


def _flux2_root_from_component_path(path: str | os.PathLike[str]) -> Path | None:
    component_path = Path(path).expanduser()
    candidates = [component_path]
    try:
        resolved = component_path.resolve()
        if resolved != component_path:
            candidates.append(resolved)
    except OSError:
        pass
    for candidate in candidates:
        for parent in (candidate.parent, *candidate.parents):
            if parent.name in {"text_encoder", "tokenizer", "vae"}:
                root = parent.parent
                if _is_flux2_root(root):
                    return root
            if _is_flux2_root(parent):
                return parent
    return None


def _flux2_standalone_text_encoder_dir(path: str | os.PathLike[str]) -> Path | None:
    component_path = Path(path).expanduser()
    candidates = [component_path]
    if component_path.is_file():
        candidates.append(component_path.parent)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if not (candidate / "config.json").is_file() or not (candidate / "model.safetensors.index.json").is_file():
            continue
        try:
            with (candidate / "config.json").open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception:
            continue
        architectures = " ".join(str(item) for item in config.get("architectures") or [])
        if str(config.get("model_type") or "").lower() == "qwen3" or "qwen3" in architectures.lower():
            return candidate
    return None


def _flux2_qwen3_text_encoder_info(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = Path(path).expanduser()
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return None
    try:
        with safe_open(str(path), framework="np") as handle:
            keys = set(handle.keys())
            if "model.embed_tokens.weight" not in keys or "model.norm.weight" not in keys:
                return None
            if "lm_head.weight" in keys or any(key.startswith("visual.") for key in keys):
                return None
            if not any(key.startswith("model.layers.0.") for key in keys):
                return None
            embed_shape = handle.get_slice("model.embed_tokens.weight").get_shape()
            hidden_size = int(embed_shape[1]) if len(embed_shape) == 2 else 0
            intermediate_size = 0
            gate_key = "model.layers.0.mlp.gate_proj.weight"
            if gate_key in keys:
                gate_shape = handle.get_slice(gate_key).get_shape()
                if len(gate_shape) == 2:
                    effective_gate_shape = _flux2_comfy_quant_effective_shape(handle, gate_key, gate_shape)
                    if hidden_size and int(effective_gate_shape[1]) != hidden_size:
                        return None
                    intermediate_size = int(effective_gate_shape[0])
            layer_count = 0
            for key in keys:
                match = re.match(r"model\.layers\.(\d+)\.", key)
                if match:
                    layer_count = max(layer_count, int(match.group(1)) + 1)
            return {
                "path": path,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "layer_count": layer_count,
            }
    except Exception:
        return None


def _flux2_comfy_quant_config(handle: Any, base_key: str) -> dict[str, Any] | None:
    quant_key = f"{base_key}.comfy_quant"
    try:
        keys = set(handle.keys())
    except Exception:
        keys = set()
    if quant_key not in keys:
        return None
    try:
        tensor = handle.get_tensor(quant_key)
        if hasattr(tensor, "numpy"):
            payload = tensor.numpy().tobytes()
        else:
            payload = np.asarray(tensor).tobytes()
        return json.loads(payload)
    except Exception:
        return None


def _flux2_comfy_quant_effective_shape(
    handle: Any,
    weight_key: str,
    stored_shape: tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    shape = tuple(int(value) for value in stored_shape)
    if len(shape) != 2:
        return shape
    config = _flux2_comfy_quant_config(handle, weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key)
    if not config:
        return shape
    if str(config.get("format") or "").lower() == "nvfp4":
        return (shape[0], shape[1] * 2)
    return shape


def is_flux2_text_encoder_file(path: str | os.PathLike[str]) -> bool:
    root = _flux2_root_from_component_path(path)
    component_path = Path(path).expanduser()
    return (
        root is not None and (component_path.name == "text_encoder" or component_path.parent.name == "text_encoder")
    ) or _flux2_standalone_text_encoder_dir(component_path) is not None or _flux2_qwen3_text_encoder_info(component_path) is not None


def flux2_clip_from_model_root(model_path: str | os.PathLike[str], name: str | None = None) -> dict[str, Any]:
    root = Path(model_path).expanduser()
    try:
        root = root.resolve()
    except OSError:
        pass
    if not _is_flux2_root(root):
        raise RuntimeError(f"SDMLX FLUX.2 Klein CLIP: incomplete model root: {model_path}")
    manifest = _flux2_read_package_manifest(root)
    text_encoder_component = _flux2_resolved_package_component_path(root, "text_encoder", manifest)
    if text_encoder_component is None:
        raise RuntimeError(f"SDMLX FLUX.2 Klein CLIP: no text encoder resolved for {model_path}")
    if text_encoder_component.is_dir():
        text_encoder_files = sorted(text_encoder_component.glob("*.safetensors"))
        text_encoder_path = text_encoder_files[0] if text_encoder_files else text_encoder_component
    else:
        text_encoder_path = text_encoder_component
    return {
        "type": FLUX2_MODEL_FAMILY,
        "cache_key": str(root),
        "model_path": str(root),
        "text_encoder_path": str(text_encoder_path),
        "tokenizer_path": str(root / "tokenizer"),
        "name": str(name or text_encoder_path.name),
        "unused": False,
    }


def flux2_clip_from_text_encoder(text_encoder_path: str | os.PathLike[str], name: str | None = None) -> dict[str, Any]:
    path = Path(text_encoder_path).expanduser()
    root = _flux2_root_from_component_path(path)
    if root is not None:
        return flux2_clip_from_model_root(root, name=name or path.name)
    standalone_info = _flux2_qwen3_text_encoder_info(path)
    if standalone_info is not None:
        try:
            resolved_path = path.resolve()
        except OSError:
            resolved_path = path
        return {
            "type": FLUX2_MODEL_FAMILY,
            "cache_key": str(resolved_path),
            "text_encoder_path": str(resolved_path),
            "name": str(name or path.name),
            "hidden_size": standalone_info.get("hidden_size") or 0,
            "intermediate_size": standalone_info.get("intermediate_size") or 0,
            "layer_count": standalone_info.get("layer_count") or 0,
            "unused": False,
        }
    standalone_dir = _flux2_standalone_text_encoder_dir(path)
    if standalone_dir is None:
        raise RuntimeError(
            "SDMLX FLUX.2 Klein CLIP: select a Qwen3 .safetensors text encoder or a text_encoder inside a FLUX.2 Klein .sdmlx/runtime root."
        )
    try:
        standalone_dir = standalone_dir.resolve()
    except OSError:
        pass
    clip = {
        "type": FLUX2_MODEL_FAMILY,
        "cache_key": str(standalone_dir),
        "text_encoder_path": str(standalone_dir),
        "name": str(name or standalone_dir.name),
        "unused": False,
    }
    config_path = standalone_dir / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        clip["hidden_size"] = int(config.get("hidden_size") or 0)
    except Exception:
        pass
    return clip


def _is_flux2_small_decoder_vae_file(path: str | os.PathLike[str]) -> bool:
    path = Path(path)
    if path.name == "full_encoder_small_decoder.safetensors":
        return True
    if path.name != "diffusion_pytorch_model.safetensors" or not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="np") as handle:
            keys = set(handle.keys())
        return (
            "bn.num_batches_tracked" in keys
            and (
                "decoder.mid.attn_1.q.weight" in keys
                or "decoder.mid_block.attentions.0.to_q.weight" in keys
            )
        )
    except Exception:
        return False


def _is_flux2_vae_weight_file(path: str | os.PathLike[str]) -> bool:
    path = Path(path).expanduser()
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return False
    try:
        with safe_open(str(path), framework="np") as handle:
            keys = set(handle.keys())
        return (
            "bn.num_batches_tracked" in keys
            and "encoder.conv_in.weight" in keys
            and "decoder.conv_in.weight" in keys
            and (
                "decoder.mid.attn_1.q.weight" in keys
                or "decoder.mid_block.attentions.0.to_q.weight" in keys
            )
        )
    except Exception:
        return False


def is_flux2_vae_file(path: str | os.PathLike[str]) -> bool:
    root = _flux2_root_from_component_path(path)
    component_path = Path(path).expanduser()
    if root is not None and (component_path.name == "vae" or component_path.parent.name == "vae"):
        return True
    return _is_flux2_small_decoder_vae_file(path) or _is_flux2_vae_weight_file(path)


def flux2_vae_from_file(
    vae_path: str | os.PathLike[str],
    name: str | None = None,
    model_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    logical_path = Path(vae_path).expanduser()
    root = Path(model_path).expanduser() if model_path is not None else _flux2_root_from_component_path(logical_path)
    path = logical_path
    if logical_path.is_file():
        try:
            path = logical_path.resolve()
        except OSError:
            pass
    if root is not None:
        try:
            root = root.resolve()
        except OSError:
            pass
    variant = _FLUX2_VAE_SMALL_DECODER if path.is_file() and _is_flux2_small_decoder_vae_file(path) else _FLUX2_VAE_STANDARD
    vae = {
        "type": FLUX2_MODEL_FAMILY,
        "cache_key": str(path),
        "vae_path": str(path),
        "vae_variant": variant,
        "name": str(name or path.name),
        "unused": False,
    }
    if root is not None:
        vae["model_path"] = str(root)
    return vae


def flux2_vae_from_model_root(model_path: str | os.PathLike[str], name: str | None = None) -> dict[str, Any]:
    root = Path(model_path).expanduser()
    try:
        root = root.resolve()
    except OSError:
        pass
    if not _is_flux2_root(root):
        raise RuntimeError(f"SDMLX FLUX.2 Klein VAE: incomplete model root: {model_path}")
    manifest = _flux2_read_package_manifest(root)
    package_vae = _flux2_resolved_package_component_path(root, "vae", manifest)
    if package_vae is None:
        package_vae = root / "vae"
    if package_vae.is_file() or package_vae.is_symlink():
        return flux2_vae_from_file(package_vae, name=name or package_vae.name, model_path=root)
    return flux2_vae_from_file(_resolve_flux2_small_decoder_vae(), name=name or "small_decoder", model_path=root)


def flux2_placeholders_from_model_root(model_path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    return flux2_clip_from_model_root(model_path), flux2_vae_from_model_root(model_path)


def _resolve_flux2_small_decoder_vae() -> Path:
    candidates = (
        "full_encoder_small_decoder.safetensors",
        "diffusion_pytorch_model.safetensors",
    )
    try:
        for name in candidates:
            path = folder_paths.get_full_path("vae", name)
            if path:
                return Path(path)
        for name in folder_paths.get_filename_list("vae"):
            if Path(name).name in candidates:
                path = folder_paths.get_full_path("vae", name)
                if path:
                    return Path(path)
    except Exception:
        pass

    models_dir = Path(getattr(folder_paths, "models_dir", SUITE_ROOT.parent / "models"))
    for name in candidates:
        path = models_dir / "vae" / name
        if path.exists():
            return path
    raise RuntimeError(
        "SDMLX FLUX.2 Klein Loader: small_decoder VAE requested, but no "
        "full_encoder_small_decoder.safetensors was found in ComfyUI models/vae."
    )


def _normalize_flux2_vae_variant(vae_variant: str | None) -> str:
    value = str(vae_variant or _FLUX2_VAE_STANDARD).strip().lower().replace("-", "_")
    if value in {"small", "small_decoder"}:
        return _FLUX2_VAE_SMALL_DECODER
    return _FLUX2_VAE_STANDARD


def _flux2_default_decode_tiling_config():
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.vae.tiling_config import TilingConfig

    decode_tiles_per_dim: int | None = None
    override = os.environ.get("SDMLX_FLUX2_VAE_DECODE_TILES")
    if override is not None:
        value = override.strip().lower()
        if value in {"0", "none", "off", "false", "no"}:
            decode_tiles_per_dim = None
        else:
            try:
                decode_tiles_per_dim = max(1, int(value))
            except ValueError:
                decode_tiles_per_dim = None
        print(f"SDMLX FLUX.2 diagnostic: VAE decode tiles={decode_tiles_per_dim or 'off'}")

    return TilingConfig(
        vae_decode_tiles_per_dim=decode_tiles_per_dim,
        vae_decode_overlap=8,
        vae_encode_tiled=False,
    )


def _flux2_fallback_decode_tiling_config():
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.vae.tiling_config import TilingConfig

    return TilingConfig(
        vae_decode_tiles_per_dim=2,
        vae_decode_overlap=8,
        vae_encode_tiled=False,
    )


def _apply_flux2_decode_tiling_default(model: Any) -> None:
    if getattr(model, "tiling_config", None) is not None:
        return
    model.tiling_config = _flux2_default_decode_tiling_config()


def _is_flux2_distilled_model(sdmlx_model: dict[str, Any]) -> bool:
    model_name = str(sdmlx_model.get("model_name") or "").lower()
    model_config = str(sdmlx_model.get("model_config") or "").lower()
    return "base" not in model_name and "base" not in model_config


@contextmanager
def _flux2_sampling_acceleration_env(sdmlx_model: dict[str, Any], profile: str = "edit"):
    old = {key: os.environ.get(key) for key in _FLUX2_ACCEL_ENV_KEYS}
    for key in _FLUX2_ACCEL_ENV_KEYS:
        os.environ.pop(key, None)
    if profile == "txt2img":
        os.environ["SDMLX_FLUX2_MIXED_FP16"] = "1"
        os.environ["SDMLX_FLUX2_SINGLE_TO_OUT_FP16_PRETRANSPOSE"] = "1"
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _flux2_sampling_profile_for_model(sdmlx_model: dict[str, Any], references: list[dict[str, Any]]) -> str:
    if references:
        return "edit"
    quant_kind = _flux2_quant_contract_kind(sdmlx_model.get("transformer_quant_contract"))
    if quant_kind == _FLUX2_QUANT_DENSE_BF16:
        return "txt2img"
    return "edit"


def _flux2_packed_dims(model_config) -> tuple[int, int, int, int, int]:
    overrides = getattr(model_config, "transformer_overrides", {}) or {}
    heads = int(overrides.get("num_attention_heads", 24))
    hidden = heads * _FLUX2_HEAD_DIM
    mlp = hidden * 3
    double_blocks = int(overrides.get("num_layers", 5))
    single_blocks = int(overrides.get("num_single_layers", 20))
    joint_attention_dim = int(overrides.get("joint_attention_dim", hidden + 4608))
    return hidden, mlp, double_blocks, single_blocks, joint_attention_dim


def _flux2_packed_params(model_config) -> list[_Flux2PackedParam]:
    hidden, mlp, double_blocks, single_blocks, joint_attention_dim = _flux2_packed_dims(model_config)
    inner = hidden
    params: list[_Flux2PackedParam] = [
        _Flux2PackedParam("x_embedder.weight", "img_in.weight", (hidden, 128)),
        _Flux2PackedParam("context_embedder.weight", "txt_in.weight", (hidden, joint_attention_dim)),
        _Flux2PackedParam("time_guidance_embed.linear_1.weight", "time_in.in_layer.weight", (hidden, 256)),
        _Flux2PackedParam("time_guidance_embed.linear_2.weight", "time_in.out_layer.weight", (hidden, hidden)),
        _Flux2PackedParam(
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_img.lin.weight",
            (hidden * 6, hidden),
        ),
        _Flux2PackedParam(
            "double_stream_modulation_txt.linear.weight",
            "double_stream_modulation_txt.lin.weight",
            (hidden * 6, hidden),
        ),
        _Flux2PackedParam(
            "single_stream_modulation.linear.weight",
            "single_stream_modulation.lin.weight",
            (hidden * 3, hidden),
        ),
        _Flux2PackedParam("norm_out.linear.weight", "final_layer.adaLN_modulation.1.weight", (hidden * 2, hidden)),
        _Flux2PackedParam("proj_out.weight", "final_layer.linear.weight", (128, hidden)),
    ]

    for block in range(double_blocks):
        img_qkv = f"double_blocks.{block}.img_attn.qkv.weight"
        txt_qkv = f"double_blocks.{block}.txt_attn.qkv.weight"
        for part, start in (("to_q", 0), ("to_k", inner), ("to_v", inner * 2)):
            params.append(
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.attn.{part}.weight",
                    img_qkv,
                    (inner, hidden),
                    (start, inner),
                )
            )
        for part, start in (("add_q_proj", 0), ("add_k_proj", inner), ("add_v_proj", inner * 2)):
            params.append(
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.attn.{part}.weight",
                    txt_qkv,
                    (inner, hidden),
                    (start, inner),
                )
            )
        params.extend(
            [
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.attn.to_out.weight",
                    f"double_blocks.{block}.img_attn.proj.weight",
                    (hidden, inner),
                ),
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.attn.to_add_out.weight",
                    f"double_blocks.{block}.txt_attn.proj.weight",
                    (hidden, inner),
                ),
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.attn.norm_q.weight",
                    f"double_blocks.{block}.img_attn.norm.query_norm.scale",
                    (_FLUX2_HEAD_DIM,),
                ),
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.attn.norm_k.weight",
                    f"double_blocks.{block}.img_attn.norm.key_norm.scale",
                    (_FLUX2_HEAD_DIM,),
                ),
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.attn.norm_added_q.weight",
                    f"double_blocks.{block}.txt_attn.norm.query_norm.scale",
                    (_FLUX2_HEAD_DIM,),
                ),
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.attn.norm_added_k.weight",
                    f"double_blocks.{block}.txt_attn.norm.key_norm.scale",
                    (_FLUX2_HEAD_DIM,),
                ),
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.ff.linear_in.weight",
                    f"double_blocks.{block}.img_mlp.0.weight",
                    (mlp * 2, hidden),
                ),
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.ff.linear_out.weight",
                    f"double_blocks.{block}.img_mlp.2.weight",
                    (hidden, mlp),
                ),
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.ff_context.linear_in.weight",
                    f"double_blocks.{block}.txt_mlp.0.weight",
                    (mlp * 2, hidden),
                ),
                _Flux2PackedParam(
                    f"transformer_blocks.{block}.ff_context.linear_out.weight",
                    f"double_blocks.{block}.txt_mlp.2.weight",
                    (hidden, mlp),
                ),
            ]
        )

    for block in range(single_blocks):
        params.extend(
            [
                _Flux2PackedParam(
                    f"single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.weight",
                    f"single_blocks.{block}.linear1.weight",
                    (inner * 3 + mlp * 2, hidden),
                ),
                _Flux2PackedParam(
                    f"single_transformer_blocks.{block}.attn.norm_q.weight",
                    f"single_blocks.{block}.norm.query_norm.scale",
                    (_FLUX2_HEAD_DIM,),
                ),
                _Flux2PackedParam(
                    f"single_transformer_blocks.{block}.attn.norm_k.weight",
                    f"single_blocks.{block}.norm.key_norm.scale",
                    (_FLUX2_HEAD_DIM,),
                ),
                _Flux2PackedParam(
                    f"single_transformer_blocks.{block}.attn.to_out.weight",
                    f"single_blocks.{block}.linear2.weight",
                    (hidden, inner + mlp),
                ),
            ]
        )
    return params


def _flux2_transformer_file(root: Path) -> Path | None:
    candidates = [
        root / "transformer" / "diffusion_pytorch_model.safetensors",
        root / "flux-2-klein-4b.safetensors",
        root / "flux-2-klein-9b.safetensors",
        root / "flux-2-klein-9b-kv.safetensors",
        root / "flux-2-klein-9b-fp8.safetensors",
        root / "flux-2-klein-base-4b.safetensors",
        root / "flux-2-klein-base-9b.safetensors",
    ]
    candidates.extend(sorted(root.glob("*.safetensors")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


_FLUX2_CHECKPOINT_KEY_PREFIXES = ("", "model.diffusion_model.", "diffusion_model.")


def _flux2_resolve_source_key(key_set: set[str], source_key: str) -> str | None:
    for prefix in _FLUX2_CHECKPOINT_KEY_PREFIXES:
        candidate = f"{prefix}{source_key}"
        if candidate in key_set:
            return candidate
        if source_key.endswith(".scale") and (".query_norm." in source_key or ".key_norm." in source_key):
            norm_weight_candidate = f"{prefix}{source_key[:-len('.scale')]}.weight"
            if norm_weight_candidate in key_set:
                return norm_weight_candidate
    return None


def _is_packed_flux2_transformer(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
    except Exception:
        return False
    return (
        _flux2_resolve_source_key(keys, "double_blocks.0.img_attn.qkv.weight") is not None
        and _flux2_resolve_source_key(keys, "single_blocks.0.linear1.weight") is not None
        and _flux2_resolve_source_key(keys, "final_layer.adaLN_modulation.1.weight") is not None
    )


def _flux2_safetensors_header_index(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        header = comfy.utils.safetensors_header(str(path))
    except Exception:
        header = None
    if not header:
        return {}, {}
    try:
        payload = json.loads(header)
    except Exception:
        return {}, {}
    metadata = dict(payload.pop("__metadata__", {}) or {})
    tensors = {
        key: value
        for key, value in payload.items()
        if isinstance(value, dict) and isinstance(value.get("shape"), list)
    }
    return tensors, metadata


def _flux2_dtype_is_fp8(dtype: Any) -> bool:
    text = str(dtype or "").upper()
    return "F8" in text or "FLOAT8" in text


def _flux2_dtype_is_bf16(dtype: Any) -> bool:
    text = str(dtype or "").upper()
    return text in {"BF16", "BFLOAT16", "TORCH.BFLOAT16"}


def _flux2_comfy_quant_formats(path: Path, comfy_quant_keys: list[str]) -> list[str]:
    if not comfy_quant_keys:
        return []
    formats: set[str] = set()
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in comfy_quant_keys:
                base_key = key[: -len(".comfy_quant")]
                config = _flux2_comfy_quant_config(handle, base_key)
                if config:
                    quant_format = str(config.get("format") or "").strip().lower()
                    if quant_format:
                        formats.add(quant_format)
    except Exception:
        return []
    return sorted(formats)


def _flux2_runtime_policy_for_quant_kind(kind: str) -> str:
    if kind == _FLUX2_QUANT_DENSE_BF16:
        return "fast_dense"
    if kind == _FLUX2_QUANT_SCALED_FP8:
        return "scaled_fp8_dequant_q8"
    if kind in {_FLUX2_QUANT_COMFY, _FLUX2_QUANT_COMFY_MXFP8}:
        return "comfy_quant_dequant_q8"
    if kind == _FLUX2_QUANT_RAW_FP8:
        return "raw_fp8_unscaled_dense_bf16"
    return "default"


def _flux2_inspect_transformer_quant_contract(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "kind": _FLUX2_QUANT_UNKNOWN,
            "runtime_policy": _flux2_runtime_policy_for_quant_kind(_FLUX2_QUANT_UNKNOWN),
            "counts": {},
        }

    tensors, metadata = _flux2_safetensors_header_index(path)
    if not tensors:
        return {
            "kind": _FLUX2_QUANT_UNKNOWN,
            "runtime_policy": _flux2_runtime_policy_for_quant_kind(_FLUX2_QUANT_UNKNOWN),
            "counts": {},
        }

    dtype_counts: dict[str, int] = {}
    two_d_weight_count = 0
    two_d_fp8_count = 0
    two_d_bf16_count = 0
    comfy_quant_keys: list[str] = []
    weight_scale_count = 0
    input_scale_count = 0
    scaled_fp8_marker_count = 0

    for key, spec in tensors.items():
        dtype = str(spec.get("dtype") or "")
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        shape = spec.get("shape") or []
        if key.endswith(".comfy_quant"):
            comfy_quant_keys.append(key)
        if key.endswith((".weight_scale", ".scale_weight")):
            weight_scale_count += 1
        if key.endswith((".input_scale", ".scale_input")):
            input_scale_count += 1
        if key.endswith("scaled_fp8"):
            scaled_fp8_marker_count += 1
        if key.endswith(".weight") and len(shape) == 2:
            two_d_weight_count += 1
            if _flux2_dtype_is_fp8(dtype):
                two_d_fp8_count += 1
            if _flux2_dtype_is_bf16(dtype):
                two_d_bf16_count += 1

    comfy_formats = _flux2_comfy_quant_formats(path, comfy_quant_keys)
    has_quant_metadata = "_quantization_metadata" in metadata
    has_scale_siblings = weight_scale_count > 0 or input_scale_count > 0

    if two_d_fp8_count == 0 and two_d_weight_count > 0 and two_d_bf16_count == two_d_weight_count:
        kind = _FLUX2_QUANT_DENSE_BF16
    elif comfy_quant_keys:
        kind = _FLUX2_QUANT_COMFY_MXFP8 if comfy_formats == ["mxfp8"] else _FLUX2_QUANT_COMFY
    elif two_d_fp8_count > 0 and (has_scale_siblings or has_quant_metadata or scaled_fp8_marker_count > 0):
        kind = _FLUX2_QUANT_SCALED_FP8
    elif two_d_fp8_count > 0:
        kind = _FLUX2_QUANT_RAW_FP8
    elif two_d_weight_count > 0:
        kind = _FLUX2_QUANT_MIXED_DENSE
    else:
        kind = _FLUX2_QUANT_UNKNOWN

    counts = {
        "tensors": len(tensors),
        "two_dimensional_weights": two_d_weight_count,
        "two_dimensional_fp8_weights": two_d_fp8_count,
        "two_dimensional_bf16_weights": two_d_bf16_count,
        "comfy_quant": len(comfy_quant_keys),
        "weight_scale": weight_scale_count,
        "input_scale": input_scale_count,
        "scaled_fp8_markers": scaled_fp8_marker_count,
    }
    contract = {
        "kind": kind,
        "runtime_policy": _flux2_runtime_policy_for_quant_kind(kind),
        "counts": counts,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "has_quantization_metadata": bool(has_quant_metadata),
    }
    if comfy_formats:
        contract["comfy_quant_formats"] = comfy_formats
        contract["format"] = comfy_formats[0] if len(comfy_formats) == 1 else "+".join(comfy_formats)
    return contract


def _flux2_quant_contract_kind(contract: dict[str, Any] | None) -> str:
    return str((contract or {}).get("kind") or _FLUX2_QUANT_UNKNOWN)


def _flux2_quant_contract_uses_dense_compat(contract: dict[str, Any] | None) -> bool:
    return False


def _flux2_quant_contract_summary(contract: dict[str, Any] | None) -> str:
    contract = contract or {}
    counts = dict(contract.get("counts") or {})
    return (
        f"{_flux2_quant_contract_kind(contract)}"
        f" policy={contract.get('runtime_policy') or 'default'}"
        f" 2d={int(counts.get('two_dimensional_weights') or 0)}"
        f" fp8={int(counts.get('two_dimensional_fp8_weights') or 0)}"
        f" bf16={int(counts.get('two_dimensional_bf16_weights') or 0)}"
        f" comfy_quant={int(counts.get('comfy_quant') or 0)}"
        f" scales={int(counts.get('weight_scale') or 0)}"
        f" input_scales={int(counts.get('input_scale') or 0)}"
    )


def _flux2_refresh_manifest_transformer_quant_contract(
    package_path: Path,
    manifest: dict[str, Any],
    *,
    write: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = dict(manifest or {})
    transformer_path = _flux2_transformer_file(package_path)
    contract = _flux2_inspect_transformer_quant_contract(transformer_path)
    if manifest.get("transformer_quant_contract") != contract:
        manifest["transformer_quant_contract"] = contract
        if write:
            _write_flux2_manifest(package_path, manifest)
    return manifest, contract


def _flux2_log_quant_contract_once(package_path: Path, contract: dict[str, Any] | None) -> None:
    try:
        key = str(package_path.resolve())
    except Exception:
        key = str(package_path)
    if key in _FLUX2_QUANT_CONTRACT_LOGGED:
        return
    _FLUX2_QUANT_CONTRACT_LOGGED.add(key)
    print(
        f"SDMLX FLUX.2 Klein: loaded SDMLX container '{package_path.name}'",
        flush=True,
    )
    _flux2_log(
        "SDMLX FLUX.2 Klein package: "
        f"quant-contract: {_flux2_quant_contract_summary(contract)}",
        verbose=True,
    )


def _flux2_transformer_hidden_from_file(path: Path) -> int | None:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for key in (
                "double_blocks.0.img_attn.qkv.weight",
                "transformer_blocks.0.attn.to_q.weight",
                "x_embedder.weight",
                "img_in.weight",
            ):
                resolved_key = _flux2_resolve_source_key(keys, key)
                if resolved_key is None:
                    continue
                shape = tuple(handle.get_tensor(resolved_key).shape)
                if len(shape) != 2:
                    continue
                if key.endswith("qkv.weight"):
                    return int(shape[1])
                return int(shape[0])
    except Exception:
        return None
    return None


def _flux2_block_counts_from_file(path: Path) -> tuple[int | None, int | None]:
    double_blocks: set[int] = set()
    single_blocks: set[int] = set()
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                for prefix in _FLUX2_CHECKPOINT_KEY_PREFIXES[1:]:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        break
                match = re.match(r"double_blocks\.(\d+)\.", key)
                if match:
                    double_blocks.add(int(match.group(1)))
                    continue
                match = re.match(r"single_blocks\.(\d+)\.", key)
                if match:
                    single_blocks.add(int(match.group(1)))
                    continue
                match = re.match(r"transformer_blocks\.(\d+)\.", key)
                if match:
                    double_blocks.add(int(match.group(1)))
                    continue
                match = re.match(r"single_transformer_blocks\.(\d+)\.", key)
                if match:
                    single_blocks.add(int(match.group(1)))
    except Exception:
        return None, None
    return (max(double_blocks) + 1 if double_blocks else None, max(single_blocks) + 1 if single_blocks else None)


def _flux2_config_name_from_checkpoint(path: Path) -> str:
    lowered = path.name.lower()
    hidden = _flux2_transformer_hidden_from_file(path)
    double_blocks, single_blocks = _flux2_block_counts_from_file(path)
    is_9b = hidden == 4096 or double_blocks == 8 or single_blocks == 24 or "9b" in lowered
    is_base = "base" in lowered
    is_kv = "kv" in lowered
    if is_9b and is_base:
        return "flux2-klein-base-9b"
    if is_9b and is_kv:
        return "flux2-klein-9b-kv"
    if is_9b:
        return "flux2-klein-9b"
    if is_base:
        return "flux2-klein-base-4b"
    return "flux2-klein-4b"


def is_flux2_checkpoint_file(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.suffix.lower() != ".safetensors":
        return False
    lowered = candidate.name.lower()
    try:
        with safe_open(str(candidate), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = set(handle.keys())
    except Exception:
        return False
    architecture = " ".join(
        str(metadata.get(key, ""))
        for key in ("modelspec.architecture", "modelspec.title", "architecture", "model_type")
    ).lower()
    if "flux.1" in architecture or "flux1" in architecture:
        return False
    if "flux.2" in architecture or "flux2" in architecture or "klein" in architecture:
        return True
    if "flux" in lowered and ("klein" in lowered or "flux2" in lowered or "flux.2" in lowered):
        return True
    hidden = _flux2_transformer_hidden_from_file(candidate)
    double_blocks, single_blocks = _flux2_block_counts_from_file(candidate)
    return (hidden, double_blocks, single_blocks) in {(3072, 5, 20), (4096, 8, 24)}


def _flux2_config_key(config_name: str) -> str:
    name = str(config_name or "").strip().lower()
    name = name.rsplit("/", 1)[-1]
    name = name.replace("flux.2", "flux2")
    name = name.replace("_", "-")
    return name


def _flux2_model_config_by_name(config_name: str):
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.config import ModelConfig

    name = _flux2_config_key(config_name)
    if name == "flux2-klein-base-9b":
        return ModelConfig.flux2_klein_base_9b()
    if name == "flux2-klein-base-4b":
        return ModelConfig.flux2_klein_base_4b()
    if name == "flux2-klein-9b-kv":
        return ModelConfig.flux2_klein_9b_kv()
    if name == "flux2-klein-9b":
        return ModelConfig.flux2_klein_9b()
    return ModelConfig.flux2_klein_4b()


def _find_flux2_component_donor(config_name: str, *, exclude: Path | None = None) -> Path | None:
    wanted = _flux2_config_key(config_name)
    model_config = _flux2_model_config_by_name(config_name)
    roots = _find_flux2_roots()
    excluded: Path | None = None
    if exclude is not None:
        try:
            excluded = exclude.expanduser().resolve()
        except Exception:
            excluded = exclude.expanduser()

    def complete(root: Path) -> bool:
        return (
            all((root / name).exists() for name in ("text_encoder", "tokenizer", "vae"))
            and _flux2_tokenizer_has_chat_template(root / "tokenizer")
        )

    def usable(root: Path) -> bool:
        try:
            resolved = root.expanduser().resolve()
        except Exception:
            resolved = root.expanduser()
        if excluded is not None and resolved == excluded:
            return False
        return complete(root) and _flux2_text_encoder_matches_config(root / "text_encoder", model_config)

    for root in roots:
        if _flux2_is_cache_local_root(root):
            continue
        if not usable(root):
            continue
        try:
            if _flux2_config_key(_model_config_for_root(root).model_name) == wanted:
                return root
        except Exception:
            continue
    for root in roots:
        if _flux2_is_cache_local_root(root):
            continue
        if usable(root):
            return root
    if not _flux2_legacy_component_fallback_enabled():
        return None
    try:
        from .sdmlx_asset_registry import find_flux2_component_root

        registry_root = find_flux2_component_root(config_name, exclude=excluded)
    except Exception:
        registry_root = None
    if registry_root is not None and usable(registry_root):
        return registry_root
    return None


def _write_flux2_manifest(package_path: Path, manifest: dict[str, Any]) -> None:
    package_path.mkdir(parents=True, exist_ok=True)
    with (package_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _iter_flux2_sdmlx_packages() -> list[Path]:
    packages: list[Path] = []
    seen: set[str] = set()
    skip_dirs = {"cache", "registry", "AccelerationPatches", "SpeedPatches"}

    def visit(root: Path) -> None:
        try:
            entries = sorted(root.iterdir(), key=lambda path: path.name.lower())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in skip_dirs:
                continue
            try:
                key = str(entry.resolve())
            except Exception:
                key = str(entry)
            if entry.name.endswith(".sdmlx"):
                if key not in seen:
                    seen.add(key)
                    packages.append(entry)
                continue
            visit(entry)

    for root in _sdmlx_model_roots():
        visit(root)
    return packages


def _flux2_package_label(package_path: Path) -> str:
    for root in _sdmlx_model_roots():
        try:
            return package_path.relative_to(root).as_posix()
        except ValueError:
            continue
    return package_path.name


def _flux2_try_existing_checkpoint_package(package_path: Path, checkpoint: Path, identity: dict[str, Any]) -> Path | None:
    manifest_path = package_path / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        config_name = str(manifest.get("model_config") or _flux2_config_name_from_checkpoint(checkpoint))
        model_config = _flux2_model_config_by_name(config_name)
        if (
            is_flux2_manifest(manifest)
            and str(manifest.get("source_content_digest") or "") == identity["source_content_digest"]
            and _flux2_transformer_file(package_path) is not None
        ):
            if (
                not _flux2_package_components_match_config(package_path, model_config, manifest)
                or _flux2_package_needs_component_layout_refresh(package_path, config_name, model_config)
            ):
                print(
                    "SDMLX FLUX.2 Klein conversion: "
                    f"repairing shared components for {_flux2_package_label(package_path)}",
                    flush=True,
                )
                materialized = _flux2_materialize_package_components(
                    package_path,
                    config_name,
                    model_config,
                    exclude=package_path,
                )
                components = dict(manifest.get("components") or {})
                components.update(materialized.entries)
                manifest["components"] = components
                manifest["component_modes"] = {
                    **dict(manifest.get("component_modes") or {}),
                    **materialized.modes,
                }
                _write_flux2_manifest(package_path, manifest)
            if _flux2_package_components_match_config(package_path, model_config, manifest):
                manifest, _quant_contract = _flux2_refresh_manifest_transformer_quant_contract(package_path, manifest)
                print(
                    "SDMLX FLUX.2 Klein conversion: "
                    f"cache hit {checkpoint.name} -> {_flux2_package_label(package_path)}",
                    flush=True,
                )
                return package_path
            print(
                "SDMLX FLUX.2 Klein conversion: "
                f"cache components for {_flux2_package_label(package_path)} do not match {model_config.model_name}; rebuilding",
                flush=True,
            )
    except Exception:
        return None
    return None


def _flux2_package_from_checkpoint(path: str | os.PathLike[str]) -> Path:
    start_s = time.perf_counter()
    checkpoint = Path(path).expanduser().resolve()
    if not is_flux2_checkpoint_file(checkpoint):
        raise RuntimeError(f"SDMLX FLUX.2 Klein: unsupported checkpoint file: {checkpoint.name}")

    identity = _flux2_file_identity(checkpoint)
    package_path = _flux2_primary_sdmlx_root() / f"{_flux2_safe_package_name(checkpoint)}.sdmlx"
    existing_package = _flux2_try_existing_checkpoint_package(package_path, checkpoint, identity)
    if existing_package is not None:
        return existing_package

    try:
        default_resolved = package_path.resolve()
    except Exception:
        default_resolved = package_path
    for candidate in _iter_flux2_sdmlx_packages():
        try:
            candidate_resolved = candidate.resolve()
        except Exception:
            candidate_resolved = candidate
        if candidate_resolved == default_resolved:
            continue
        existing_package = _flux2_try_existing_checkpoint_package(candidate, checkpoint, identity)
        if existing_package is not None:
            return existing_package

    config_name = _flux2_config_name_from_checkpoint(checkpoint)
    model_config = _flux2_model_config_by_name(config_name)
    print(
        "SDMLX FLUX.2 Klein conversion: "
        f"preparing {checkpoint.name} as {model_config.model_name} -> {package_path.name}",
        flush=True,
    )
    package_path.mkdir(parents=True, exist_ok=True)
    transformer_name = checkpoint.name
    copy_mode = _flux2_clone_or_copy(checkpoint, package_path / transformer_name)
    print(
        "SDMLX FLUX.2 Klein conversion: "
        f"transformer {copy_mode} complete ({checkpoint.stat().st_size / (1024 ** 3):.2f} GiB)",
        flush=True,
    )
    transformer_quant_contract = _flux2_inspect_transformer_quant_contract(package_path / transformer_name)

    materialized = _flux2_materialize_package_components(
        package_path,
        config_name,
        model_config,
        exclude=package_path,
    )
    component_modes: dict[str, str] = {"transformer": copy_mode, **materialized.modes}

    manifest = {
        "package_format": FLUX2_PACKAGE_FORMAT,
        "model_family": FLUX2_MODEL_FAMILY,
        "model_config": model_config.model_name,
        "model_version": "flux2-klein",
        "runtime": "sdmlx_qwen_native.flux2",
        "materialized": True,
        "model_path": ".",
        **identity,
        "components": {
            "transformer": {
                "storage": "package",
                "path": transformer_name,
                "bytes": int((package_path / transformer_name).stat().st_size),
                "copy_mode": copy_mode,
            },
            **materialized.entries,
        },
        "component_modes": component_modes,
        "transformer_quant_contract": transformer_quant_contract,
        "recommendations": {
            "steps": 20 if "base" in model_config.model_name else 4,
            "guidance": 5.0 if "base" in model_config.model_name else 1.0,
            "scheduler": "flow_match_euler_discrete",
            "flow_shift": _flux2_default_flow_shift_for_model_config(model_config),
        },
    }
    _write_flux2_manifest(package_path, manifest)
    print(
        "SDMLX FLUX.2 Klein conversion: "
        f"done in {time.perf_counter() - start_s:.2f}s -> {package_path}",
        flush=True,
    )
    return package_path


def _flux2_default_flow_shift_for_model_config(model_config: Any) -> float | None:
    model_name = str(getattr(model_config, "model_name", "") or "").lower()
    return None if "base" in model_name else 3.0


def flux2_model_from_manifest(package_path: str | os.PathLike[str], manifest: dict[str, Any], preload: bool = False):
    package_root = Path(package_path).expanduser().resolve()
    manifest = dict(manifest or {})
    config_name = str(manifest.get("model_config") or _flux2_config_name_from_checkpoint(_flux2_transformer_file(package_root) or package_root))
    model_config = _flux2_model_config_by_name(config_name)
    vae_variant = _FLUX2_VAE_SMALL_DECODER
    vae_path = ""
    if not _is_flux2_root(package_root):
        if _flux2_transformer_file(package_root) is None:
            raise RuntimeError(
                "SDMLX FLUX.2 Klein package is not a runnable runtime root yet. "
                "Missing: transformer. Rebuild the package from the original checkpoint."
            )
        print(
            "SDMLX FLUX.2 Klein package: "
            f"materializing missing components for {package_root.name}",
            flush=True,
        )
        materialized = _flux2_materialize_package_components(
            package_root,
            config_name,
            model_config,
            exclude=package_root,
        )
        components = dict(manifest.get("components") or {})
        components.update(materialized.entries)
        manifest["components"] = components
        manifest["component_modes"] = {
            **dict(manifest.get("component_modes") or {}),
            **materialized.modes,
        }
        _write_flux2_manifest(package_root, manifest)
        if not _is_flux2_root(package_root):
            missing = [
                name
                for name in ("text_encoder", "tokenizer", "vae")
                if (
                    not _flux2_tokenizer_has_chat_template(package_root / "tokenizer")
                    if name == "tokenizer"
                    else _flux2_resolved_package_component_path(package_root, name, manifest) is None
                )
            ]
            raise RuntimeError(
                "SDMLX FLUX.2 Klein package is not a runnable runtime root yet. "
                f"Missing: {', '.join(missing)}. Rebuild the package from the original checkpoint."
            )
    if not _flux2_package_components_match_config(package_root, model_config, manifest):
        print(
            "SDMLX FLUX.2 Klein package: "
            f"repairing shared components for {package_root.name}",
            flush=True,
        )
        materialized = _flux2_materialize_package_components(
            package_root,
            config_name,
            model_config,
            exclude=package_root,
        )
        components = dict(manifest.get("components") or {})
        components.update(materialized.entries)
        manifest["components"] = components
        manifest["component_modes"] = {
            **dict(manifest.get("component_modes") or {}),
            **materialized.modes,
        }
        _write_flux2_manifest(package_root, manifest)
        if not _flux2_package_components_match_config(package_root, model_config, manifest):
            text_encoder = _flux2_resolved_package_component_path(package_root, "text_encoder", manifest)
            actual_hidden = _flux2_text_encoder_hidden_size(text_encoder) if text_encoder is not None else None
            expected_hidden = int((model_config.text_encoder_overrides or {}).get("hidden_size") or 0)
            tokenizer_ok = _flux2_tokenizer_has_chat_template(package_root / "tokenizer")
            raise RuntimeError(
                "SDMLX FLUX.2 Klein package has incompatible shared components. "
                f"{package_root.name} uses text_encoder hidden_size={actual_hidden}, "
                f"but {model_config.model_name} expects hidden_size={expected_hidden}; "
                f"tokenizer_chat_template={tokenizer_ok}. "
                "Reload the original checkpoint once so SDMLX can repair/rebuild the package."
            )
    elif _flux2_package_needs_component_layout_refresh(package_root, config_name, model_config):
        print(
            "SDMLX FLUX.2 Klein package: "
            f"refreshing component layout for {package_root.name}",
            flush=True,
        )
        materialized = _flux2_materialize_package_components(
            package_root,
            config_name,
            model_config,
            exclude=package_root,
        )
        components = dict(manifest.get("components") or {})
        components.update(materialized.entries)
        manifest["components"] = components
        manifest["component_modes"] = {
            **dict(manifest.get("component_modes") or {}),
            **materialized.modes,
        }
        _write_flux2_manifest(package_root, manifest)
    manifest, transformer_quant_contract = _flux2_refresh_manifest_transformer_quant_contract(package_root, manifest)
    _flux2_log_quant_contract_once(package_root, transformer_quant_contract)
    default_flow_shift = _flux2_default_flow_shift_for_model_config(model_config)
    recommendations = dict(manifest.get("recommendations") or {})
    if recommendations.get("flow_shift") != default_flow_shift:
        recommendations["flow_shift"] = default_flow_shift
        manifest["recommendations"] = recommendations
        _write_flux2_manifest(package_root, manifest)
    clip_placeholder, vae_placeholder = flux2_placeholders_from_model_root(package_root)
    vae_variant = _normalize_flux2_vae_variant(vae_placeholder.get("vae_variant"))
    vae_path = str(vae_placeholder.get("vae_path") or "")
    clip_path = str(clip_placeholder.get("text_encoder_path") or "")
    tokenizer_path = str(clip_placeholder.get("tokenizer_path") or "")
    manifest, _prepared_entry_changed = _flux2_refresh_package_text_encoder_cache_manifest_entry(
        package_root,
        manifest,
        clip_path,
        model_config,
        transformer_quant_contract,
    )
    model = {
        "model_family": FLUX2_MODEL_FAMILY,
        "model_path": str(package_root),
        "model_name": package_root.name,
        "model_config": model_config.model_name,
        "package_path": str(package_root),
        "vae_variant": vae_variant,
        "vae_path": vae_path,
        "clip_path": clip_path,
        "tokenizer_path": tokenizer_path,
        "packed_transformer": bool(_is_packed_flux2_transformer(_flux2_transformer_file(package_root))),
        "transformer_quant_contract": transformer_quant_contract,
        "default_steps": 20 if "base" in model_config.model_name else 4,
        "default_guidance": 5.0 if "base" in model_config.model_name else 1.0,
        "default_flow_shift": default_flow_shift,
        "loras": list(manifest.get("loras") or []),
        "recommendations": recommendations,
    }
    if preload:
        _load_flux2_model(
            package_root,
            model_config.model_name,
            vae_variant=vae_variant,
            vae_path=vae_path,
            clip_path=clip_path,
            tokenizer_path=tokenizer_path,
        )
    return model, clip_placeholder, vae_placeholder


def flux2_model_from_checkpoint(path: str | os.PathLike[str], preload: bool = False, name: str | None = None):
    package_path = _flux2_package_from_checkpoint(path)
    with (package_path / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    model, _clip_placeholder, _vae_placeholder = flux2_model_from_manifest(package_path, manifest, preload=preload)
    if name:
        model["model_name"] = str(name)
    return model


def _build_flux2_packed_transformer_weights(
    path: Path,
    model_config,
    quant_contract: dict[str, Any] | None = None,
) -> _Flux2PackedWeights:
    quant_contract = quant_contract or _flux2_inspect_transformer_quant_contract(path)
    dense_compat = _flux2_quant_contract_uses_dense_compat(quant_contract)
    raw_fp8_dense_dtype = _flux2_raw_fp8_dense_dtype(quant_contract)
    flat: list[tuple[str, mx.array]] = []
    quantized_modules: set[str] = set()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        key_set = set(handle.keys())
        for param in _flux2_packed_params(model_config):
            source_key = _flux2_resolve_source_key(key_set, param.source)
            if source_key is None:
                raise KeyError(f"FLUX.2 packed transformer tensor not found: {param.source}")
            tensor = handle.get_tensor(source_key)
            is_fp8 = "float8" in str(tensor.dtype)
            is_comfy_quant_weight = (
                param.source.endswith(".weight")
                and f"{source_key[: -len('.weight')]}.comfy_quant" in key_set
            )
            if is_comfy_quant_weight:
                tensor = _flux2_dequantize_comfy_quant_weight(
                    handle,
                    source_key,
                    tensor,
                    model_config,
                    context="packed transformer",
                )
                tensor = tensor.to(torch.float16)
            elif "float8" in str(tensor.dtype):
                tensor = tensor.float()
                if param.source.endswith(".weight") and tensor.ndim == 2:
                    base_key = source_key[: -len(".weight")]
                    for scale_key in (f"{base_key}.weight_scale", f"{base_key}.scale_weight"):
                        if scale_key in key_set:
                            scale = handle.get_tensor(scale_key).float()
                            if scale.ndim == 1:
                                scale = scale[:, None]
                            tensor = tensor * scale
                            break
                if raw_fp8_dense_dtype == "bf16":
                    tensor = tensor.to(torch.bfloat16)
                else:
                    tensor = tensor.to(torch.float16)
            elif tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)

            if param.source_slice is not None:
                start, length = param.source_slice
                tensor = tensor[start : start + length]
            if param.target == "norm_out.linear.weight":
                # Packed FLUX.2 stores final AdaLN modulation as shift, scale.
                # The mflux/MLX module consumes scale, shift.
                shift, scale = torch.split(tensor, tensor.shape[0] // 2, dim=0)
                tensor = torch.cat([scale, shift], dim=0)
            if tensor.dtype == torch.bfloat16:
                weight = mx.array(tensor.float().detach().cpu().numpy(), dtype=mx.bfloat16)
            else:
                weight = mx.array(tensor.detach().cpu().numpy())
            should_quantize = (
                is_fp8
                and not dense_compat
                and raw_fp8_dense_dtype is None
                and param.target.endswith(".weight")
                and tensor.ndim == 2
                and tensor.shape[-1] % 64 == 0
            )
            if should_quantize:
                qweight, scales, *biases = mx.quantize(weight, group_size=64, bits=8)
                module_name = param.target[: -len(".weight")]
                flat.append((param.target, qweight))
                flat.append((f"{module_name}.scales", scales))
                if biases:
                    flat.append((f"{module_name}.biases", biases[0]))
                    mx.eval(qweight, scales, biases[0])
                else:
                    mx.eval(qweight, scales)
                quantized_modules.add(module_name)
            else:
                flat.append((param.target, weight))
    return _Flux2PackedWeights(flat=flat, quantized_modules=frozenset(quantized_modules))


def _iter_flux2_packed_transformer_param_tensors(
    path: Path,
    model_config,
    quant_contract: dict[str, Any] | None = None,
):
    quant_contract = quant_contract or _flux2_inspect_transformer_quant_contract(path)
    keep_bf16_weights = _flux2_preserve_packed_bf16_weights(model_config, quant_contract)
    dense_compat = _flux2_quant_contract_uses_dense_compat(quant_contract)
    raw_fp8_dense_dtype = _flux2_raw_fp8_dense_dtype(quant_contract)

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        key_set = set(handle.keys())
        for param in _flux2_packed_params(model_config):
            source_key = _flux2_resolve_source_key(key_set, param.source)
            if source_key is None:
                raise KeyError(f"FLUX.2 packed transformer tensor not found: {param.source}")
            tensor = handle.get_tensor(source_key)
            is_fp8 = "float8" in str(tensor.dtype)
            is_comfy_quant_weight = (
                param.source.endswith(".weight")
                and f"{source_key[: -len('.weight')]}.comfy_quant" in key_set
            )
            if is_comfy_quant_weight:
                tensor = _flux2_dequantize_comfy_quant_weight(
                    handle,
                    source_key,
                    tensor,
                    model_config,
                    context="packed transformer",
                )
                tensor = tensor.to(torch.float16)
            elif is_fp8:
                tensor = tensor.float()
                if param.source.endswith(".weight") and tensor.ndim == 2:
                    base_key = source_key[: -len(".weight")]
                    for scale_key in (f"{base_key}.weight_scale", f"{base_key}.scale_weight"):
                        if scale_key in key_set:
                            scale = handle.get_tensor(scale_key).float()
                            if scale.ndim == 1:
                                scale = scale[:, None]
                            tensor = tensor * scale
                            break
                if raw_fp8_dense_dtype == "bf16":
                    tensor = tensor.to(torch.bfloat16)
                else:
                    tensor = tensor.to(torch.float16)
            elif tensor.dtype == torch.bfloat16:
                if not keep_bf16_weights:
                    tensor = tensor.to(torch.float16)

            if param.source_slice is not None:
                start, length = param.source_slice
                tensor = tensor[start : start + length]
            if param.target == "norm_out.linear.weight":
                # Packed FLUX.2 stores final AdaLN modulation as shift, scale.
                # The mflux/MLX module consumes scale, shift.
                shift, scale = torch.split(tensor, tensor.shape[0] // 2, dim=0)
                tensor = torch.cat([scale, shift], dim=0)
            if tensor.dtype == torch.bfloat16:
                # NumPy has no stable BF16 carrier here, so bridge through FP32
                # and restore MLX BF16 immediately.
                weight = mx.array(tensor.float().detach().cpu().numpy(), dtype=mx.bfloat16)
            else:
                weight = mx.array(tensor.detach().cpu().numpy())
            should_quantize = (
                is_fp8
                and not dense_compat
                and raw_fp8_dense_dtype is None
                and param.target.endswith(".weight")
                and tensor.ndim == 2
                and tensor.shape[-1] % 64 == 0
            )
            yield param.target, weight, should_quantize


def _load_flux2_packed_transformer_into_model(
    path: Path,
    model_config,
    transformer: Any,
    quant_contract: dict[str, Any] | None = None,
) -> bool:
    quantized_modules: set[str] = set()
    quant_contract = quant_contract or _flux2_inspect_transformer_quant_contract(path)
    for target, weight, should_quantize in _iter_flux2_packed_transformer_param_tensors(
        path,
        model_config,
        quant_contract=quant_contract,
    ):
        if should_quantize:
            qweight, scales, *biases = mx.quantize(weight, group_size=64, bits=8)
            module_name = target[: -len(".weight")]
            if module_name not in quantized_modules:
                _flux2_quantize_exact_modules(transformer, frozenset({module_name}))
                quantized_modules.add(module_name)
            flat = [(target, qweight), (f"{module_name}.scales", scales)]
            if biases:
                flat.append((f"{module_name}.biases", biases[0]))
                mx.eval(qweight, scales, biases[0])
            else:
                mx.eval(qweight, scales)
            transformer.update(tree_unflatten(flat), strict=False)
            del weight, qweight, scales, biases, flat
        else:
            transformer.update(tree_unflatten([(target, weight)]), strict=False)
            mx.eval(weight)
            del weight
    gc.collect()
    mx.clear_cache()
    return bool(quantized_modules)


def _flux2_resolve_module(root: Any, module_path: str) -> Any:
    current = root
    for part in module_path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _flux2_replace_module(root: Any, module_path: str, replacement: Any) -> None:
    parts = module_path.split(".")
    parent = root
    for part in parts[:-1]:
        if isinstance(parent, list):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    leaf = parts[-1]
    if isinstance(parent, list):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _flux2_quantize_exact_modules(root: Any, module_paths: frozenset[str]) -> None:
    for module_path in sorted(module_paths):
        module = _flux2_resolve_module(root, module_path)
        if not hasattr(module, "to_quantized"):
            raise RuntimeError(f"SDMLX FLUX.2 Klein: module cannot be quantized: {module_path}")
        _flux2_replace_module(root, module_path, module.to_quantized(group_size=64, bits=8))


def _find_flux2_roots() -> list[Path]:
    roots: list[Path] = []
    for base in _sdmlx_model_roots():
        if not base.exists():
            continue
        if _is_flux2_root(base) and not _flux2_is_cache_local_root(base):
            roots.append(base)
        try:
            for child in base.rglob("*.sdmlx"):
                if _is_flux2_root(child) and not _flux2_is_cache_local_root(child):
                    roots.append(child)
        except OSError:
            continue
        components = base / "components"
        if components.exists():
            try:
                for child in components.iterdir():
                    if _is_flux2_root(child) and not _flux2_is_cache_local_root(child):
                        roots.append(child)
            except OSError:
                continue
    if FLUX2_DEFAULT_ROOT is not None and FLUX2_DEFAULT_ROOT.exists():
        roots.append(FLUX2_DEFAULT_ROOT)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.resolve()).lower()
        except OSError:
            resolved = str(root).lower()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return sorted(unique, key=lambda item: item.name.lower())


def _flux2_model_options() -> list[str]:
    global _FLUX2_ROOT_DISPLAY_TO_PATH
    options: list[str] = []
    mapping: dict[str, Path] = {}
    for root in _find_flux2_roots():
        label = root.name
        parent = root.parent.name
        if parent and parent != "cache":
            label = f"{parent}/{label}"
        original = label
        counter = 2
        while label in mapping:
            label = f"{original} ({counter})"
            counter += 1
        mapping[label] = root
        options.append(label)
    _FLUX2_ROOT_DISPLAY_TO_PATH = mapping
    return options or ["<no FLUX.2 Klein model roots found>"]


def _resolve_flux2_root(model_root: str) -> Path:
    path = _FLUX2_ROOT_DISPLAY_TO_PATH.get(model_root)
    if path is None:
        _flux2_model_options()
        path = _FLUX2_ROOT_DISPLAY_TO_PATH.get(model_root)
    if path is None:
        candidate = Path(str(model_root)).expanduser()
        if candidate.exists():
            path = candidate
    if path is None:
        for candidate in _find_flux2_roots():
            if candidate.name == model_root:
                path = candidate
                break
    if path is None or not _is_flux2_root(path):
        raise FileNotFoundError(f"SDMLX FLUX.2 Klein: model root not found or incomplete: {model_root}")
    return path


def _model_config_for_root(root: Path):
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.config import ModelConfig

    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            config_name = str(manifest.get("model_config") or "").strip().lower()
            if config_name:
                return _flux2_model_config_by_name(config_name)
        except Exception:
            pass

    name = root.name.lower()
    if "base-9b" in name:
        return ModelConfig.flux2_klein_base_9b()
    if "base-4b" in name:
        return ModelConfig.flux2_klein_base_4b()
    if "9b-kv" in name:
        return ModelConfig.flux2_klein_9b_kv()
    if "9b" in name:
        return ModelConfig.flux2_klein_9b()
    return ModelConfig.flux2_klein_4b()


def is_flux2_sdmlx_model(model: Any) -> bool:
    return isinstance(model, dict) and model.get("model_family") == FLUX2_MODEL_FAMILY


def _flux2_lora_file_identity(path: str) -> tuple[str, int | None, int | None]:
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except Exception:
        pass
    try:
        stat = resolved.stat()
        return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    except Exception:
        return (str(resolved), None, None)


def _normalize_flux2_lora_identity(identity: Any, path: str) -> tuple:
    if isinstance(identity, dict):
        return (
            str(identity.get("path") or os.path.abspath(path)),
            identity.get("size"),
            identity.get("mtime_ns"),
        )
    if isinstance(identity, (list, tuple)):
        return tuple(identity)
    return _flux2_lora_file_identity(path)


def flux2_lora_specs_from_model(model: dict[str, Any]) -> tuple[tuple[str, float, tuple], ...]:
    specs: list[tuple[str, float, tuple]] = []
    for item in model.get("loras") or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not path:
            continue
        strength = float(item.get("strength_model", item.get("scale", 1.0)))
        if strength == 0.0:
            continue
        abs_path = os.path.abspath(str(path))
        identity = _normalize_flux2_lora_identity(item.get("identity"), abs_path)
        specs.append((abs_path, strength, identity))
    return tuple(specs)


def _normalize_flux2_lab_patch_policy(value: Any) -> str:
    text = str(value or _FLUX2_LAB_AUTO_COMFY_LAYERED).strip().lower().replace("-", "_")
    aliases = {
        "auto": _FLUX2_LAB_AUTO_COMFY_LAYERED,
        "comfy": _FLUX2_LAB_AUTO_COMFY_LAYERED,
        "comfy_layered": _FLUX2_LAB_AUTO_COMFY_LAYERED,
        "current": _FLUX2_LAB_PRODUCT_CURRENT,
        "product": _FLUX2_LAB_PRODUCT_CURRENT,
        "native": _FLUX2_LAB_PRODUCT_CURRENT,
        "rebind": _FLUX2_LAB_RUNTIME_REBIND,
        "runtime": _FLUX2_LAB_RUNTIME_REBIND,
        "dense": _FLUX2_LAB_DENSE_WEIGHT_PATCH,
        "dense_patch": _FLUX2_LAB_DENSE_WEIGHT_PATCH,
        "weight_patch": _FLUX2_LAB_DENSE_WEIGHT_PATCH,
        "requantize": _FLUX2_LAB_QUANTIZED_REQUANTIZE,
        "quantized": _FLUX2_LAB_QUANTIZED_REQUANTIZE,
        "dense_touched": _FLUX2_LAB_QUANTIZED_DENSE_TOUCHED,
        "quantized_dense": _FLUX2_LAB_QUANTIZED_DENSE_TOUCHED,
    }
    text = aliases.get(text, text)
    allowed = {
        _FLUX2_LAB_AUTO_COMFY_LAYERED,
        _FLUX2_LAB_PRODUCT_CURRENT,
        _FLUX2_LAB_RUNTIME_REBIND,
        _FLUX2_LAB_DENSE_WEIGHT_PATCH,
        _FLUX2_LAB_QUANTIZED_REQUANTIZE,
        _FLUX2_LAB_QUANTIZED_DENSE_TOUCHED,
    }
    return text if text in allowed else _FLUX2_LAB_AUTO_COMFY_LAYERED


def _normalize_flux2_lab_lora_patch_strategy(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    allowed = {
        "",
        _FLUX2_LAB_PRODUCT_CURRENT,
        _FLUX2_LAB_RUNTIME_REBIND,
        _FLUX2_LAB_DENSE_WEIGHT_PATCH,
        _FLUX2_LAB_QUANTIZED_REQUANTIZE,
        _FLUX2_LAB_QUANTIZED_DENSE_TOUCHED,
        _FLUX2_LAB_NO_LORA,
    }
    return text if text in allowed else ""


def _flux2_lab_effective_lora_strategy(
    patch_policy: Any,
    model_config: Any,
    quant_contract: dict[str, Any] | None,
    lora_specs: tuple[tuple[str, float, tuple], ...],
) -> str:
    policy = _normalize_flux2_lab_patch_policy(patch_policy)
    kind = _flux2_quant_contract_kind(quant_contract)

    if not lora_specs:
        if policy == _FLUX2_LAB_PRODUCT_CURRENT:
            return _FLUX2_LAB_NO_LORA
        if policy == _FLUX2_LAB_DENSE_WEIGHT_PATCH:
            return _FLUX2_LAB_DENSE_WEIGHT_PATCH
        if policy == _FLUX2_LAB_AUTO_COMFY_LAYERED:
            if kind == _FLUX2_QUANT_RAW_FP8 and _flux2_raw_fp8_dense_dtype(quant_contract) is not None:
                return _FLUX2_LAB_DENSE_WEIGHT_PATCH
        return _FLUX2_LAB_NO_LORA

    if policy != _FLUX2_LAB_AUTO_COMFY_LAYERED:
        return policy

    model_name = str(getattr(model_config, "model_name", "") or "").lower()
    if kind == _FLUX2_QUANT_DENSE_BF16:
        if _flux2_is_9b_model_config(model_config) and "kv" not in model_name:
            return _FLUX2_LAB_DENSE_WEIGHT_PATCH
        return _FLUX2_LAB_RUNTIME_REBIND
    if kind == _FLUX2_QUANT_RAW_FP8:
        return _FLUX2_LAB_DENSE_WEIGHT_PATCH
    if kind in {_FLUX2_QUANT_SCALED_FP8, _FLUX2_QUANT_COMFY, _FLUX2_QUANT_COMFY_MXFP8}:
        return _FLUX2_LAB_QUANTIZED_REQUANTIZE
    return _FLUX2_LAB_RUNTIME_REBIND


def _flux2_lab_strategy_notes(
    strategy: str,
    model_config: Any,
    quant_contract: dict[str, Any] | None,
) -> list[str]:
    kind = _flux2_quant_contract_kind(quant_contract)
    notes: list[str] = []
    if strategy == _FLUX2_LAB_DENSE_WEIGHT_PATCH:
        notes.append("LoRA stack patches dense target weights before sampling.")
        if kind == _FLUX2_QUANT_RAW_FP8:
            notes.append("raw-FP8 source uses dense compatibility materialization first.")
    elif strategy == _FLUX2_LAB_RUNTIME_REBIND:
        notes.append("LoRA wrappers are stripped/rebound on warm cache reuse.")
    elif strategy == _FLUX2_LAB_QUANTIZED_REQUANTIZE:
        notes.append("LoRA stack patches quantized targets through dequantize/add/requantize.")
    elif strategy == _FLUX2_LAB_QUANTIZED_DENSE_TOUCHED:
        notes.append("Diagnostic: touched quantized targets remain dense after LoRA patch.")
    elif strategy == _FLUX2_LAB_PRODUCT_CURRENT:
        notes.append("Uses the current validated product decision tree.")
    elif strategy == _FLUX2_LAB_NO_LORA:
        notes.append("No active LoRA stack.")

    if strategy == _FLUX2_LAB_DENSE_WEIGHT_PATCH and kind == _FLUX2_QUANT_RAW_FP8:
        notes.append("Lab keeps the raw-FP8 dense compatibility branch stable even with no active LoRA.")

    if _flux2_is_4b_model_config(model_config):
        notes.append("4B branch remains conservative in Lab V1 unless explicitly overridden.")
    return notes


def _flux2_lab_conditioning_ref_count(*conditionings: Any) -> int:
    total = 0
    for conditioning in conditionings:
        if isinstance(conditioning, dict):
            total += len(conditioning.get("reference_latents") or [])
    return total


def _flux2_build_lab_runtime_plan(
    sdmlx_model: dict[str, Any],
    *,
    patch_policy: Any = _FLUX2_LAB_AUTO_COMFY_LAYERED,
    positive: Any = None,
    negative: Any = None,
) -> dict[str, Any]:
    if not is_flux2_sdmlx_model(sdmlx_model):
        raise RuntimeError("SDMLX FLUX.2 Lab Runtime Plan: connect a FLUX.2 Klein model.")
    root = Path(str(sdmlx_model["model_path"])).expanduser()
    config_name = str(sdmlx_model.get("model_config") or "")
    model_config = _flux2_model_config_by_name(config_name) if config_name else _model_config_for_root(root)
    transformer_path = _flux2_transformer_file(root)
    quant_contract = sdmlx_model.get("transformer_quant_contract")
    if not isinstance(quant_contract, dict) and transformer_path is not None:
        quant_contract = _flux2_inspect_transformer_quant_contract(transformer_path)
    kind = _flux2_quant_contract_kind(quant_contract)
    lora_specs = flux2_lora_specs_from_model(sdmlx_model)
    policy = _normalize_flux2_lab_patch_policy(patch_policy)
    strategy = _flux2_lab_effective_lora_strategy(policy, model_config, quant_contract, lora_specs)
    model_name = str(getattr(model_config, "model_name", "") or config_name or root.name)
    reference_count = _flux2_lab_conditioning_ref_count(positive, negative)
    supports_kv = bool(getattr(model_config, "supports_kv_cache", False))
    lora_identity = [
        {
            "path": path,
            "strength": float(strength),
            "identity": list(identity) if isinstance(identity, tuple) else identity,
        }
        for path, strength, identity in lora_specs
    ]
    plan = {
        "type": _FLUX2_LAB_RUNTIME_PLAN_TYPE,
        "model_family": FLUX2_MODEL_FAMILY,
        "model_path": str(root),
        "model_name": str(sdmlx_model.get("model_name") or root.name),
        "model_config": model_name,
        "model_size": "9B" if _flux2_is_9b_model_config(model_config) else "4B" if _flux2_is_4b_model_config(model_config) else "unknown",
        "quant_kind": kind,
        "quant_policy": (quant_contract or {}).get("policy"),
        "patch_policy": policy,
        "lora_patch_strategy": strategy,
        "lora_count": len(lora_specs),
        "lora_identity": lora_identity,
        "reference_count": int(reference_count),
        "supports_kv_cache": supports_kv,
        "raw_fp8_dense_dtype": _flux2_raw_fp8_dense_dtype(quant_contract),
        "notes": _flux2_lab_strategy_notes(strategy, model_config, quant_contract),
    }
    return plan


def _flux2_lab_plan_summary(plan: dict[str, Any]) -> str:
    notes = "; ".join(str(item) for item in plan.get("notes") or [])
    return (
        f"model={plan.get('model_name')} config={plan.get('model_config')} "
        f"size={plan.get('model_size')} quant={plan.get('quant_kind')} "
        f"policy={plan.get('patch_policy')} strategy={plan.get('lora_patch_strategy')} "
        f"loras={plan.get('lora_count')} refs={plan.get('reference_count')} "
        f"kv={plan.get('supports_kv_cache')}"
        + (f"\n{notes}" if notes else "")
    )


def _flux2_lora_strategy_from_runtime_plan(plan: Any) -> str | None:
    if not isinstance(plan, dict):
        return None
    strategy = _normalize_flux2_lab_lora_patch_strategy(plan.get("lora_patch_strategy"))
    if strategy in {"", _FLUX2_LAB_PRODUCT_CURRENT, _FLUX2_LAB_NO_LORA}:
        return None
    return strategy


def _flux2_suite_lora_patch_strategy(
    sdmlx_model: dict[str, Any],
    *,
    positive: Any = None,
    negative: Any = None,
) -> str | None:
    explicit = _normalize_flux2_lab_lora_patch_strategy(sdmlx_model.get("_flux2_lab_lora_patch_strategy"))
    if explicit not in {"", _FLUX2_LAB_PRODUCT_CURRENT, _FLUX2_LAB_NO_LORA}:
        return explicit
    plan = _flux2_build_lab_runtime_plan(
        sdmlx_model,
        patch_policy=_FLUX2_LAB_AUTO_COMFY_LAYERED,
        positive=positive,
        negative=negative,
    )
    return _flux2_lora_strategy_from_runtime_plan(plan)


def _validate_flux2_lora_specs(lora_specs: tuple[tuple[str, float, tuple], ...], model_config: Any) -> None:
    if not lora_specs:
        return
    overrides = model_config.transformer_overrides
    max_double = int(overrides.get("num_layers", 0)) - 1
    max_single = int(overrides.get("num_single_layers", 0)) - 1
    inner_dim = int(overrides.get("num_attention_heads", 0)) * _FLUX2_HEAD_DIM
    joint_attention_dim = int(overrides.get("joint_attention_dim", 0))
    expected_large_dims = {
        128,
        256,
        inner_dim,
        joint_attention_dim,
        inner_dim * 3,
        inner_dim * 4,
        inner_dim * 6,
        inner_dim * 9,
    }
    model_size_hint = "9B" if inner_dim == 4096 else "4B" if inner_dim == 3072 else f"inner_dim={inner_dim}"
    opposite_size_hint = "4B" if inner_dim == 4096 else "9B" if inner_dim == 3072 else "another FLUX.2 size"
    double_pattern = re.compile(r"(?:double_blocks_|double_blocks\.)(\d+)")
    single_pattern = re.compile(r"(?:single_blocks_|single_blocks\.)(\d+)")

    for path, _scale, _identity in lora_specs:
        lora_path = Path(path)
        if not lora_path.exists():
            raise FileNotFoundError(f"SDMLX FLUX.2 Klein: LoRA not found: {path}")
        double_blocks: set[int] = set()
        single_blocks: set[int] = set()
        try:
            with safe_open(str(lora_path), framework="np", device="cpu") as handle:
                metadata = handle.metadata() or {}
                keys = list(handle.keys())
                shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
        except Exception as exc:
            raise RuntimeError(f"SDMLX FLUX.2 Klein: failed to inspect LoRA {lora_path.name}: {exc}") from exc
        metadata_text = " ".join(str(value).lower() for value in metadata.values())
        metadata_arch = (
            metadata.get("modelspec.architecture")
            or metadata.get("ss_base_model_version")
            or metadata.get("ss_sd_model_name")
            or "unknown"
        )
        for key in keys:
            match = double_pattern.search(key)
            if match:
                double_blocks.add(int(match.group(1)))
            match = single_pattern.search(key)
            if match:
                single_blocks.add(int(match.group(1)))
            if any(token in key for token in (".lora_A.", ".lora_B.", ".lora_down.", ".lora_up.")):
                for dim in shapes.get(key, ()):
                    dim = int(dim)
                    if dim > 1024 and dim not in expected_large_dims:
                        looks_like_flux1_single = (
                            inner_dim == 3072
                            and dim == inner_dim * 7
                            and ("single_blocks" in key or "single_blocks_" in key)
                            and "linear1" in key
                        )
                        if "flux1" in metadata_text or "flux-1" in metadata_text or looks_like_flux1_single:
                            raise RuntimeError(
                                "SDMLX FLUX.2 Klein: LoRA appears to target FLUX.1, not FLUX.2 Klein: "
                                f"{lora_path.name} contains {key} with tensor dimension {dim}. "
                                f"The loaded {model_size_hint} checkpoint ({model_config.model_name}) "
                                f"expects FLUX.2 Klein single linear1 dimension {inner_dim * 9}; "
                                f"this LoRA metadata reports {metadata_arch!r}. "
                                "Use a FLUX.2 Klein LoRA that matches the loaded checkpoint."
                            )
                        raise RuntimeError(
                            "SDMLX FLUX.2 Klein: LoRA appears incompatible with this checkpoint: "
                            f"{lora_path.name} contains tensor dimension {dim}, "
                            f"but the loaded {model_size_hint} checkpoint "
                            f"({model_config.model_name}) expects one of "
                            f"{sorted(expected_large_dims)} for FLUX.2 Klein adapters. "
                            f"Use a {model_size_hint} LoRA for this checkpoint, or switch to the "
                            f"{opposite_size_hint} checkpoint that matches this LoRA."
                        )
        if double_blocks and max(double_blocks) > max_double:
            raise RuntimeError(
                "SDMLX FLUX.2 Klein: LoRA appears incompatible with this checkpoint: "
                f"{lora_path.name} uses double block {max(double_blocks)}, "
                f"but {model_config.model_name} has blocks 0..{max_double}."
            )
        if single_blocks and max(single_blocks) > max_single:
            raise RuntimeError(
                "SDMLX FLUX.2 Klein: LoRA appears incompatible with this checkpoint: "
                f"{lora_path.name} uses single block {max(single_blocks)}, "
                f"but {model_config.model_name} has blocks 0..{max_single}."
            )


def _flux2_strip_lora_wrappers(module: Any) -> int:
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
    from sdmlx_qwen_native.models.common.lora.layer.linear_lora_layer import LoRALinear
    from sdmlx_qwen_native.models.common.lora.layer.lokr_linear_layer import LoKrLinear

    visited: set[int] = set()

    def _base_from_wrapper(obj: Any) -> Any | None:
        if isinstance(obj, FusedLoRALinear):
            return obj.base_linear
        if isinstance(obj, (LoRALinear, LoKrLinear)):
            return obj.linear
        return None

    def _walk(obj: Any) -> int:
        obj_id = id(obj)
        if obj_id in visited:
            return 0
        visited.add(obj_id)
        count = 0

        if isinstance(obj, list):
            for index, child in enumerate(list(obj)):
                base = _base_from_wrapper(child)
                if base is not None:
                    obj[index] = base
                    child = base
                    count += 1
                count += _walk(child)
            return count

        if isinstance(obj, dict):
            for key, child in list(obj.items()):
                base = _base_from_wrapper(child)
                if base is not None:
                    obj[key] = base
                    child = base
                    count += 1
                count += _walk(child)
            return count

        if not isinstance(obj, nn.Module):
            return 0

        for name, child in list(obj.items()):
            base = _base_from_wrapper(child)
            if base is not None:
                obj[name] = base
                child = base
                count += 1
            count += _walk(child)
        return count

    return _walk(module)


def _flux2_rebind_runtime_loras(model: Any, lora_specs: tuple[tuple[str, float, tuple], ...]) -> None:
    current_specs = tuple(getattr(model, "_sdmlx_lora_specs", ()) or ())
    if hasattr(model, "_sdmlx_lora_specs") and current_specs == tuple(lora_specs):
        return

    removed = _flux2_strip_lora_wrappers(model.transformer)
    lora_paths = [path for path, _scale, _identity in lora_specs] or None
    lora_scales = [scale for _path, scale, _identity in lora_specs] or None

    from sdmlx_qwen_native.models.flux2.flux2_initializer import Flux2Initializer

    Flux2Initializer._apply_lora(model, lora_paths, lora_scales)
    model._sdmlx_lora_specs = tuple(lora_specs)
    if removed or lora_specs:
        gc.collect()
        mx.clear_cache()
    _flux2_log(
        "SDMLX FLUX.2 Klein: rebound runtime LoRA stack "
        f"(removed={removed}, active={len(lora_specs)})",
        verbose=True,
    )


def _flux2_dense_lora_bake_allowed(
    model_config,
    transformer_quant_contract: dict[str, Any] | None,
    *,
    has_quantized_modules: bool,
    lora_specs: tuple[tuple[str, float, tuple], ...],
) -> bool:
    if not lora_specs or has_quantized_modules:
        return False
    if _flux2_quant_contract_kind(transformer_quant_contract) != _FLUX2_QUANT_DENSE_BF16:
        return False
    overrides = getattr(model_config, "text_encoder_overrides", None) or {}
    hidden_size = int(overrides.get("hidden_size") or 0)
    model_name = str(getattr(model_config, "model_name", "") or "").lower()
    if hidden_size != 4096 or "9b" not in model_name or "kv" in model_name:
        return False
    value = str(os.environ.get("SDMLX_FLUX2_DENSE_LORA_BAKE") or "").strip().lower()
    return value in {"1", "true", "on", "yes", "bf16", "dense"}


def _flux2_bake_dense_lora_with_stats(module: Any) -> dict[str, int | float]:
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
    from sdmlx_qwen_native.models.common.lora.layer.linear_lora_layer import LoRALinear
    from sdmlx_qwen_native.models.common.lora.layer.lokr_linear_layer import LoKrLinear

    import time

    stats: dict[str, int | float] = {
        "wrappers": 0,
        "baked_modules": 0,
        "baked_loras": 0,
        "passthrough": 0,
        "skipped": 0,
        "seconds": 0.0,
    }

    def _assign(parent: Any, attr_name: str | None, idx: int | None, new_child: Any) -> None:
        if parent is None:
            return
        if isinstance(parent, list) and idx is not None:
            parent[idx] = new_child
        elif isinstance(parent, dict) and attr_name is not None:
            parent[attr_name] = new_child
        elif attr_name is not None:
            setattr(parent, attr_name, new_child)

    def _apply_lora_delta(base_linear: Any, lora_layer: Any) -> bool:
        if not isinstance(lora_layer, LoRALinear) or not hasattr(base_linear, "weight"):
            return False
        if isinstance(base_linear, nn.QuantizedLinear):
            return False

        weight = base_linear.weight
        delta = mx.matmul(lora_layer.lora_A.astype(mx.float32), lora_layer.lora_B.astype(mx.float32))
        delta = mx.transpose(delta) * float(getattr(lora_layer, "scale", 1.0))
        if tuple(weight.shape) != tuple(delta.shape):
            stats["skipped"] = int(stats["skipped"]) + 1
            return False
        base_linear.weight = (weight.astype(mx.float32) + delta).astype(weight.dtype)
        mx.eval(base_linear.weight)
        stats["baked_loras"] = int(stats["baked_loras"]) + 1
        return True

    def _bake(base_linear: Any, loras: list[Any]) -> Any:
        if isinstance(base_linear, nn.QuantizedLinear) or not hasattr(base_linear, "weight"):
            stats["skipped"] = int(stats["skipped"]) + len(loras)
            return FusedLoRALinear(base_linear=base_linear, loras=loras) if loras else base_linear

        passthrough_loras: list[Any] = []
        baked_here = 0
        for lora in loras:
            if isinstance(lora, LoRALinear) and _apply_lora_delta(base_linear, lora):
                baked_here += 1
            else:
                passthrough_loras.append(lora)

        if baked_here:
            stats["baked_modules"] = int(stats["baked_modules"]) + 1
        if passthrough_loras:
            stats["passthrough"] = int(stats["passthrough"]) + len(passthrough_loras)
            return FusedLoRALinear(base_linear=base_linear, loras=passthrough_loras)
        return base_linear

    def _walk(obj: Any, parent: Any = None, attr_name: str | None = None, idx: int | None = None) -> None:
        if isinstance(obj, FusedLoRALinear):
            stats["wrappers"] = int(stats["wrappers"]) + 1
            new_child = _bake(obj.base_linear, list(obj.loras))
            _assign(parent, attr_name, idx, new_child)
            return
        if isinstance(obj, LoRALinear):
            stats["wrappers"] = int(stats["wrappers"]) + 1
            new_child = _bake(obj.linear, [obj])
            _assign(parent, attr_name, idx, new_child)
            return
        if isinstance(obj, LoKrLinear):
            return

        if isinstance(obj, list):
            for index, child in enumerate(list(obj)):
                _walk(child, obj, None, index)
            return
        if isinstance(obj, tuple):
            temp_list = list(obj)
            for index, child in enumerate(temp_list):
                _walk(child, temp_list, None, index)
            if parent is not None:
                _assign(parent, attr_name, idx, type(obj)(temp_list))
            return
        if isinstance(obj, dict):
            for key, child in list(obj.items()):
                _walk(child, obj, key, None)
            return
        if isinstance(obj, nn.Module):
            for name, child in list(vars(obj).items()):
                if isinstance(child, (nn.Module, list, tuple, dict)):
                    _walk(child, obj, name, None)

    t0 = time.perf_counter()
    _walk(module)
    mx.clear_cache()
    gc.collect()
    stats["seconds"] = time.perf_counter() - t0
    return stats


def _flux2_negated_lora_specs(
    lora_specs: tuple[tuple[str, float, tuple], ...],
) -> tuple[tuple[str, float, tuple], ...]:
    return tuple((path, -float(scale), identity) for path, scale, identity in lora_specs)


def _flux2_lora_specs_dense_reversible_supported(
    lora_specs: tuple[tuple[str, float, tuple], ...],
) -> bool:
    for path, _scale, _identity in lora_specs:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = [str(key).lower() for key in handle.keys()]
        except Exception:
            return False
        joined = "\n".join(keys)
        unsupported_markers = (
            "lokr",
            "hada",
            "ia3",
            "boft",
            ".oft",
            "dora",
        )
        if any(marker in joined for marker in unsupported_markers):
            return False
        if not any(token in joined for token in ("lora_a", "lora_b", "lora_down", "lora_up")):
            return False
    return True


def _flux2_dense_lora_weight_patch_can_reverse(
    model: Any,
    current_specs: tuple[tuple[str, float, tuple], ...],
    requested_specs: tuple[tuple[str, float, tuple], ...],
) -> bool:
    if not bool(getattr(model, "_sdmlx_lora_patch_reversible", False)):
        return False
    return (
        _flux2_lora_specs_dense_reversible_supported(current_specs)
        and _flux2_lora_specs_dense_reversible_supported(requested_specs)
    )


def _flux2_dense_lora_weight_patch_requires_reload(
    model: Any,
    lora_specs: tuple[tuple[str, float, tuple], ...],
) -> bool:
    current_specs = tuple(getattr(model, "_sdmlx_lora_specs", ()) or ())
    patch_mode = str(getattr(model, "_sdmlx_lora_patch_mode", "") or "")
    requested = tuple(lora_specs)
    if current_specs == requested:
        return False
    # Product dense-BF16 keeps the cached model either clean or patched for one
    # LoRA stack. The Lab raw-FP8 branch may opt into reversible A/B patch
    # switching to avoid full base reloads while testing Comfy-like layering.
    if patch_mode == "dense_weight_patch" and bool(current_specs):
        return not _flux2_dense_lora_weight_patch_can_reverse(model, current_specs, requested)
    return patch_mode == "dense_weight_patch" and bool(current_specs)


def _flux2_apply_dense_lora_delta_specs(
    model: Any,
    lora_specs: tuple[tuple[str, float, tuple], ...],
    *,
    strict_ab_only: bool,
) -> tuple[dict[str, int | float], int]:
    stats: dict[str, int | float] = {
        "wrappers": 0,
        "baked_modules": 0,
        "baked_loras": 0,
        "passthrough": 0,
        "skipped": 0,
        "seconds": 0.0,
    }
    removed = _flux2_strip_lora_wrappers(model.transformer)
    if not lora_specs:
        return stats, removed
    if strict_ab_only and not _flux2_lora_specs_dense_reversible_supported(lora_specs):
        raise RuntimeError(
            "SDMLX FLUX.2 Lab: reversible dense LoRA switching supports only normal A/B LoRAs; "
            "reload the clean base for this adapter type."
        )

    from sdmlx_qwen_native.models.flux2.flux2_initializer import Flux2Initializer

    lora_paths = [path for path, _scale, _identity in lora_specs]
    lora_scales = [scale for _path, scale, _identity in lora_specs]
    Flux2Initializer._apply_lora(model, lora_paths, lora_scales)
    stats = _flux2_bake_dense_lora_with_stats(model.transformer)
    remaining = _flux2_strip_lora_wrappers(model.transformer)
    if strict_ab_only and (
        int(stats.get("passthrough") or 0) > 0
        or int(stats.get("skipped") or 0) > 0
        or int(remaining or 0) > 0
    ):
        raise RuntimeError(
            "SDMLX FLUX.2 Lab: reversible dense LoRA switch found unbaked adapter wrappers; "
            "reload the clean base for this adapter stack."
        )
    return stats, removed + remaining


def _flux2_apply_dense_lora_weight_patch(
    model: Any,
    lora_specs: tuple[tuple[str, float, tuple], ...],
) -> None:
    requested = tuple(lora_specs)
    current_specs = tuple(getattr(model, "_sdmlx_lora_specs", ()) or ())
    patch_mode = str(getattr(model, "_sdmlx_lora_patch_mode", "") or "")
    if patch_mode == "dense_weight_patch" and current_specs == requested:
        return
    reversible_switch = (
        patch_mode == "dense_weight_patch"
        and bool(current_specs)
        and current_specs != requested
        and _flux2_dense_lora_weight_patch_can_reverse(model, current_specs, requested)
    )
    if _flux2_dense_lora_weight_patch_requires_reload(model, requested):
        raise RuntimeError(
            "SDMLX FLUX.2 Klein: dense LoRA weight patch stack changed on an already patched model; "
            "reload the clean base model before applying the new LoRA stack."
        )

    removed = 0
    undo_stats = None
    if reversible_switch:
        undo_stats, removed = _flux2_apply_dense_lora_delta_specs(
            model,
            _flux2_negated_lora_specs(current_specs),
            strict_ab_only=True,
        )

    if not requested:
        if not reversible_switch:
            removed += _flux2_strip_lora_wrappers(model.transformer)
        model._sdmlx_lora_specs = ()
        model._sdmlx_lora_patch_mode = "dense_weight_patch"
        if removed:
            gc.collect()
            mx.clear_cache()
        if reversible_switch:
            _flux2_log(
                "SDMLX FLUX.2 Lab: reversible dense LoRA switch "
                f"(undo_loras={int((undo_stats or {}).get('baked_loras') or 0)}, "
                "apply_loras=0)",
                verbose=True,
            )
        return

    patch_stats, patch_removed = _flux2_apply_dense_lora_delta_specs(
        model,
        requested,
        strict_ab_only=bool(getattr(model, "_sdmlx_lora_patch_reversible", False)),
    )
    removed += patch_removed
    model._sdmlx_lora_specs = requested
    model._sdmlx_lora_patch_mode = "dense_weight_patch"
    if removed or int(patch_stats.get("wrappers") or 0) > 0:
        gc.collect()
        mx.clear_cache()
    if reversible_switch:
        _flux2_log(
            "SDMLX FLUX.2 Lab: reversible dense LoRA switch "
            f"(undo_loras={int((undo_stats or {}).get('baked_loras') or 0)}, "
            f"apply_loras={int(patch_stats.get('baked_loras') or 0)}, "
            f"time={float((undo_stats or {}).get('seconds') or 0.0) + float(patch_stats.get('seconds') or 0.0):.2f}s)",
            verbose=True,
        )
    else:
        _flux2_log(
            "SDMLX FLUX.2 Klein: dense LoRA weight patch "
            f"(modules={int(patch_stats.get('baked_modules') or 0)}, "
            f"loras={int(patch_stats.get('baked_loras') or 0)}, "
            f"passthrough={int(patch_stats.get('passthrough') or 0)}, "
            f"skipped={int(patch_stats.get('skipped') or 0)}, "
            f"time={float(patch_stats.get('seconds') or 0.0):.2f}s)",
            verbose=True,
        )


def _validate_flux2_text_encoder_for_model(text_encoder_path: Path, model_config) -> None:
    expected_hidden = int((model_config.text_encoder_overrides or {}).get("hidden_size") or 0)
    actual_hidden = int(_flux2_text_encoder_hidden_size(text_encoder_path) or 0)
    projection_input = int(_flux2_text_encoder_projection_input_size(text_encoder_path) or 0)
    projection_mismatches = _flux2_text_encoder_projection_mismatches(text_encoder_path, expected_hidden)
    if expected_hidden and actual_hidden and expected_hidden != actual_hidden:
        raise RuntimeError(
            "SDMLX FLUX.2 Klein CLIP: selected Qwen3 text encoder does not match this checkpoint "
            f"({text_encoder_path.name}: hidden_size={actual_hidden}, model expects {expected_hidden})."
        )
    if expected_hidden and projection_input and projection_input != expected_hidden:
        raise RuntimeError(
            "SDMLX FLUX.2 Klein CLIP: selected Qwen3 text encoder has incompatible projection weights "
            f"({text_encoder_path.name}: first projection input={projection_input}, "
            f"model expects {expected_hidden})."
        )
    if projection_mismatches:
        preview = ", ".join(f"{key}:{shape}" for key, shape in projection_mismatches[:3])
        raise RuntimeError(
            "SDMLX FLUX.2 Klein CLIP: selected Qwen3 text encoder has mixed/incompatible projection shapes "
            f"({text_encoder_path.name}: {len(projection_mismatches)} projection tensors do not match "
            f"hidden_size={expected_hidden}; first mismatches: {preview}). "
            "Use the FLUX.2 Klein package CLIP output or the matching FLUX.2 Klein Qwen3 text_encoder folder."
        )


def _flux2_qwen3_expected_weight_shape(
    key: str,
    model_config: Any,
    stored_shape: tuple[int, ...],
    quant_format: str | None = None,
) -> tuple[int, ...]:
    if len(stored_shape) != 2:
        return stored_shape
    overrides = model_config.text_encoder_overrides or {}
    hidden_size = int(overrides.get("hidden_size") or stored_shape[1])
    intermediate_size = int(overrides.get("intermediate_size") or 0)
    head_dim = int(overrides.get("head_dim") or _FLUX2_HEAD_DIM)
    num_attention_heads = int(overrides.get("num_attention_heads") or (hidden_size // head_dim))
    num_key_value_heads = int(overrides.get("num_key_value_heads") or max(1, num_attention_heads // 4))
    q_out = num_attention_heads * head_dim
    kv_out = num_key_value_heads * head_dim

    if key == "model.embed_tokens.weight":
        return (stored_shape[0], hidden_size)
    if re.match(r"model\.layers\.\d+\.self_attn\.q_proj\.weight$", key):
        return (q_out, hidden_size)
    if re.match(r"model\.layers\.\d+\.self_attn\.(?:k_proj|v_proj)\.weight$", key):
        return (kv_out, hidden_size)
    if re.match(r"model\.layers\.\d+\.self_attn\.o_proj\.weight$", key):
        return (hidden_size, q_out)
    if re.match(r"model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj)\.weight$", key):
        return (stored_shape[0], hidden_size)
    if re.match(r"model\.layers\.\d+\.mlp\.down_proj\.weight$", key):
        if intermediate_size:
            return (hidden_size, intermediate_size)
        return (hidden_size, stored_shape[1] * 2 if quant_format == "nvfp4" else stored_shape[1])
    if quant_format == "nvfp4":
        return (stored_shape[0], stored_shape[1] * 2)
    return stored_shape


def _flux2_dequantize_comfy_quant_weight(
    handle: Any,
    key: str,
    tensor: torch.Tensor,
    model_config: Any,
    *,
    expected_shape: tuple[int, ...] | None = None,
    context: str = "CLIP",
) -> torch.Tensor:
    base_key = key[: -len(".weight")]
    config = _flux2_comfy_quant_config(handle, base_key)
    if not config:
        return tensor

    quant_format = str(config.get("format") or "").lower()
    try:
        from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
    except Exception as exc:
        raise RuntimeError(
            f"SDMLX FLUX.2 Klein {context}: this component uses Comfy mixed-precision "
            f"weights ({quant_format}), but Comfy quantization support is unavailable."
        ) from exc

    if quant_format not in QUANT_ALGOS:
        raise RuntimeError(f"SDMLX FLUX.2 Klein {context}: unsupported Comfy quantization format: {quant_format}")

    qconfig = QUANT_ALGOS[quant_format]
    layout_type = qconfig["comfy_tensor_layout"]
    layout_cls = get_layout_class(layout_type)
    if layout_cls is None:
        raise RuntimeError(f"SDMLX FLUX.2 Klein {context}: missing Comfy quantized layout: {layout_type}")

    stored_shape = tuple(int(value) for value in tensor.shape)
    config_shape = config.get("orig_shape")
    if isinstance(config_shape, (list, tuple)) and config_shape:
        orig_shape = tuple(int(value) for value in config_shape)
    elif expected_shape is not None:
        orig_shape = tuple(int(value) for value in expected_shape)
    else:
        orig_shape = _flux2_qwen3_expected_weight_shape(key, model_config, stored_shape, quant_format)
    qdata = tensor.to(dtype=qconfig["storage_t"])

    if quant_format in {"float8_e4m3fn", "float8_e5m2"}:
        scale_key = f"{base_key}.weight_scale"
        if scale_key not in set(handle.keys()):
            raise RuntimeError(f"SDMLX FLUX.2 Klein {context}: missing Comfy quant scale: {scale_key}")
        scale = handle.get_tensor(scale_key).float()
        params = layout_cls.Params(scale=scale, orig_dtype=torch.bfloat16, orig_shape=orig_shape)
    elif quant_format == "nvfp4":
        tensor_scale_key = f"{base_key}.weight_scale_2"
        block_scale_key = f"{base_key}.weight_scale"
        keys = set(handle.keys())
        if tensor_scale_key not in keys or block_scale_key not in keys:
            raise RuntimeError(f"SDMLX FLUX.2 Klein {context}: missing NVFP4 scales for {base_key}")
        tensor_scale = handle.get_tensor(tensor_scale_key)
        block_scale = handle.get_tensor(block_scale_key)
        if block_scale.dtype == torch.uint8:
            block_scale = block_scale.view(torch.float8_e4m3fn)
        params = layout_cls.Params(
            scale=tensor_scale,
            block_scale=block_scale,
            orig_dtype=torch.bfloat16,
            orig_shape=orig_shape,
        )
    elif quant_format == "mxfp8":
        scale_key = f"{base_key}.weight_scale"
        if scale_key not in set(handle.keys()):
            raise RuntimeError(f"SDMLX FLUX.2 Klein {context}: missing MXFP8 scale: {scale_key}")
        scale = handle.get_tensor(scale_key)
        float8_e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if scale.dtype == torch.uint8 and float8_e8m0 is not None:
            scale = scale.view(float8_e8m0)
        params = layout_cls.Params(scale=scale, orig_dtype=torch.bfloat16, orig_shape=orig_shape)
    else:
        raise RuntimeError(f"SDMLX FLUX.2 Klein {context}: unsupported Comfy quantization format: {quant_format}")

    quantized = QuantizedTensor(qdata, layout_type, params)
    return quantized.dequantize()


def _flux2_text_encoder_uses_comfy_quant(path: Path) -> bool:
    try:
        with safe_open(str(path), framework="np") as handle:
            return any(key.endswith(".comfy_quant") for key in handle.keys())
    except Exception:
        return False


def _flux2_text_encoder_cache_model_config_payload(model_config) -> dict[str, Any]:
    overrides = model_config.text_encoder_overrides or {}
    hidden_size = int(overrides.get("hidden_size") or 0)
    head_dim = int(overrides.get("head_dim") or _FLUX2_HEAD_DIM)
    num_attention_heads = int(
        overrides.get("num_attention_heads")
        or (hidden_size // head_dim if hidden_size and head_dim else 0)
    )
    num_key_value_heads = int(
        overrides.get("num_key_value_heads")
        or (max(1, num_attention_heads // 4) if num_attention_heads else 0)
    )
    return {
        "hidden_size": hidden_size,
        "intermediate_size": int(overrides.get("intermediate_size") or 0),
        "num_hidden_layers": int(overrides.get("num_hidden_layers") or 0),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "prepared_dtype": "bfloat16",
    }


def _flux2_text_encoder_cache_normalize_model_config(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    hidden_size = int(payload.get("hidden_size") or 0)
    head_dim = int(payload.get("head_dim") or _FLUX2_HEAD_DIM)
    num_attention_heads = int(
        payload.get("num_attention_heads")
        or (hidden_size // head_dim if hidden_size and head_dim else 0)
    )
    num_key_value_heads = int(
        payload.get("num_key_value_heads")
        or (max(1, num_attention_heads // 4) if num_attention_heads else 0)
    )
    return {
        "hidden_size": hidden_size,
        "intermediate_size": int(payload.get("intermediate_size") or 0),
        "num_hidden_layers": int(payload.get("num_hidden_layers") or 0),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "prepared_dtype": str(payload.get("prepared_dtype") or "bfloat16"),
    }


def _flux2_text_encoder_cache_metadata(path: Path, model_config) -> dict[str, str]:
    identity = _flux2_file_identity(path)
    config_payload = _flux2_text_encoder_cache_model_config_payload(model_config)
    return {
        "cache_format": _FLUX2_TEXT_ENCODER_CACHE_FORMAT,
        "source_path": str(identity["source_path"]),
        "source_name": str(identity["source_name"]),
        "source_size": str(identity["source_size"]),
        "source_mtime_ns": str(identity["source_mtime_ns"]),
        "source_content_digest": str(identity["source_content_digest"]),
        "model_config": json.dumps(config_payload, sort_keys=True, separators=(",", ":")),
    }


def _flux2_text_encoder_cache_key_payload(path: Path, model_config) -> dict[str, Any]:
    identity = _flux2_file_identity(path)
    return {
        "cache_format": _FLUX2_TEXT_ENCODER_CACHE_FORMAT,
        "source_size": int(identity["source_size"]),
        "source_mtime_ns": int(identity["source_mtime_ns"]),
        "source_content_digest": str(identity["source_content_digest"]),
        "model_config": _flux2_text_encoder_cache_model_config_payload(model_config),
    }


def _flux2_text_encoder_cache_path(path: Path, model_config) -> Path:
    key_payload = _flux2_text_encoder_cache_key_payload(path, model_config)
    digest_source = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
    safe_stem = _flux2_safe_package_name(path)
    hidden = (model_config.text_encoder_overrides or {}).get("hidden_size") or "qwen3"
    return _flux2_text_encoder_prepared_cache_dir() / f"{safe_stem}-h{hidden}-{digest}" / "weights.safetensors"


def _flux2_text_encoder_cache_source_matches(candidate: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(candidate, dict):
        return False
    try:
        return (
            int(candidate.get("source_size") or -1) == int(expected["source_size"])
            and int(candidate.get("source_mtime_ns") or -1)
            == int(expected["source_mtime_ns"])
            and str(candidate.get("source_content_digest") or "")
            == str(expected["source_content_digest"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _flux2_text_encoder_cache_manifest_matches(
    manifest: Any,
    source_identity: dict[str, Any],
    model_config,
) -> bool:
    if not isinstance(manifest, dict):
        return False
    cache_format = str(manifest.get("cache_format") or manifest.get("format") or "")
    if cache_format != _FLUX2_TEXT_ENCODER_CACHE_FORMAT:
        return False
    if not _flux2_text_encoder_cache_source_matches(
        manifest.get("source_identity"),
        source_identity,
    ):
        return False
    actual_config = _flux2_text_encoder_cache_normalize_model_config(
        manifest.get("model_config")
    )
    expected_config = _flux2_text_encoder_cache_model_config_payload(model_config)
    return actual_config == expected_config


def _flux2_text_encoder_cache_metadata_matches(
    metadata: Any,
    source_identity: dict[str, Any],
    model_config,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    if str(metadata.get("cache_format") or "") != _FLUX2_TEXT_ENCODER_CACHE_FORMAT:
        return False
    if not _flux2_text_encoder_cache_source_matches(metadata, source_identity):
        return False
    try:
        actual_payload = json.loads(str(metadata.get("model_config") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    actual_config = _flux2_text_encoder_cache_normalize_model_config(actual_payload)
    expected_config = _flux2_text_encoder_cache_model_config_payload(model_config)
    return actual_config == expected_config


def _flux2_text_encoder_cache_candidates(path: Path, model_config) -> list[Path]:
    source_identity = _flux2_file_identity(path)
    canonical_path = _flux2_text_encoder_cache_path(path, model_config)
    candidates: dict[str, Path] = {}
    if canonical_path.is_file() and canonical_path.stat().st_size > 0:
        candidates[str(canonical_path.resolve())] = canonical_path

    cache_root = _flux2_text_encoder_prepared_cache_dir()
    for manifest_path in cache_root.glob("*/manifest.json"):
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            continue
        if not _flux2_text_encoder_cache_manifest_matches(
            manifest,
            source_identity,
            model_config,
        ):
            continue
        cache_path = manifest_path.parent / "weights.safetensors"
        try:
            if not cache_path.is_file() or cache_path.stat().st_size <= 0:
                continue
            candidates[str(cache_path.resolve())] = cache_path
        except OSError:
            continue

    def recency(cache_path: Path) -> tuple[int, str]:
        try:
            return cache_path.stat().st_mtime_ns, cache_path.parent.name
        except OSError:
            return 0, cache_path.parent.name

    return sorted(candidates.values(), key=recency, reverse=True)


def _flux2_resolve_text_encoder_cache_path(path: Path, model_config) -> Path:
    candidates = _flux2_text_encoder_cache_candidates(path, model_config)
    if candidates:
        return candidates[0]
    return _flux2_text_encoder_cache_path(path, model_config)


def _flux2_text_encoder_cache_manifest_path(cache_path: Path) -> Path:
    return cache_path.parent / "manifest.json"


def _flux2_text_encoder_cache_manifest_entry(
    source_path: Path,
    model_config,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    cache_path = cache_path or _flux2_resolve_text_encoder_cache_path(
        source_path,
        model_config,
    )
    identity = _flux2_file_identity(source_path)
    return {
        "format": _FLUX2_TEXT_ENCODER_CACHE_MANIFEST_FORMAT,
        "kind": "prepared_text_encoder",
        "model_family": FLUX2_MODEL_FAMILY,
        "component": "text_encoder",
        "cache_format": _FLUX2_TEXT_ENCODER_CACHE_FORMAT,
        "storage": "sdmlx_cache",
        "path": _flux2_manifest_path(cache_path),
        "manifest": _flux2_manifest_path(_flux2_text_encoder_cache_manifest_path(cache_path)),
        "source_identity": identity,
        "model_config": _flux2_text_encoder_cache_model_config_payload(model_config),
        "derived_from": _flux2_manifest_path(source_path),
    }


def _flux2_write_text_encoder_cache_manifest(
    source_path: Path,
    model_config,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    cache_path = cache_path or _flux2_text_encoder_cache_path(source_path, model_config)
    manifest = _flux2_text_encoder_cache_manifest_entry(source_path, model_config, cache_path)
    manifest_path = _flux2_text_encoder_cache_manifest_path(cache_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(".manifest.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, manifest_path)
    return manifest


def _flux2_refresh_package_text_encoder_cache_manifest_entry(
    package_path: Path,
    manifest: dict[str, Any],
    text_encoder_path: str | os.PathLike[str] | None,
    model_config,
    transformer_quant_contract: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    if not text_encoder_path or not _flux2_prepared_text_encoder_cache_allowed(model_config, transformer_quant_contract):
        return manifest, False
    source_path = Path(text_encoder_path).expanduser()
    if not source_path.is_file() or not _flux2_text_encoder_uses_comfy_quant(source_path):
        return manifest, False
    prepared = _flux2_text_encoder_cache_manifest_entry(source_path, model_config)
    updated = dict(manifest or {})
    components = dict(updated.get("components") or {})
    text_entry = dict(components.get("text_encoder") or {})
    if text_entry.get("prepared_cache") == prepared:
        return manifest, False
    text_entry["prepared_cache"] = prepared
    components["text_encoder"] = text_entry
    updated["components"] = components
    _write_flux2_manifest(package_path, updated)
    return updated, True


def _flux2_flatten_weight_tree(tree: Any, prefix: str = "") -> dict[str, mx.array]:
    flat: dict[str, mx.array] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flux2_flatten_weight_tree(value, child))
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            child = f"{prefix}.{index}" if prefix else str(index)
            flat.update(_flux2_flatten_weight_tree(value, child))
    else:
        if not prefix:
            raise RuntimeError("SDMLX FLUX.2 Klein CLIP cache: empty tensor key while flattening weights.")
        flat[prefix] = tree
    return flat


def _flux2_load_prepared_text_encoder_cache(path: Path, model_config) -> dict[str, Any] | None:
    source_identity = _flux2_file_identity(path)
    for cache_path in _flux2_text_encoder_cache_candidates(path, model_config):
        try:
            arrays, metadata = mx.load(str(cache_path), return_metadata=True)
            if not _flux2_text_encoder_cache_metadata_matches(
                metadata or {},
                source_identity,
                model_config,
            ):
                continue
            try:
                _flux2_write_text_encoder_cache_manifest(path, model_config, cache_path)
            except Exception:
                pass
            _flux2_log(f"SDMLX FLUX.2 Klein CLIP cache: hit {cache_path.parent.name}", verbose=True)
            return tree_unflatten(list(dict(arrays.items()).items()))
        except Exception:
            continue
    return None


def _flux2_store_prepared_text_encoder_cache(path: Path, model_config, weights: dict[str, Any], elapsed_s: float) -> None:
    cache_path = _flux2_text_encoder_cache_path(path, model_config)
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return
    flat = _flux2_flatten_weight_tree(weights)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.stem}.tmp.safetensors")
    metadata = _flux2_text_encoder_cache_metadata(path, model_config)
    try:
        mx.save_safetensors(str(tmp_path), flat, metadata=metadata)
        os.replace(tmp_path, cache_path)
        _flux2_write_text_encoder_cache_manifest(path, model_config, cache_path)
        size_gib = cache_path.stat().st_size / (1024**3)
        _flux2_log(
            "SDMLX FLUX.2 Klein CLIP cache: stored "
            f"{cache_path.parent.name} ({size_gib:.2f} GiB, source load {elapsed_s:.2f}s)",
            verbose=True,
        )
    except Exception as exc:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        _flux2_log(
            f"SDMLX FLUX.2 Klein CLIP cache: store skipped ({type(exc).__name__}: {exc})",
            verbose=True,
        )


def _load_flux2_text_encoder_weights(
    text_encoder_path: str | os.PathLike[str],
    model_config,
    *,
    prepared_cache: bool = False,
) -> dict[str, Any]:
    path = Path(text_encoder_path).expanduser()
    _validate_flux2_text_encoder_for_model(path, model_config)
    standalone_dir = _flux2_standalone_text_encoder_dir(path)
    if standalone_dir is not None:
        from dataclasses import replace
        from sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader
        from sdmlx_qwen_native.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

        component = next(
            component
            for component in Flux2KleinWeightDefinition.get_components()
            if component.name == "text_encoder"
        )
        component = replace(component, hf_subdir=standalone_dir.name)
        weights, _q_level, _version = WeightLoader._load_component(standalone_dir.parent, component)
        return weights

    info = _flux2_qwen3_text_encoder_info(path)
    if info is None:
        raise RuntimeError(f"SDMLX FLUX.2 Klein CLIP: unsupported Qwen3 text encoder file: {path.name}")

    use_prepared_cache = (
        bool(prepared_cache)
        and _flux2_text_encoder_uses_comfy_quant(path)
    )
    if use_prepared_cache:
        cached_weights = _flux2_load_prepared_text_encoder_cache(path, model_config)
        if cached_weights is not None:
            return cached_weights

    from safetensors.torch import safe_open as torch_safe_open
    from sdmlx_qwen_native.models.common.config.model_config import ModelConfig
    from sdmlx_qwen_native.models.common.weights.mapping.weight_mapper import WeightMapper
    from sdmlx_qwen_native.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping

    t_load = time.perf_counter()
    raw_weights: dict[str, mx.array] = {}
    skipped_suffixes = (".weight_scale", ".scale_weight", ".input_scale", ".comfy_quant")
    with torch_safe_open(str(path), framework="pt", device="cpu") as handle:
        key_set = set(handle.keys())
        for key in handle.keys():
            if key.endswith(skipped_suffixes):
                continue
            if not key.startswith("model."):
                continue
            tensor = handle.get_tensor(key)
            is_comfy_quant_weight = key.endswith(".weight") and f"{key[: -len('.weight')]}.comfy_quant" in key_set
            if is_comfy_quant_weight:
                tensor = _flux2_dequantize_comfy_quant_weight(handle, key, tensor, model_config)
                tensor = tensor.to(torch.float16)
            elif "float8" in str(tensor.dtype):
                tensor = tensor.float()
                if key.endswith(".weight"):
                    base_key = key[: -len(".weight")]
                    for scale_key in (f"{base_key}.weight_scale", f"{base_key}.scale_weight"):
                        if scale_key in key_set:
                            scale = handle.get_tensor(scale_key).float()
                            while scale.ndim < tensor.ndim:
                                scale = scale.unsqueeze(-1)
                            tensor = tensor * scale
                            break
                tensor = tensor.to(torch.float16)
            elif tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            raw_weights[key] = mx.array(tensor.detach().cpu().numpy())

    precision = ModelConfig.precision
    raw_weights = {
        key: value if value.dtype == precision else value.astype(precision)
        for key, value in raw_weights.items()
    }
    weights = WeightMapper.apply_mapping(
        hf_weights=raw_weights,
        mapping=Flux2WeightMapping.get_text_encoder_mapping(),
    )
    if use_prepared_cache:
        _flux2_store_prepared_text_encoder_cache(path, model_config, weights, time.perf_counter() - t_load)
    return weights


def _flux2_runtime_text_encoder_projection_input(model: Any) -> int | None:
    text_encoder = getattr(model, "text_encoder", None)
    try:
        layer0 = text_encoder.layers[0]
        weight = layer0.self_attn.k_proj.weight
        if len(weight.shape) == 2:
            return int(weight.shape[1])
    except Exception:
        return None
    return None


def _flux2_runtime_text_encoder_projection_mismatches(
    model: Any,
    expected_hidden: int,
) -> list[tuple[str, tuple[int, ...]]]:
    if not expected_hidden:
        return []
    text_encoder = getattr(model, "text_encoder", None)
    mismatches: list[tuple[str, tuple[int, ...]]] = []
    try:
        layers = list(text_encoder.layers)
    except Exception:
        return mismatches
    for index, layer in enumerate(layers):
        for name, module in (
            ("self_attn.q_proj", layer.self_attn.q_proj),
            ("self_attn.k_proj", layer.self_attn.k_proj),
            ("self_attn.v_proj", layer.self_attn.v_proj),
            ("mlp.gate_proj", layer.mlp.gate_proj),
        ):
            try:
                shape = tuple(int(value) for value in module.weight.shape)
            except Exception:
                continue
            if len(shape) == 2 and int(shape[1]) != expected_hidden:
                mismatches.append((f"layers.{index}.{name}.weight", shape))
    return mismatches


def _load_flux2_packed_model(
    root: Path,
    *,
    edit: bool,
    model_config,
    vae_variant: str = _FLUX2_VAE_STANDARD,
    vae_path: str | None = None,
    clip_path: str | None = None,
    tokenizer_path: str | None = None,
    lora_specs: tuple[tuple[str, float, tuple], ...] = (),
    lora_quant_bake_mode: str | None = None,
):
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader
    from sdmlx_qwen_native.models.flux2.flux2_initializer import Flux2Initializer
    from sdmlx_qwen_native.models.flux2.variants import Flux2Klein, Flux2KleinEdit
    from sdmlx_qwen_native.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    transformer_path = _flux2_transformer_file(root)
    if not _is_packed_flux2_transformer(transformer_path):
        raise RuntimeError(f"SDMLX FLUX.2 Klein: packed transformer not found in {root}")
    transformer_quant_contract = _flux2_inspect_transformer_quant_contract(transformer_path)

    cls = Flux2KleinEdit if edit else Flux2Klein
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    Flux2Initializer._init_config(model, model_config)
    model.transformer_quant_contract = transformer_quant_contract
    tokenizer_root = Path(tokenizer_path).expanduser().parent if tokenizer_path else root
    Flux2Initializer._init_tokenizers(model, str(tokenizer_root))
    Flux2Initializer._init_models(model, vae_variant=vae_variant)

    for component in Flux2KleinWeightDefinition.get_components():
        if component.name == "transformer":
            continue
        if component.name == "vae" and vae_variant == _FLUX2_VAE_SMALL_DECODER:
            continue
        if component.name == "text_encoder" and clip_path:
            weights = _load_flux2_text_encoder_weights(
                clip_path,
                model_config,
                prepared_cache=_flux2_prepared_text_encoder_cache_allowed(model_config, transformer_quant_contract),
            )
        elif component.name == "vae" and vae_path:
            weights = Flux2Initializer._load_vae_from_file(vae_path).components["vae"]
        else:
            weights, _q_level, _version = WeightLoader._load_component(root, component)
        if component.name == "vae":
            model.vae.update(weights, strict=False)
        elif component.name == "text_encoder":
            model.text_encoder.update(weights, strict=False)
        del weights
        gc.collect()
        mx.clear_cache()
    if vae_variant == _FLUX2_VAE_SMALL_DECODER:
        if not vae_path:
            raise RuntimeError("SDMLX FLUX.2 Klein: small_decoder VAE requested without a VAE path.")
        vae_weights = Flux2Initializer._load_vae_from_file(vae_path).components["vae"]
        model.vae.update(vae_weights, strict=False)
        del vae_weights
        gc.collect()
        mx.clear_cache()

    has_quantized_modules = _load_flux2_packed_transformer_into_model(
        transformer_path,
        model_config,
        model.transformer,
        quant_contract=transformer_quant_contract,
    )
    if has_quantized_modules:
        model.bits = 8
    else:
        model.bits = None
    lora_paths = [path for path, _scale, _identity in lora_specs] or None
    lora_scales = [scale for _path, scale, _identity in lora_specs] or None
    Flux2Initializer._apply_lora(model, lora_paths, lora_scales)
    lora_quant_bake_mode = str(lora_quant_bake_mode or _flux2_lora_quant_bake_mode()).strip().lower().replace("-", "_")
    if lora_quant_bake_mode == _FLUX2_LAB_QUANTIZED_REQUANTIZE:
        lora_quant_bake_mode = _FLUX2_LORA_QUANT_BAKE_REQUANTIZE
    elif lora_quant_bake_mode == _FLUX2_LAB_QUANTIZED_DENSE_TOUCHED:
        lora_quant_bake_mode = _FLUX2_LORA_QUANT_BAKE_DENSE_TOUCHED
    if _flux2_dense_lora_bake_allowed(
        model_config,
        transformer_quant_contract,
        has_quantized_modules=has_quantized_modules,
        lora_specs=lora_specs,
    ):
        bake_stats = _flux2_bake_dense_lora_with_stats(model.transformer)
        if int(bake_stats.get("baked_loras") or 0) > 0:
            _flux2_log(
                "SDMLX FLUX.2 Klein: baked LoRA into dense BF16 weights "
                f"(modules={int(bake_stats.get('baked_modules') or 0)}, "
                f"loras={int(bake_stats.get('baked_loras') or 0)}, "
                f"passthrough={int(bake_stats.get('passthrough') or 0)}, "
                f"skipped={int(bake_stats.get('skipped') or 0)}, "
                f"time={float(bake_stats.get('seconds') or 0.0):.2f}s)",
                verbose=True,
            )
    elif has_quantized_modules and lora_specs and lora_quant_bake_mode != _FLUX2_LORA_QUANT_BAKE_OFF:
        from sdmlx_qwen_native.models.common.lora.mapping.lora_saver import LoRASaver

        keep_touched_dense = lora_quant_bake_mode == _FLUX2_LORA_QUANT_BAKE_DENSE_TOUCHED
        bake_stats = LoRASaver.bake_quantized_lora_with_stats(
            model.transformer,
            keep_touched_dense=keep_touched_dense,
        )
        if int(bake_stats.get("baked_loras") or 0) > 0:
            _flux2_log(
                "SDMLX FLUX.2 Klein: baked LoRA into quantized weights "
                f"(mode={lora_quant_bake_mode}, "
                f"modules={int(bake_stats.get('baked_modules') or 0)}, "
                f"loras={int(bake_stats.get('baked_loras') or 0)}, "
                f"dense={int(bake_stats.get('dense_modules') or 0)}, "
                f"requantized={int(bake_stats.get('requantized_modules') or 0)}, "
                f"passthrough={int(bake_stats.get('passthrough') or 0)}, "
                f"skipped={int(bake_stats.get('skipped') or 0)}, "
                f"time={float(bake_stats.get('seconds') or 0.0):.2f}s)",
                verbose=True,
            )
    return model


def _flux2_model_supports_argument(model: Any, argument: str) -> bool:
    try:
        return argument in inspect.signature(model.generate_image).parameters
    except (TypeError, ValueError):
        return False


def _ensure_flux2_text_encoder_loaded(model: Any, root: Path, clip_path: str | None = None) -> None:
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader
    from sdmlx_qwen_native.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
    from sdmlx_qwen_native.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    expected_hidden = int((model.model_config.text_encoder_overrides or {}).get("hidden_size") or 0)
    existing_input = _flux2_runtime_text_encoder_projection_input(model)
    existing_mismatches = _flux2_runtime_text_encoder_projection_mismatches(model, expected_hidden)
    if "text_encoder" in model and (not expected_hidden or (existing_input == expected_hidden and not existing_mismatches)):
        return
    if "text_encoder" in model:
        del model.text_encoder
        gc.collect()
        mx.clear_cache()

    model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)
    component = next(
        component
        for component in Flux2KleinWeightDefinition.get_components()
        if component.name == "text_encoder"
    )
    if clip_path:
        weights = _load_flux2_text_encoder_weights(
            clip_path,
            model.model_config,
            prepared_cache=_flux2_prepared_text_encoder_cache_allowed(
                model.model_config,
                getattr(model, "transformer_quant_contract", None),
            ),
        )
    else:
        weights, _q_level, _version = WeightLoader._load_component(root, component)
    model.text_encoder.update(weights, strict=False)
    del weights
    gc.collect()
    mx.clear_cache()
    loaded_input = _flux2_runtime_text_encoder_projection_input(model)
    loaded_mismatches = _flux2_runtime_text_encoder_projection_mismatches(model, expected_hidden)
    if expected_hidden and (loaded_input != expected_hidden or loaded_mismatches):
        preview = ", ".join(f"{key}:{shape}" for key, shape in loaded_mismatches[:3])
        raise RuntimeError(
            "SDMLX FLUX.2 Klein text encoder runtime shape mismatch after load "
            f"(projection input={loaded_input}, model expects {expected_hidden}"
            + (f"; first mismatches: {preview}" if preview else "")
            + ")."
        )


def _load_flux2_model(
    root: Path,
    config_name: str,
    *,
    edit: bool = False,
    vae_variant: str = _FLUX2_VAE_STANDARD,
    vae_path: str | None = None,
    clip_path: str | None = None,
    tokenizer_path: str | None = None,
    lora_specs: tuple[tuple[str, float, tuple], ...] = (),
    require_text_encoder: bool = True,
    lora_patch_strategy: str | None = None,
):
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.flux2.flux2_initializer import Flux2Initializer
    from sdmlx_qwen_native.models.flux2.variants import Flux2Klein, Flux2KleinEdit

    mode = "edit" if edit else "txt2img"
    vae_variant = _normalize_flux2_vae_variant(vae_variant)
    lab_lora_patch_strategy = _normalize_flux2_lab_lora_patch_strategy(lora_patch_strategy)
    lora_quant_bake_mode = _flux2_lora_quant_bake_mode()
    transformer_path = _flux2_transformer_file(root)
    transformer_quant_contract = (
        _flux2_inspect_transformer_quant_contract(transformer_path)
        if _is_packed_flux2_transformer(transformer_path)
        else None
    )
    model_config = _flux2_model_config_by_name(config_name) if config_name else _model_config_for_root(root)
    quant_kind = _flux2_quant_contract_kind(transformer_quant_contract)
    raw_fp8_dense_dtype = _flux2_raw_fp8_dense_dtype(transformer_quant_contract) or ""
    dense_lora_reversible_enabled = False
    dense_lora_weight_patch_enabled = _flux2_dense_lora_weight_patch_enabled(transformer_quant_contract, model_config)
    lora_rebind_enabled = (
        not dense_lora_weight_patch_enabled
        and _flux2_runtime_lora_rebind_enabled(transformer_quant_contract, model_config)
    )
    if lab_lora_patch_strategy and lab_lora_patch_strategy not in {
        _FLUX2_LAB_PRODUCT_CURRENT,
        _FLUX2_LAB_NO_LORA,
    }:
        if lab_lora_patch_strategy == _FLUX2_LAB_RUNTIME_REBIND:
            dense_lora_weight_patch_enabled = False
            lora_rebind_enabled = bool(lora_specs)
        elif lab_lora_patch_strategy == _FLUX2_LAB_DENSE_WEIGHT_PATCH:
            if quant_kind not in {_FLUX2_QUANT_DENSE_BF16, _FLUX2_QUANT_RAW_FP8}:
                raise RuntimeError(
                    "SDMLX FLUX.2 Lab: dense_weight_patch requires dense_bf16 or raw_fp8_unscaled dense "
                    f"compatibility, got {quant_kind}."
                )
            if quant_kind == _FLUX2_QUANT_RAW_FP8 and not raw_fp8_dense_dtype:
                raise RuntimeError(
                    "SDMLX FLUX.2 Lab: dense_weight_patch for raw_fp8_unscaled requires the dense "
                    "compatibility route. Set raw-FP8 dense mode to bf16/fp16 or use another lab policy."
                )
            dense_lora_weight_patch_enabled = True
            dense_lora_reversible_enabled = quant_kind == _FLUX2_QUANT_RAW_FP8
            lora_rebind_enabled = False
        elif lab_lora_patch_strategy in {_FLUX2_LAB_QUANTIZED_REQUANTIZE, _FLUX2_LAB_QUANTIZED_DENSE_TOUCHED}:
            if quant_kind not in {_FLUX2_QUANT_SCALED_FP8, _FLUX2_QUANT_COMFY, _FLUX2_QUANT_COMFY_MXFP8}:
                raise RuntimeError(
                    "SDMLX FLUX.2 Lab: quantized LoRA patching requires scaled_fp8/comfy_quant, "
                    f"got {quant_kind}."
                )
            dense_lora_weight_patch_enabled = False
            lora_rebind_enabled = False
            lora_quant_bake_mode = (
                _FLUX2_LORA_QUANT_BAKE_DENSE_TOUCHED
                if lab_lora_patch_strategy == _FLUX2_LAB_QUANTIZED_DENSE_TOUCHED
                else _FLUX2_LORA_QUANT_BAKE_REQUANTIZE
            )
    lora_cache_key = () if (lora_rebind_enabled or dense_lora_weight_patch_enabled) else tuple(lora_specs)
    if lab_lora_patch_strategy and lab_lora_patch_strategy not in {_FLUX2_LAB_PRODUCT_CURRENT, _FLUX2_LAB_NO_LORA}:
        lora_mode_key = f"strategy:{lab_lora_patch_strategy}:{lora_quant_bake_mode}"
    elif dense_lora_weight_patch_enabled:
        lora_mode_key = "dense_weight_patch"
    elif lora_rebind_enabled:
        lora_mode_key = "runtime_rebind"
    else:
        lora_mode_key = lora_quant_bake_mode
    key = (
        str(root.resolve()),
        config_name,
        mode,
        vae_variant,
        str(vae_path or ""),
        str(clip_path or ""),
        str(tokenizer_path or ""),
        raw_fp8_dense_dtype,
        lora_cache_key,
        lora_mode_key,
    )
    cached = _FLUX2_MODEL_CACHE.get(key)
    if cached is not None:
        cached._sdmlx_lora_patch_reversible = bool(dense_lora_reversible_enabled)
        if edit and not _flux2_model_supports_argument(cached, "reference_sizes"):
            _FLUX2_MODEL_CACHE.pop(key, None)
            _ensure_suite_qwen_native_runtime()
        elif dense_lora_weight_patch_enabled and _flux2_dense_lora_weight_patch_requires_reload(cached, lora_specs):
            _FLUX2_MODEL_CACHE.pop(key, None)
            gc.collect()
            mx.clear_cache()
            cached = None
        else:
            if require_text_encoder:
                _ensure_flux2_text_encoder_loaded(cached, root, clip_path=clip_path)
            if dense_lora_weight_patch_enabled:
                _flux2_apply_dense_lora_weight_patch(cached, lora_specs)
            elif lora_rebind_enabled:
                _flux2_rebind_runtime_loras(cached, lora_specs)
            _apply_flux2_kv_first_step_barriers(cached, 0)
            _apply_flux2_decode_tiling_default(cached)
            _clear_flux2_model_cache(keep_key=key)
            return cached
    if key not in _FLUX2_MODEL_CACHE:
        _clear_flux2_model_cache()
        _ensure_suite_qwen_native_runtime()
        from sdmlx_qwen_native.models.flux2.variants import Flux2Klein, Flux2KleinEdit
    else:
        return cached
    _validate_flux2_lora_specs(lora_specs, model_config)
    load_lora_specs = () if (lora_rebind_enabled or dense_lora_weight_patch_enabled) else lora_specs
    if _is_packed_flux2_transformer(transformer_path):
        model = _load_flux2_packed_model(
            root,
            edit=edit,
            model_config=model_config,
            vae_variant=vae_variant,
            vae_path=vae_path,
            clip_path=clip_path,
            tokenizer_path=tokenizer_path,
            lora_specs=load_lora_specs,
            lora_quant_bake_mode=lora_quant_bake_mode,
        )
    else:
        cls = Flux2KleinEdit if edit else Flux2Klein
        lora_paths = [path for path, _scale, _identity in load_lora_specs] or None
        lora_scales = [scale for _path, scale, _identity in load_lora_specs] or None
        model = cls(
            model_path=str(root),
            model_config=model_config,
            vae_variant=vae_variant,
            vae_path=vae_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
        if tokenizer_path:
            tokenizer_root = Path(tokenizer_path).expanduser().parent
            Flux2Initializer._init_tokenizers(model, str(tokenizer_root))
        if clip_path:
            from sdmlx_qwen_native.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder

            model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)
            weights = _load_flux2_text_encoder_weights(
                clip_path,
                model_config,
                prepared_cache=_flux2_prepared_text_encoder_cache_allowed(model_config, transformer_quant_contract),
            )
            model.text_encoder.update(weights, strict=False)
            del weights
            gc.collect()
            mx.clear_cache()
    if edit and not _flux2_model_supports_argument(model, "reference_sizes"):
        module = sys.modules.get(model.__class__.__module__)
        source = getattr(module, "__file__", model.__class__.__module__)
        raise RuntimeError(
            "SDMLX FLUX.2 Klein Edit loaded an incompatible native runtime "
            f"without reference_sizes support: {source}"
        )
    model._sdmlx_model_path = str(root)
    model._sdmlx_clip_path = str(clip_path or "")
    model._sdmlx_lora_patch_reversible = bool(dense_lora_reversible_enabled)
    if require_text_encoder:
        _ensure_flux2_text_encoder_loaded(model, root, clip_path=clip_path)
    model._sdmlx_lora_specs = tuple(load_lora_specs)
    if dense_lora_weight_patch_enabled:
        _flux2_apply_dense_lora_weight_patch(model, lora_specs)
    elif lora_rebind_enabled:
        _flux2_rebind_runtime_loras(model, lora_specs)
    _apply_flux2_kv_first_step_barriers(model, 0)
    _apply_flux2_decode_tiling_default(model)
    _FLUX2_MODEL_CACHE[key] = model
    _clear_flux2_model_cache(keep_key=key)
    return model


def _pil_to_comfy_image(image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


def _comfy_image_to_pil(image: torch.Tensor):
    from PIL import Image

    tensor = image
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    if tuple(tensor.shape[:1]) == (1,) and tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise RuntimeError(f"SDMLX FLUX.2 Klein Edit: expected BHWC/BWC RGB image tensor, got {tuple(image.shape)}")
    array = np.clip(np.asarray(tensor, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def _comfy_image_to_flux2_array(image: torch.Tensor) -> mx.array:
    tensor = image
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    array = np.asarray(tensor, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise RuntimeError(f"SDMLX FLUX.2 VAE Encode: expected BHWC RGB image tensor, got {tuple(image.shape)}")
    array = np.clip(array, 0.0, 1.0)
    mlx_image = mx.array(array).transpose(0, 3, 1, 2)
    return 2.0 * mlx_image - 1.0


def _flux2_decoded_to_comfy_image(decoded: mx.array) -> torch.Tensor:
    if decoded.ndim == 5 and decoded.shape[2] == 1:
        decoded = mx.squeeze(decoded, axis=2)
    image = decoded.transpose(0, 2, 3, 1)
    image = mx.clip((image / 2.0) + 0.5, 0.0, 1.0).astype(mx.float32)
    mx.eval(image)
    array = np.asarray(image, dtype=np.float32)
    result = torch.from_numpy(array)
    del image
    mx.clear_cache()
    return result


def _load_flux2_vae_component(mlx_vae: dict[str, Any] | None, fallback: dict[str, Any] | None = None):
    component = dict(fallback or {})
    if isinstance(mlx_vae, dict):
        component.update(mlx_vae)
    if component.get("type") != FLUX2_MODEL_FAMILY:
        raise RuntimeError("SDMLX FLUX.2 VAE: connect a FLUX.2 Klein VAE from the SDMLX VAE Loader.")

    vae_variant = _normalize_flux2_vae_variant(component.get("vae_variant") or _FLUX2_VAE_STANDARD)
    vae_path = str(component.get("vae_path") or "")
    model_path = str(component.get("model_path") or "")
    key = (vae_variant, vae_path, model_path)
    cached = _FLUX2_VAE_CACHE.get(key)
    if cached is not None:
        return cached

    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader
    from sdmlx_qwen_native.models.flux2.flux2_initializer import Flux2Initializer
    from sdmlx_qwen_native.models.flux2.model.flux2_vae.vae import Flux2VAE
    from sdmlx_qwen_native.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    decoder_channels = (
        Flux2Initializer.SMALL_DECODER_CHANNELS if vae_variant == _FLUX2_VAE_SMALL_DECODER else None
    )
    vae = Flux2VAE(decoder_block_out_channels=decoder_channels)
    if vae_variant == _FLUX2_VAE_SMALL_DECODER:
        path = vae_path or str(_resolve_flux2_small_decoder_vae())
        weights = Flux2Initializer._load_vae_from_file(path).components["vae"]
    elif vae_path:
        weights = Flux2Initializer._load_vae_from_file(vae_path).components["vae"]
    elif model_path:
        weights, _q_level, _version = WeightLoader._load_component(
            Path(model_path).expanduser(),
            Flux2KleinWeightDefinition.get_vae_component(),
        )
    else:
        raise RuntimeError("SDMLX FLUX.2 VAE: no VAE file or model root is available.")
    vae.update(weights, strict=False)
    del weights
    gc.collect()
    mx.clear_cache()
    _FLUX2_VAE_CACHE[key] = vae
    return vae


def flux2_conditioning_from_text(text: str, mlx_clip: dict[str, Any] | None = None) -> dict[str, Any]:
    if mlx_clip is not None and (not isinstance(mlx_clip, dict) or mlx_clip.get("type") != FLUX2_MODEL_FAMILY):
        raise RuntimeError("SDMLX FLUX.2 CLIP Text Encode: connect a FLUX.2 Klein CLIP output.")
    return {
        "type": FLUX2_MODEL_FAMILY,
        "text": str(text or ""),
        "clip": dict(mlx_clip) if isinstance(mlx_clip, dict) else None,
        "reference_latents": [],
    }


def encode_flux2_image_with_vae(image: torch.Tensor, mlx_vae: dict[str, Any]) -> dict[str, Any]:
    vae = _load_flux2_vae_component(mlx_vae)
    from sdmlx_qwen_native.models.common.vae.vae_util import VAEUtil
    from sdmlx_qwen_native.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
    from sdmlx_qwen_native.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers

    if hasattr(image, "shape"):
        pixel_height = int(image.shape[1])
        pixel_width = int(image.shape[2])
    else:
        array = np.asarray(image, dtype=np.float32)
        pixel_height = int(array.shape[1])
        pixel_width = int(array.shape[2])
    encoded = VAEUtil.encode(vae=vae, image=_comfy_image_to_flux2_array(image), tiling_config=None)
    encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
    encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
    encoded = Flux2LatentCreator.patchify_latents(encoded)
    encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=vae)
    packed = Flux2LatentCreator.pack_latents(encoded)
    mx.eval(packed)
    return {
        "type": FLUX2_MODEL_FAMILY,
        "role": "reference_latent",
        "samples": packed,
        "width": pixel_width,
        "height": pixel_height,
        "batch_size": int(packed.shape[0]),
        "flux2_grid_height": int(encoded.shape[2]),
        "flux2_grid_width": int(encoded.shape[3]),
        "flux2_channels": int(encoded.shape[1]),
    }


def decode_flux2_latent_with_vae(mlx_latent: dict[str, Any], mlx_vae: dict[str, Any]) -> torch.Tensor:
    profiler = _Flux2PhaseProfiler("VAE Decode FLUX.2-klein")
    profiler.mark("start")
    if not isinstance(mlx_latent, dict) or mlx_latent.get("type") != FLUX2_MODEL_FAMILY:
        raise RuntimeError("SDMLX FLUX.2 VAE Decode: expected a FLUX.2 Klein latent.")
    vae = _load_flux2_vae_component(mlx_vae, fallback=mlx_latent.get("vae_component"))
    profiler.mark("vae ready")
    latents = mlx_latent.get("samples")
    if latents is None:
        raise RuntimeError("SDMLX FLUX.2 VAE Decode: latent does not contain samples.")
    if not isinstance(latents, mx.array):
        latents = mx.array(np.asarray(latents, dtype=np.float32))
    profiler.mark("latents ready")
    latent_height = int(mlx_latent.get("latent_height") or 0)
    latent_width = int(mlx_latent.get("latent_width") or 0)
    if latent_height <= 0 or latent_width <= 0:
        width = int(mlx_latent.get("width") or 0)
        height = int(mlx_latent.get("height") or 0)
        latent_height = height // 16
        latent_width = width // 16
    packed = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)
    profiler.mark("packed")
    tiling_config = _flux2_default_decode_tiling_config()
    try:
        decoded = vae.decode_packed_latents(packed, tiling_config=tiling_config)
    except Exception:
        if (
            os.environ.get("SDMLX_FLUX2_VAE_DECODE_TILES") is not None
            or getattr(tiling_config, "vae_decode_tiles_per_dim", None) is not None
        ):
            raise
        print("SDMLX FLUX.2 VAE Decode: regular decode failed, retrying with 2x2 tiled decode.")
        mx.clear_cache()
        decoded = vae.decode_packed_latents(packed, tiling_config=_flux2_fallback_decode_tiling_config())
    mx.eval(decoded)
    profiler.mark("decode eval")
    image = _flux2_decoded_to_comfy_image(decoded)
    profiler.mark("to comfy image")
    del decoded, packed
    gc.collect()
    mx.clear_cache()
    profiler.mark("cleanup")
    profiler.emit()
    return image


def _flux2_reference_ids(batch_size: int, height: int, width: int, index: int) -> mx.array:
    from sdmlx_qwen_native.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator

    dummy = mx.zeros((batch_size, 1, height, width), dtype=mx.float32)
    return Flux2LatentCreator.prepare_grid_ids(dummy, t_coord=10 + 10 * index)


def _floor_to_flux2_multiple(value: float, multiple: int = 16) -> int:
    return max(multiple, int(float(value)) // multiple * multiple)


def _scale_image_size_to_megapixels(image_size: tuple[int, int], megapixels: float) -> tuple[int, int]:
    source_width, source_height = image_size
    if source_width <= 0 or source_height <= 0:
        return 16, 16
    total = max(0.01, float(megapixels)) * 1024 * 1024
    scale_by = math.sqrt(total / float(source_width * source_height))
    width = round(source_width * scale_by)
    height = round(source_height * scale_by)
    return _floor_to_flux2_multiple(width), _floor_to_flux2_multiple(height)


def _flux2_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    return _floor_to_flux2_multiple(width), _floor_to_flux2_multiple(height)


def _flux2_validate_component_root(sdmlx_model: dict[str, Any], component: dict[str, Any], label: str) -> None:
    if not isinstance(component, dict) or component.get("type") != FLUX2_MODEL_FAMILY:
        raise RuntimeError(f"SDMLX FLUX.2 Klein {label}: connect a FLUX.2 Klein {label.lower()} output.")
    model_config_name = str(sdmlx_model.get("model_config") or "")
    if model_config_name:
        model_config = _flux2_model_config_by_name(model_config_name)
    else:
        model_config = _model_config_for_root(Path(str(sdmlx_model["model_path"])))
    if label == "CLIP" and component.get("text_encoder_path"):
        _validate_flux2_text_encoder_for_model(Path(str(component["text_encoder_path"])), model_config)
    component_model_path = component.get("model_path")
    if not component_model_path:
        return
    model_root = Path(sdmlx_model["model_path"]).expanduser()
    component_root = Path(component_model_path).expanduser()
    model_key = str(model_root.absolute()).lower()
    component_key = str(component_root.absolute()).lower()
    if component_key != model_key:
        raise RuntimeError(
            "SDMLX FLUX.2 Klein: model/CLIP/VAE components must come from the same .sdmlx runtime root "
            f"({model_root.name} != {component_root.name})."
        )


def _flux2_model_with_components(
    sdmlx_model: dict[str, Any],
    mlx_clip: Any | None = None,
    mlx_vae: Any | None = None,
) -> dict[str, Any]:
    model = dict(sdmlx_model)
    if mlx_clip is not None:
        _flux2_validate_component_root(model, mlx_clip, "CLIP")
        model["clip_path"] = mlx_clip.get("text_encoder_path")
        model["tokenizer_path"] = mlx_clip.get("tokenizer_path")
    if mlx_vae is not None:
        _flux2_validate_component_root(model, mlx_vae, "VAE")
        if mlx_vae.get("vae_variant"):
            model["vae_variant"] = str(mlx_vae["vae_variant"])
        if mlx_vae.get("vae_path"):
            model["vae_path"] = str(mlx_vae["vae_path"])
    return model


class SDMLXFlux2ScaleToMegapixels:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.upscale_methods, {"default": "lanczos"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resolution_steps": ("INT", {"default": 1, "min": 1, "max": 256}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "scale"
    CATEGORY = "SDMLX/Image"

    def scale(self, image, upscale_method: str, megapixels: float, resolution_steps: int = 1):
        samples = image.movedim(-1, 1)
        total = float(megapixels) * 1024 * 1024
        source_height = int(samples.shape[2])
        source_width = int(samples.shape[3])
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError("SDMLX FLUX.2 Scale To Megapixels: input image has invalid dimensions.")
        resolution_steps = max(1, int(resolution_steps))
        scale_by = math.sqrt(total / float(source_width * source_height))
        width = round(source_width * scale_by / resolution_steps) * resolution_steps
        height = round(source_height * scale_by / resolution_steps) * resolution_steps
        scaled = comfy.utils.common_upscale(samples, int(width), int(height), str(upscale_method), "disabled")
        return (scaled.movedim(1, -1),)


class SDMLXFlux2EmptyLatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "flux2_dimensions": (_flux2_dimension_options(), {"default": "custom"}),
            }
        }

    RETURN_TYPES = ("LATENT,mlx_latent",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "SDMLX/Latent"

    def generate(self, width: int, height: int, batch_size: int = 1, flux2_dimensions: str = "custom"):
        width, height = _parse_flux2_dimension_option(flux2_dimensions, width, height)
        latent = torch.zeros(
            [int(batch_size), 128, int(height) // 16, int(width) // 16],
            device="cpu",
        )
        return (
            {
                "type": FLUX2_MODEL_FAMILY,
                "samples": latent,
                "width": int(width),
                "height": int(height),
                "batch_size": int(batch_size),
            },
        )


class SDMLXFlux2ReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("mlx_conditioning",),
                "latent": ("LATENT,mlx_latent",),
            }
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "append"
    CATEGORY = "SDMLX/Conditioning"

    def append(self, conditioning, latent):
        if isinstance(conditioning, dict) and conditioning.get("type") == "flux1":
            if not isinstance(latent, dict) or "samples" not in latent:
                raise RuntimeError("SDMLX Reference Latent: connect a FLUX.1 VAE-encoded latent.")
            refs = list(conditioning.get("reference_latents") or [])
            refs.append(latent["samples"])
            out = dict(conditioning)
            out["reference_latents"] = refs
            out["reference_latents_method"] = str(out.get("reference_latents_method") or "offset")
            return (out,)

        if not isinstance(conditioning, dict) or conditioning.get("type") != FLUX2_MODEL_FAMILY:
            raise RuntimeError("SDMLX FLUX.2 Reference Latent: connect a FLUX.2 Klein conditioning.")
        if not isinstance(latent, dict) or latent.get("type") != FLUX2_MODEL_FAMILY:
            raise RuntimeError("SDMLX FLUX.2 Reference Latent: connect a FLUX.2 Klein VAE-encoded latent.")
        packed = latent.get("samples")
        if packed is None or latent.get("role") != "reference_latent":
            raise RuntimeError("SDMLX FLUX.2 Reference Latent: input latent must come from SDMLX VAE Encode.")
        refs = list(conditioning.get("reference_latents") or [])
        index = len(refs)
        batch_size = int(latent.get("batch_size") or 1)
        grid_height = int(latent.get("flux2_grid_height") or 0)
        grid_width = int(latent.get("flux2_grid_width") or 0)
        if grid_height <= 0 or grid_width <= 0:
            raise RuntimeError("SDMLX FLUX.2 Reference Latent: encoded latent is missing FLUX.2 grid metadata.")
        ref = {
            "latents": packed,
            "ids": _flux2_reference_ids(batch_size, grid_height, grid_width, index),
            "width": int(latent.get("width") or 0),
            "height": int(latent.get("height") or 0),
        }
        refs.append(ref)
        out = dict(conditioning)
        out["reference_latents"] = refs
        return (out,)


_FLUX2_ENHANCED_DOUBLE = "0-7:mid_img=0.55"
_FLUX2_ENHANCED_SINGLE = (
    "0:mid_img=0.22; "
    "1:mid_img=0.24; "
    "3:mid_img=0.28; "
    "4:mid_img=0.22; "
    "6:mid_img=0.26; "
    "7:mid_img=0.27; "
    "8:mid_img=0.25; "
    "10:mid_img=0.27; "
    "13:mid_img=0.27"
)
_FLUX2_ENHANCED_PRESETS = {
    "hard_lock": (0.040, 0.0250),
    "mid_lock": (0.200, 0.0700),
    "soft_lock": (0.500, 0.0700),
}
_FLUX2_KLEIN_ENHANCER_CONFIG_TYPE = "flux2_klein_enhancer_config"
_FLUX2_ENHANCER_ADVANCED_DEFAULTS: dict[str, Any] = {
    "similarity_floor": 0.200,
    "softmax_temperature": 0.0700,
    "mask_threshold": 1.0,
    "double_blocks": _FLUX2_ENHANCED_DOUBLE,
    "single_blocks": _FLUX2_ENHANCED_SINGLE,
    "reference_indices": "all",
    "active_scale": 1.0,
    "per_token_whiten": 0.0,
    "early_layer_scale": 1.0,
    "mid_layer_scale": 1.0,
    "late_layer_scale": 1.0,
    "debug": False,
}


def _flux2_parse_enhanced_schedule(text: str, max_block: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in str(text or "").split(";"):
        row = row.strip()
        if not row or ":" not in row:
            continue
        block_part, value_part = row.split(":", 1)
        value_part = value_part.strip()
        if "=" in value_part:
            key, value_part = value_part.split("=", 1)
            if key.strip().lower() not in {"mid", "mid_img"}:
                continue
        try:
            strength = float(value_part.strip())
        except ValueError:
            continue
        try:
            if "-" in block_part:
                lo_s, hi_s = block_part.split("-", 1)
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
            else:
                lo = hi = int(block_part.strip())
        except ValueError:
            continue
        if lo > hi:
            lo, hi = hi, lo
        for idx in range(max(0, lo), min(max_block, hi) + 1):
            out[idx] = strength
    return out


def _flux2_parse_reference_indices(text: str, count: int) -> list[int]:
    if count <= 0:
        return []
    value = str(text or "all").strip().lower()
    if value in {"", "all", "*"}:
        return list(range(count))
    selected: set[int] = set()
    for part in re.split(r"[;, ]+", value):
        if not part:
            continue
        try:
            if "-" in part:
                a_s, b_s = part.split("-", 1)
                a, b = int(a_s), int(b_s)
                if a > b:
                    a, b = b, a
                for idx in range(a, b + 1):
                    if 0 <= idx < count:
                        selected.add(idx)
            else:
                idx = int(part)
                if 0 <= idx < count:
                    selected.add(idx)
        except ValueError:
            continue
    return sorted(selected)


def _flux2_enhancer_advanced_values(
    enhancer_advanced: Any,
    optional: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    values = dict(_FLUX2_ENHANCER_ADVANCED_DEFAULTS)
    connected = False
    if isinstance(enhancer_advanced, dict) and enhancer_advanced.get("type") == _FLUX2_KLEIN_ENHANCER_CONFIG_TYPE:
        connected = True
        for key in values:
            if key in enhancer_advanced:
                values[key] = enhancer_advanced[key]

    # Keep old saved/test workflows from silently losing their custom controls
    # while the public sampler UI moves those widgets into the advanced node.
    legacy_keys = [key for key in values if key in optional]
    if legacy_keys:
        connected = True
        for key in legacy_keys:
            values[key] = optional[key]
    return values, connected


def _flux2_prepare_mask_2d(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None or not torch.is_tensor(mask):
        return None
    x = mask.detach().float().cpu()
    if x.dim() == 4:
        if x.shape[-1] in (1, 3, 4):
            x = x[0].mean(dim=-1)
        else:
            x = x[0, 0]
    elif x.dim() == 3:
        if x.shape[-1] in (1, 3, 4) and x.shape[0] != 1:
            x = x.mean(dim=-1)
        else:
            x = x[0]
    elif x.dim() != 2:
        return None
    return x.contiguous()


def _flux2_token_grid_for_reference(ref: dict[str, Any], token_count: int) -> tuple[int, int]:
    token_count = max(1, int(token_count))
    width = max(1, int(ref.get("width") or 1))
    height = max(1, int(ref.get("height") or 1))
    target = height / width
    best = (1, token_count)
    best_err = float("inf")
    limit = int(token_count ** 0.5) + 3
    for h in range(1, limit):
        if token_count % h != 0:
            continue
        w = token_count // h
        for hh, ww in ((h, w), (w, h)):
            err = abs((hh / max(ww, 1)) - target)
            if err < best_err:
                best = (hh, ww)
                best_err = err
    return best


def _flux2_mask_keep_indices(
    mask: torch.Tensor | None,
    ref: dict[str, Any],
    token_count: int,
    threshold: float,
) -> np.ndarray | None:
    x = _flux2_prepare_mask_2d(mask)
    if x is None:
        return None
    grid = _flux2_token_grid_for_reference(ref, token_count)
    resized = F.interpolate(x[None, None], size=grid, mode="bilinear", align_corners=False).view(-1)
    keep = resized >= float(threshold)
    if keep.numel() <= 0:
        return np.zeros((0,), dtype=np.int32)
    return torch.nonzero(keep, as_tuple=False).flatten().to(torch.int32).numpy()


def _flux2_clip_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _flux2_enhanced_text_scales(reference_balance: float) -> tuple[float, float]:
    balance = _flux2_clip_float(reference_balance, 0.0, 1.0)
    if balance <= 0.5:
        return balance * 2.0, 1.0
    return 1.0, (1.0 - balance) * 2.0


class _Flux2KleinEnhanceState:
    def __init__(
        self,
        *,
        ref_token_counts: list[int],
        text_token_count: int,
        selected_reference_indices: list[int],
        mask_keep_indices: list[np.ndarray | None],
        mask_behavior: str,
        identity_strength: float,
        similarity_floor: float,
        softmax_temperature: float,
        double_map: dict[int, float],
        single_map: dict[int, float],
        text_scale: float,
        reference_scale: float,
        active_scale: float,
        per_token_whiten: float,
        early_layer_scale: float,
        mid_layer_scale: float,
        late_layer_scale: float,
        color_anchor_means: mx.array | None,
        color_anchor_strength: float,
        debug: bool,
    ):
        self.ref_token_counts = [max(0, int(v)) for v in ref_token_counts]
        self.total_ref_tokens = int(sum(self.ref_token_counts))
        self.text_token_count = max(0, int(text_token_count))
        self.selected_reference_indices = [
            idx for idx in selected_reference_indices if 0 <= idx < len(self.ref_token_counts)
        ]
        if not self.selected_reference_indices and self.ref_token_counts:
            self.selected_reference_indices = list(range(len(self.ref_token_counts)))
        self.mask_keep_indices = mask_keep_indices
        self.mask_behavior = mask_behavior if mask_behavior in {"focus_only", "zero_unmasked_tokens"} else "focus_only"
        self.identity_strength = max(0.0, float(identity_strength))
        self.similarity_floor = _flux2_clip_float(similarity_floor, 0.0, 0.95)
        self.softmax_temperature = max(1e-6, float(softmax_temperature))
        self.double_map = dict(double_map)
        self.single_map = dict(single_map)
        self.text_scale = float(text_scale)
        self.reference_scale = float(reference_scale)
        self.active_scale = float(active_scale)
        self.per_token_whiten = float(per_token_whiten)
        self.early_layer_scale = float(early_layer_scale)
        self.mid_layer_scale = float(mid_layer_scale)
        self.late_layer_scale = float(late_layer_scale)
        self.color_anchor_means = color_anchor_means
        self.color_anchor_strength = _flux2_clip_float(color_anchor_strength, 0.0, 1.0)
        self.debug = bool(debug)
        self._mask_bias_cache: dict[tuple[int, int, str], mx.array | None] = {}

    @property
    def feature_transfer_enabled(self) -> bool:
        return self.identity_strength > 0.0 and self.total_ref_tokens > 0 and bool(self.double_map or self.single_map)

    @property
    def reference_steering_enabled(self) -> bool:
        return (
            self.feature_transfer_enabled
            or self.text_scale != 1.0
            or self.reference_scale != 1.0
            or (self.mask_behavior == "zero_unmasked_tokens" and any(v is not None for v in self.mask_keep_indices))
        )

    @property
    def color_anchor_enabled(self) -> bool:
        return self.color_anchor_strength > 0.0 and self.color_anchor_means is not None

    def apply_text_controls(self, prompt_embeds: mx.array) -> mx.array:
        changed = (
            self.active_scale != 1.0
            or self.per_token_whiten != 0.0
            or self.early_layer_scale != 1.0
            or self.mid_layer_scale != 1.0
            or self.late_layer_scale != 1.0
        )
        if not changed:
            return prompt_embeds
        active_end = int(prompt_embeds.shape[1])
        active = prompt_embeds[:, :active_end, :].astype(mx.float32)
        embed_dim = int(active.shape[-1])
        if self.per_token_whiten != 0.0:
            seq_mean = mx.mean(active, axis=1, keepdims=True)
            active = seq_mean + (active - seq_mean) * (1.0 + self.per_token_whiten)
        if self.active_scale != 1.0:
            active = active * self.active_scale
        if embed_dim % 3 == 0:
            slice_w = embed_dim // 3
            if self.early_layer_scale != 1.0:
                active = mx.concatenate(
                    [
                        active[:, :, :slice_w] * self.early_layer_scale,
                        active[:, :, slice_w:],
                    ],
                    axis=-1,
                )
            if self.mid_layer_scale != 1.0:
                active = mx.concatenate(
                    [
                        active[:, :, :slice_w],
                        active[:, :, slice_w : 2 * slice_w] * self.mid_layer_scale,
                        active[:, :, 2 * slice_w :],
                    ],
                    axis=-1,
                )
            if self.late_layer_scale != 1.0:
                active = mx.concatenate(
                    [
                        active[:, :, : 2 * slice_w],
                        active[:, :, 2 * slice_w :] * self.late_layer_scale,
                    ],
                    axis=-1,
                )
        return active.astype(prompt_embeds.dtype)

    def _ref_slices(self, total_seq: int) -> list[tuple[int, int, int]]:
        base = int(total_seq) - self.total_ref_tokens
        out: list[tuple[int, int, int]] = []
        offset = base
        for ref_idx, count in enumerate(self.ref_token_counts):
            end = offset + int(count)
            if count > 0:
                out.append((ref_idx, offset, end))
            offset = end
        return out

    def _selected_ref_slices(self, total_seq: int) -> list[tuple[int, int, int]]:
        selected = set(self.selected_reference_indices)
        return [item for item in self._ref_slices(total_seq) if item[0] in selected]

    def _masked_ref_tokens(self, tokens: mx.array, ref_idx: int, start: int, end: int) -> mx.array | None:
        ref = tokens[:, start:end, :]
        if ref_idx >= len(self.mask_keep_indices):
            return ref
        keep = self.mask_keep_indices[ref_idx]
        if keep is None:
            return ref
        if keep.size <= 0:
            return None
        indices = mx.array(keep.astype(np.int32))
        return mx.take(ref, indices, axis=1)

    def _source_bias(self, total_seq: int, text_len: int, dtype) -> mx.array | None:
        if self.mask_behavior != "zero_unmasked_tokens" or not any(v is not None for v in self.mask_keep_indices):
            return None
        key = (int(text_len), int(total_seq), str(dtype))
        if key in self._mask_bias_cache:
            return self._mask_bias_cache[key]
        bias = np.zeros((int(total_seq),), dtype=np.float32)
        for ref_idx, start, end in self._ref_slices(total_seq):
            if ref_idx >= len(self.mask_keep_indices):
                continue
            keep = self.mask_keep_indices[ref_idx]
            if keep is None:
                continue
            bias[start:end] = -1.0e9
            if keep.size > 0:
                keep_abs = keep.astype(np.int64) + int(start)
                keep_abs = keep_abs[(keep_abs >= start) & (keep_abs < end)]
                bias[keep_abs] = 0.0
        arr = mx.array(bias, dtype=dtype).reshape(1, 1, 1, int(total_seq))
        self._mask_bias_cache[key] = arr
        return arr

    def apply_kv_controls(
        self,
        key: mx.array,
        value: mx.array,
        *,
        text_len: int,
        total_seq: int,
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        parts_k: list[mx.array] = []
        parts_v: list[mx.array] = []
        cursor = 0
        text_len = max(0, min(int(text_len), int(total_seq)))
        if text_len > 0:
            scale = self.text_scale
            parts_k.append(key[:, :, :text_len, :] * scale)
            parts_v.append(value[:, :, :text_len, :] * scale)
            cursor = text_len
        gen_end = max(cursor, int(total_seq) - self.total_ref_tokens)
        if gen_end > cursor:
            parts_k.append(key[:, :, cursor:gen_end, :])
            parts_v.append(value[:, :, cursor:gen_end, :])
            cursor = gen_end
        if int(total_seq) > cursor:
            scale = self.reference_scale
            parts_k.append(key[:, :, cursor:, :] * scale)
            parts_v.append(value[:, :, cursor:, :] * scale)
        if parts_k:
            key = mx.concatenate(parts_k, axis=2)
            value = mx.concatenate(parts_v, axis=2)
        return key, value, self._source_bias(total_seq, text_len, key.dtype)

    def _strength_for(self, block_type: str, block_index: int) -> float:
        if block_type == "double":
            return float(self.double_map.get(int(block_index), 0.0)) * self.identity_strength
        if block_type == "single":
            return float(self.single_map.get(int(block_index), 0.0)) * self.identity_strength
        return 0.0

    def apply_feature_pull(
        self,
        attn: mx.array,
        *,
        text_len: int,
        total_seq: int,
        block_type: str,
        block_index: int,
    ) -> mx.array:
        strength = self._strength_for(block_type, block_index)
        if strength <= 0.0 or self.total_ref_tokens <= 0:
            return attn
        text_len = max(0, min(int(text_len), int(total_seq)))
        gen_start = text_len
        gen_end = int(total_seq) - self.total_ref_tokens
        if gen_end <= gen_start:
            return attn
        ref_parts: list[mx.array] = []
        for ref_idx, start, end in self._selected_ref_slices(total_seq):
            ref = self._masked_ref_tokens(attn, ref_idx, start, end)
            if ref is not None and int(ref.shape[1]) > 0:
                ref_parts.append(ref)
        if not ref_parts:
            return attn
        ref = mx.concatenate(ref_parts, axis=1).astype(mx.float32)
        gen = attn[:, gen_start:gen_end, :].astype(mx.float32)
        gen_centered = gen - mx.mean(gen, axis=1, keepdims=True)
        ref_centered = ref - mx.mean(ref, axis=1, keepdims=True)
        gen_norm = gen_centered / mx.sqrt(mx.sum(gen_centered * gen_centered, axis=-1, keepdims=True) + 1.0e-8)
        ref_norm = ref_centered / mx.sqrt(mx.sum(ref_centered * ref_centered, axis=-1, keepdims=True) + 1.0e-8)
        sim = gen_norm @ mx.transpose(ref_norm, (0, 2, 1))
        neg = mx.full(sim.shape, -1.0e9, dtype=sim.dtype)
        sim = mx.where(sim >= self.similarity_floor, sim, neg)
        weights = mx.softmax(sim / self.softmax_temperature, axis=-1)
        pooled = weights @ ref
        best = mx.max(sim, axis=-1)
        confidence = mx.clip((best - self.similarity_floor) / max(1.0 - self.similarity_floor, 1.0e-6), 0.0, 1.0)
        delta = (pooled - gen) * (confidence[..., None] * strength)
        adjusted = (gen + delta).astype(attn.dtype)
        return mx.concatenate([attn[:, :gen_start, :], adjusted, attn[:, gen_end:, :]], axis=1)

    def apply_color_anchor(self, latents: mx.array, noise: mx.array, sigma) -> mx.array:
        if not self.color_anchor_enabled:
            return noise
        try:
            sigma_value = float(np.asarray(sigma).reshape(-1)[0])
        except Exception:
            sigma_value = 0.0
        if sigma_value <= 1.0e-8:
            return noise
        x0 = latents - noise * sigma_value
        ref = self.color_anchor_means
        if ref is None:
            return noise
        ref = ref.astype(x0.dtype)
        cur = mx.mean(x0, axis=1, keepdims=True)
        corrected = x0 + (ref - cur) * self.color_anchor_strength
        return ((latents - corrected) / sigma_value).astype(noise.dtype)


def _flux2_enhanced_color_anchor_means(references: list[dict[str, Any]], ref_index: int = 0) -> mx.array | None:
    if not references:
        return None
    idx = max(0, min(int(ref_index), len(references) - 1))
    latents = references[idx].get("latents")
    if latents is None:
        return None
    # Packed reference latents are [B, tokens, channels]; keep only per-channel
    # means so spatial/identity structure is not copied by this control.
    return mx.mean(latents.astype(mx.float32), axis=1, keepdims=True)


def _flux2_enhanced_reference_token_counts(references: list[dict[str, Any]]) -> list[int]:
    counts: list[int] = []
    for ref in references:
        latents = ref.get("latents") if isinstance(ref, dict) else None
        counts.append(int(latents.shape[1]) if hasattr(latents, "shape") and len(latents.shape) >= 2 else 0)
    return counts


def _flux2_enhanced_direct_reference(
    image: torch.Tensor,
    mlx_vae: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    latent = _flux2_get_cached_direct_reference(image, mlx_vae)
    if latent is None:
        latent = encode_flux2_image_with_vae(image, mlx_vae)
        _flux2_store_direct_reference(image, mlx_vae, latent)
    packed = latent.get("samples")
    grid_height = int(latent.get("flux2_grid_height") or 0)
    grid_width = int(latent.get("flux2_grid_width") or 0)
    batch_size = int(latent.get("batch_size") or 1)
    if packed is None or grid_height <= 0 or grid_width <= 0:
        raise RuntimeError("SDMLX FLUX.2 Enhanced Edit: direct reference image could not be encoded.")
    return {
        "latents": packed,
        "ids": _flux2_reference_ids(batch_size, grid_height, grid_width, index),
        "width": int(latent.get("width") or 0),
        "height": int(latent.get("height") or 0),
    }


def _flux2_enhanced_apply_markers_hint(text: str) -> str:
    # Keep prompt text semantically intact while allowing future section-aware
    # controls to detect explicit markers. No arbitrary position fallback.
    return str(text or "")


class SDMLXFlux2KleinKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdmlx_model": (MODEL_TYPE,),
                "positive": ("mlx_conditioning",),
                "negative": ("mlx_conditioning",),
                "latent_image": ("LATENT,mlx_latent",),
                "seed": ("INT", {"default": 1234567890, "min": 0, "max": 2**63 - 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "preview": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("mlx_latent",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "SDMLX/Sampling"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        if _flux2_env_enabled("SDMLX_FLUX2_FORCE_RERUN"):
            return time.time_ns()
        return False

    @staticmethod
    def _conditioning_text(conditioning: Any, label: str) -> str:
        if not isinstance(conditioning, dict) or conditioning.get("type") != FLUX2_MODEL_FAMILY:
            raise RuntimeError(f"SDMLX KSampler (FLUX.2-klein): connect a FLUX.2 Klein {label} conditioning.")
        return str(conditioning.get("text") or "")

    @staticmethod
    def _conditioning_clip(*conditionings: Any) -> dict[str, Any] | None:
        for conditioning in conditionings:
            if isinstance(conditioning, dict) and isinstance(conditioning.get("clip"), dict):
                return conditioning["clip"]
        return None

    @staticmethod
    def _conditioning_refs(*conditionings: Any) -> list[dict[str, Any]]:
        for conditioning in conditionings:
            if isinstance(conditioning, dict):
                refs = conditioning.get("reference_latents") or []
                if refs:
                    return list(refs)
        return []

    @staticmethod
    def _latent_dimensions(latent_image: Any) -> tuple[int, int, int]:
        if not isinstance(latent_image, dict):
            raise RuntimeError("SDMLX KSampler (FLUX.2-klein): connect an Empty Latent Image FLUX.2.")
        width = int(latent_image.get("width") or 0)
        height = int(latent_image.get("height") or 0)
        batch_size = int(latent_image.get("batch_size") or 1)
        samples = latent_image.get("samples")
        if (width <= 0 or height <= 0) and hasattr(samples, "shape") and len(samples.shape) >= 4:
            height = int(samples.shape[-2]) * 16
            width = int(samples.shape[-1]) * 16
            batch_size = int(samples.shape[0])
        if width <= 0 or height <= 0:
            raise RuntimeError("SDMLX KSampler (FLUX.2-klein): latent image is missing width/height metadata.")
        return width, height, batch_size

    def sample(
        self,
        sdmlx_model,
        positive,
        negative,
        latent_image,
        seed: int,
        steps: int,
        guidance: float,
        preview: bool = False,
    ):
        profiler = _Flux2PhaseProfiler("KSampler FLUX.2-klein")
        profiler.mark("start")
        if not is_flux2_sdmlx_model(sdmlx_model):
            raise RuntimeError("SDMLX KSampler (FLUX.2-klein): connect a FLUX.2 Klein model.")
        clip = self._conditioning_clip(positive, negative)
        sdmlx_model = _flux2_model_with_components(sdmlx_model, clip, None)
        profiler.mark("components resolved")
        root = Path(sdmlx_model["model_path"])
        lora_specs = flux2_lora_specs_from_model(sdmlx_model)
        width, height, batch_size = self._latent_dimensions(latent_image)
        positive_text = self._conditioning_text(positive, "positive")
        negative_text = self._conditioning_text(negative, "negative")
        references = self._conditioning_refs(positive, negative)
        needs_negative_text = float(guidance) > 1.0
        cached_positive_text = _flux2_get_cached_text_conditioning(sdmlx_model, positive_text)
        cached_negative_text = (
            _flux2_get_cached_text_conditioning(sdmlx_model, negative_text)
            if needs_negative_text
            else None
        )
        require_text_encoder = cached_positive_text is None or (needs_negative_text and cached_negative_text is None)
        lora_patch_strategy = _flux2_suite_lora_patch_strategy(
            sdmlx_model,
            positive=positive,
            negative=negative,
        )
        model = _load_flux2_model(
            root,
            str(sdmlx_model.get("model_config") or ""),
            edit=True,
            vae_variant=str(sdmlx_model.get("vae_variant") or _FLUX2_VAE_STANDARD),
            vae_path=sdmlx_model.get("vae_path"),
            clip_path=sdmlx_model.get("clip_path"),
            tokenizer_path=sdmlx_model.get("tokenizer_path"),
            lora_specs=lora_specs,
            require_text_encoder=require_text_encoder,
            lora_patch_strategy=lora_patch_strategy,
        )
        profiler.mark("model ready")

        profiler.mark("conditioning metadata")

        supports_kv = bool(getattr(model.model_config, "supports_kv_cache", False))
        _apply_flux2_kv_first_step_barriers(model, 0, supports_kv=supports_kv)
        keep_text_encoder = _flux2_should_keep_text_encoder_standard(model, references)
        _apply_flux2_pre_encode_memory_policy(model, keep_text_encoder=keep_text_encoder)
        flow_shift = sdmlx_model.get("default_flow_shift")

        _ensure_suite_qwen_native_runtime()
        from sdmlx_qwen_native.models.common.config.config import Config
        from sdmlx_qwen_native.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers

        config = Config(
            model_config=model.model_config,
            num_inference_steps=int(steps),
            height=int(height),
            width=int(width),
            guidance=float(guidance),
            scheduler=_FLUX2_DEFAULT_SCHEDULER,
            flow_shift=float(flow_shift) if flow_shift is not None else None,
        )
        profiler.mark("scheduler config")

        try:
            if cached_positive_text is not None:
                prompt_embeds, text_ids = cached_positive_text
                profiler.mark("positive text cache")
            else:
                _flux2_clear_cache_before_text_encode(model, profiler, "positive text cache hygiene")
                prompt_embeds, text_ids = _Flux2KleinEditHelpers.encode_text(
                    positive_text,
                    tokenizer=model.tokenizers["qwen3"],
                    text_encoder=model.text_encoder,
                )
                _flux2_store_text_conditioning(sdmlx_model, positive_text, prompt_embeds, text_ids)
                profiler.mark("positive text encode")
            negative_prompt_embeds = None
            negative_text_ids = None
            if needs_negative_text:
                if cached_negative_text is not None:
                    negative_prompt_embeds, negative_text_ids = cached_negative_text
                    profiler.mark("negative text cache")
                else:
                    _flux2_clear_cache_before_text_encode(model, profiler, "negative text cache hygiene")
                    negative_prompt_embeds, negative_text_ids = _Flux2KleinEditHelpers.encode_text(
                        negative_text,
                        tokenizer=model.tokenizers["qwen3"],
                        text_encoder=model.text_encoder,
                    )
                    _flux2_store_text_conditioning(
                        sdmlx_model,
                        negative_text,
                        negative_prompt_embeds,
                        negative_text_ids,
                    )
                    profiler.mark("negative text encode")
            had_text_encoder = "text_encoder" in model
            model._release_text_encoder_after_encode(
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )
            if keep_text_encoder:
                profiler.mark("text encoder retained")
            elif had_text_encoder:
                profiler.mark("text encoder released")
            else:
                profiler.mark("text encoder skipped")

            latents, latent_ids, latent_height, latent_width = _Flux2KleinEditHelpers.prepare_generation_latents(
                seed=int(seed),
                height=config.height,
                width=config.width,
            )
            profiler.mark("generation latents")
            if batch_size != int(latents.shape[0]):
                latents = mx.broadcast_to(latents, (batch_size, latents.shape[1], latents.shape[2]))
                latent_ids = mx.broadcast_to(latent_ids, (batch_size, latent_ids.shape[1], latent_ids.shape[2]))

            if references:
                image_latents = mx.concatenate([ref["latents"] for ref in references], axis=1)
                image_latent_ids = mx.concatenate([ref["ids"] for ref in references], axis=1)
            else:
                image_latents = mx.zeros((batch_size, 0, latents.shape[-1]), dtype=latents.dtype)
                image_latent_ids = mx.zeros((batch_size, 0, latent_ids.shape[-1]), dtype=latent_ids.dtype)
            if image_latents.shape[0] != batch_size:
                image_latents = mx.broadcast_to(image_latents, (batch_size, image_latents.shape[1], image_latents.shape[2]))
            if image_latent_ids.shape[0] != batch_size:
                image_latent_ids = mx.broadcast_to(
                    image_latent_ids,
                    (batch_size, image_latent_ids.shape[1], image_latent_ids.shape[2]),
                )
            mx.eval(image_latents, image_latent_ids)
            mx.clear_cache()
            profiler.mark("reference latents ready")

            cache_enabled = supports_kv and image_latents.shape[1] > 0
            _apply_flux2_sampling_memory_policy(model, cache_enabled=bool(cache_enabled))
            num_ref_tokens = int(image_latents.shape[1]) if cache_enabled else 0
            total_steps = int(len(config.time_steps))
            if cache_enabled:
                print("SDMLX FLUX.2: kv-cache", flush=True)
            kv_cache_obj, negative_kv_cache = model._create_kv_caches(
                cache_enabled=cache_enabled,
                needs_negative_cache=negative_prompt_embeds is not None,
            )
            predict = model._predict(model.transformer)
            cached_predict = model._cached_predict(model.transformer) if cache_enabled else None
            pbar = comfy.utils.ProgressBar(total_steps)
            profiler.mark("sampling prepared")

            sampling_profile = _flux2_sampling_profile_for_model(sdmlx_model, references)
            with _flux2_sampling_acceleration_env(sdmlx_model, profile=sampling_profile):
                for step_idx, t in enumerate(config.time_steps):
                    _throw_if_comfy_interrupted()
                    model.transformer.sdmlx_phase_step_idx = step_idx
                    if cache_enabled and step_idx == 0:
                        model.transformer.sdmlx_phase_step_mode = "extract"
                        model._configure_kv_caches(
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                            mode="extract",
                            num_ref_tokens=num_ref_tokens,
                        )
                        model._start_kv_cache_step(
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                            step_idx=step_idx,
                        )
                        noise = predict(
                            latents=latents,
                            image_latents=image_latents,
                            latent_ids=latent_ids,
                            image_latent_ids=image_latent_ids,
                            prompt_embeds=prompt_embeds,
                            text_ids=text_ids,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_text_ids=negative_text_ids,
                            guidance=float(guidance),
                            timestep=config.scheduler.timesteps[t],
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                        )
                        mx.eval(noise)
                        image_latents = None
                        image_latent_ids = None
                        if bool(getattr(model, "sdmlx_clear_cache_each_step", False)):
                            mx.clear_cache()
                    elif cache_enabled:
                        model.transformer.sdmlx_phase_step_mode = "cached"
                        model._configure_kv_caches(
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                            mode="cached",
                            num_ref_tokens=num_ref_tokens,
                        )
                        model._start_kv_cache_step(
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                            step_idx=step_idx,
                        )
                        assert cached_predict is not None
                        noise = cached_predict(
                            latents=latents,
                            latent_ids=latent_ids,
                            prompt_embeds=prompt_embeds,
                            text_ids=text_ids,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_text_ids=negative_text_ids,
                            guidance=float(guidance),
                            timestep=config.scheduler.timesteps[t],
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                        )
                    else:
                        model.transformer.sdmlx_phase_step_mode = "full"
                        noise = predict(
                            latents=latents,
                            image_latents=image_latents,
                            latent_ids=latent_ids,
                            image_latent_ids=image_latent_ids,
                            prompt_embeds=prompt_embeds,
                            text_ids=text_ids,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_text_ids=negative_text_ids,
                            guidance=float(guidance),
                            timestep=config.scheduler.timesteps[t],
                        )

                    latents = config.scheduler.step(
                        noise=noise,
                        timestep=t,
                        latents=latents,
                        sigmas=config.scheduler.sigmas,
                    )
                    mx.eval(latents)
                    if bool(getattr(model, "sdmlx_clear_cache_each_step", False)):
                        mx.clear_cache()
                    pbar.update(1)
                    profiler.mark(f"sampling step {step_idx + 1}/{total_steps}")
                    _throw_if_comfy_interrupted()
        finally:
            mx.clear_cache()
            profiler.mark("final clear_cache")
            profiler.emit()

        return (
            {
                "type": FLUX2_MODEL_FAMILY,
                "samples": latents,
                "width": config.width,
                "height": config.height,
                "batch_size": batch_size,
                "latent_height": int(latent_height),
                "latent_width": int(latent_width),
                "model_path": str(root),
                "model_config": str(sdmlx_model.get("model_config") or ""),
                "vae_component": {
                    "type": FLUX2_MODEL_FAMILY,
                    "model_path": str(root),
                    "vae_variant": str(sdmlx_model.get("vae_variant") or _FLUX2_VAE_STANDARD),
                    "vae_path": str(sdmlx_model.get("vae_path") or ""),
                },
            },
        )


class SDMLXFlux2LabRuntimePlan:
    _patch_policies = [
        _FLUX2_LAB_AUTO_COMFY_LAYERED,
        _FLUX2_LAB_PRODUCT_CURRENT,
        _FLUX2_LAB_RUNTIME_REBIND,
        _FLUX2_LAB_DENSE_WEIGHT_PATCH,
        _FLUX2_LAB_QUANTIZED_REQUANTIZE,
        _FLUX2_LAB_QUANTIZED_DENSE_TOUCHED,
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdmlx_model": (MODEL_TYPE,),
                "patch_policy": (cls._patch_policies, {"default": _FLUX2_LAB_AUTO_COMFY_LAYERED}),
            },
            "optional": {
                "positive": ("mlx_conditioning",),
                "negative": ("mlx_conditioning",),
            },
        }

    RETURN_TYPES = (_FLUX2_LAB_RUNTIME_PLAN_TYPE, "STRING")
    RETURN_NAMES = ("runtime_plan", "summary")
    FUNCTION = "plan"
    CATEGORY = "SDMLX/FLUX.2 Lab"

    def plan(
        self,
        sdmlx_model,
        patch_policy: str = _FLUX2_LAB_AUTO_COMFY_LAYERED,
        positive=None,
        negative=None,
    ):
        runtime_plan = _flux2_build_lab_runtime_plan(
            sdmlx_model,
            patch_policy=patch_policy,
            positive=positive,
            negative=negative,
        )
        return runtime_plan, _flux2_lab_plan_summary(runtime_plan)


class SDMLXFlux2LabKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdmlx_model": (MODEL_TYPE,),
                "positive": ("mlx_conditioning",),
                "negative": ("mlx_conditioning",),
                "latent_image": ("LATENT,mlx_latent",),
                "seed": ("INT", {"default": 1234567890, "min": 0, "max": 2**63 - 1}),
                "steps": ("INT", {"default": 20, "min": 2, "max": 100}),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "preview": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "runtime_plan": (_FLUX2_LAB_RUNTIME_PLAN_TYPE,),
            },
        }

    RETURN_TYPES = ("mlx_latent",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "SDMLX/FLUX.2 Lab"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        if _flux2_env_enabled("SDMLX_FLUX2_FORCE_RERUN"):
            return time.time_ns()
        return False

    def sample(
        self,
        sdmlx_model,
        positive,
        negative,
        latent_image,
        seed: int,
        steps: int,
        guidance: float,
        preview: bool = False,
        runtime_plan: dict[str, Any] | None = None,
    ):
        if not is_flux2_sdmlx_model(sdmlx_model):
            raise RuntimeError("SDMLX KSampler (FLUX.2 Lab): connect a FLUX.2 Klein model.")
        if int(steps) < 2:
            raise RuntimeError(
                "SDMLX KSampler (FLUX.2 Lab): FLUX.2 scheduler needs at least 2 steps; "
                "use 4 for distilled checkpoints or 20+ for base checkpoints."
            )
        patch_policy = (
            runtime_plan.get("patch_policy")
            if isinstance(runtime_plan, dict) and runtime_plan.get("type") == _FLUX2_LAB_RUNTIME_PLAN_TYPE
            else _FLUX2_LAB_AUTO_COMFY_LAYERED
        )
        current_plan = _flux2_build_lab_runtime_plan(
            sdmlx_model,
            patch_policy=patch_policy,
            positive=positive,
            negative=negative,
        )
        strategy = _flux2_lora_strategy_from_runtime_plan(current_plan)
        lab_model = dict(sdmlx_model)
        if strategy:
            lab_model["_flux2_lab_lora_patch_strategy"] = strategy
            lab_model["_flux2_lab_runtime_plan"] = current_plan
        return SDMLXFlux2KleinKSampler().sample(
            lab_model,
            positive,
            negative,
            latent_image,
            seed=seed,
            steps=steps,
            guidance=guidance,
            preview=preview,
        )


class SDMLXFlux2KleinEnhancerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "similarity_floor": ("FLOAT", {"default": 0.200, "min": 0.0, "max": 0.95, "step": 0.001}),
                "softmax_temperature": ("FLOAT", {"default": 0.0700, "min": 0.0001, "max": 0.25, "step": 0.0001}),
                "mask_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "double_blocks": ("STRING", {"default": _FLUX2_ENHANCED_DOUBLE, "multiline": False}),
                "single_blocks": ("STRING", {"default": _FLUX2_ENHANCED_SINGLE, "multiline": False}),
                "reference_indices": ("STRING", {"default": "all", "multiline": False}),
                "active_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "per_token_whiten": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "early_layer_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "mid_layer_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "late_layer_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (_FLUX2_KLEIN_ENHANCER_CONFIG_TYPE,)
    RETURN_NAMES = ("enhancer_advanced",)
    FUNCTION = "configure"
    CATEGORY = "SDMLX/Advanced"

    def configure(
        self,
        similarity_floor: float = 0.200,
        softmax_temperature: float = 0.0700,
        mask_threshold: float = 1.0,
        double_blocks: str = _FLUX2_ENHANCED_DOUBLE,
        single_blocks: str = _FLUX2_ENHANCED_SINGLE,
        reference_indices: str = "all",
        active_scale: float = 1.0,
        per_token_whiten: float = 0.0,
        early_layer_scale: float = 1.0,
        mid_layer_scale: float = 1.0,
        late_layer_scale: float = 1.0,
        debug: bool = False,
    ):
        return ({
            "type": _FLUX2_KLEIN_ENHANCER_CONFIG_TYPE,
            "similarity_floor": float(similarity_floor),
            "softmax_temperature": float(softmax_temperature),
            "mask_threshold": float(mask_threshold),
            "double_blocks": str(double_blocks or ""),
            "single_blocks": str(single_blocks or ""),
            "reference_indices": str(reference_indices or "all"),
            "active_scale": float(active_scale),
            "per_token_whiten": float(per_token_whiten),
            "early_layer_scale": float(early_layer_scale),
            "mid_layer_scale": float(mid_layer_scale),
            "late_layer_scale": float(late_layer_scale),
            "debug": bool(debug),
        },)


class SDMLXFlux2KleinEnhancedEditSampler:
    @classmethod
    def INPUT_TYPES(cls):
        optional: dict[str, Any] = {
            "enhance_preset": (["native_only", "soft_lock", "mid_lock", "hard_lock"], {"default": "mid_lock"}),
            "identity_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            "reference_balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "color_anchor_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "mask_behavior": (["focus_only", "zero_unmasked_tokens"], {"default": "focus_only"}),
            "preview": ("BOOLEAN", {"default": False}),
            "enhancer_advanced": (_FLUX2_KLEIN_ENHANCER_CONFIG_TYPE,),
        }
        for i in range(1, 9):
            optional[f"reference_image_{i}"] = ("IMAGE",)
        for i in range(1, 9):
            optional[f"subject_mask_{i}"] = ("MASK",)
        return {
            "required": {
                "sdmlx_model": (MODEL_TYPE,),
                "positive": ("mlx_conditioning",),
                "negative": ("mlx_conditioning",),
                "latent_image": ("LATENT,mlx_latent",),
                "mlx_vae": ("mlx_vae",),
                "seed": ("INT", {"default": 1234567890, "min": 0, "max": 2**63 - 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("mlx_latent",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "SDMLX/Sampling"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        if _flux2_env_enabled("SDMLX_FLUX2_FORCE_RERUN"):
            return time.time_ns()
        return False

    @staticmethod
    def _conditioning_refs_ordered(positive: Any, negative: Any) -> list[dict[str, Any]]:
        source = "conditioning_positive"
        conditioning = positive
        refs = positive.get("reference_latents") or [] if isinstance(positive, dict) else []
        if not refs:
            source = "conditioning_negative"
            conditioning = negative
            refs = negative.get("reference_latents") or [] if isinstance(negative, dict) else []

        ordered: list[dict[str, Any]] = []
        if not isinstance(conditioning, dict):
            return ordered
        for index, ref in enumerate(refs, start=1):
            if not isinstance(ref, dict) or ref.get("latents") is None:
                continue
            item = dict(ref)
            item["_sdmlx_reference_source"] = source
            item["_sdmlx_reference_slot"] = index
            ordered.append(item)
        return ordered

    @staticmethod
    def _direct_references(
        mlx_vae: dict[str, Any],
        start_index: int,
        optional: dict[str, Any],
    ) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        index = int(start_index)
        for slot in range(1, 9):
            image = optional.get(f"reference_image_{slot}")
            if image is None:
                continue
            ref = _flux2_enhanced_direct_reference(image, mlx_vae, index=index)
            ref["_sdmlx_reference_source"] = "direct"
            ref["_sdmlx_reference_slot"] = slot
            refs.append(ref)
            index += 1
        return refs

    @staticmethod
    def _finalize_reference_sequence(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for position, ref in enumerate(references, start=1):
            ref["_sdmlx_reference_sequence"] = position
        return references

    @staticmethod
    def _mask_keep_list(
        references: list[dict[str, Any]],
        ref_token_counts: list[int],
        optional: dict[str, Any],
        threshold: float,
    ) -> list[np.ndarray | None]:
        out: list[np.ndarray | None] = []
        for idx, ref in enumerate(references):
            mask = None
            if ref.get("_sdmlx_reference_source") == "direct":
                slot = int(ref.get("_sdmlx_reference_slot") or 0)
                if 1 <= slot <= 8:
                    mask = optional.get(f"subject_mask_{slot}")
            count = ref_token_counts[idx] if idx < len(ref_token_counts) else 0
            out.append(_flux2_mask_keep_indices(mask, ref, count, threshold))
        return out

    @staticmethod
    def _build_state(
        *,
        references: list[dict[str, Any]],
        text_token_count: int,
        enhance_preset: str,
        identity_strength: float,
        reference_balance: float,
        color_anchor_strength: float,
        mask_behavior: str,
        similarity_floor: float,
        softmax_temperature: float,
        mask_threshold: float,
        double_blocks: str,
        single_blocks: str,
        reference_indices: str,
        active_scale: float,
        per_token_whiten: float,
        early_layer_scale: float,
        mid_layer_scale: float,
        late_layer_scale: float,
        optional: dict[str, Any],
        debug: bool,
    ) -> _Flux2KleinEnhanceState:
        preset = str(enhance_preset or "mid_lock").strip().lower()
        if preset in _FLUX2_ENHANCED_PRESETS:
            similarity_floor, softmax_temperature = _FLUX2_ENHANCED_PRESETS[preset]
            double_blocks = _FLUX2_ENHANCED_DOUBLE
            single_blocks = _FLUX2_ENHANCED_SINGLE
        elif preset == "native_only":
            identity_strength = 0.0
            reference_balance = 0.5
            double_blocks = ""
            single_blocks = ""
        ref_token_counts = _flux2_enhanced_reference_token_counts(references)
        selected = _flux2_parse_reference_indices(reference_indices, len(ref_token_counts))
        mask_keep = SDMLXFlux2KleinEnhancedEditSampler._mask_keep_list(
            references,
            ref_token_counts,
            optional,
            mask_threshold,
        )
        text_scale, ref_scale = _flux2_enhanced_text_scales(reference_balance)
        return _Flux2KleinEnhanceState(
            ref_token_counts=ref_token_counts,
            text_token_count=text_token_count,
            selected_reference_indices=selected,
            mask_keep_indices=mask_keep,
            mask_behavior=mask_behavior,
            identity_strength=identity_strength,
            similarity_floor=similarity_floor,
            softmax_temperature=softmax_temperature,
            double_map=_flux2_parse_enhanced_schedule(double_blocks, 7),
            single_map=_flux2_parse_enhanced_schedule(single_blocks, 23),
            text_scale=text_scale,
            reference_scale=ref_scale,
            active_scale=active_scale,
            per_token_whiten=per_token_whiten,
            early_layer_scale=early_layer_scale,
            mid_layer_scale=mid_layer_scale,
            late_layer_scale=late_layer_scale,
            color_anchor_means=_flux2_enhanced_color_anchor_means(references),
            color_anchor_strength=color_anchor_strength,
            debug=debug,
        )

    def sample(
        self,
        sdmlx_model,
        positive,
        negative,
        latent_image,
        mlx_vae,
        seed: int,
        steps: int,
        guidance: float,
        enhance_preset: str = "mid_lock",
        identity_strength: float = 1.0,
        reference_balance: float = 0.5,
        color_anchor_strength: float = 0.0,
        mask_behavior: str = "focus_only",
        preview: bool = False,
        enhancer_advanced: dict[str, Any] | None = None,
        **optional,
    ):
        profiler = _Flux2PhaseProfiler("KSampler FLUX.2-klein Enhanced Edit")
        profiler.mark("start")
        if not is_flux2_sdmlx_model(sdmlx_model):
            raise RuntimeError("SDMLX KSampler (FLUX.2-klein Enhanced Edit): connect a FLUX.2 Klein model.")
        if not isinstance(mlx_vae, dict) or mlx_vae.get("type") != FLUX2_MODEL_FAMILY:
            raise RuntimeError("SDMLX KSampler (FLUX.2-klein Enhanced Edit): connect a FLUX.2 Klein VAE.")

        clip = SDMLXFlux2KleinKSampler._conditioning_clip(positive, negative)
        sdmlx_model = _flux2_model_with_components(sdmlx_model, clip, mlx_vae)
        root = Path(sdmlx_model["model_path"])
        lora_specs = flux2_lora_specs_from_model(sdmlx_model)
        width, height, batch_size = SDMLXFlux2KleinKSampler._latent_dimensions(latent_image)
        positive_text = _flux2_enhanced_apply_markers_hint(
            SDMLXFlux2KleinKSampler._conditioning_text(positive, "positive")
        )
        negative_text = _flux2_enhanced_apply_markers_hint(
            SDMLXFlux2KleinKSampler._conditioning_text(negative, "negative")
        )
        needs_negative_text = float(guidance) > 1.0
        cached_positive_text = _flux2_get_cached_text_conditioning(sdmlx_model, positive_text)
        cached_negative_text = (
            _flux2_get_cached_text_conditioning(sdmlx_model, negative_text)
            if needs_negative_text
            else None
        )
        require_text_encoder = cached_positive_text is None or (needs_negative_text and cached_negative_text is None)
        lora_patch_strategy = _flux2_suite_lora_patch_strategy(
            sdmlx_model,
            positive=positive,
            negative=negative,
        )
        model = _load_flux2_model(
            root,
            str(sdmlx_model.get("model_config") or ""),
            edit=True,
            vae_variant=str(sdmlx_model.get("vae_variant") or _FLUX2_VAE_STANDARD),
            vae_path=sdmlx_model.get("vae_path"),
            clip_path=sdmlx_model.get("clip_path"),
            tokenizer_path=sdmlx_model.get("tokenizer_path"),
            lora_specs=lora_specs,
            require_text_encoder=require_text_encoder,
            lora_patch_strategy=lora_patch_strategy,
        )
        profiler.mark("model ready")

        references = self._conditioning_refs_ordered(positive, negative)
        references.extend(self._direct_references(mlx_vae, len(references), optional))
        self._finalize_reference_sequence(references)
        profiler.mark("references ready")

        supports_kv = bool(getattr(model.model_config, "supports_kv_cache", False))
        _apply_flux2_kv_first_step_barriers(model, 0, supports_kv=supports_kv)
        keep_text_encoder = _flux2_is_4b_model_config(model.model_config) and not references
        _apply_flux2_pre_encode_memory_policy(model, keep_text_encoder=keep_text_encoder)
        flow_shift = sdmlx_model.get("default_flow_shift")

        _ensure_suite_qwen_native_runtime()
        from sdmlx_qwen_native.models.common.config.config import Config
        from sdmlx_qwen_native.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers

        config = Config(
            model_config=model.model_config,
            num_inference_steps=int(steps),
            height=int(height),
            width=int(width),
            guidance=float(guidance),
            scheduler=_FLUX2_DEFAULT_SCHEDULER,
            flow_shift=float(flow_shift) if flow_shift is not None else None,
        )
        profiler.mark("scheduler config")

        state: _Flux2KleinEnhanceState | None = None
        try:
            if cached_positive_text is not None:
                prompt_embeds, text_ids = cached_positive_text
                profiler.mark("positive text cache")
            else:
                _flux2_clear_cache_before_text_encode(model, profiler, "positive text cache hygiene")
                prompt_embeds, text_ids = _Flux2KleinEditHelpers.encode_text(
                    positive_text,
                    tokenizer=model.tokenizers["qwen3"],
                    text_encoder=model.text_encoder,
                )
                _flux2_store_text_conditioning(sdmlx_model, positive_text, prompt_embeds, text_ids)
                profiler.mark("positive text encode")
            advanced_values, advanced_connected = _flux2_enhancer_advanced_values(enhancer_advanced, optional)
            state = self._build_state(
                references=references,
                text_token_count=int(prompt_embeds.shape[1]),
                enhance_preset="advanced" if advanced_connected else enhance_preset,
                identity_strength=float(identity_strength),
                reference_balance=float(reference_balance),
                color_anchor_strength=float(color_anchor_strength),
                mask_behavior=mask_behavior,
                similarity_floor=float(advanced_values["similarity_floor"]),
                softmax_temperature=float(advanced_values["softmax_temperature"]),
                mask_threshold=float(advanced_values["mask_threshold"]),
                double_blocks=str(advanced_values["double_blocks"]),
                single_blocks=str(advanced_values["single_blocks"]),
                reference_indices=str(advanced_values["reference_indices"]),
                active_scale=float(advanced_values["active_scale"]),
                per_token_whiten=float(advanced_values["per_token_whiten"]),
                early_layer_scale=float(advanced_values["early_layer_scale"]),
                mid_layer_scale=float(advanced_values["mid_layer_scale"]),
                late_layer_scale=float(advanced_values["late_layer_scale"]),
                optional=optional,
                debug=bool(advanced_values["debug"]),
            )
            prompt_embeds = state.apply_text_controls(prompt_embeds)
            negative_prompt_embeds = None
            negative_text_ids = None
            if needs_negative_text:
                if cached_negative_text is not None:
                    negative_prompt_embeds, negative_text_ids = cached_negative_text
                    profiler.mark("negative text cache")
                else:
                    _flux2_clear_cache_before_text_encode(model, profiler, "negative text cache hygiene")
                    negative_prompt_embeds, negative_text_ids = _Flux2KleinEditHelpers.encode_text(
                        negative_text,
                        tokenizer=model.tokenizers["qwen3"],
                        text_encoder=model.text_encoder,
                    )
                    _flux2_store_text_conditioning(
                        sdmlx_model,
                        negative_text,
                        negative_prompt_embeds,
                        negative_text_ids,
                    )
                    profiler.mark("negative text encode")
            had_text_encoder = "text_encoder" in model
            model._release_text_encoder_after_encode(
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )
            if keep_text_encoder:
                profiler.mark("text encoder retained")
            elif had_text_encoder:
                profiler.mark("text encoder released")
            else:
                profiler.mark("text encoder skipped")

            latents, latent_ids, latent_height, latent_width = _Flux2KleinEditHelpers.prepare_generation_latents(
                seed=int(seed),
                height=config.height,
                width=config.width,
            )
            profiler.mark("generation latents")
            if batch_size != int(latents.shape[0]):
                latents = mx.broadcast_to(latents, (batch_size, latents.shape[1], latents.shape[2]))
                latent_ids = mx.broadcast_to(latent_ids, (batch_size, latent_ids.shape[1], latent_ids.shape[2]))

            if references:
                image_latents = mx.concatenate([ref["latents"] for ref in references], axis=1)
                image_latent_ids = mx.concatenate([ref["ids"] for ref in references], axis=1)
            else:
                image_latents = mx.zeros((batch_size, 0, latents.shape[-1]), dtype=latents.dtype)
                image_latent_ids = mx.zeros((batch_size, 0, latent_ids.shape[-1]), dtype=latent_ids.dtype)
            if image_latents.shape[0] != batch_size:
                image_latents = mx.broadcast_to(image_latents, (batch_size, image_latents.shape[1], image_latents.shape[2]))
            if image_latent_ids.shape[0] != batch_size:
                image_latent_ids = mx.broadcast_to(
                    image_latent_ids,
                    (batch_size, image_latent_ids.shape[1], image_latent_ids.shape[2]),
                )
            mx.eval(image_latents, image_latent_ids)
            mx.clear_cache()
            profiler.mark("reference latents ready")

            cache_enabled = (
                supports_kv
                and image_latents.shape[1] > 0
                and state is not None
                and not state.reference_steering_enabled
            )
            if supports_kv and image_latents.shape[1] > 0 and state is not None and state.reference_steering_enabled:
                print("SDMLX FLUX.2 enhanced-edit: kv-cache disabled for reference-token steering", flush=True)
            elif cache_enabled:
                print("SDMLX FLUX.2: kv-cache", flush=True)
            _apply_flux2_sampling_memory_policy(model, cache_enabled=bool(cache_enabled))
            model.transformer.sdmlx_flux2_enhance_state = state if state.reference_steering_enabled else None

            num_ref_tokens = int(image_latents.shape[1]) if cache_enabled else 0
            total_steps = int(len(config.time_steps))
            kv_cache_obj, negative_kv_cache = model._create_kv_caches(
                cache_enabled=cache_enabled,
                needs_negative_cache=negative_prompt_embeds is not None,
            )
            predict = model._predict(model.transformer)
            cached_predict = model._cached_predict(model.transformer) if cache_enabled else None
            pbar = comfy.utils.ProgressBar(total_steps)
            profiler.mark("sampling prepared")

            with _flux2_sampling_acceleration_env(sdmlx_model):
                for step_idx, t in enumerate(config.time_steps):
                    _throw_if_comfy_interrupted()
                    model.transformer.sdmlx_phase_step_idx = step_idx
                    if cache_enabled and step_idx == 0:
                        model.transformer.sdmlx_phase_step_mode = "extract"
                        model._configure_kv_caches(
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                            mode="extract",
                            num_ref_tokens=num_ref_tokens,
                        )
                        model._start_kv_cache_step(
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                            step_idx=step_idx,
                        )
                        noise = predict(
                            latents=latents,
                            image_latents=image_latents,
                            latent_ids=latent_ids,
                            image_latent_ids=image_latent_ids,
                            prompt_embeds=prompt_embeds,
                            text_ids=text_ids,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_text_ids=negative_text_ids,
                            guidance=float(guidance),
                            timestep=config.scheduler.timesteps[t],
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                        )
                        mx.eval(noise)
                        image_latents = None
                        image_latent_ids = None
                        if bool(getattr(model, "sdmlx_clear_cache_each_step", False)):
                            mx.clear_cache()
                    elif cache_enabled:
                        model.transformer.sdmlx_phase_step_mode = "cached"
                        model._configure_kv_caches(
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                            mode="cached",
                            num_ref_tokens=num_ref_tokens,
                        )
                        model._start_kv_cache_step(
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                            step_idx=step_idx,
                        )
                        assert cached_predict is not None
                        noise = cached_predict(
                            latents=latents,
                            latent_ids=latent_ids,
                            prompt_embeds=prompt_embeds,
                            text_ids=text_ids,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_text_ids=negative_text_ids,
                            guidance=float(guidance),
                            timestep=config.scheduler.timesteps[t],
                            kv_cache=kv_cache_obj,
                            negative_kv_cache=negative_kv_cache,
                        )
                    else:
                        model.transformer.sdmlx_phase_step_mode = "enhanced_full"
                        noise = predict(
                            latents=latents,
                            image_latents=image_latents,
                            latent_ids=latent_ids,
                            image_latent_ids=image_latent_ids,
                            prompt_embeds=prompt_embeds,
                            text_ids=text_ids,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_text_ids=negative_text_ids,
                            guidance=float(guidance),
                            timestep=config.scheduler.timesteps[t],
                        )

                    if state is not None and state.color_anchor_enabled:
                        noise = state.apply_color_anchor(latents, noise, config.scheduler.sigmas[t])
                    latents = config.scheduler.step(
                        noise=noise,
                        timestep=t,
                        latents=latents,
                        sigmas=config.scheduler.sigmas,
                    )
                    mx.eval(latents)
                    if bool(getattr(model, "sdmlx_clear_cache_each_step", False)):
                        mx.clear_cache()
                    pbar.update(1)
                    profiler.mark(f"sampling step {step_idx + 1}/{total_steps}")
                    _throw_if_comfy_interrupted()
        finally:
            if hasattr(model, "transformer"):
                model.transformer.sdmlx_flux2_enhance_state = None
            mx.clear_cache()
            profiler.mark("final clear_cache")
            profiler.emit()

        return (
            {
                "type": FLUX2_MODEL_FAMILY,
                "samples": latents,
                "width": config.width,
                "height": config.height,
                "batch_size": batch_size,
                "latent_height": int(latent_height),
                "latent_width": int(latent_width),
                "model_path": str(root),
                "model_config": str(sdmlx_model.get("model_config") or ""),
                "vae_component": {
                    "type": FLUX2_MODEL_FAMILY,
                    "model_path": str(root),
                    "vae_variant": str(sdmlx_model.get("vae_variant") or _FLUX2_VAE_STANDARD),
                    "vae_path": str(sdmlx_model.get("vae_path") or ""),
                },
            },
        )


NODE_CLASS_MAPPINGS = {
    "SDMLXFlux2ScaleToMegapixels": SDMLXFlux2ScaleToMegapixels,
    "SDMLXFlux2EmptyLatentImage": SDMLXFlux2EmptyLatentImage,
    "SDMLXFlux2ReferenceLatent": SDMLXFlux2ReferenceLatent,
    "SDMLXFlux2KleinKSampler": SDMLXFlux2KleinKSampler,
    "SDMLXFlux2LabRuntimePlan": SDMLXFlux2LabRuntimePlan,
    "SDMLXFlux2LabKSampler": SDMLXFlux2LabKSampler,
    "SDMLXFlux2KleinEnhancerAdvanced": SDMLXFlux2KleinEnhancerAdvanced,
    "SDMLXFlux2KleinEnhancedEditSampler": SDMLXFlux2KleinEnhancedEditSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDMLXFlux2ScaleToMegapixels": "🍏 SDMLX Scale To Megapixels",
    "SDMLXFlux2EmptyLatentImage": "🍏 SDMLX Empty Latent Image FLUX.2",
    "SDMLXFlux2ReferenceLatent": "🍏 SDMLX Reference Latent",
    "SDMLXFlux2KleinKSampler": "🍏 SDMLX KSampler (FLUX.2-klein)",
    "SDMLXFlux2LabRuntimePlan": "🍏 SDMLX FLUX.2 Lab Runtime Plan",
    "SDMLXFlux2LabKSampler": "🍏 SDMLX KSampler (FLUX.2 Lab)",
    "SDMLXFlux2KleinEnhancerAdvanced": "🍏 SDMLX FLUX.2 Klein Enhancer Advanced",
    "SDMLXFlux2KleinEnhancedEditSampler": "🍏 SDMLX KSampler (FLUX.2-klein Enhanced Edit)",
}
