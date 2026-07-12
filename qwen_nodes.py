from __future__ import annotations

import contextlib
import json
import hashlib
import inspect
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image


SUITE_ROOT = Path(__file__).resolve().parent
QWEN_NATIVE_ROOT = SUITE_ROOT / "sdmlx_qwen_native"

for path in (SUITE_ROOT, QWEN_NATIVE_ROOT):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

import folder_paths  # noqa: E402


MODEL_TYPE = "sdmlx_model"
QWEN_ACCEL_NONE = "None"
QWEN_DEFAULT_MODEL = "mlx-community/qwen-image-edit-2511-8bit"
QWEN_MODEL_FAMILY = "qwen-image-edit"
QWEN_MODEL_VERSION = "2511"
QWEN_PACKAGE_FORMAT = "sdmlx-qwen-package-v1"
QWEN_EDIT_PLUS_VL_AREA = 384 * 384
QWEN_REFERENCE_AREA = 1024 * 1024
QWEN_PHR00T_PLUS_REFERENCE_SIZE = 896
QWEN_PHR00T_PLUS_REFERENCE_AREA = QWEN_PHR00T_PLUS_REFERENCE_SIZE * QWEN_PHR00T_PLUS_REFERENCE_SIZE
QWEN_PHR00T_PLUS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe key details of the input image (including any objects, characters, poses, facial features, clothing, "
    "setting, textures and style), then explain how the user's text instruction should alter, modify or recreate the "
    "image. Generate a new image that meets the user's requirements, which can vary from a small change to a "
    "completely new image using inputs as a guide.<|im_end|>\n"
    "<|im_start|>user\n"
    "{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
QWEN_TOKEN_BUDGET_OFF = "off"
QWEN_TOKEN_BUDGET_PRESETS = {
    # New names for the previous Qwen Fidelity presets:
    # min  = old fast, two refs ~= total_reference_budget=393216
    # low  = old med, useful middle result from early tests
    # med  = old high, two refs ~= total_reference_budget=786432
    # high = old max, upper budgeted preset below Comfy parity
    "min": 393216 // 2,
    "low": 262144,
    "med": 786432 // 2,
    "high": 1024 * 1024 // 2,
}
QWEN_TOKEN_BUDGET_OPTIONS = ["min", "low", "med", "high", QWEN_TOKEN_BUDGET_OFF]
QWEN_ACCEL_PATCH_PACKAGE_DIR = "AccelerationPatches"
QWEN_ACCEL_PATCH_PACKAGE_BF16 = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.sdmlxpatch"
QWEN_ACCEL_PATCH_PACKAGE_FP32 = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.sdmlxpatch"
QWEN_ACCEL_PATCH_LABEL_BF16 = "Qwen Image Edit 2511 Lightning 4-step (bf16)"
QWEN_ACCEL_PATCH_LABEL_FP32 = "Qwen Image Edit 2511 Lightning 4-step (fp32)"
QWEN_ACCEL_PATCH_REPO_ID = "lightx2v/Qwen-Image-Edit-2511-Lightning"
QWEN_ACCEL_PATCH_FILENAME_BF16 = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
QWEN_ACCEL_PATCH_FILENAME_FP32 = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"
QWEN_ACCEL_PATCH_PACKAGE = QWEN_ACCEL_PATCH_PACKAGE_BF16
QWEN_ACCEL_PATCH_LABEL = QWEN_ACCEL_PATCH_LABEL_BF16
QWEN_ACCEL_PATCH_FILENAME = QWEN_ACCEL_PATCH_FILENAME_BF16
QWEN_ACCEL_PATCH_REGISTRY = {
    QWEN_ACCEL_PATCH_PACKAGE_BF16: {
        "label": QWEN_ACCEL_PATCH_LABEL_BF16,
        "base_model_family": "qwen-image-edit",
        "model_version": "2511",
        "recommended_steps": 4,
        "recommended_scheduler": "linear",
        "source_repo": QWEN_ACCEL_PATCH_REPO_ID,
        "source_file": QWEN_ACCEL_PATCH_FILENAME_BF16,
    },
    QWEN_ACCEL_PATCH_PACKAGE_FP32: {
        "label": QWEN_ACCEL_PATCH_LABEL_FP32,
        "base_model_family": "qwen-image-edit",
        "model_version": "2511",
        "recommended_steps": 4,
        "recommended_scheduler": "linear",
        "source_repo": QWEN_ACCEL_PATCH_REPO_ID,
        "source_file": QWEN_ACCEL_PATCH_FILENAME_FP32,
    },
    "Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.sdmlxpatch": {
        "label": "Qwen Image 2512 Wuli Turbo 2-step",
        "base_model_family": "qwen-image",
        "model_version": "2512",
        "recommended_steps": 2,
        "recommended_scheduler": "comfy_aura_simple",
        "source_repo": "Wuli-art/Qwen-Image-2512-Turbo-LoRA-2-Steps",
        "source_file": "Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors",
    },
    "Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V3.0-bf16.sdmlxpatch": {
        "label": "Qwen Image 2512 Wuli Turbo 4-step",
        "base_model_family": "qwen-image",
        "model_version": "2512",
        "recommended_steps": 4,
        "recommended_scheduler": "comfy_aura_simple",
        "source_repo": "Wuli-art/Qwen-Image-2512-Turbo-LoRA",
        "source_file": "Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V3.0-bf16.safetensors",
    },
}
QWEN_ACCEL_PATCH_LABELS = {name: data["label"] for name, data in QWEN_ACCEL_PATCH_REGISTRY.items()}
QWEN_ACCEL_PATCH_BY_LABEL = {label: name for name, label in QWEN_ACCEL_PATCH_LABELS.items()}
QWEN_ACCEL_PATCH_LABEL_ALIASES = {
    "Qwen Image Edit 2511 Lightning 4-step": QWEN_ACCEL_PATCH_PACKAGE_BF16,
}
_QWEN_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_QWEN_VAE_CACHE: dict[tuple[Any, ...], Any] = {}
_QWEN_ACCEL_OPTIONS_CACHE: tuple[str, ...] | None = None
QWEN_AURAFLOW_SCHEDULER = "comfy_aura_simple"
QWEN_AURAFLOW_DEFAULT_SHIFT = 3.1
QWEN_ROOT_REQUIRED_DIRS = ("transformer", "text_encoder", "vae", "tokenizer")
QWEN_FP16_QUANT_PARAMS_DEFAULT_REGEX = r"transformer\.transformer_blocks\.[0-9]+\.(attn|img_ff|txt_ff)\."
QWEN_FP16_QUANT_PARAMS_QWEN_IMAGE_REGEX = r"transformer\.transformer_blocks\.[0-9]+\.(img_ff|txt_ff)\."
QWEN_VARIANT_VERSIONS = {
    "qwen-image-edit": "2511",
    "qwen-image": "2512",
}


def _clear_qwen_reference_vae_cache() -> None:
    try:
        from sdmlx_qwen_native.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil

        QwenEditUtil.clear_image_conditioning_cache()
    except Exception:
        pass


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_disabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"0", "false", "no", "off", "disabled"}


def _qwen_verbose_logs_enabled() -> bool:
    return _env_flag("SDMLX_QWEN_VERBOSE") or _env_flag("SDMLX_QWEN_DEBUG")


def _qwen_debug_logs_enabled() -> bool:
    return _env_flag("SDMLX_QWEN_DEBUG")


def _qwen_log(message: str, *, verbose: bool = False, debug: bool = False) -> None:
    if debug and not _qwen_debug_logs_enabled():
        return
    if verbose and not _qwen_verbose_logs_enabled():
        return
    print(message)


def _qwen_comfy_progress_bar(total_steps: int) -> Any | None:
    try:
        import comfy.utils  # type: ignore

        return comfy.utils.ProgressBar(max(1, int(total_steps)))
    except Exception as exc:
        _qwen_log(f"SDMLX Qwen: Comfy progress unavailable ({exc})", debug=True)
        return None


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

    for path in (SUITE_ROOT, QWEN_NATIVE_ROOT):
        path_s = str(path)
        if path_s in sys.path:
            sys.path.remove(path_s)
        sys.path.insert(0, path_s)


def _sdmlx_model_root() -> Path:
    path = None
    try:
        folder_map = getattr(folder_paths, "folder_names_and_paths", {})
        candidate_paths = []
        for key in ("sdmlx", "SDMLX"):
            if key in folder_map:
                candidate_paths.extend(Path(p) for p in folder_paths.get_folder_paths(key))
        seen = set()
        candidate_paths = [p for p in candidate_paths if not (str(p) in seen or seen.add(str(p)))]
        for candidate in candidate_paths:
            if candidate.is_dir():
                try:
                    entries = {entry.name for entry in candidate.iterdir()}
                except OSError:
                    entries = set()
                if "cache" in entries or "AccelerationPatches" in entries or any(entry.endswith(".sdmlx") for entry in entries):
                    path = candidate
                    break
        if path is None:
            for candidate in candidate_paths:
                if candidate.is_dir():
                    path = candidate
                    break
        if path is None and candidate_paths:
            path = candidate_paths[0]
    except Exception:
        path = None
    if path is None:
        models_dir = Path(getattr(folder_paths, "models_dir", SUITE_ROOT.parent / "models"))
        path = models_dir / "SDMLX"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _qwen_acceleration_patch_dir() -> Path:
    path = _sdmlx_model_root() / QWEN_ACCEL_PATCH_PACKAGE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _qwen_acceleration_package_path(package_name: str) -> Path:
    return _qwen_acceleration_patch_dir() / package_name


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _normalize_qwen_variant(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "qwen-image-edit": "qwen-image-edit",
        "qwen-image-edit-2511": "qwen-image-edit",
        "qwen-image": "qwen-image",
        "qwen-image-2512": "qwen-image",
    }
    return aliases.get(normalized)


def _normalize_qwen_version(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"2511", "qwen-image-edit-2511"}:
        return "2511"
    if normalized in {"2512", "qwen-image-2512"}:
        return "2512"
    return None


def _qwen_identity_from_text(values: list[Any] | tuple[Any, ...]) -> tuple[str, str] | None:
    text = " ".join(str(value or "") for value in values).lower().replace("_", "-")
    has_2511 = "2511" in text or "qwen-image-edit" in text
    has_2512 = "2512" in text
    if has_2511 and has_2512:
        raise RuntimeError(
            "SDMLX Qwen: contradictory source identity contains both Qwen Image Edit 2511 and Qwen Image 2512 markers."
        )
    if has_2511:
        return "qwen-image-edit", "2511"
    if has_2512:
        return "qwen-image", "2512"
    return None


def resolve_qwen_model_identity(
    *,
    manifest: dict[str, Any] | None = None,
    transformer_keys: set[str] | list[str] | tuple[str, ...] | None = None,
    source_identifiers: list[Any] | tuple[Any, ...] = (),
    root_hint: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Resolve 2511/Edit versus 2512/Image without cache-name guessing."""
    manifest = manifest if isinstance(manifest, dict) else {}
    raw_variant = manifest.get("qwen_variant")
    raw_version = manifest.get("model_version")
    manifest_variant = _normalize_qwen_variant(raw_variant)
    manifest_version = _normalize_qwen_version(raw_version)
    if raw_variant not in (None, "") and manifest_variant is None:
        raise RuntimeError(f"SDMLX Qwen: unsupported qwen_variant in manifest: {raw_variant!r}.")
    if raw_version not in (None, "") and manifest_version is None:
        raise RuntimeError(f"SDMLX Qwen: unsupported model_version in manifest: {raw_version!r}.")
    if manifest_variant and manifest_version:
        expected_version = QWEN_VARIANT_VERSIONS[manifest_variant]
        if manifest_version != expected_version:
            raise RuntimeError(
                "SDMLX Qwen: contradictory manifest identity: "
                f"qwen_variant={manifest_variant!r} requires model_version={expected_version}, "
                f"not {manifest_version}."
            )
    elif manifest_variant:
        manifest_version = QWEN_VARIANT_VERSIONS[manifest_variant]
    elif manifest_version:
        manifest_variant = "qwen-image-edit" if manifest_version == "2511" else "qwen-image"

    marker_identity = None
    if transformer_keys and any(str(key).split(".")[-1] == "__index_timestep_zero__" for key in transformer_keys):
        marker_identity = ("qwen-image-edit", "2511")
    source_identity = _qwen_identity_from_text(tuple(source_identifiers))

    manifest_identity = (
        (manifest_variant, manifest_version)
        if manifest_variant is not None and manifest_version is not None
        else None
    )
    if manifest_identity is not None:
        if marker_identity is not None and manifest_identity != marker_identity:
            raise RuntimeError(
                "SDMLX Qwen: manifest identifies Qwen Image 2512, but the transformer contains the 2511 Edit marker."
            )
        if source_identity is not None and manifest_identity != source_identity:
            raise RuntimeError(
                "SDMLX Qwen: manifest identity conflicts with the original checkpoint/repository identity."
            )
        return manifest_identity

    if marker_identity is not None:
        if source_identity is not None and source_identity != marker_identity:
            raise RuntimeError(
                "SDMLX Qwen: transformer structure identifies 2511 Edit, but the original source identifies 2512."
            )
        return marker_identity
    if source_identity is not None:
        return source_identity

    weak_identity = _qwen_identity_from_text((root_hint,)) if root_hint is not None else None
    if weak_identity is not None:
        return weak_identity
    raise RuntimeError(
        "SDMLX Qwen: model identity is ambiguous. The package needs qwen_variant/model_version, "
        "or the original checkpoint/repository identity must contain 2511/edit or 2512."
    )


def _qwen_transformer_identity_keys(root: Path) -> set[str]:
    transformer_dir = root / "transformer"
    identity_keys: set[str] = set()
    for index_path in sorted(transformer_dir.glob("*.index.json")):
        try:
            weight_map = _read_json(index_path).get("weight_map") or {}
            identity_keys.update(
                key for key in weight_map if str(key).split(".")[-1] == "__index_timestep_zero__"
            )
        except Exception:
            continue
    if identity_keys:
        return identity_keys
    try:
        from safetensors import safe_open

        for tensor_path in sorted(transformer_dir.glob("*.safetensors")):
            if tensor_path.name.startswith("._"):
                continue
            with safe_open(str(tensor_path), framework="np") as handle:
                for key in handle.keys():
                    if str(key).split(".")[-1] == "__index_timestep_zero__":
                        identity_keys.add(str(key))
    except Exception:
        pass
    return identity_keys


def _qwen_root_source_identifiers(root: Path, name: str | None = None) -> list[str]:
    identifiers = [str(name or "")]
    for marker_name in (
        "sdmlx_qwen_aio_root.json",
        "sdmlx_qwen_split_root.json",
        "sdmlx_qwen_prepared_cache.json",
    ):
        marker_path = root / marker_name
        if not marker_path.is_file():
            continue
        try:
            marker = _read_json(marker_path)
        except Exception:
            continue
        identifiers.extend(
            str(marker.get(key) or "")
            for key in ("source_checkpoint", "source_root", "source_repo")
        )
    return identifiers


def _qwen_identity_root_hint(root: Path) -> str | None:
    if any(part.lower() in {"runtime-roots", "qwen-runtime-roots"} for part in root.parts):
        return None
    return str(root)


def _backfill_qwen_manifest_identity(
    package_path: Path,
    manifest: dict[str, Any],
    qwen_variant: str,
    model_version: str,
) -> dict[str, Any]:
    if manifest.get("qwen_variant") and manifest.get("model_version"):
        return manifest
    updated = dict(manifest)
    updated.setdefault("qwen_variant", qwen_variant)
    updated.setdefault("model_version", model_version)
    manifest_path = package_path / "manifest.json"
    if manifest_path.is_file():
        _write_json(manifest_path, updated)
    return updated


def is_qwen_manifest(manifest: dict[str, Any] | None) -> bool:
    if not isinstance(manifest, dict):
        return False
    family = str(manifest.get("model_family") or manifest.get("base_model_family") or "").strip().lower()
    package_format = str(manifest.get("package_format") or manifest.get("format") or "").strip().lower()
    return family in {
        "qwen",
        QWEN_MODEL_FAMILY,
        "qwen-image-edit-2511",
        "qwen-image",
        "qwen-image-2512",
    } or package_format == QWEN_PACKAGE_FORMAT


def is_qwen_sdmlx_model(model: Any) -> bool:
    return isinstance(model, dict) and str(model.get("model_family", "")).strip().lower() == QWEN_MODEL_FAMILY


def is_qwen_model_root(path: str | os.PathLike[str]) -> bool:
    root = Path(path).expanduser()
    if not root.is_dir():
        return False
    return all((root / name).exists() for name in QWEN_ROOT_REQUIRED_DIRS)


def _qwen_root_from_component_path(path: str | os.PathLike[str]) -> Path | None:
    current = Path(path).expanduser()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if is_qwen_model_root(candidate):
            return candidate
    return None


def qwen_model_from_root(model_path: str | os.PathLike[str], preload: bool = False, name: str | None = None):
    root = Path(model_path).expanduser()
    if not is_qwen_model_root(root):
        raise RuntimeError(
            "SDMLX Qwen: selected diffusion model is not a complete Qwen root. "
            "Expected transformer/, text_encoder/, vae/, and tokenizer/."
        )
    model_path_s = str(root.resolve())
    manifest: dict[str, Any] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        candidate = _read_json(manifest_path)
        if is_qwen_manifest(candidate):
            manifest = candidate
    source_identifiers = _qwen_root_source_identifiers(root, name=name)
    source_identifiers.extend(
        str(manifest.get(key) or "")
        for key in ("source_root", "source_repo")
    )
    qwen_variant, model_version = resolve_qwen_model_identity(
        manifest=manifest,
        transformer_keys=_qwen_transformer_identity_keys(root),
        source_identifiers=source_identifiers,
        root_hint=_qwen_identity_root_hint(root),
    )
    if manifest:
        _backfill_qwen_manifest_identity(root, manifest, qwen_variant, model_version)
    model = {
        "model_family": QWEN_MODEL_FAMILY,
        "qwen_variant": qwen_variant,
        "model_version": model_version,
        "model_path": model_path_s,
        "name": str(name or root.name),
        "cache_key": model_path_s,
        "loras": [],
        "mod_lora_scale": 0.0,
        "recommendations": {
            "steps": 4,
            "guidance": 1.0,
            "scheduler": "linear",
            "acceleration_patch": QWEN_ACCEL_PATCH_LABEL,
        },
        "source_repo": model_path_s,
    }
    if preload:
        _load_qwen_model(model_path=model_path_s, lora_specs=[])
    return model


def _qwen_runtime_cache_dir(name: str) -> Path:
    path = Path(tempfile.gettempdir()) / "sdmlx" / "qwen" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _qwen_runtime_roots_dir() -> Path:
    return _qwen_runtime_cache_dir("runtime-roots")


def _qwen_vae_roots_dir() -> Path:
    return _qwen_runtime_cache_dir("vae-roots")


def _resolve_comfy_model_file(folder_names: tuple[str, ...], candidates: tuple[str, ...], label: str) -> Path:
    scanned_roots: list[Path] = []
    for folder_name in folder_names:
        try:
            names = folder_paths.get_filename_list(folder_name)
        except Exception:
            names = []
        for candidate in candidates:
            if candidate not in names:
                continue
            try:
                path = folder_paths.get_full_path(folder_name, candidate)
            except Exception:
                path = None
            if path:
                return Path(path).expanduser().resolve()
        for name in names:
            lowered = str(name).lower()
            if all(part in lowered for part in ("qwen",)):
                try:
                    path = folder_paths.get_full_path(folder_name, name)
                except Exception:
                    path = None
                if path and label.lower() in {"text encoder", "vae"}:
                    if label.lower() == "text encoder" and ("vl" in lowered or "2.5" in lowered or "qwen_" in lowered):
                        return Path(path).expanduser().resolve()
                    if label.lower() == "vae" and "vae" in lowered:
                        return Path(path).expanduser().resolve()
        try:
            scanned_roots.extend(Path(path) for path in folder_paths.get_folder_paths(folder_name))
        except Exception:
            pass
    try:
        models_dir = Path(folder_paths.models_dir)
        for folder_name in folder_names:
            scanned_roots.append(models_dir / folder_name)
    except Exception:
        pass
    seen: set[str] = set()
    for root in scanned_roots:
        try:
            root = root.expanduser()
            key = str(root.resolve())
        except Exception:
            continue
        if key in seen or not root.exists():
            continue
        seen.add(key)
        for candidate in candidates:
            path = root / candidate
            if path.exists():
                return path.resolve()
        for path in root.rglob("*qwen*"):
            if not path.is_file() or path.suffix.lower() not in {".safetensors", ".ckpt"}:
                continue
            lowered = path.name.lower()
            if label.lower() == "text encoder" and ("vl" in lowered or "2.5" in lowered or "qwen_" in lowered):
                return path.resolve()
            if label.lower() == "vae" and "vae" in lowered:
                return path.resolve()
    raise FileNotFoundError(f"SDMLX Qwen: could not find a default Qwen {label}.")


def _default_qwen_model_root() -> Path | None:
    roots: list[Path] = []
    try:
        roots.extend(Path(root) for root in folder_paths.get_folder_paths("diffusion_models"))
    except Exception:
        pass
    try:
        roots.append(Path(folder_paths.models_dir) / "diffusion_models")
    except Exception:
        pass
    roots.append(_sdmlx_model_root())

    seen: set[str] = set()
    candidates: list[Path] = []
    for root in roots:
        try:
            root = root.expanduser()
            key = str(root.resolve())
        except Exception:
            continue
        if key in seen or not root.exists():
            continue
        seen.add(key)
        for candidate in root.rglob("*"):
            if not candidate.is_dir():
                continue
            lowered = candidate.name.lower()
            if "qwen" not in lowered:
                continue
            if is_qwen_model_root(candidate):
                candidates.append(candidate.resolve())

    def score(path: Path) -> tuple[int, int, str]:
        lowered = path.name.lower()
        return (
            0 if QWEN_MODEL_VERSION in lowered else 1,
            0 if "8bit" in lowered else 1,
            str(path),
        )

    if not candidates:
        return None
    return sorted(candidates, key=score)[0]


def _default_qwen_text_encoder_path() -> Path:
    return _resolve_comfy_model_file(
        ("text_encoders", "clip"),
        (
            "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "qwen_2.5_vl_7b_fp16.safetensors",
            "qwen_2.5_vl_7b.safetensors",
        ),
        "text encoder",
    )


def _default_qwen_vae_path() -> Path:
    return _resolve_comfy_model_file(
        ("vae",),
        (
            "qwen-edit-vae.safetensors",
            "qwen_image_edit_vae.safetensors",
            "qwen_image_vae.safetensors",
            "split_files/vae/qwen_image_vae.safetensors",
        ),
        "VAE",
    )


def _default_qwen_tokenizer_path() -> Path:
    model_root = _default_qwen_model_root()
    if model_root is not None and (model_root / "tokenizer" / "tokenizer.json").exists():
        return (model_root / "tokenizer").resolve()

    roots = []
    try:
        for root in folder_paths.get_folder_paths("diffusion_models"):
            roots.append(Path(root))
    except Exception:
        pass
    try:
        roots.append(Path(folder_paths.models_dir) / "diffusion_models")
    except Exception:
        pass
    roots.append(_sdmlx_model_root())
    seen: set[str] = set()
    for root in roots:
        try:
            root = root.expanduser()
            key = str(root.resolve())
        except Exception:
            continue
        if key in seen or not root.exists():
            continue
        seen.add(key)
        for candidate in root.rglob("tokenizer"):
            if not candidate.is_dir():
                continue
            if (candidate / "tokenizer.json").exists() and "qwen" in str(candidate).lower():
                return candidate.resolve()
    raise FileNotFoundError("SDMLX Qwen: could not find a Qwen tokenizer directory.")


def _safe_symlink_or_copy(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            if destination.is_symlink() and Path(os.readlink(destination)) == source:
                return
        except Exception:
            pass
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    try:
        os.symlink(str(source), str(destination), target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _safetensors_weight_map(path: Path, prefixes: tuple[str, ...] | None = None) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as handle:
        return {
            key: path.name
            for key in handle.keys()
            if prefixes is None or any(key.startswith(prefix) for prefix in prefixes)
        }


def _write_single_file_index(component_dir: Path, file_name: str) -> None:
    index_path = component_dir / "model.safetensors.index.json"
    if index_path.exists():
        return
    weight_map = _safetensors_weight_map(component_dir / file_name)
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": {}, "weight_map": weight_map}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_filtered_single_file_index(component_dir: Path, file_name: str, prefixes: tuple[str, ...]) -> None:
    index_path = component_dir / "model.safetensors.index.json"
    if index_path.exists():
        return
    weight_map = _safetensors_weight_map(component_dir / file_name, prefixes=prefixes)
    if not weight_map:
        raise RuntimeError(f"SDMLX Qwen: no indexed tensors matching {prefixes} in {component_dir / file_name}")
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": {}, "weight_map": weight_map}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _is_qwen_aio_checkpoint(path: str | os.PathLike[str]) -> bool:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as handle:
            has_transformer = False
            has_text_encoder = False
            has_vae = False
            for key in handle.keys():
                if key.startswith("model.diffusion_model."):
                    has_transformer = True
                elif key.startswith("text_encoders."):
                    has_text_encoder = True
                elif key.startswith("vae."):
                    has_vae = True
                if has_transformer and has_text_encoder and has_vae:
                    return True
    except Exception:
        return False
    return False


def _qwen_runtime_root_from_aio_checkpoint(checkpoint_path: str | os.PathLike[str]) -> Path:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    donor_root = _default_qwen_model_root()
    tokenizer = _default_qwen_tokenizer_path()
    vae = (
        (donor_root / "vae").resolve()
        if donor_root is not None and (donor_root / "vae").exists()
        else _default_qwen_vae_path()
    )
    digest_source = "|".join(str(_qwen_file_identity(path)[0:3]) for path in (checkpoint, vae, tokenizer))
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
    root = _qwen_runtime_roots_dir() / f"{checkpoint.stem}-aio-{digest}"
    marker = root / "sdmlx_qwen_aio_root.json"

    if is_qwen_model_root(root) and marker.is_file():
        return root

    _safe_symlink_or_copy(checkpoint, root / "transformer" / checkpoint.name)
    _safe_symlink_or_copy(checkpoint, root / "text_encoder" / checkpoint.name)
    if vae.is_dir():
        _safe_symlink_or_copy(vae, root / "vae")
    else:
        _safe_symlink_or_copy(vae, root / "vae" / vae.name)
    _write_filtered_single_file_index(root / "text_encoder", checkpoint.name, ("text_encoders.",))
    _safe_symlink_or_copy(tokenizer, root / "tokenizer")
    _write_json(
        marker,
        {
            "runtime_root_format": "sdmlx-qwen-aio-root-v1",
            "source_checkpoint": str(checkpoint),
            "source_identity": repr(_qwen_file_identity(checkpoint)),
            "vae": str(vae),
            "tokenizer": str(tokenizer),
            "created_at": time.time(),
        },
    )
    return root


def _qwen_runtime_root_from_split_files(
    transformer_path: str | os.PathLike[str],
    text_encoder_path: str | os.PathLike[str] | None = None,
    vae_path: str | os.PathLike[str] | None = None,
    tokenizer_path: str | os.PathLike[str] | None = None,
) -> Path:
    transformer = Path(transformer_path).expanduser().resolve()
    donor_root = _default_qwen_model_root()
    text_encoder = (
        Path(text_encoder_path).expanduser().resolve()
        if text_encoder_path
        else (donor_root / "text_encoder").resolve()
        if donor_root is not None and (donor_root / "text_encoder").exists()
        else _default_qwen_text_encoder_path()
    )
    vae = (
        Path(vae_path).expanduser().resolve()
        if vae_path
        else (donor_root / "vae").resolve()
        if donor_root is not None and (donor_root / "vae").exists()
        else _default_qwen_vae_path()
    )
    tokenizer = (
        Path(tokenizer_path).expanduser().resolve()
        if tokenizer_path
        else (donor_root / "tokenizer").resolve()
        if donor_root is not None and (donor_root / "tokenizer" / "tokenizer.json").exists()
        else _default_qwen_tokenizer_path()
    )
    digest_source = "|".join(
        str(_qwen_file_identity(path)[0:3])
        for path in (transformer, text_encoder, vae, tokenizer)
    )
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
    root = _qwen_runtime_roots_dir() / f"{transformer.stem}-{digest}"

    _safe_symlink_or_copy(transformer, root / "transformer" / transformer.name)
    if text_encoder.is_dir():
        _safe_symlink_or_copy(text_encoder, root / "text_encoder")
    else:
        _safe_symlink_or_copy(text_encoder, root / "text_encoder" / text_encoder.name)
        _write_single_file_index(root / "text_encoder", text_encoder.name)
    if vae.is_dir():
        _safe_symlink_or_copy(vae, root / "vae")
    else:
        _safe_symlink_or_copy(vae, root / "vae" / vae.name)
    _safe_symlink_or_copy(tokenizer, root / "tokenizer")
    _write_json(
        root / "sdmlx_qwen_split_root.json",
        {
            "runtime_root_format": "sdmlx-qwen-split-root-v1",
            "source_checkpoint": str(transformer),
            "source_identity": repr(_qwen_file_identity(transformer)),
            "text_encoder": str(text_encoder),
            "vae": str(vae),
            "tokenizer": str(tokenizer),
            "created_at": time.time(),
        },
    )
    return root


def is_qwen_checkpoint_file(path: str | os.PathLike[str]) -> bool:
    lowered = str(path).lower()
    if "flux2" in lowered or "flux.2" in lowered or ("flux" in lowered and "klein" in lowered):
        return False
    if not lowered.endswith((".safetensors", ".ckpt")):
        return False
    if not lowered.endswith(".safetensors"):
        return "qwen" in lowered
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as handle:
            keys = set(handle.keys())
        if (
            "model.diffusion_model.input_blocks.0.0.weight" in keys
            or "first_stage_model.encoder.conv_in.weight" in keys
            or any(key.startswith("conditioner.embedders.") for key in keys)
        ):
            return False
        if (
            "double_blocks.0.img_attn.qkv.weight" in keys
            or "single_blocks.0.linear1.weight" in keys
            or "final_layer.adaLN_modulation.1.weight" in keys
        ):
            return False
        qwen_aio_transformer = (
            "model.diffusion_model.img_in.weight" in keys
            and any(key.startswith("model.diffusion_model.transformer_blocks.0.attn.") for key in keys)
        )
        qwen_aio_bundle = (
            qwen_aio_transformer
            and any(key.startswith("text_encoders.") for key in keys)
            and any(key.startswith("vae.") for key in keys)
        )
        qwen_bare_transformer = (
            "__index_timestep_zero__" in keys
            or (
                "img_in.weight" in keys
                and any(key.startswith("transformer_blocks.0.attn.") for key in keys)
            )
            or qwen_aio_transformer
        )
        return (
            qwen_aio_bundle
            or qwen_bare_transformer
        )
    except Exception:
        return False


def qwen_model_from_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    preload: bool = False,
    name: str | None = None,
):
    if _is_qwen_aio_checkpoint(checkpoint_path):
        root = _qwen_runtime_root_from_aio_checkpoint(checkpoint_path)
        _qwen_log(f"SDMLX Qwen: AIO checkpoint detected -> runtime root {root.name}", verbose=True)
    else:
        root = _qwen_runtime_root_from_split_files(checkpoint_path)
    return qwen_model_from_root(root, preload=preload, name=name or Path(checkpoint_path).name)


def qwen_placeholders_from_model_root(model_path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(model_path).expanduser()
    if not is_qwen_model_root(root):
        raise RuntimeError("SDMLX Qwen: cannot build placeholders without a complete Qwen model root.")
    text_encoder_files = sorted((root / "text_encoder").glob("*.safetensors"))
    vae_files = sorted((root / "vae").glob("*.safetensors"))
    if not text_encoder_files:
        raise FileNotFoundError(f"SDMLX Qwen: no text_encoder safetensors found in {root}.")
    if not vae_files:
        raise FileNotFoundError(f"SDMLX Qwen: no VAE safetensors found in {root}.")
    return (
        qwen_clip_from_text_encoder(text_encoder_files[0], name=text_encoder_files[0].name),
        qwen_vae_from_file(vae_files[0], name=vae_files[0].name),
    )


def qwen_clip_from_text_encoder(text_encoder_path: str | os.PathLike[str], name: str | None = None) -> dict[str, Any]:
    path = Path(text_encoder_path).expanduser()
    root = _qwen_root_from_component_path(path)
    clip = {
        "type": QWEN_MODEL_FAMILY,
        "cache_key": str(path.resolve() if path.exists() else path),
        "text_encoder_path": str(path.resolve() if path.exists() else path),
        "name": str(name or path.name),
        "unused": False,
    }
    if root is not None:
        clip["model_path"] = str(root.resolve())
        clip["tokenizer_path"] = str((root / "tokenizer").resolve())
    return clip


def is_qwen_vae_file(path: str | os.PathLike[str]) -> bool:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as handle:
            keys = set(handle.keys())
        native_markers = {
            "decoder.conv_in.conv3d.weight",
            "decoder.conv_out.conv3d.weight",
        }
        comfy_markers = {
            "decoder.middle.1.to_qkv.weight",
            "encoder.head.0.gamma",
        }
        return native_markers.issubset(keys) or comfy_markers.issubset(keys)
    except Exception:
        return False


def qwen_vae_from_file(vae_path: str | os.PathLike[str], name: str | None = None) -> dict[str, Any]:
    path = Path(vae_path).expanduser()
    root = _qwen_root_from_component_path(path)
    vae_root = root if root is not None else _qwen_vae_runtime_root_from_file(path)
    vae = {
        "type": QWEN_MODEL_FAMILY,
        "cache_key": str(path.resolve() if path.exists() else path),
        "vae_path": str(path.resolve() if path.exists() else path),
        "vae_root": str(vae_root.resolve()),
        "name": str(name or path.name),
        "unused": False,
    }
    if root is not None:
        vae["model_path"] = str(root.resolve())
    return vae


def qwen_model_from_manifest(package_path: str | os.PathLike[str], manifest: dict[str, Any], preload: bool = False):
    package_path = Path(package_path)
    manifest = _qwen_materialize_hf_package_if_needed(package_path, manifest)
    model_path = manifest.get("model_path")
    components = manifest.get("components") or {}
    model_component = components.get("model") if isinstance(components, dict) else None
    if not model_path and isinstance(model_component, dict):
        model_path = model_component.get("path")
    model_path = str(model_path or manifest.get("source_repo") or QWEN_DEFAULT_MODEL)
    if model_path and not os.path.isabs(model_path) and not model_path.startswith(("mlx-community/", "Qwen/", "http")):
        model_path = str((package_path / model_path).resolve())
    local_root = Path(model_path).expanduser() if os.path.isabs(model_path) else None
    transformer_keys = (
        _qwen_transformer_identity_keys(local_root)
        if local_root is not None and is_qwen_model_root(local_root)
        else set()
    )
    qwen_variant, model_version = resolve_qwen_model_identity(
        manifest=manifest,
        transformer_keys=transformer_keys,
        source_identifiers=(manifest.get("source_root"), manifest.get("source_repo")),
        root_hint=package_path,
    )
    manifest = _backfill_qwen_manifest_identity(package_path, manifest, qwen_variant, model_version)
    source_repo = manifest.get("source_repo") or model_path

    model = {
        "model_family": QWEN_MODEL_FAMILY,
        "qwen_variant": qwen_variant,
        "model_version": model_version,
        "model_path": model_path,
        "cache_key": str(package_path),
        "package_path": str(package_path),
        "prepared_cache": bool(manifest.get("prepared_cache")),
        "prepared_cache_format": manifest.get("prepared_cache_format"),
        "loras": list(manifest.get("loras") or []),
        "mod_lora_scale": float(manifest.get("mod_lora_scale") or 0.0),
        "recommendations": dict(manifest.get("recommendations") or {}),
        "source_repo": source_repo,
    }
    clip_placeholder = {
        "type": QWEN_MODEL_FAMILY,
        "cache_key": str(package_path),
        "model_path": model_path,
        "source_repo": source_repo,
        "unused": True,
    }
    vae_placeholder = {
        "type": QWEN_MODEL_FAMILY,
        "cache_key": str(package_path),
        "model_path": model_path,
        "source_repo": source_repo,
        "unused": True,
    }
    if preload:
        _load_qwen_model(model_path=model_path, lora_specs=[])
    return model, clip_placeholder, vae_placeholder


def _qwen_materialized_manifest(package_path: Path, manifest: dict[str, Any], source_repo: str) -> dict[str, Any]:
    updated = dict(manifest)
    updated["model_path"] = "."
    updated["materialized"] = True
    updated["source_repo"] = source_repo
    components = dict(updated.get("components") or {})
    components.update(
        {
            "model": {"storage": "package_dir", "path": "."},
            "transformer": {"storage": "package_dir", "path": "transformer"},
            "text_encoder": {"storage": "package_dir", "path": "text_encoder"},
            "vae": {"storage": "package_dir", "path": "vae"},
            "tokenizer": {"storage": "package_dir", "path": "tokenizer"},
        }
    )
    updated["components"] = components
    _write_json(package_path / "manifest.json", updated)
    return updated


def _qwen_materialize_hf_package_if_needed(package_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if is_qwen_model_root(package_path):
        source_repo = str(manifest.get("source_repo") or QWEN_DEFAULT_MODEL)
        if str(manifest.get("model_path") or "") != "." or not manifest.get("materialized"):
            return _qwen_materialized_manifest(package_path, manifest, source_repo)
        return manifest

    components = manifest.get("components") or {}
    model_component = components.get("model") if isinstance(components, dict) else None
    storage = str(model_component.get("storage") if isinstance(model_component, dict) else "").strip()
    if storage != "huggingface_repo":
        return manifest

    repo_id = str(
        (model_component.get("path") if isinstance(model_component, dict) else None)
        or manifest.get("source_repo")
        or QWEN_DEFAULT_MODEL
    ).strip()
    if not repo_id:
        raise RuntimeError("SDMLX Qwen: empty Hugging Face repo id in .sdmlx package manifest.")

    print(f"SDMLX Qwen: downloading model package {repo_id} into {package_path.name}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(package_path),
            ignore_patterns=[".git/*", ".cache/*"],
        )
    except TypeError:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(package_path),
        )

    if not is_qwen_model_root(package_path):
        raise RuntimeError(
            "SDMLX Qwen: downloaded package is not a complete Qwen root. "
            "Expected transformer/, text_encoder/, vae/, and tokenizer/."
        )
    return _qwen_materialized_manifest(package_path, manifest, repo_id)


def create_qwen_dummy_package(
    package_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str] | None = None,
    source_repo: str | None = None,
) -> Path:
    package_path = Path(package_path)
    package_path.mkdir(parents=True, exist_ok=True)
    resolved_source = str(source_repo or model_path or QWEN_DEFAULT_MODEL)
    qwen_variant, model_version = resolve_qwen_model_identity(
        source_identifiers=(model_path, resolved_source),
        root_hint=package_path,
    )
    manifest = {
        "package_format": QWEN_PACKAGE_FORMAT,
        "model_family": QWEN_MODEL_FAMILY,
        "qwen_variant": qwen_variant,
        "model_version": model_version,
        "runtime": "sdmlx_qwen_native",
        "source_repo": resolved_source,
        "model_path": str(model_path) if model_path else resolved_source,
        "mod_lora_scale": 0.0,
        "recommendations": {
            "steps": 4,
            "guidance": 1.0,
            "scheduler": "linear",
            "acceleration_patch": QWEN_ACCEL_PATCH_LABEL,
        },
        "components": {
            "model": {
                "storage": "external_dir" if model_path else "huggingface_repo",
                "path": str(model_path) if model_path else resolved_source,
            }
        },
    }
    _write_json(package_path / "manifest.json", manifest)
    return package_path


def _is_qwen_acceleration_patch_package(package_path: Path) -> bool:
    try:
        manifest = _read_json(package_path / "manifest.json")
    except Exception:
        return False
    if manifest.get("format") != "sdmlx-acceleration-patch-v1":
        return False
    family = str(manifest.get("base_model_family", "")).strip().lower()
    return family in {"qwen", QWEN_MODEL_FAMILY, "qwen-image-edit", "qwen-image-edit-2511", "qwen-image", "qwen-image-2512"}


def _qwen_acceleration_patch_metadata(package_name: str) -> dict[str, Any]:
    metadata = dict(QWEN_ACCEL_PATCH_REGISTRY.get(package_name, {}))
    if metadata:
        return metadata
    try:
        manifest = _read_json(_qwen_acceleration_package_path(package_name) / "manifest.json")
    except Exception:
        return {}
    return {
        "label": str(manifest.get("name") or package_name.removesuffix(".sdmlxpatch")),
        "base_model_family": str(manifest.get("base_model_family") or "qwen"),
        "model_version": str(manifest.get("model_version") or ""),
        "recommended_steps": manifest.get("recommended_steps", 4),
        "recommended_scheduler": str(manifest.get("recommended_scheduler") or "linear"),
        "source_repo": str(manifest.get("source_repo") or ""),
        "source_file": str(manifest.get("source_file") or ""),
    }


def _qwen_acceleration_patch_applies(package_name: str, model_variant: str) -> bool:
    family = str(_qwen_acceleration_patch_metadata(package_name).get("base_model_family") or "").strip().lower()
    if family == "qwen":
        return True
    if model_variant == "qwen-image":
        return family in {"qwen-image", "qwen-image-2512"}
    return family in {"qwen-image-edit", "qwen-image-edit-2511"}


def qwen_acceleration_label(package_name: str) -> str:
    return QWEN_ACCEL_PATCH_LABELS.get(package_name, package_name.removesuffix(".sdmlxpatch"))


def qwen_acceleration_package_name(selection: str | None) -> str | None:
    if not selection or selection == QWEN_ACCEL_NONE:
        return None
    value = str(selection)
    package_name = QWEN_ACCEL_PATCH_BY_LABEL.get(value, QWEN_ACCEL_PATCH_LABEL_ALIASES.get(value, value))
    package_name = os.path.basename(package_name)
    if not package_name.endswith(".sdmlxpatch"):
        package_name += ".sdmlxpatch"
    return package_name


def is_qwen_acceleration_patch_selection(selection: str | None) -> bool:
    package_name = qwen_acceleration_package_name(selection)
    if package_name is None:
        return False
    if package_name in QWEN_ACCEL_PATCH_LABELS:
        return True
    return _is_qwen_acceleration_patch_package(_qwen_acceleration_package_path(package_name))


def qwen_acceleration_patch_options() -> list[str]:
    global _QWEN_ACCEL_OPTIONS_CACHE
    if _QWEN_ACCEL_OPTIONS_CACHE is not None:
        return list(_QWEN_ACCEL_OPTIONS_CACHE)

    names = set(QWEN_ACCEL_PATCH_LABELS)
    try:
        for entry in _qwen_acceleration_patch_dir().iterdir():
            if entry.is_dir() and entry.name.endswith(".sdmlxpatch") and _is_qwen_acceleration_patch_package(entry):
                names.add(entry.name)
    except Exception:
        pass

    options = sorted((qwen_acceleration_label(name) for name in names), key=str.lower)
    _QWEN_ACCEL_OPTIONS_CACHE = tuple(options)
    return list(options)


def _lora_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.extend(Path(path) for path in folder_paths.get_folder_paths("loras"))
    except Exception:
        pass
    try:
        roots.append(Path(folder_paths.models_dir) / "loras")
    except Exception:
        pass
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            root = root.expanduser()
            if not root.exists():
                continue
            key = str(root.resolve())
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _find_local_qwen_acceleration_lora(filename: str) -> Path | None:
    for root in _lora_roots():
        candidate = root / "qwen" / filename
        if candidate.exists():
            return candidate
        for path in root.rglob(filename):
            if path.exists():
                return path
    return None


def ensure_qwen_acceleration_patch(selection: str | None) -> tuple[str, Path] | None:
    package_name = qwen_acceleration_package_name(selection)
    if package_name is None:
        return None
    if package_name not in QWEN_ACCEL_PATCH_LABELS and not _is_qwen_acceleration_patch_package(_qwen_acceleration_package_path(package_name)):
        return None

    package_path = _qwen_acceleration_package_path(package_name)
    patch_path = package_path / "patch.safetensors"
    metadata = _qwen_acceleration_patch_metadata(package_name)
    if not patch_path.exists():
        package_path.mkdir(parents=True, exist_ok=True)
        source_file = str(metadata.get("source_file") or "")
        source_repo = str(metadata.get("source_repo") or "")
        local_source = _find_local_qwen_acceleration_lora(source_file) if source_file else None
        if local_source is None:
            if not source_repo or not source_file:
                raise RuntimeError(f"SDMLX Qwen: acceleration-patch package has no source file: {package_name}")
            print(f"SDMLX Qwen: downloading acceleration-patch {metadata.get('label') or package_name}")
            downloaded = Path(
                hf_hub_download(
                    repo_id=source_repo,
                    filename=source_file,
                )
            )
            local_source = downloaded
        shutil.copy2(local_source, patch_path)

    manifest_path = package_path / "manifest.json"
    if not manifest_path.exists():
        _write_json(
            manifest_path,
            {
                "format": "sdmlx-acceleration-patch-v1",
                "base_model_family": metadata.get("base_model_family") or QWEN_MODEL_FAMILY,
                "model_version": metadata.get("model_version") or QWEN_MODEL_VERSION,
                "patch_kind": "lora",
                "name": metadata.get("label") or qwen_acceleration_label(package_name),
                "recommended_steps": metadata.get("recommended_steps", 4),
                "recommended_guidance": 1.0,
                "recommended_scheduler": metadata.get("recommended_scheduler") or "linear",
                "source_repo": metadata.get("source_repo") or "",
                "source_file": metadata.get("source_file") or "",
            },
        )
    source_metadata_path = package_path / "source_metadata.json"
    if not source_metadata_path.exists():
        _write_json(
            source_metadata_path,
            {
                "source_repo": metadata.get("source_repo") or "",
                "source_file": metadata.get("source_file") or "",
                "local_patch_file": "patch.safetensors",
            },
        )

    if not _is_qwen_acceleration_patch_package(package_path):
        raise RuntimeError(f"SDMLX Qwen: acceleration-patch is not a Qwen package: {package_path}")
    return (qwen_acceleration_label(package_name), patch_path)


def _image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image[0].detach().float().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _pil_to_image_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _qwen_model_cache_key(
    model_path: str,
    lora_specs: list[tuple[str, float, float]],
    *,
    bake_lora_to_quantized: bool = False,
    model_variant: str = "qwen-image-edit",
) -> tuple[Any, ...]:
    return (
        _qwen_file_identity(model_path),
        str(model_variant),
        bool(bake_lora_to_quantized),
        tuple(
            (
                _qwen_file_identity(path),
                round(float(scale), 8),
                round(float(mod_scale), 8),
            )
            for path, scale, mod_scale in lora_specs
        ),
    )


def _bake_qwen_lora_wrappers(module: Any) -> dict[str, int | float]:
    _ensure_suite_qwen_native_runtime()
    from mlx import nn
    from sdmlx_qwen_native.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
    from sdmlx_qwen_native.models.common.lora.layer.linear_lora_layer import LoRALinear
    from sdmlx_qwen_native.models.common.lora.layer.lokr_linear_layer import LoKrLinear

    stats: dict[str, int | float] = {
        "lora_wrappers": 0,
        "fused_wrappers": 0,
        "baked": 0,
        "passthrough": 0,
        "skipped": 0,
    }

    def assign(parent: Any, attr_name: str | None, idx: int | None, new_child: Any) -> None:
        if parent is None:
            return
        if isinstance(parent, list) and idx is not None:
            parent[idx] = new_child
        elif isinstance(parent, dict) and attr_name is not None:
            parent[attr_name] = new_child
        elif attr_name is not None:
            setattr(parent, attr_name, new_child)

    def bake_one(base_linear: Any, loras: list[Any]) -> Any:
        if not isinstance(base_linear, nn.QuantizedLinear):
            stats["skipped"] = int(stats["skipped"]) + len(loras)
            return FusedLoRALinear(base_linear=base_linear, loras=loras) if loras else base_linear

        bakeable_loras: list[Any] = []
        passthrough_loras: list[Any] = []
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
                _qwen_log(
                    "SDMLX Qwen: acceleration-patch bake skipped shape mismatch "
                    f"base={tuple(dense.shape)} delta={tuple(delta.shape)}",
                    verbose=True,
                )
                stats["skipped"] = int(stats["skipped"]) + len(loras)
                return FusedLoRALinear(base_linear=base_linear, loras=loras)
            dense = dense + delta

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
        stats["baked"] = int(stats["baked"]) + 1
        if passthrough_loras:
            stats["passthrough"] = int(stats["passthrough"]) + len(passthrough_loras)
            return FusedLoRALinear(base_linear=base_linear, loras=passthrough_loras)
        return base_linear

    def walk(obj: Any, parent: Any = None, attr_name: str | None = None, idx: int | None = None) -> None:
        if isinstance(obj, FusedLoRALinear):
            stats["fused_wrappers"] = int(stats["fused_wrappers"]) + 1
            new_child = bake_one(obj.base_linear, list(obj.loras))
            assign(parent, attr_name, idx, new_child)
            return
        elif isinstance(obj, LoRALinear):
            stats["lora_wrappers"] = int(stats["lora_wrappers"]) + 1
            new_child = bake_one(obj.linear, [obj])
            assign(parent, attr_name, idx, new_child)
            return
        elif isinstance(obj, LoKrLinear):
            stats["passthrough"] = int(stats["passthrough"]) + 1
            return

        if isinstance(obj, list):
            for i, child in enumerate(list(obj)):
                walk(child, obj, None, i)
        elif isinstance(obj, tuple):
            temp = list(obj)
            for i, child in enumerate(temp):
                walk(child, temp, None, i)
            assign(parent, attr_name, idx, type(obj)(temp))
        elif isinstance(obj, dict):
            for key, child in list(obj.items()):
                walk(child, obj, key, None)
        elif isinstance(obj, nn.Module):
            for name, child in vars(obj).items():
                if isinstance(child, (nn.Module, list, tuple, dict)):
                    walk(child, obj, name, None)

    t0 = time.perf_counter()
    walk(module)
    mx.clear_cache()
    stats["seconds"] = time.perf_counter() - t0
    return stats


def _qwen_fp16_quant_params_enabled() -> bool:
    value = str(os.environ.get("SDMLX_QWEN_FP16_QUANT_PARAMS", "")).strip().lower()
    if value:
        return value in {"1", "true", "on", "yes"}
    return True


def _apply_qwen_fp16_quant_params(model: Any) -> None:
    if not _qwen_fp16_quant_params_enabled():
        return
    _ensure_suite_qwen_native_runtime()
    from mlx import nn

    regex = str(os.environ.get("SDMLX_QWEN_FP16_QUANT_REGEX", "")).strip() or _qwen_default_fp16_quant_regex(model)
    pattern = re.compile(regex) if regex else None
    converted = 0
    for component_name in ("transformer", "text_encoder"):
        component = getattr(model, component_name, None)
        if component is None or not isinstance(component, nn.Module):
            continue
        for name, module in component.named_modules():
            if not isinstance(module, nn.QuantizedLinear):
                continue
            full_name = f"{component_name}.{name}"
            if pattern is not None and not pattern.search(full_name):
                continue
            changed = False
            for attr in ("scales", "biases", "bias"):
                value = module.get(attr)
                if value is not None and value.dtype != mx.float16:
                    setattr(module, attr, value.astype(mx.float16))
                    changed = True
            if changed:
                converted += 1
    if converted:
        suffix = f", regex={regex}" if regex else ""
        _qwen_log(f"SDMLX Qwen: fp16 quantized-linear params refreshed ({converted} layers{suffix})", verbose=True)


def _qwen_default_fp16_quant_regex(model: Any) -> str:
    model_config = getattr(model, "model_config", None)
    model_id = " ".join(
        str(part or "")
        for part in (
            getattr(model_config, "model_name", ""),
            getattr(model_config, "base_model", ""),
            " ".join(getattr(model_config, "aliases", []) or []),
        )
    ).lower()
    if "qwen-image" in model_id and "edit" not in model_id:
        return QWEN_FP16_QUANT_PARAMS_QWEN_IMAGE_REGEX
    return QWEN_FP16_QUANT_PARAMS_DEFAULT_REGEX


def _qwen_component_identity(path: str | os.PathLike[str]) -> tuple[Any, ...]:
    path_obj = Path(path).expanduser()
    if path_obj.is_dir():
        return tuple(_qwen_file_identity(file) for file in sorted(path_obj.glob("*.safetensors")))
    return (_qwen_file_identity(path_obj),)


def _qwen_source_identity(source_root: Path) -> tuple[Any, ...]:
    return (
        _qwen_component_identity(source_root / "transformer"),
        _qwen_component_identity(source_root / "text_encoder"),
        _qwen_component_identity(source_root / "vae"),
    )


def _qwen_source_identity_repr(source_root: Path) -> str:
    return repr(_qwen_source_identity(source_root))


def _qwen_canonical_identity_path(path: str) -> str:
    parts = Path(path).parts
    lowered = [part.lower() for part in parts]
    for index in range(len(parts) - 1):
        if lowered[index] == "models" and lowered[index + 1] == "sdmlx":
            tail = "/".join(parts[index + 2 :])
            return "models/sdmlx" + (f"/{tail}" if tail else "")
    return path


def _qwen_normalize_identity_repr(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"(?:/[^\n,'\")]+)+/models/(?:SDMLX|sdmlx)", "models/sdmlx", text)
    # Package sorting must not invalidate prepared-cache identity. If a shared
    # donor package moves from models/SDMLX/foo.sdmlx to
    # models/SDMLX/sorted/foo.sdmlx, the component file contents are unchanged.
    return re.sub(
        r"models/sdmlx/(?:[^,'\")]+/)*([^/,'\")]+\.sdmlx/)",
        r"models/sdmlx/\1",
        text,
    )


def _qwen_identity_repr_matches(stored: Any, current: Any) -> bool:
    return _qwen_normalize_identity_repr(stored) == _qwen_normalize_identity_repr(current)


def _qwen_file_identity(path: str | os.PathLike[str]) -> tuple[str, int | None, int | None]:
    path_obj = Path(path).expanduser()
    try:
        resolved = path_obj.resolve()
    except Exception:
        resolved = path_obj
    try:
        stat = resolved.stat()
        return (_qwen_canonical_identity_path(str(resolved)), int(stat.st_size), int(stat.st_mtime_ns))
    except Exception:
        return (_qwen_canonical_identity_path(str(resolved)), None, None)


def _qwen_saved_component_format(path: Path) -> bool:
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader

    shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
    if not shard_files:
        return False
    metadata = WeightLoader._read_safetensors_metadata(shard_files[0])
    return metadata.get("quantization_level") is not None or metadata.get("mflux_version") is not None


def _qwen_root_uses_comfy_fp8_transformer(root: Path) -> bool:
    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader

    transformer = root / "transformer"
    if not transformer.is_dir():
        return False
    return any(
        WeightLoader._is_comfy_fp8_safetensors(path) or WeightLoader._has_float8_safetensors(path)
        for path in sorted(transformer.glob("*.safetensors"))
        if not path.name.startswith("._")
    )


def _qwen_prepared_model_package_root(source_root: Path) -> Path:
    return _qwen_prepared_model_root(source_root, parent=_sdmlx_model_root(), suffix=".sdmlx")


def _qwen_prepared_package_base_name(source_root: Path) -> str:
    transformer_files = sorted((source_root / "transformer").glob("*.safetensors"))
    if transformer_files:
        base = transformer_files[0].stem
    else:
        base = source_root.name
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in base).strip(" .") or "qwen"


def _qwen_prepared_model_root(source_root: Path, parent: Path, suffix: str) -> Path:
    identity = _qwen_source_identity(source_root)
    digest = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:16]
    safe_name = _qwen_prepared_package_base_name(source_root)
    package_root = parent / f"{safe_name}{suffix}"
    if not package_root.exists():
        return package_root
    try:
        manifest = _read_json(package_root / "manifest.json")
    except Exception:
        manifest = {}
    if _qwen_identity_repr_matches(manifest.get("source_identity"), repr(identity)):
        return package_root
    return parent / f"{safe_name}-{digest}{suffix}"


def _qwen_iter_sdmlx_packages(root: Path):
    skip_dirs = {"cache", "AccelerationPatches", "SpeedPatches"}
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return
    for entry in entries:
        if not entry.is_dir() or entry.name.startswith(".") or entry.name in skip_dirs:
            continue
        if entry.name.endswith(".sdmlx"):
            yield entry
            continue
        yield from _qwen_iter_sdmlx_packages(entry)


def _qwen_find_prepared_package_by_source_identity(source_identity: str) -> Path | None:
    root = _sdmlx_model_root()
    if not root.is_dir():
        return None
    for package_root in _qwen_iter_sdmlx_packages(root):
        try:
            manifest = _read_json(package_root / "manifest.json")
        except Exception:
            continue
        if not _qwen_identity_repr_matches(manifest.get("source_identity"), source_identity):
            continue
        if _qwen_prepared_model_is_complete(package_root):
            return package_root
    return None


def _qwen_prepared_model_is_complete(cache_root: Path) -> bool:
    if not is_qwen_model_root(cache_root):
        return False
    marker = cache_root / "sdmlx_qwen_prepared_cache.json"
    if not marker.is_file():
        return False
    return all(_qwen_saved_component_format(cache_root / name) for name in ("transformer", "text_encoder", "vae"))


def _qwen_variant_and_version_from_root(source_root: Path) -> tuple[str, str]:
    manifest: dict[str, Any] = {}
    manifest_path = source_root / "manifest.json"
    if manifest_path.is_file():
        candidate = _read_json(manifest_path)
        if is_qwen_manifest(candidate):
            manifest = candidate
    source_identifiers = _qwen_root_source_identifiers(source_root)
    source_identifiers.extend(
        str(manifest.get(key) or "")
        for key in ("source_root", "source_repo")
    )
    return resolve_qwen_model_identity(
        manifest=manifest,
        transformer_keys=_qwen_transformer_identity_keys(source_root),
        source_identifiers=source_identifiers,
        root_hint=_qwen_identity_root_hint(source_root),
    )


def _write_qwen_prepared_package_manifest(package_root: Path, source_root: Path) -> None:
    marker_path = package_root / "sdmlx_qwen_prepared_cache.json"
    marker = {}
    if marker_path.is_file():
        try:
            marker = _read_json(marker_path)
        except Exception:
            marker = {}
    qwen_variant, model_version = _qwen_variant_and_version_from_root(source_root)
    manifest = {
        "package_format": QWEN_PACKAGE_FORMAT,
        "model_family": QWEN_MODEL_FAMILY,
        "qwen_variant": qwen_variant,
        "model_version": model_version,
        "runtime": "sdmlx_qwen_native",
        "source_repo": str(source_root),
        "model_path": ".",
        "prepared_cache": True,
        "prepared_cache_format": marker.get("cache_format") or "sdmlx-qwen-prepared-model-v1",
        "source_root": marker.get("source_root") or str(source_root),
        "source_identity": marker.get("source_identity"),
        "created_at": marker.get("created_at") or time.time(),
        "mod_lora_scale": 0.0,
        "recommendations": {
            "steps": 4,
            "guidance": 1.0,
            "scheduler": "linear",
            "acceleration_patch": QWEN_ACCEL_PATCH_LABEL,
        },
        "components": {
            "model": {
                "storage": "package_dir",
                "path": ".",
            },
            "transformer": {
                "storage": "package_dir",
                "path": "transformer",
            },
            "text_encoder": {
                "storage": "package_dir",
                "path": "text_encoder",
            },
            "vae": {
                "storage": "package_dir",
                "path": "vae",
            },
            "tokenizer": {
                "storage": "package_dir",
                "path": "tokenizer",
            },
        },
    }
    _write_json(package_root / "manifest.json", manifest)


def _qwen_migrate_prepared_package_to_readable_name(source_root: Path, existing_package: Path, package_root: Path) -> Path:
    if existing_package == package_root or package_root.exists():
        return existing_package
    try:
        if existing_package.parent.resolve() != package_root.parent.resolve():
            return existing_package
    except Exception:
        if existing_package.parent != package_root.parent:
            return existing_package
    source_identity = _qwen_source_identity_repr(source_root)
    try:
        manifest = _read_json(existing_package / "manifest.json")
    except Exception:
        return existing_package
    if manifest.get("source_identity") != source_identity or not _qwen_prepared_model_is_complete(existing_package):
        return existing_package
    try:
        shutil.move(str(existing_package), str(package_root))
        _write_qwen_prepared_package_manifest(package_root, source_root)
        print(f"SDMLX Qwen: prepared model package renamed to {package_root.name}")
        return package_root
    except Exception:
        return existing_package


def _qwen_cached_runtime_model_path(model_path: str) -> str:
    source_root = Path(model_path).expanduser()
    if not source_root.is_dir() or not is_qwen_model_root(source_root):
        return model_path

    try:
        source_root = source_root.resolve()
    except Exception:
        pass

    if _qwen_saved_component_format(source_root / "transformer"):
        return str(source_root)

    if not _qwen_root_uses_comfy_fp8_transformer(source_root):
        return str(source_root)

    package_root = _qwen_prepared_model_package_root(source_root)
    if _qwen_prepared_model_is_complete(package_root):
        _write_qwen_prepared_package_manifest(package_root, source_root)
        _qwen_log(f"SDMLX Qwen: prepared model package hit: {package_root.name}", verbose=True)
        return str(package_root)

    source_identity = _qwen_source_identity_repr(source_root)
    existing_package = _qwen_find_prepared_package_by_source_identity(source_identity)
    if existing_package is not None:
        existing_package = _qwen_migrate_prepared_package_to_readable_name(source_root, existing_package, package_root)
        _qwen_log(f"SDMLX Qwen: prepared model package hit by identity: {existing_package.name}", verbose=True)
        return str(existing_package)

    _ensure_suite_qwen_native_runtime()
    qwen_variant, _model_version = _qwen_variant_and_version_from_root(source_root)
    if qwen_variant == "qwen-image":
        from sdmlx_qwen_native.models.qwen.variants.txt2img.qwen_image import QwenImage as QwenPreparedModel
    else:
        from sdmlx_qwen_native.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit as QwenPreparedModel

    print(f"SDMLX Qwen: preparing MLX Q8 model cache for {source_root.name}...")
    parent = package_root.parent
    tmp_root = Path(tempfile.mkdtemp(prefix=f".{package_root.name}.", dir=str(parent)))
    t0 = time.perf_counter()
    model = None
    try:
        model = QwenPreparedModel(model_path=str(source_root))
        model.save_model(str(tmp_root))
        marker = {
            "cache_format": "sdmlx-qwen-prepared-model-v1",
            "source_root": str(source_root),
            "source_identity": source_identity,
            "created_at": time.time(),
        }
        with open(tmp_root / "sdmlx_qwen_prepared_cache.json", "w", encoding="utf-8") as handle:
            json.dump(marker, handle, indent=2, sort_keys=True)
        _write_qwen_prepared_package_manifest(tmp_root, source_root)
        if package_root.exists():
            shutil.rmtree(package_root)
        os.replace(tmp_root, package_root)
        print(f"SDMLX Qwen: prepared model package stored: {package_root.name} ({time.perf_counter() - t0:.2f}s)")
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    finally:
        try:
            del model
        except Exception:
            pass
        mx.clear_cache()

    return str(package_root)


def _qwen_lora_summary(lora_specs: list[tuple[str, float, float]]) -> str:
    return (
        ", ".join(
            f"{Path(path).name}@{scale:.2f}/mod{mod_scale:.2f}"
            for path, scale, mod_scale in lora_specs
        )
        or "none"
    )


def _load_qwen_model(
    model_path: str,
    lora_specs: list[tuple[str, float, float]],
    *,
    bake_lora_to_quantized: bool = False,
    model_variant: str = "qwen-image-edit",
):
    runtime_model_path = _qwen_cached_runtime_model_path(model_path)
    model_variant = "qwen-image" if str(model_variant).strip().lower() == "qwen-image" else "qwen-image-edit"
    cache_key = _qwen_model_cache_key(
        runtime_model_path,
        lora_specs,
        bake_lora_to_quantized=bake_lora_to_quantized,
        model_variant=model_variant,
    )
    cached = _QWEN_MODEL_CACHE.get(cache_key)
    if cached is not None:
        _qwen_log(
            "SDMLX Qwen: model cache hit "
            f"(loras={_qwen_lora_summary(lora_specs)}, baked={bool(bake_lora_to_quantized)})",
            verbose=True,
        )
        _apply_qwen_fp16_quant_params(cached)
        return cached

    _ensure_suite_qwen_native_runtime()
    if model_variant == "qwen-image":
        from sdmlx_qwen_native.models.qwen.variants.txt2img.qwen_image import QwenImage
    else:
        from sdmlx_qwen_native.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

    lora_paths = [path for path, _scale, _mod_scale in lora_specs] or None
    lora_scales = [scale for _path, scale, _mod_scale in lora_specs] or None
    lora_mod_scales = [mod_scale for _path, _scale, mod_scale in lora_specs] or None
    lora_summary = _qwen_lora_summary(lora_specs)
    _qwen_log(
        f"SDMLX Qwen: loading {model_variant} "
        f"model={Path(runtime_model_path).name or runtime_model_path}, loras={lora_summary}, "
        f"baked={bool(bake_lora_to_quantized)}",
        verbose=True,
    )
    t0 = time.perf_counter()
    if model_variant == "qwen-image":
        model = QwenImage(
            model_path=runtime_model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_mod_scales=lora_mod_scales,
        )
    else:
        model = QwenImageEdit(
            model_path=runtime_model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_mod_scales=lora_mod_scales,
        )
    _apply_qwen_fp16_quant_params(model)
    if bake_lora_to_quantized and lora_specs:
        bake_stats = _bake_qwen_lora_wrappers(model.transformer)
        _apply_qwen_fp16_quant_params(model)
        _qwen_log(
            "SDMLX Qwen: acceleration-patch baked into quantized weights "
            f"(baked={int(bake_stats['baked'])}, passthrough={int(bake_stats.get('passthrough', 0))}, "
            f"skipped={int(bake_stats['skipped'])}, "
            f"time={float(bake_stats['seconds']):.2f}s)",
            verbose=True,
        )
    _clear_qwen_reference_vae_cache()
    _QWEN_MODEL_CACHE.clear()
    _QWEN_MODEL_CACHE[cache_key] = model
    _qwen_log(f"SDMLX Qwen: model load={time.perf_counter() - t0:.2f}s", verbose=True)
    return model


def _qwen_vae_runtime_root_from_file(vae_path: str | os.PathLike[str]) -> Path:
    path = Path(vae_path).expanduser().resolve()
    root = _qwen_root_from_component_path(path)
    if root is not None:
        return root.resolve()
    digest_source = str(_qwen_file_identity(path))
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
    root = _qwen_vae_roots_dir() / f"{path.stem}-{digest}"
    _safe_symlink_or_copy(path, root / "vae" / path.name)
    return root


def _qwen_vae_root_from_placeholder(mlx_vae: dict[str, Any]) -> str:
    vae_root = mlx_vae.get("vae_root")
    if vae_root:
        return str(vae_root)

    model_path = mlx_vae.get("model_path")
    if model_path:
        return str(model_path)

    vae_path = mlx_vae.get("vae_path")
    if vae_path:
        root = _qwen_root_from_component_path(vae_path)
        if root is not None:
            return str(root.resolve())
        return str(_qwen_vae_runtime_root_from_file(vae_path))

    cache_key = mlx_vae.get("cache_key")
    if cache_key:
        try:
            package_path = Path(cache_key)
            manifest = _read_json(package_path / "manifest.json")
            manifest_model_path = manifest.get("model_path")
            components = manifest.get("components") or {}
            model_component = components.get("model") if isinstance(components, dict) else None
            if not manifest_model_path and isinstance(model_component, dict):
                manifest_model_path = model_component.get("path")
            manifest_model_path = str(manifest_model_path or manifest.get("source_repo") or QWEN_DEFAULT_MODEL)
            if (
                manifest_model_path
                and not os.path.isabs(manifest_model_path)
                and not manifest_model_path.startswith(("mlx-community/", "Qwen/", "http"))
            ):
                manifest_model_path = str((package_path / manifest_model_path).resolve())
            return manifest_model_path
        except Exception:
            pass

    return str(mlx_vae.get("source_repo") or QWEN_DEFAULT_MODEL)


def _load_qwen_vae(vae_root: str):
    cache_key = ("qwen_vae", _qwen_component_identity(Path(vae_root) / "vae"))
    cached = _QWEN_VAE_CACHE.get(cache_key)
    if cached is not None:
        _qwen_log("SDMLX Qwen VAE: cache hit", verbose=True)
        return cached

    _ensure_suite_qwen_native_runtime()
    from sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader
    from sdmlx_qwen_native.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from sdmlx_qwen_native.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
    from mlx.utils import tree_flatten

    component = next(c for c in QwenWeightDefinition.get_components() if c.name == "vae")
    t0 = time.perf_counter()
    weights, _q_level, _version = WeightLoader._load_component(Path(vae_root), component)
    vae = QwenVAE()
    expected = dict(tree_flatten(vae.parameters()))
    loaded = dict(tree_flatten(weights))
    missing = sorted(set(expected) - set(loaded))
    if missing:
        raise RuntimeError(
            "SDMLX Qwen VAE: unsupported or incomplete weight mapping: "
            f"missing={len(missing)} ({', '.join(missing[:3])})"
        )
    vae.update(weights, strict=True)
    mx.eval(vae.parameters())
    _clear_qwen_reference_vae_cache()
    _QWEN_VAE_CACHE.clear()
    _QWEN_VAE_CACHE[cache_key] = vae
    _qwen_log(f"SDMLX Qwen VAE: load={time.perf_counter() - t0:.2f}s", verbose=True)
    return vae


def _qwen_vae_from_placeholder(mlx_vae: Any):
    if not isinstance(mlx_vae, dict) or mlx_vae.get("type") != QWEN_MODEL_FAMILY:
        raise RuntimeError("SDMLX Qwen VAE: connect a Qwen mlx_vae output from SDMLX VAE Loader or Loader Universal.")
    return _load_qwen_vae(_qwen_vae_root_from_placeholder(mlx_vae))


def _qwen_tensor_to_mx_image(image: Any) -> mx.array:
    if hasattr(image, "detach"):
        image_np = image.detach().cpu().float().numpy()
    else:
        image_np = np.asarray(image, dtype=np.float32)
    if image_np.ndim != 4 or image_np.shape[-1] != 3:
        raise RuntimeError(f"SDMLX Qwen VAE Encode: expected BHWC RGB image tensor, got {tuple(image_np.shape)}.")
    if int(image_np.shape[0]) != 1:
        raise RuntimeError("SDMLX Qwen VAE Encode currently supports batch_size=1.")
    image_nchw = np.transpose(image_np.astype(np.float32), (0, 3, 1, 2))
    return mx.array(image_nchw) * 2.0 - 1.0


def _qwen_latent_to_mx(samples: Any) -> mx.array:
    if isinstance(samples, mx.array):
        latents = samples
    elif hasattr(samples, "detach"):
        latents = mx.array(samples.detach().cpu().float().numpy())
    else:
        latents = mx.array(np.asarray(samples, dtype=np.float32))
    if latents.ndim == 4 and latents.shape[-1] == 16 and latents.shape[1] != 16:
        latents = mx.transpose(latents, (0, 3, 1, 2))
    if latents.ndim not in (4, 5) or int(latents.shape[1]) != 16:
        raise RuntimeError(f"SDMLX Qwen VAE Decode: expected 16-channel Qwen latent, got shape {tuple(latents.shape)}.")
    return latents


def encode_qwen_image_with_vae(pixels: Any, mlx_vae: Any) -> dict[str, Any]:
    from sdmlx_qwen_native.models.common.vae.vae_util import VAEUtil

    t0 = time.perf_counter()
    vae = _qwen_vae_from_placeholder(mlx_vae)
    image = _qwen_tensor_to_mx_image(pixels)
    latents = VAEUtil.encode(vae=vae, image=image)
    mx.eval(latents)
    _qwen_log(f"SDMLX Qwen VAE Encode: total={time.perf_counter() - t0:.3f}s", verbose=True)
    return {"samples": latents, "model_family": QWEN_MODEL_FAMILY}


def decode_qwen_latent_with_vae(mlx_latent: dict[str, Any], mlx_vae: Any) -> torch.Tensor:
    from sdmlx_qwen_native.models.common.vae.vae_util import VAEUtil

    t0 = time.perf_counter()
    vae = _qwen_vae_from_placeholder(mlx_vae)
    latents = _qwen_latent_to_mx(mlx_latent["samples"])
    decoded = VAEUtil.decode(vae=vae, latent=latents)
    image = _qwen_decoded_to_image_tensor(decoded)
    _qwen_log(f"SDMLX Qwen VAE Decode: total={time.perf_counter() - t0:.3f}s", verbose=True)
    return image


def _qwen_decoded_to_image_tensor(decoded: mx.array) -> torch.Tensor:
    if decoded.ndim == 5 and decoded.shape[2] == 1:
        decoded = decoded[:, :, 0, :, :]
    decoded = mx.transpose(decoded, (0, 2, 3, 1))
    decoded = mx.clip(((decoded / 2.0) + 0.5) * 255.0 + 0.5, 0.0, 255.0).astype(mx.uint8)
    mx.eval(decoded)
    image = np.array(decoded).astype(np.float32) / 255.0
    return torch.from_numpy(image).contiguous()


def _qwen_conditioning_entry(
    prompt: str,
    images: list[torch.Tensor],
    negative_prompt: str = "",
    reference_latents: bool = False,
    conditioning_mode: str = "qwen_image_edit_plus",
    use_picture_prefix: bool = True,
    vl_target_area: int = QWEN_EDIT_PLUS_VL_AREA,
    reference_target_area: int | None = None,
    reference_target_areas: list[int] | None = None,
    reference_target_multiple: int | None = None,
    prompt_template: str | None = None,
    image_slots: list[int] | None = None,
    clip: dict[str, Any] | None = None,
    vae: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_image_slots = list(image_slots) if image_slots is not None else list(range(1, len(images) + 1))
    resolved_reference_target_areas = (
        [max(16 * 16, int(area)) for area in reference_target_areas]
        if reference_target_areas is not None
        else None
    )
    entry = {
        "model_family": QWEN_MODEL_FAMILY,
        "conditioning_mode": str(conditioning_mode),
        "prompt": str(prompt or ""),
        "negative_prompt": str(negative_prompt or ""),
        "images": images,
        "image_slots": [int(slot) for slot in resolved_image_slots],
        "reference_latents": bool(reference_latents),
        "use_picture_prefix": bool(use_picture_prefix),
        "vl_target_area": int(vl_target_area),
        "reference_target_area": int(reference_target_area) if reference_target_area is not None else None,
        "reference_target_areas": resolved_reference_target_areas,
        "reference_target_multiple": int(reference_target_multiple) if reference_target_multiple is not None else None,
        "prompt_template": str(prompt_template) if prompt_template is not None else None,
    }
    if isinstance(clip, dict):
        for key in ("model_path", "text_encoder_path", "tokenizer_path"):
            if clip.get(key):
                entry[key] = str(clip[key])
    if isinstance(vae, dict):
        for key in ("model_path", "vae_path"):
            if vae.get(key):
                entry[key] = str(vae[key])
    return entry


def _extract_qwen_entry(conditioning: Any) -> dict[str, Any] | None:
    if isinstance(conditioning, dict) and conditioning.get("model_family") == QWEN_MODEL_FAMILY:
        return conditioning
    if isinstance(conditioning, (list, tuple)):
        for item in conditioning:
            entry = _extract_qwen_entry(item)
            if entry is not None:
                return entry
    return None


def qwen_conditioning_has_entry(conditioning: Any) -> bool:
    return _extract_qwen_entry(conditioning) is not None


def qwen_conditioning_with_reference_image(
    conditioning: Any,
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[Any, bool, bool, int]:
    if isinstance(conditioning, dict) and conditioning.get("model_family") == QWEN_MODEL_FAMILY:
        images = list(conditioning.get("images") or [])
        added = False
        if not images:
            images = [image]
            added = True
        updated = dict(conditioning)
        updated["images"] = images
        updated["reference_latents"] = bool(updated.get("reference_latents", False))
        updated["inpaint_source_image"] = image
        if mask is not None:
            updated["inpaint_mask"] = mask
        return updated, True, added, len(images)

    if isinstance(conditioning, list):
        updated_items = []
        changed = False
        added = False
        image_count = 0
        for item in conditioning:
            if changed:
                updated_items.append(item)
                continue
            updated_item, item_changed, item_added, item_image_count = qwen_conditioning_with_reference_image(item, image, mask)
            updated_items.append(updated_item)
            if item_changed:
                changed = True
                added = item_added
                image_count = item_image_count
        return updated_items, changed, added, image_count

    if isinstance(conditioning, tuple):
        updated_items = []
        changed = False
        added = False
        image_count = 0
        for item in conditioning:
            if changed:
                updated_items.append(item)
                continue
            updated_item, item_changed, item_added, item_image_count = qwen_conditioning_with_reference_image(item, image, mask)
            updated_items.append(updated_item)
            if item_changed:
                changed = True
                added = item_added
                image_count = item_image_count
        return tuple(updated_items), changed, added, image_count

    return conditioning, False, False, 0


def _patch_qwen_conditioning_entries(conditioning: Any, patch_fn) -> tuple[Any, int]:
    if isinstance(conditioning, dict) and conditioning.get("model_family") == QWEN_MODEL_FAMILY:
        return patch_fn(conditioning), 1
    if isinstance(conditioning, list):
        patched_items = []
        count = 0
        for item in conditioning:
            patched_item, patched_count = _patch_qwen_conditioning_entries(item, patch_fn)
            patched_items.append(patched_item)
            count += patched_count
        return patched_items, count
    if isinstance(conditioning, tuple):
        patched_items = []
        count = 0
        for item in conditioning:
            patched_item, patched_count = _patch_qwen_conditioning_entries(item, patch_fn)
            patched_items.append(patched_item)
            count += patched_count
        return tuple(patched_items), count
    return conditioning, 0


def _qwen_distribute_budget(total_area: int, image_count: int, weights: list[int] | None = None) -> list[int]:
    image_count = max(1, int(image_count))
    total_area = max(16 * 16 * image_count, int(total_area))
    if not weights:
        weights = [1] * image_count
    weights = [max(1, int(weight)) for weight in weights[:image_count]]
    if len(weights) < image_count:
        weights.extend([1] * (image_count - len(weights)))

    weight_sum = max(1, sum(weights))
    raw = [(total_area * weight / weight_sum) for weight in weights]
    areas = [max(16 * 16, int(value)) for value in raw]
    delta = total_area - sum(areas)
    if delta > 0:
        order = sorted(range(image_count), key=lambda index: raw[index] - int(raw[index]), reverse=True)
        for offset in range(delta):
            areas[order[offset % image_count]] += 1
    elif delta < 0:
        order = sorted(range(image_count), key=lambda index: raw[index] - int(raw[index]))
        remaining = -delta
        while remaining > 0:
            changed = False
            for index in order:
                if remaining <= 0:
                    break
                if areas[index] > 16 * 16:
                    areas[index] -= 1
                    remaining -= 1
                    changed = True
            if not changed:
                break
    return areas


def _to_torch_float(value: Any) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    if hasattr(value, "detach"):
        return value.detach().cpu().float()
    return torch.as_tensor(value).float()


def _resize_image_nhwc(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    image = _to_torch_float(image)
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"SDMLX Qwen: inpaint source image must be [B,H,W,C], got {tuple(image.shape)}.")
    image = image[..., :3]
    if int(image.shape[1]) == int(height) and int(image.shape[2]) == int(width):
        return torch.clamp(image, 0.0, 1.0).contiguous()
    image_nchw = image.permute(0, 3, 1, 2).contiguous()
    resized = torch.nn.functional.interpolate(image_nchw, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    return torch.clamp(resized.permute(0, 2, 3, 1).contiguous(), 0.0, 1.0)


def _resize_mask_nhw1(mask: torch.Tensor, height: int, width: int, batch: int) -> torch.Tensor:
    mask = _to_torch_float(mask)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if mask.shape[-1] == 1 and mask.shape[1] != 1:
            mask = mask.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"SDMLX Qwen: inpaint mask must be 2D/3D/4D, got {tuple(mask.shape)}.")

    if int(mask.shape[-2]) != int(height) or int(mask.shape[-1]) != int(width):
        mask = torch.nn.functional.interpolate(mask, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    if int(mask.shape[0]) == 1 and int(batch) > 1:
        mask = mask.repeat(int(batch), 1, 1, 1)
    elif int(mask.shape[0]) != int(batch):
        mask = mask[:1].repeat(int(batch), 1, 1, 1)
    return torch.clamp(mask, 0.0, 1.0).permute(0, 2, 3, 1).contiguous()


def _apply_qwen_inpaint_mask(generated: torch.Tensor, source_image: Any, mask: Any) -> torch.Tensor:
    generated = _to_torch_float(generated)
    height = int(generated.shape[1])
    width = int(generated.shape[2])
    batch = int(generated.shape[0])
    source = _resize_image_nhwc(source_image, height, width)
    if int(source.shape[0]) == 1 and batch > 1:
        source = source.repeat(batch, 1, 1, 1)
    elif int(source.shape[0]) != batch:
        source = source[:1].repeat(batch, 1, 1, 1)
    mask_nhw1 = _resize_mask_nhw1(mask, height, width, batch)
    return torch.clamp(generated * mask_nhw1 + source * (1.0 - mask_nhw1), 0.0, 1.0).contiguous()


def qwen_lora_specs_from_model(model: dict[str, Any]) -> list[tuple[str, float, float]]:
    specs: list[tuple[str, float, float]] = []
    if str(model.get("qwen_variant") or "").strip().lower() == "qwen-image" and not model.get("qwen_lora_policy"):
        default_mod_lora_scale = 1.0
    else:
        default_mod_lora_scale = float(model.get("mod_lora_scale") or 0.0)
    for item in list(model.get("lora_specs") or []) + list(model.get("loras") or []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            path, scale = item[0], item[1]
            mod_scale = item[2] if len(item) >= 3 else default_mod_lora_scale
        elif isinstance(item, dict):
            path = item.get("path")
            scale = item.get("strength_model", item.get("scale", 1.0))
            mod_scale = item.get("mod_lora_scale", item.get("modulation_scale", default_mod_lora_scale))
        else:
            continue
        if path is None:
            continue
        scale = float(scale)
        if abs(scale) <= 1e-8:
            continue
        specs.append((str(path), scale, float(mod_scale)))
    return specs


def _qwen_sampling_conditioning(positive: Any, negative: Any) -> tuple[dict[str, Any], str, str, list[Any]]:
    positive_entry = _extract_qwen_entry(positive)
    if positive_entry is None:
        raise RuntimeError("SDMLX Qwen: connect a Qwen Image Edit Conditioning node to the positive input.")
    negative_entry = _extract_qwen_entry(negative)
    prompt = str(positive_entry.get("prompt") or "")
    negative_prompt = ""
    if negative_entry is not None:
        negative_prompt = str(negative_entry.get("prompt") or negative_entry.get("negative_prompt") or "")
    return positive_entry, prompt, negative_prompt, list(positive_entry.get("images") or [])


def _qwen_generated_latent_output(generated: Any, latent_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    generated_latent = getattr(generated, "latent", None)
    if generated_latent is None:
        raise RuntimeError("SDMLX Qwen: native runtime did not return the generated latent.")
    latent_output = {
        "samples": generated_latent,
        "model_family": QWEN_MODEL_FAMILY,
    }
    if isinstance(latent_metadata, dict):
        for key in ("sdmlx_decode_crop", "sdmlx_original_size", "sdmlx_padded_size"):
            if key in latent_metadata:
                latent_output[key] = latent_metadata[key]
    return latent_output


def sample_qwen_image_edit(
    sdmlx_model: dict[str, Any],
    positive: Any,
    negative: Any,
    width: int,
    height: int,
    seed: int,
    steps: int,
    guidance: float,
    scheduler: str,
    speed_patch: str | None,
    patch_strength: float = 1.0,
    latent_metadata: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    positive_entry, prompt, negative_prompt, images = _qwen_sampling_conditioning(positive, negative)
    model_variant, _model_version = resolve_qwen_model_identity(
        manifest={
            "qwen_variant": sdmlx_model.get("qwen_variant"),
            "model_version": sdmlx_model.get("model_version"),
        }
    )
    is_txt2img = model_variant == "qwen-image" and not images
    if not images and not is_txt2img:
        raise RuntimeError("SDMLX Qwen: at least one conditioning image is required.")
    if images and model_variant == "qwen-image":
        raise RuntimeError(
            "SDMLX Qwen: this loaded Qwen Image model does not accept conditioning images. "
            "Use a Qwen Image Edit 2511 model for image-edit workflows."
        )
    image_slots = list(positive_entry.get("image_slots") or range(1, len(images) + 1))

    lora_specs = qwen_lora_specs_from_model(sdmlx_model)
    bake_acceleration_patch = False
    patch_package_name = qwen_acceleration_package_name(speed_patch)
    patch_info = ensure_qwen_acceleration_patch(speed_patch)
    if patch_info is not None and patch_package_name is not None and not _qwen_acceleration_patch_applies(patch_package_name, model_variant):
        patch_info = None
    if patch_info is None:
        if speed_patch and speed_patch != QWEN_ACCEL_NONE:
            _qwen_log("SDMLX Qwen: acceleration-patch: off (not applicable)", verbose=True)
        else:
            _qwen_log("SDMLX Qwen: acceleration-patch: off", verbose=True)
    else:
        patch_label, patch_path = patch_info
        patch_strength = float(patch_strength)
        if abs(patch_strength) > 1e-8:
            lora_specs.insert(0, (str(patch_path), patch_strength, 0.0))
        _qwen_log(f"SDMLX Qwen: acceleration-patch: {patch_label}", verbose=True)

    qwen_scheduler = str(scheduler or "linear")
    qwen_flow_shift = None
    if isinstance(sdmlx_model, dict):
        qwen_scheduler = str(sdmlx_model.get("qwen_scheduler") or qwen_scheduler)
        if sdmlx_model.get("qwen_flow_shift") is not None:
            qwen_flow_shift = float(sdmlx_model["qwen_flow_shift"])
    if qwen_scheduler not in {"linear", "comfy_aura_simple"}:
        qwen_scheduler = "linear"
    if qwen_scheduler == QWEN_AURAFLOW_SCHEDULER:
        if qwen_flow_shift is None:
            qwen_flow_shift = QWEN_AURAFLOW_DEFAULT_SHIFT
        qwen_flow_shift = max(0.01, min(100.0, float(qwen_flow_shift)))

    model = _load_qwen_model(
        model_path=str(sdmlx_model.get("model_path") or QWEN_DEFAULT_MODEL),
        lora_specs=lora_specs,
        bake_lora_to_quantized=bake_acceleration_patch,
        model_variant=model_variant,
    )

    pil_images = [_image_tensor_to_pil(tensor) for tensor in images]
    generate_parameters = inspect.signature(model.generate_image).parameters
    use_direct_images = bool(not is_txt2img and "images" in generate_parameters)
    if not is_txt2img and not use_direct_images and "image_paths" not in generate_parameters:
        raise RuntimeError(
            "SDMLX Qwen: loaded runtime is Qwen Image, but the workflow is Qwen Image Edit. "
            "Reload the correct Qwen Image Edit 2511 model or restart Comfy if the model cache was stale."
        )
    image_context = (
        contextlib.nullcontext(None)
        if is_txt2img or use_direct_images
        else tempfile.TemporaryDirectory(prefix="sdmlx_qwen_")
    )
    with image_context as temp_dir:
        _qwen_log(
            "SDMLX Qwen: sampling "
            f"steps={int(steps)}, guidance={float(guidance):.3f}, scheduler={qwen_scheduler}"
            + (f", flow_shift={qwen_flow_shift:.3f}" if qwen_flow_shift is not None else "")
            + ", "
            f"size={int(width)}x{int(height)}",
            verbose=True,
        )
        t0 = time.perf_counter()
        generate_kwargs = {
            "seed": int(seed),
            "prompt": prompt,
            "num_inference_steps": int(steps),
            "height": int(height),
            "width": int(width),
            "guidance": float(guidance),
            "scheduler": qwen_scheduler,
            "negative_prompt": negative_prompt,
        }
        if not is_txt2img:
            if use_direct_images:
                generate_kwargs["images"] = pil_images
            else:
                temp_path = Path(temp_dir)
                image_paths = []
                for index, image in enumerate(pil_images, start=1):
                    input_path = temp_path / f"input_{index}.png"
                    image.save(input_path)
                    image_paths.append(str(input_path))
                generate_kwargs["image_paths"] = image_paths
        if "use_reference_latents" in generate_parameters:
            generate_kwargs["use_reference_latents"] = bool(positive_entry.get("reference_latents", False))
        if "use_picture_prefix" in generate_parameters:
            generate_kwargs["use_picture_prefix"] = bool(positive_entry.get("use_picture_prefix", True))
        if "vl_target_area" in generate_parameters and positive_entry.get("vl_target_area") is not None:
            generate_kwargs["vl_target_area"] = int(positive_entry["vl_target_area"])
        if "reference_target_area" in generate_parameters and positive_entry.get("reference_target_area") is not None:
            generate_kwargs["reference_target_area"] = int(positive_entry["reference_target_area"])
        if "reference_target_areas" in generate_parameters and positive_entry.get("reference_target_areas") is not None:
            generate_kwargs["reference_target_areas"] = [
                max(16 * 16, int(area))
                for area in list(positive_entry.get("reference_target_areas") or [])
            ]
        if "reference_target_multiple" in generate_parameters and positive_entry.get("reference_target_multiple") is not None:
            generate_kwargs["reference_target_multiple"] = max(16, int(positive_entry["reference_target_multiple"]))
        if "prompt_template" in generate_parameters and positive_entry.get("prompt_template") is not None:
            generate_kwargs["prompt_template"] = str(positive_entry["prompt_template"])
        if "image_slots" in generate_parameters:
            generate_kwargs["image_slots"] = [int(slot) for slot in image_slots]
        if "flow_shift" in generate_parameters:
            generate_kwargs["flow_shift"] = qwen_flow_shift
        elif qwen_flow_shift is not None:
            print("SDMLX Qwen: flow_shift ignored (loaded runtime does not support it; restart/update SDMLX)")
        if "progress_callback" in generate_parameters:
            pbar = _qwen_comfy_progress_bar(int(steps))

            def progress_callback(step: int, total: int) -> None:
                if pbar is not None:
                    pbar.update_absolute(int(step), int(total))

            generate_kwargs["progress_callback"] = progress_callback
        generated = model.generate_image(**generate_kwargs)
        sample_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    mx.clear_cache()
    clear_time = time.perf_counter() - t0
    image = _pil_to_image_tensor(generated.image)
    inpaint_source = positive_entry.get("inpaint_source_image")
    inpaint_mask = positive_entry.get("inpaint_mask")
    if inpaint_source is not None and inpaint_mask is not None:
        image = _apply_qwen_inpaint_mask(image, inpaint_source, inpaint_mask)
        _qwen_log("SDMLX Qwen: inpaint mask composite active", verbose=True)
    latent_output = _qwen_generated_latent_output(generated, latent_metadata)
    _qwen_log(f"SDMLX Qwen: prompt executed in {sample_time + clear_time:.2f}s", verbose=True)
    return image, latent_output


class SDMLXQwenImageEditConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("mlx_clip",),
                "prompt": (
                    "STRING",
                    {
                        "default": "Change the image while preserving the original composition.",
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "vae": ("mlx_vae",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "SDMLX/Conditioning"

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None):
        if isinstance(clip, dict) and clip.get("type") not in (None, QWEN_MODEL_FAMILY):
            raise RuntimeError("SDMLX Qwen: connect the mlx_clip output from the SDMLX CLIP Loader.")
        if vae is not None and (not isinstance(vae, dict) or vae.get("type") != QWEN_MODEL_FAMILY):
            raise RuntimeError("SDMLX Qwen: connect a Qwen mlx_vae from the SDMLX VAE Loader.")
        image_pairs = [(1, image1), (2, image2), (3, image3)]
        images = [tensor for _slot, tensor in image_pairs if tensor is not None]
        image_slots = [slot for slot, tensor in image_pairs if tensor is not None]
        return (
            _qwen_conditioning_entry(
                prompt,
                images,
                reference_latents=vae is not None,
                conditioning_mode="qwen_image_edit_plus",
                use_picture_prefix=True,
                vl_target_area=QWEN_EDIT_PLUS_VL_AREA,
                reference_target_area=QWEN_REFERENCE_AREA,
                image_slots=image_slots,
                clip=clip,
                vae=vae,
            ),
        )


class SDMLXQwenImageEditPlusConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("mlx_clip",),
                "prompt": (
                    "STRING",
                    {
                        "default": "Change the image while preserving the original composition.",
                        "multiline": True,
                    },
                ),
                "target_size": (
                    "INT",
                    {
                        "default": QWEN_PHR00T_PLUS_REFERENCE_SIZE,
                        "min": 128,
                        "max": 2048,
                        "step": 32,
                    },
                ),
            },
            "optional": {
                "vae": ("mlx_vae",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "SDMLX/Conditioning"

    def encode(self, clip, prompt, target_size, vae=None, image1=None, image2=None, image3=None, image4=None):
        if isinstance(clip, dict) and clip.get("type") not in (None, QWEN_MODEL_FAMILY):
            raise RuntimeError("SDMLX Qwen: connect the mlx_clip output from the SDMLX CLIP Loader.")
        if vae is not None and (not isinstance(vae, dict) or vae.get("type") != QWEN_MODEL_FAMILY):
            raise RuntimeError("SDMLX Qwen: connect a Qwen mlx_vae from the SDMLX VAE Loader.")
        target_size = max(128, min(2048, int(target_size)))
        target_size = max(32, int(round(target_size / 32)) * 32)
        image_pairs = [(1, image1), (2, image2), (3, image3), (4, image4)]
        images = [tensor for _slot, tensor in image_pairs if tensor is not None]
        image_slots = [slot for slot, tensor in image_pairs if tensor is not None]
        return (
            _qwen_conditioning_entry(
                prompt,
                images,
                reference_latents=vae is not None,
                conditioning_mode="qwen_image_edit_plus_phr00t",
                use_picture_prefix=True,
                vl_target_area=QWEN_EDIT_PLUS_VL_AREA,
                reference_target_area=target_size * target_size,
                reference_target_multiple=32,
                prompt_template=QWEN_PHR00T_PLUS_TEMPLATE,
                image_slots=image_slots,
                clip=clip,
                vae=vae,
            ),
        )


class SDMLXQwenTokenBudget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("mlx_conditioning",),
                "token_budget": (QWEN_TOKEN_BUDGET_OPTIONS, {"default": QWEN_TOKEN_BUDGET_OFF}),
                "casting": ("BOOLEAN", {"default": False}),
                "budget_limiter": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "SDMLX/Conditioning"

    def apply(self, conditioning, token_budget, casting=False, budget_limiter=True):
        token_budget = str(token_budget or QWEN_TOKEN_BUDGET_OFF)
        notes = []

        def patch_entry(entry: dict[str, Any]) -> dict[str, Any]:
            updated = dict(entry)
            images = list(updated.get("images") or [])
            image_count = max(1, len(images))

            if token_budget == QWEN_TOKEN_BUDGET_OFF:
                if updated.get("reference_target_area") is None:
                    updated["reference_target_area"] = QWEN_REFERENCE_AREA
                updated["reference_target_areas"] = None
                notes.append(f"token_budget=off, reference_target_area={int(updated['reference_target_area'])} per image")
                return updated

            per_image_area = QWEN_TOKEN_BUDGET_PRESETS.get(token_budget, QWEN_TOKEN_BUDGET_PRESETS["low"])
            per_image_area = max(16 * 16, int(per_image_area))
            total_area = per_image_area * image_count
            if bool(budget_limiter):
                total_area = min(total_area, per_image_area * 2)

            weights = [2] + [1] * (image_count - 1) if bool(casting) and image_count > 1 else None
            target_areas = _qwen_distribute_budget(total_area, image_count, weights=weights)
            updated["reference_target_area"] = None
            updated["reference_target_areas"] = target_areas
            notes.append(
                f"token_budget={token_budget}, reference_target_areas={target_areas}, "
                f"casting={'true' if casting else 'false'}, "
                f"budget_limiter={'true' if budget_limiter else 'false'}"
            )
            return updated

        patched, patched_count = _patch_qwen_conditioning_entries(conditioning, patch_entry)
        if patched_count == 0:
            _qwen_log("SDMLX Qwen Token Budget: no Qwen conditioning entry found", verbose=True)
        else:
            note = "; ".join(dict.fromkeys(notes))
            _qwen_log(f"SDMLX Qwen Token Budget: patched {patched_count} entry, {note}", verbose=True)
        return (patched,)


class SDMLXQwenModelSamplingAuraFlow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdmlx_model": (MODEL_TYPE,),
                "shift": (
                    "FLOAT",
                    {
                        "default": QWEN_AURAFLOW_DEFAULT_SHIFT,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.05,
                    },
                ),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("sdmlx_model",)
    FUNCTION = "apply"
    CATEGORY = "SDMLX/Advanced"

    def apply(self, sdmlx_model, shift):
        if not is_qwen_sdmlx_model(sdmlx_model):
            raise RuntimeError("SDMLX Qwen AuraFlow: connect a Qwen .sdmlx model.")
        model = dict(sdmlx_model)
        model["qwen_scheduler"] = QWEN_AURAFLOW_SCHEDULER
        model["qwen_flow_shift"] = max(0.01, min(100.0, float(shift)))
        return (model,)


NODE_CLASS_MAPPINGS = {
    "SDMLXQwenImageEditConditioning": SDMLXQwenImageEditConditioning,
    "SDMLXQwenImageEditPlusConditioning": SDMLXQwenImageEditPlusConditioning,
    "SDMLXQwenTokenBudget": SDMLXQwenTokenBudget,
    "SDMLXQwenModelSamplingAuraFlow": SDMLXQwenModelSamplingAuraFlow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDMLXQwenImageEditConditioning": "🍏 SDMLX Qwen Image Edit Conditioning",
    "SDMLXQwenImageEditPlusConditioning": "🍏 SDMLX Qwen Image Edit Conditioning Plus",
    "SDMLXQwenTokenBudget": "🍏 SDMLX Qwen Token Budget",
    "SDMLXQwenModelSamplingAuraFlow": "🍏 SDMLX Qwen ModelSampling AuraFlow",
}
