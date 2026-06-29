from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Flux2AssetEntry:
    key: str
    repo_id: str
    expected_text_hidden_size: int
    preferred_text_encoders: tuple[str, ...]


_FLUX2_ENTRIES: dict[str, Flux2AssetEntry] = {
    "flux2-klein-4b": Flux2AssetEntry(
        key="flux2-klein-4b",
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        expected_text_hidden_size=2560,
        preferred_text_encoders=(
            "flux2-klein-4b-qwen3-text_encoder",
            "qwen_3_4b_fp8_mixed.safetensors",
            "qwen_3_4b.safetensors",
        ),
    ),
    "flux2-klein-base-4b": Flux2AssetEntry(
        key="flux2-klein-base-4b",
        repo_id="black-forest-labs/FLUX.2-klein-base-4B",
        expected_text_hidden_size=2560,
        preferred_text_encoders=(
            "flux2-klein-4b-qwen3-text_encoder",
            "qwen_3_4b_fp8_mixed.safetensors",
            "qwen_3_4b.safetensors",
        ),
    ),
    "flux2-klein-9b": Flux2AssetEntry(
        key="flux2-klein-9b",
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        expected_text_hidden_size=4096,
        preferred_text_encoders=(
            "flux2-klein-9b-qwen3-text_encoder",
            "qwen_3_8b_fp8mixed.safetensors",
        ),
    ),
    "flux2-klein-base-9b": Flux2AssetEntry(
        key="flux2-klein-base-9b",
        repo_id="black-forest-labs/FLUX.2-klein-base-9B",
        expected_text_hidden_size=4096,
        preferred_text_encoders=(
            "flux2-klein-9b-qwen3-text_encoder",
            "qwen_3_8b_fp8mixed.safetensors",
        ),
    ),
    "flux2-klein-9b-kv": Flux2AssetEntry(
        key="flux2-klein-9b-kv",
        repo_id="black-forest-labs/FLUX.2-klein-9B-KV",
        expected_text_hidden_size=4096,
        preferred_text_encoders=(
            "flux2-klein-9b-qwen3-text_encoder",
            "qwen_3_8b_fp8mixed.safetensors",
        ),
    ),
}


def flux2_registry_key(config_name: str) -> str:
    name = str(config_name or "").strip().lower()
    name = name.rsplit("/", 1)[-1]
    name = name.replace("flux.2", "flux2")
    return name.replace("_", "-")


def flux2_asset_entry(config_name: str) -> Flux2AssetEntry:
    key = flux2_registry_key(config_name)
    return _FLUX2_ENTRIES.get(key) or _FLUX2_ENTRIES["flux2-klein-4b"]


def _folder_paths_module():
    import folder_paths  # type: ignore

    return folder_paths


def sdmlx_model_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        folder_paths = _folder_paths_module()
        folder_map = getattr(folder_paths, "folder_names_and_paths", {})
        for key in ("sdmlx", "SDMLX"):
            if key in folder_map:
                roots.extend(Path(path).expanduser() for path in folder_paths.get_folder_paths(key))
        models_dir = Path(getattr(folder_paths, "models_dir", Path.cwd() / "models"))
        roots.append(models_dir / "SDMLX")
        roots.append(models_dir / "sdmlx")
    except Exception:
        pass

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve()).lower()
        except Exception:
            key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _flux2_text_hidden_size(path: Path) -> int | None:
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    try:
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                hidden_size = int((json.load(handle) or {}).get("hidden_size") or 0)
            return hidden_size or None
    except Exception:
        return None
    return None


def _flux2_tokenizer_has_chat_template(path: Path) -> bool:
    if not path.exists():
        return False
    if (path / "chat_template.jinja").is_file():
        return True
    config_path = path / "tokenizer_config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle) or {}
    except Exception:
        return False
    return bool(config.get("chat_template"))


def _is_flux2_component_root(root: Path, entry: Flux2AssetEntry) -> bool:
    if not root.is_dir():
        return False
    if not all((root / name).exists() for name in ("text_encoder", "tokenizer", "vae")):
        return False
    if not _flux2_tokenizer_has_chat_template(root / "tokenizer"):
        return False
    hidden_size = _flux2_text_hidden_size(root / "text_encoder")
    return hidden_size is None or hidden_size == entry.expected_text_hidden_size


def _candidate_component_roots() -> list[Path]:
    roots: list[Path] = []
    for sdmlx_root in sdmlx_model_roots():
        if sdmlx_root.name.endswith(".sdmlx") and sdmlx_root.is_dir():
            roots.append(sdmlx_root)
        if sdmlx_root.exists():
            for child in sdmlx_root.rglob("*.sdmlx"):
                if child.is_dir():
                    roots.append(child)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve()).lower()
        except Exception:
            key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _same_path(left: Path, right: Path | None) -> bool:
    if right is None:
        return False
    try:
        return left.resolve() == right.resolve()
    except Exception:
        return left == right


def find_flux2_component_root(config_name: str, *, exclude: Path | None = None) -> Path | None:
    entry = flux2_asset_entry(config_name)
    for root in _candidate_component_roots():
        if _same_path(root, exclude):
            continue
        if _is_flux2_component_root(root, entry):
            return root
    return None


def ensure_flux2_component_root(config_name: str, *, exclude: Path | None = None, download: bool = True) -> Path | None:
    return find_flux2_component_root(config_name, exclude=exclude)


def resolve_registered_model_file(folder_name: str, relative_name: str) -> Path | None:
    try:
        folder_paths = _folder_paths_module()
        path = folder_paths.get_full_path(folder_name, relative_name)
        if path and os.path.exists(path):
            return Path(path).expanduser().resolve()
    except Exception:
        pass
    return None
