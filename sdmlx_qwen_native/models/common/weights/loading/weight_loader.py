import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten
from safetensors.torch import load_file as torch_load_file
from safetensors.torch import safe_open as torch_safe_open

from sdmlx_qwen_native.cli.defaults.defaults import MFLUX_CACHE_DIR
from sdmlx_qwen_native.models.common.resolution.path_resolution import PathResolution
from sdmlx_qwen_native.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from sdmlx_qwen_native.models.common.weights.loading.safetensors_reader import SafetensorsReader
from sdmlx_qwen_native.models.common.weights.loading.weight_definition import ComponentDefinition
from sdmlx_qwen_native.models.common.weights.mapping.weight_mapper import WeightMapper

if TYPE_CHECKING:
    from sdmlx_qwen_native.models.common.weights.loading.weight_definition import WeightDefinitionType

logger = logging.getLogger(__name__)


class WeightLoader:
    @staticmethod
    def load_single(
        component: ComponentDefinition,
        repo_id: str,
        file_pattern: str = "*.safetensors",
    ) -> LoadedWeights:
        root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=[file_pattern, "config.json"]))
        weights, q_level, version = WeightLoader._load_component(root_path, component)
        return LoadedWeights(
            components={component.name: weights},
            meta_data=MetaData(quantization_level=q_level, mflux_version=version),
        )

    @staticmethod
    def load(
        weight_definition: "WeightDefinitionType",
        model_path: str | None = None,
    ) -> LoadedWeights:
        root_path = PathResolution.resolve(
            path=model_path,
            patterns=weight_definition.get_download_patterns(),
        )

        # 2. Load each component (with caching for shared sources)
        components = {}
        quantization_level = None
        mflux_version = None
        raw_weights_cache: dict[tuple, dict] = {}  # Cache by (path, loading_mode, weight_files)

        for component in weight_definition.get_components():
            weights, q_level, version = WeightLoader._load_component(root_path, component, raw_weights_cache)
            components[component.name] = weights

            # Track metadata from first component that has it
            if quantization_level is None and q_level is not None:
                quantization_level = q_level
                mflux_version = version

        return LoadedWeights(
            components=components,
            meta_data=MetaData(
                quantization_level=quantization_level,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def _load_component(
        root_path: Path | None,
        component: ComponentDefinition,
        raw_weights_cache: dict[tuple, dict] | None = None,
    ) -> tuple[dict, int | None, str | None]:
        # Handle direct URL downloads (e.g., Apple CDN for DepthPro)
        if component.download_url is not None:
            file_path = WeightLoader._download_from_url(component.download_url, component.name)
            raw_weights = WeightLoader._load_weights_file(file_path, component.loading_mode)
        else:
            if root_path is None:
                raise ValueError(f"No root_path and no download_url for component: {component.name}")
            component_path = root_path / component.hf_subdir

            # Try sdmlx_qwen_native saved format first (including FP8 components reloaded after sdmlx_qwen_native-save).
            weights, q_level, version = WeightLoader._try_load_mflux_format(component_path)
            if weights is not None:
                return weights, q_level, version

            # Check cache for shared loading (e.g., FIBO VLM decoder + visual from same source)
            prefix_key = tuple(component.weight_prefix_filters or [])
            cache_key = (str(component_path), component.loading_mode, tuple(component.weight_files or []), prefix_key)
            if raw_weights_cache is not None and cache_key in raw_weights_cache:
                raw_weights = raw_weights_cache[cache_key]
            else:
                # Fall back to HuggingFace format with mapping
                raw_weights = WeightLoader._load_safetensors(
                    component_path,
                    component.loading_mode,
                    component.weight_files,
                    component.weight_prefix_filters,
                )
                # Cache for potential reuse by other components
                if raw_weights_cache is not None:
                    raw_weights_cache[cache_key] = raw_weights

        # Apply prefix filtering if specified (e.g., filter "model.language_model" vs "model.visual")
        if component.weight_prefix_filters is not None:
            raw_weights = {
                k: v
                for k, v in raw_weights.items()
                if any(k.startswith(prefix) for prefix in component.weight_prefix_filters)
            }

        if component.key_transform is not None:
            transformed_weights = {}
            for key, value in raw_weights.items():
                transformed_key = component.key_transform(key)
                if transformed_key is not None:
                    transformed_weights[transformed_key] = value
            raw_weights = transformed_weights

        if component.weight_transform is not None:
            raw_weights = {k: component.weight_transform(k, v) for k, v in raw_weights.items()}

        # Apply precision conversion if specified
        if component.precision is not None:
            raw_weights = WeightLoader._convert_precision(raw_weights, component.precision)

        # Passthrough mode: apply bulk transform and unflatten (no key mapping)
        if component.mapping_getter is None:
            if component.bulk_transform is not None:
                raw_weights = {k: component.bulk_transform(v) for k, v in raw_weights.items()}
            return tree_unflatten(list(raw_weights.items())), None, None

        # Standard mode: apply declarative weight mapping
        mapped_weights = WeightMapper.apply_mapping(
            hf_weights=raw_weights,
            mapping=component.mapping_getter(),
            num_blocks=component.num_blocks,
            num_layers=component.num_layers,
        )
        return mapped_weights, None, None

    @staticmethod
    def _try_load_mflux_format(path: Path) -> tuple[dict | None, int | None, str | None]:
        t_total = time.perf_counter()
        cache_debug = WeightLoader._env_flag("SDMLX_QWEN_CACHE_DEBUG") or WeightLoader._env_flag("SDMLX_QWEN_CACHE_TIMINGS")
        if not path.exists():
            return None, None, None

        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            return None, None, None

        # Check metadata on first file without loading the full tensor payload.
        t_metadata = time.perf_counter()
        metadata = WeightLoader._read_safetensors_metadata(shard_files[0])
        metadata_seconds = time.perf_counter() - t_metadata
        quantization_level_str = metadata.get("quantization_level")
        mflux_version = metadata.get("mflux_version")

        # If no sdmlx_qwen_native metadata, this isn't our format
        if quantization_level_str is None and mflux_version is None:
            return None, None, None

        # Convert quantization level from string to int
        if quantization_level_str in (None, "None", "null", ""):
            quantization_level = None
        else:
            quantization_level = int(quantization_level_str)

        # Load all shards
        all_weights: dict[str, mx.array] = {}
        shard_timings = []
        for shard in shard_files:
            t_shard = time.perf_counter()
            shard_data = mx.load(str(shard), return_metadata=True)
            shard_weights = dict(shard_data[0].items())
            all_weights.update(shard_weights)
            shard_timings.append((shard.name, len(shard_weights), time.perf_counter() - t_shard))

        t_unflatten = time.perf_counter()
        unflattened = tree_unflatten(list(all_weights.items()))
        unflatten_seconds = time.perf_counter() - t_unflatten
        total_seconds = time.perf_counter() - t_total
        if cache_debug:
            shard_total = sum(seconds for _name, _count, seconds in shard_timings)
            slowest = sorted(shard_timings, key=lambda item: item[2], reverse=True)[:3]
            slowest_text = ", ".join(f"{name}:{seconds:.2f}s/{count}" for name, count, seconds in slowest)
            print(
                "SDMLX Qwen cache timings: "
                f"component={path.name}, shards={len(shard_files)}, tensors={len(all_weights)}, "
                f"metadata={metadata_seconds:.2f}s, shard_load={shard_total:.2f}s, "
                f"unflatten={unflatten_seconds:.2f}s, total={total_seconds:.2f}s"
                + (f", slowest=[{slowest_text}]" if slowest_text else "")
            )
        return unflattened, quantization_level, mflux_version

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _download_from_url(url: str, component_name: str) -> Path:
        cache_dir = MFLUX_CACHE_DIR / component_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = url.split("/")[-1]
        file_path = cache_dir / filename

        if not file_path.exists():
            logger.info(f"Downloading {component_name} weights from {url}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                logger.info(f"Downloaded to {file_path}")
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.error(f"Failed to download: {e}")
                logger.info(f"Please manually download from: {url}")
                raise FileNotFoundError(f"Model file not found at {file_path}") from e

        return file_path

    @staticmethod
    def _load_weights_file(file_path: Path, loading_mode: str) -> dict[str, mx.array]:
        if loading_mode == "torch_checkpoint":
            return WeightLoader._load_torch_checkpoint(file_path)
        elif loading_mode in ("mlx_native", "single"):
            data = mx.load(str(file_path), return_metadata=True)
            return dict(data[0].items())
        else:
            raise ValueError(f"Unsupported loading mode for single file: {loading_mode}")

    @staticmethod
    def _load_torch_checkpoint(file_path: Path) -> dict[str, mx.array]:
        pt_weights = torch.load(file_path, map_location="cpu", weights_only=False)
        return {k: mx.array(v.numpy()) for k, v in pt_weights.items() if isinstance(v, torch.Tensor)}

    @staticmethod
    def _load_safetensors(
        path: Path,
        loading_mode: str,
        weight_files: list[str] | None = None,
        weight_prefix_filters: list[str] | None = None,
    ) -> dict[str, mx.array]:
        if loading_mode == "mlx_native":
            return WeightLoader._load_mlx_native(path, weight_files)
        elif loading_mode == "torch_convert":
            return WeightLoader._load_torch_convert(path, weight_files)
        elif loading_mode == "multi_json":
            return WeightLoader._load_multi_json(path, weight_prefix_filters)
        elif loading_mode == "torch_bfloat16":
            return WeightLoader._load_torch_bfloat16(path)
        elif loading_mode == "single":
            return WeightLoader._load_single(path, weight_prefix_filters)
        elif loading_mode == "single_mlx":
            return WeightLoader._load_single_mlx(path, weight_prefix_filters)
        elif loading_mode == "multi_glob":
            return WeightLoader._load_multi_glob(path, weight_prefix_filters)
        elif loading_mode == "fp8_safetensors":
            return WeightLoader._load_fp8_safetensors(path)
        else:
            raise ValueError(f"Unknown loading mode: {loading_mode}")

    @staticmethod
    def _load_mlx_native(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            weights = mx.load(str(shard))
            all_weights.update(weights)

        return all_weights

    @staticmethod
    def _load_torch_convert(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            torch_weights = torch_load_file(str(shard))
            for key, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                all_weights[key] = mx.array(tensor.numpy())

        return all_weights

    @staticmethod
    def _load_multi_json(path: Path, weight_prefix_filters: list[str] | None = None) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Group weights by file
        files_to_load: dict[str, list[str]] = {}
        for param_name, file_name in index["weight_map"].items():
            if not WeightLoader._matches_prefix_filters(param_name, weight_prefix_filters):
                continue
            if file_name not in files_to_load:
                files_to_load[file_name] = []
            files_to_load[file_name].append(param_name)

        all_weights: dict[str, mx.array] = {}
        for file_name, param_names in files_to_load.items():
            file_path = path / file_name
            if WeightLoader._has_float8_safetensors(file_path):
                all_weights.update(WeightLoader._load_fp8_as_mlx_q8(file_path, keys=param_names))
            else:
                all_weights.update(WeightLoader._load_selected_safetensors(file_path, keys=param_names))

        return all_weights

    @staticmethod
    def _load_torch_bfloat16(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        weight_files = sorted(set(index["weight_map"].values()))

        all_weights: dict[str, mx.array] = {}
        for wf in weight_files:
            file_path = path / wf
            data = torch_load_file(str(file_path))
            for k, v in data.items():
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float16)
                np_arr = v.detach().cpu().numpy()
                all_weights[k] = mx.array(np_arr)

        return all_weights

    @staticmethod
    def _load_single(path: Path, weight_prefix_filters: list[str] | None = None) -> dict[str, mx.array]:
        safetensors_files = [f for f in path.glob("*.safetensors") if not f.name.startswith("._")]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        weights_file = safetensors_files[0]
        if weight_prefix_filters is not None:
            return WeightLoader._load_selected_safetensors(weights_file, prefix_filters=weight_prefix_filters)
        data = mx.load(str(weights_file), return_metadata=True)
        return dict(data[0].items())

    @staticmethod
    def _load_single_mlx(path: Path, weight_prefix_filters: list[str] | None = None) -> dict[str, mx.array]:
        safetensors_files = [f for f in path.glob("*.safetensors") if not f.name.startswith("._")]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")
        return SafetensorsReader.read_file(
            safetensors_files[0],
            prefix_filters=weight_prefix_filters,
        )

    @staticmethod
    def _load_multi_glob(path: Path, weight_prefix_filters: list[str] | None = None) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            if WeightLoader._has_float8_safetensors(shard):
                print(f"SDMLX Qwen: FP8 weights detected; repacking as MLX Q8 for {shard.name}")
                all_weights.update(WeightLoader._load_fp8_as_mlx_q8(shard, prefix_filters=weight_prefix_filters))
                continue

            if weight_prefix_filters is not None:
                all_weights.update(WeightLoader._load_selected_safetensors(shard, prefix_filters=weight_prefix_filters))
                continue

            data, _ = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(data.items()))

        return all_weights

    @staticmethod
    def _load_fp8_safetensors(path: Path) -> dict[str, mx.array]:
        return SafetensorsReader.read_directory(path)

    @staticmethod
    def _is_comfy_fp8_safetensors(path: Path) -> bool:
        header = WeightLoader._read_safetensors_header(path)
        if header is None:
            return False

        has_fp8 = False
        has_weight_scale = False
        has_comfy_quant = False
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(".comfy_quant"):
                has_comfy_quant = True
            if key.endswith((".weight_scale", ".scale_weight")):
                has_weight_scale = True
            if isinstance(value, dict) and str(value.get("dtype", "")).startswith("F8_"):
                has_fp8 = True

        return has_comfy_quant or (has_fp8 and has_weight_scale)

    @staticmethod
    def _has_float8_safetensors(path: Path) -> bool:
        header = WeightLoader._read_safetensors_header(path)
        if header is None:
            return False
        return any(
            key != "__metadata__"
            and isinstance(value, dict)
            and str(value.get("dtype", "")).startswith("F8_")
            for key, value in header.items()
        )

    @staticmethod
    def _read_safetensors_header(path: Path) -> dict | None:
        try:
            with path.open("rb") as f:
                header_len = int.from_bytes(f.read(8), "little")
                return json.loads(f.read(header_len))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _read_safetensors_metadata(path: Path) -> dict:
        header = WeightLoader._read_safetensors_header(path)
        if header is None:
            return {}
        metadata = header.get("__metadata__", {})
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _matches_prefix_filters(key: str, prefix_filters: list[str] | None) -> bool:
        return prefix_filters is None or any(key.startswith(prefix) for prefix in prefix_filters)

    @staticmethod
    def _load_selected_safetensors(
        path: Path,
        *,
        keys: list[str] | None = None,
        prefix_filters: list[str] | None = None,
    ) -> dict[str, mx.array]:
        selected = set(keys) if keys is not None else None
        all_weights: dict[str, mx.array] = {}
        with torch_safe_open(str(path), framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if selected is not None and key not in selected:
                    continue
                if not WeightLoader._matches_prefix_filters(key, prefix_filters):
                    continue
                tensor = weights.get_tensor(key)
                if WeightLoader._is_torch_float8(tensor):
                    tensor = tensor.float().to(torch.float16)
                elif tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                all_weights[key] = mx.array(tensor.detach().cpu().numpy())
        return all_weights

    @staticmethod
    def _load_fp8_as_mlx_q8(
        path: Path,
        *,
        keys: list[str] | None = None,
        prefix_filters: list[str] | None = None,
    ) -> dict[str, mx.array]:
        all_weights: dict[str, mx.array] = {}
        skipped_suffixes = (".weight_scale", ".scale_weight", ".input_scale", ".comfy_quant")
        selected = set(keys) if keys is not None else None

        with torch_safe_open(str(path), framework="pt", device="cpu") as weights:
            keys = list(weights.keys())
            key_set = set(keys)
            for key in keys:
                if key.endswith(skipped_suffixes):
                    continue
                if selected is not None and key not in selected:
                    continue
                if not WeightLoader._matches_prefix_filters(key, prefix_filters):
                    continue

                tensor = weights.get_tensor(key)
                if WeightLoader._is_torch_float8(tensor):
                    tensor = tensor.float()
                    can_quantize = (
                        key.endswith(".weight")
                        and tensor.ndim == 2
                        and tensor.shape[-1] % 64 == 0
                    )
                    if can_quantize:
                        base_key = key[: -len(".weight")]
                        for scale_key in (f"{base_key}.weight_scale", f"{base_key}.scale_weight"):
                            if scale_key in key_set:
                                tensor = tensor * weights.get_tensor(scale_key).float()
                                break
                    tensor = tensor.to(torch.float16)

                    weight = mx.array(tensor.detach().cpu().numpy())
                    if can_quantize:
                        qweight, scales, *biases = mx.quantize(weight, group_size=64, bits=8)
                        values = [qweight, scales, *biases]
                        mx.eval(*values)
                        all_weights[key] = qweight
                        all_weights[f"{key[:-len('.weight')]}.scales"] = scales
                        if biases:
                            all_weights[f"{key[:-len('.weight')]}.biases"] = biases[0]
                    else:
                        all_weights[key] = weight
                    continue
                elif tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)

                all_weights[key] = mx.array(tensor.detach().cpu().numpy())

        return all_weights

    @staticmethod
    def _is_torch_float8(tensor: torch.Tensor) -> bool:
        return "float8" in str(tensor.dtype)

    @staticmethod
    def _convert_precision(weights: dict[str, mx.array], precision: mx.Dtype) -> dict[str, mx.array]:
        quantized_dtypes = {mx.uint32, mx.uint16, mx.uint8}
        return {
            k: v if v.dtype == precision or v.dtype in quantized_dtypes else v.astype(precision)
            for k, v in weights.items()
        }
