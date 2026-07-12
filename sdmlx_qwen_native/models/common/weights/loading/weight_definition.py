from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, TypeAlias

import mlx.core as mx

from sdmlx_qwen_native.models.common.weights.mapping.weight_mapping import WeightTarget

if TYPE_CHECKING:
    from sdmlx_qwen_native.models.common.tokenizer.tokenizer import BaseTokenizer
    from sdmlx_qwen_native.models.depth_pro.weights.depth_pro_weight_definition import DepthProWeightDefinition
    from sdmlx_qwen_native.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition
    from sdmlx_qwen_native.models.fibo_vlm.weights.fibo_vlm_weight_definition import FIBOVLMWeightDefinition
    from sdmlx_qwen_native.models.flux.weights.flux_weight_definition import FluxWeightDefinition
    from sdmlx_qwen_native.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
    from sdmlx_qwen_native.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
    from sdmlx_qwen_native.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition
    from sdmlx_qwen_native.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

    WeightDefinitionType: TypeAlias = type[
        FluxWeightDefinition
        | FIBOWeightDefinition
        | FIBOVLMWeightDefinition
        | QwenWeightDefinition
        | ZImageWeightDefinition
        | SeedVR2WeightDefinition
        | DepthProWeightDefinition
        | Ideogram4WeightDefinition
    ]


@dataclass
class ComponentDefinition:
    name: str
    hf_subdir: str
    mapping_getter: Callable[[], List[WeightTarget]] | None = None
    model_attr: str | None = None
    num_blocks: int | None = None
    num_layers: int | None = None
    loading_mode: str = "mlx_native"
    precision: mx.Dtype | None = None
    skip_quantization: bool = False
    bulk_transform: Callable[[mx.array], mx.array] | None = None
    weight_subkey: str | None = None
    download_url: str | None = None
    weight_prefix_filters: List[str] | None = None
    weight_files: List[str] | None = None  # Specific files to load (if None, loads all *.safetensors)
    key_transform: Callable[[str], str | None] | None = None
    weight_transform: Callable[[str, mx.array], mx.array] | None = None


@dataclass
class TokenizerDefinition:
    name: str
    hf_subdir: str
    tokenizer_class: str = "AutoTokenizer"
    fallback_subdirs: List[str] | None = None
    download_patterns: List[str] | None = None
    encoder_class: type["BaseTokenizer"] | None = None
    max_length: int = 512
    padding: str = "max_length"
    template: str | None = None
    use_chat_template: bool = False
    chat_template_kwargs: dict | None = field(default_factory=dict)
    add_special_tokens: bool = True
    processor_class: type | None = None
    image_token: str = "<|image_pad|>"
    chat_template: str | None = None  # Jinja2 template for apply_chat_template
