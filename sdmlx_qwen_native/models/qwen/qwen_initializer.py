import os
import re

import mlx.core as mx
from mlx import nn

from sdmlx_qwen_native.callbacks.callback_registry import CallbackRegistry
from sdmlx_qwen_native.models.common.config import ModelConfig
from sdmlx_qwen_native.models.common.lora.mapping.lora_loader import LoRALoader
from sdmlx_qwen_native.models.common.tokenizer import TokenizerLoader
from sdmlx_qwen_native.models.common.weights.loading.loaded_weights import LoadedWeights
from sdmlx_qwen_native.models.common.weights.loading.weight_applier import WeightApplier
from sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader
from sdmlx_qwen_native.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from sdmlx_qwen_native.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
from sdmlx_qwen_native.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer
from sdmlx_qwen_native.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from sdmlx_qwen_native.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from sdmlx_qwen_native.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor
from sdmlx_qwen_native.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer
from sdmlx_qwen_native.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping
from sdmlx_qwen_native.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition


QWEN_FP16_QUANT_PARAMS_DEFAULT_REGEX = r"transformer\.transformer_blocks\.[0-9]+\.(attn|img_ff|txt_ff)\."
QWEN_FP16_QUANT_PARAMS_QWEN_IMAGE_REGEX = r"transformer\.transformer_blocks\.[0-9]+\.(img_ff|txt_ff)\."


def _qwen_verbose_logs_enabled() -> bool:
    value = str(os.environ.get("SDMLX_QWEN_VERBOSE") or os.environ.get("SDMLX_QWEN_DEBUG") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _qwen_log(message: str) -> None:
    if _qwen_verbose_logs_enabled():
        print(message)


class QwenImageInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_mod_scales: list[float] | None = None,
        mod_lora_scale: float = 0.0,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        QwenImageInitializer._init_config(model, model_config)
        weights = QwenImageInitializer._load_weights(path)
        QwenImageInitializer._init_tokenizers(model, path)
        QwenImageInitializer._init_models(model)
        QwenImageInitializer._apply_weights(model, weights, quantize)
        QwenImageInitializer._apply_lora(
            model,
            lora_paths,
            lora_scales,
            lora_mod_scales=lora_mod_scales,
            mod_lora_scale=mod_lora_scale,
        )
        QwenImageInitializer._apply_drawthings_folding(model)
        QwenImageInitializer._apply_fp16_quant_params(model)

    @staticmethod
    def init_edit(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_mod_scales: list[float] | None = None,
        mod_lora_scale: float = 0.0,
    ) -> None:
        # Use model_path if provided, otherwise fall back to model_config.model_name
        path = model_path if model_path else model_config.model_name
        QwenImageInitializer._init_config(model, model_config)
        weights = QwenImageInitializer._load_weights(path)
        QwenImageInitializer._init_tokenizers(model, path)
        QwenImageInitializer._init_edit_models(model)
        QwenImageInitializer._apply_weights(model, weights, quantize)
        QwenImageInitializer._apply_lora(
            model,
            lora_paths,
            lora_scales,
            lora_mod_scales=lora_mod_scales,
            mod_lora_scale=mod_lora_scale,
        )
        QwenImageInitializer._apply_drawthings_folding(model)
        QwenImageInitializer._apply_fp16_quant_params(model)

        # Add vision-language tokenizer
        raw_tokenizer = model.tokenizers["qwen"].tokenizer
        processor = QwenVisionLanguageProcessor(tokenizer=raw_tokenizer)
        model.tokenizers["qwen_vl"] = QwenVisionLanguageTokenizer(
            processor=processor,
            max_length=1024,
            use_picture_prefix=True,
        )
        model.qwen_vl_encoder = QwenVisionLanguageEncoder(encoder=model.text_encoder.encoder)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=QwenWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=QwenWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = QwenVAE()
        model.transformer = QwenTransformer(activation_scaling_profile="qwen_image")
        model.text_encoder = QwenTextEncoder()

    @staticmethod
    def _init_edit_models(model) -> None:
        model.vae = QwenVAE()
        model.transformer = QwenTransformer(activation_scaling_profile="qwen_image_edit")
        model.text_encoder = QwenTextEncoder()
        model.text_encoder.encoder.visual = VisionTransformer()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=QwenWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        *,
        lora_mod_scales: list[float] | None = None,
        mod_lora_scale: float = 0.0,
    ) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=QwenLoRAMapping.get_stable_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            role="stable",
        )
        if lora_paths:
            base_scales = lora_scales or [1.0] * len(lora_paths)
            if lora_mod_scales is None:
                requested_mod_scales = [float(mod_lora_scale)] * len(lora_paths)
            else:
                requested_mod_scales = [float(scale) for scale in lora_mod_scales[: len(lora_paths)]]
                requested_mod_scales.extend([0.0] * (len(lora_paths) - len(requested_mod_scales)))

            mod_paths = []
            mod_scales = []
            for path, base_scale, requested_mod_scale in zip(lora_paths, base_scales, requested_mod_scales):
                if abs(requested_mod_scale) <= 1e-8:
                    continue
                mod_paths.append(path)
                mod_scales.append(float(base_scale) * float(requested_mod_scale))
        else:
            mod_paths = []
            mod_scales = []

        if mod_paths:
            summary = ", ".join(
                f"{path.split('/')[-1]}@{scale:.4f}" for path, scale in zip(mod_paths, mod_scales)
            )
            _qwen_log(f"SDMLX Qwen: applying modulation LoRA scale(s): {summary}")
            LoRALoader.load_and_apply_lora(
                lora_mapping=QwenLoRAMapping.get_modulation_mapping(),
                transformer=model.transformer,
                lora_paths=mod_paths,
                lora_scales=mod_scales,
                role="modulation",
            )

    @staticmethod
    def _apply_drawthings_folding(model) -> None:
        transformer = getattr(model, "transformer", None)
        if transformer is not None and hasattr(transformer, "apply_drawthings_folded_weight_scaling"):
            transformer.apply_drawthings_folded_weight_scaling()

    @staticmethod
    def _apply_fp16_quant_params(model) -> None:
        if not QwenImageInitializer._env_enabled("SDMLX_QWEN_FP16_QUANT_PARAMS"):
            return

        regex = str(os.environ.get("SDMLX_QWEN_FP16_QUANT_REGEX", "")).strip() or QwenImageInitializer._default_fp16_quant_regex(model)
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
            _qwen_log(f"SDMLX Qwen: fp16 quantized-linear params active ({converted} layers{suffix})")

    @staticmethod
    def _default_fp16_quant_regex(model) -> str:
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

    @staticmethod
    def _env_enabled(name: str) -> bool:
        value = str(os.environ.get(name, "")).strip().lower()
        if value:
            return value in {"1", "true", "on", "yes"}
        return True
