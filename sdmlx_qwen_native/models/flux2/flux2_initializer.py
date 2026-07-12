from pathlib import Path

from sdmlx_qwen_native.callbacks.callback_registry import CallbackRegistry
from sdmlx_qwen_native.models.common.config import ModelConfig
from sdmlx_qwen_native.models.common.lora.mapping.lora_loader import LoRALoader
from sdmlx_qwen_native.models.common.tokenizer import TokenizerLoader

from sdmlx_qwen_native.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from sdmlx_qwen_native.models.common.weights.loading.weight_applier import WeightApplier
from sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader
from sdmlx_qwen_native.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from sdmlx_qwen_native.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from sdmlx_qwen_native.models.flux2.model.flux2_vae.vae import Flux2VAE
from sdmlx_qwen_native.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping
from sdmlx_qwen_native.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition


class Flux2Initializer:
    SMALL_DECODER_CHANNELS = (96, 192, 384, 384)

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        vae_variant: str = "standard",
        vae_path: str | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        Flux2Initializer._init_config(model, model_config)
        weights = Flux2Initializer._load_weights(path)
        Flux2Initializer._init_tokenizers(model, path)
        Flux2Initializer._init_models(model, vae_variant=vae_variant)
        if vae_path or Flux2Initializer._is_small_decoder(vae_variant):
            weights.components.pop("vae", None)
        Flux2Initializer._apply_weights(model, weights, quantize)
        if vae_path:
            Flux2Initializer._apply_vae_from_file(model, vae_path)
        Flux2Initializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Flux2KleinWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Flux2KleinWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model, vae_variant: str = "standard") -> None:
        decoder_channels = (
            Flux2Initializer.SMALL_DECODER_CHANNELS if Flux2Initializer._is_small_decoder(vae_variant) else None
        )
        model.vae = Flux2VAE(decoder_block_out_channels=decoder_channels)
        model.transformer = Flux2Transformer(**model.model_config.transformer_overrides)
        model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Flux2KleinWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Flux2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            role="flux2",
        )

    @staticmethod
    def _is_small_decoder(vae_variant: str | None) -> bool:
        return str(vae_variant or "standard").strip().lower().replace("-", "_") in {"small_decoder", "small"}

    @staticmethod
    def _load_vae_from_file(path: str) -> LoadedWeights:
        vae_path = Path(path).expanduser()
        component = Flux2KleinWeightDefinition.get_vae_component(hf_subdir=".", weight_files=[vae_path.name])
        weights, _q_level, _version = WeightLoader._load_component(vae_path.parent, component)
        return LoadedWeights(components={"vae": weights}, meta_data=MetaData())

    @staticmethod
    def _apply_vae_from_file(model, vae_path: str | None) -> None:
        if not vae_path:
            raise ValueError("FLUX.2 VAE file requested without a vae_path")
        weights = Flux2Initializer._load_vae_from_file(vae_path)
        component = Flux2KleinWeightDefinition.get_vae_component(hf_subdir=".", weight_files=[Path(vae_path).name])
        WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=model.vae,
            component=component,
            quantize_arg=None,
        )
