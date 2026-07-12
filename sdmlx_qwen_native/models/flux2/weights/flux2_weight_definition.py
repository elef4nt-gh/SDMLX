from typing import List

import mlx.core as mx

from sdmlx_qwen_native.models.common.config.model_config import ModelConfig
from sdmlx_qwen_native.models.common.tokenizer import LanguageTokenizer
from sdmlx_qwen_native.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from sdmlx_qwen_native.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class Flux2KleinWeightDefinition:
    @staticmethod
    def normalize_vae_key(key: str) -> str | None:
        # FLUX.2 small-decoder VAE files may use the BFL/Comfy layout
        # (encoder.down.0.block.0..., decoder.mid.attn_1..., etc.).
        # Normalize it to the Diffusers-style keys consumed by the existing mapping.
        if key.endswith(".num_batches_tracked"):
            return None

        normalized = key
        normalized = normalized.replace("encoder.quant_conv.", "quant_conv.")
        normalized = normalized.replace("decoder.post_quant_conv.", "post_quant_conv.")
        normalized = normalized.replace("encoder.norm_out.", "encoder.conv_norm_out.")
        normalized = normalized.replace("decoder.norm_out.", "decoder.conv_norm_out.")
        normalized = normalized.replace(".nin_shortcut.", ".conv_shortcut.")

        for prefix in ("encoder", "decoder"):
            normalized = normalized.replace(f"{prefix}.mid.block_1.", f"{prefix}.mid_block.resnets.0.")
            normalized = normalized.replace(f"{prefix}.mid.block_2.", f"{prefix}.mid_block.resnets.1.")
            normalized = normalized.replace(f"{prefix}.mid.attn_1.norm.", f"{prefix}.mid_block.attentions.0.group_norm.")
            normalized = normalized.replace(f"{prefix}.mid.attn_1.q.", f"{prefix}.mid_block.attentions.0.to_q.")
            normalized = normalized.replace(f"{prefix}.mid.attn_1.k.", f"{prefix}.mid_block.attentions.0.to_k.")
            normalized = normalized.replace(f"{prefix}.mid.attn_1.v.", f"{prefix}.mid_block.attentions.0.to_v.")
            normalized = normalized.replace(f"{prefix}.mid.attn_1.proj_out.", f"{prefix}.mid_block.attentions.0.to_out.0.")

            for block in range(4):
                for res in range(3):
                    normalized = normalized.replace(
                        f"{prefix}.down.{block}.block.{res}.",
                        f"{prefix}.down_blocks.{block}.resnets.{res}.",
                    )
                normalized = normalized.replace(
                    f"{prefix}.down.{block}.downsample.conv.",
                    f"{prefix}.down_blocks.{block}.downsamplers.0.conv.",
                )

        for raw_block in range(4):
            model_block = 3 - raw_block
            for res in range(3):
                normalized = normalized.replace(
                    f"decoder.up.{raw_block}.block.{res}.",
                    f"decoder.up_blocks.{model_block}.resnets.{res}.",
                )
            normalized = normalized.replace(
                f"decoder.up.{raw_block}.upsample.conv.",
                f"decoder.up_blocks.{model_block}.upsamplers.0.conv.",
            )

        return normalized

    @staticmethod
    def normalize_vae_weight(key: str, value: mx.array) -> mx.array:
        if (
            ".mid_block.attentions.0." in key
            and key.endswith(".weight")
            and len(value.shape) == 4
            and tuple(value.shape[2:]) == (1, 1)
        ):
            return value.reshape(value.shape[0], value.shape[1])
        return value

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_vae_mapping,
                key_transform=Flux2KleinWeightDefinition.normalize_vae_key,
                weight_transform=Flux2KleinWeightDefinition.normalize_vae_weight,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_text_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_vae_component(
        *,
        hf_subdir: str = "vae",
        weight_files: list[str] | None = None,
    ) -> ComponentDefinition:
        return ComponentDefinition(
            name="vae",
            hf_subdir=hf_subdir,
            precision=ModelConfig.precision,
            mapping_getter=Flux2WeightMapping.get_vae_mapping,
            key_transform=Flux2KleinWeightDefinition.normalize_vae_key,
            weight_transform=Flux2KleinWeightDefinition.normalize_vae_weight,
            weight_files=weight_files,
            skip_quantization=True,
        )

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3",
                hf_subdir="tokenizer",
                tokenizer_class="Qwen2TokenizerFast",
                encoder_class=LanguageTokenizer,
                max_length=512,
                use_chat_template=True,
                chat_template_kwargs={"enable_thinking": False},
                download_patterns=["tokenizer/**", "added_tokens.json", "chat_template.jinja"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/**",
            "added_tokens.json",
            "chat_template.jinja",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")
