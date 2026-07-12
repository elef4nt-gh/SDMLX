from typing import List

from sdmlx_qwen_native.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class QwenLoRAMapping(LoRAMapping):
    @staticmethod
    def get_mapping() -> List[LoRATarget]:
        return QwenLoRAMapping.get_stable_mapping()

    @staticmethod
    def get_stable_mapping() -> List[LoRATarget]:
        targets = [
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=[
                    "transformer_blocks.{block}.attn.to_q.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora.up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_q.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_q.lora_B.weight",
                    "transformer_blocks.{block}.attn.to_q.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_q.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.attn.to_q.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora.down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_q.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_q.lora_A.weight",
                    "transformer_blocks.{block}.attn.to_q.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_q.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.attn.to_q.alpha",
                    "transformer.transformer_blocks.{block}.attn.to_q.alpha",
                    "lora_unet_transformer_blocks_{block}_attn_to_q.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=[
                    "transformer_blocks.{block}.attn.to_k.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora.up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_k.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_k.lora_B.weight",
                    "transformer_blocks.{block}.attn.to_k.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_k.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.attn.to_k.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora.down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_k.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_k.lora_A.weight",
                    "transformer_blocks.{block}.attn.to_k.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_k.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.attn.to_k.alpha",
                    "transformer.transformer_blocks.{block}.attn.to_k.alpha",
                    "lora_unet_transformer_blocks_{block}_attn_to_k.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=[
                    "transformer_blocks.{block}.attn.to_v.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora.up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_v.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_v.lora_B.weight",
                    "transformer_blocks.{block}.attn.to_v.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_v.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.attn.to_v.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora.down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_v.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_v.lora_A.weight",
                    "transformer_blocks.{block}.attn.to_v.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_v.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.attn.to_v.alpha",
                    "transformer.transformer_blocks.{block}.attn.to_v.alpha",
                    "lora_unet_transformer_blocks_{block}_attn_to_v.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.attn_to_out.0",
                possible_up_patterns=[
                    "transformer_blocks.{block}.attn.to_out.0.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora.up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_out.0.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_out.0.lora_B.weight",
                    "transformer_blocks.{block}.attn.to_out.0.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_out_0.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.attn.to_out.0.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora.down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_out.0.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_out.0.lora_A.weight",
                    "transformer_blocks.{block}.attn.to_out.0.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_out_0.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.attn.to_out.0.alpha",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.alpha",
                    "lora_unet_transformer_blocks_{block}_attn_to_out_0.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_q_proj",
                possible_up_patterns=[
                    "transformer_blocks.{block}.attn.add_q_proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_q_proj.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_q_proj.lora_B.weight",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_add_q_proj.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.attn.add_q_proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_q_proj.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_q_proj.lora_A.weight",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_add_q_proj.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.attn.add_q_proj.alpha",
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.alpha",
                    "lora_unet_transformer_blocks_{block}_attn_add_q_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_k_proj",
                possible_up_patterns=[
                    "transformer_blocks.{block}.attn.add_k_proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_k_proj.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_k_proj.lora_B.weight",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_add_k_proj.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.attn.add_k_proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_k_proj.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_k_proj.lora_A.weight",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_add_k_proj.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.attn.add_k_proj.alpha",
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.alpha",
                    "lora_unet_transformer_blocks_{block}_attn_add_k_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_v_proj",
                possible_up_patterns=[
                    "transformer_blocks.{block}.attn.add_v_proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_v_proj.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_v_proj.lora_B.weight",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_add_v_proj.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.attn.add_v_proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_v_proj.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.add_v_proj.lora_A.weight",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_add_v_proj.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.attn.add_v_proj.alpha",
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.alpha",
                    "lora_unet_transformer_blocks_{block}_attn_add_v_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_add_out",
                possible_up_patterns=[
                    "transformer_blocks.{block}.attn.to_add_out.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_up.weight",
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_add_out.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_add_out.lora_B.weight",
                    "transformer_blocks.{block}.attn.to_add_out.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_add_out.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.attn.to_add_out.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_down.weight",
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_add_out.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.attn.to_add_out.lora_A.weight",
                    "transformer_blocks.{block}.attn.to_add_out.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_attn_to_add_out.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.attn.to_add_out.alpha",
                    "transformer.transformer_blocks.{block}.attn.to_add_out.alpha",
                    "lora_unet_transformer_blocks_{block}_attn_to_add_out.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.img_ff.mlp_in",
                possible_up_patterns=[
                    "transformer_blocks.{block}.img_mlp.net.0.proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.img_mlp.net.0.proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.img_mlp.net.0.proj.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mlp.net.0.proj.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mlp.net.0.proj.lora_B.weight",
                    "transformer_blocks.{block}.img_mlp.net.0.proj.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_img_mlp_net_0_proj.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.img_mlp.net.0.proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.img_mlp.net.0.proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.img_mlp.net.0.proj.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mlp.net.0.proj.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mlp.net.0.proj.lora_A.weight",
                    "transformer_blocks.{block}.img_mlp.net.0.proj.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_img_mlp_net_0_proj.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.img_mlp.net.0.proj.alpha",
                    "transformer.transformer_blocks.{block}.img_mlp.net.0.proj.alpha",
                    "lora_unet_transformer_blocks_{block}_img_mlp_net_0_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.img_ff.mlp_out",
                possible_up_patterns=[
                    "transformer_blocks.{block}.img_mlp.net.2.lora_up.weight",
                    "transformer.transformer_blocks.{block}.img_mlp.net.2.lora_up.weight",
                    "transformer.transformer_blocks.{block}.img_mlp.net.2.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mlp.net.2.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mlp.net.2.lora_B.weight",
                    "transformer_blocks.{block}.img_mlp.net.2.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_img_mlp_net_2.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.img_mlp.net.2.lora_down.weight",
                    "transformer.transformer_blocks.{block}.img_mlp.net.2.lora_down.weight",
                    "transformer.transformer_blocks.{block}.img_mlp.net.2.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mlp.net.2.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mlp.net.2.lora_A.weight",
                    "transformer_blocks.{block}.img_mlp.net.2.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_img_mlp_net_2.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.img_mlp.net.2.alpha",
                    "transformer.transformer_blocks.{block}.img_mlp.net.2.alpha",
                    "lora_unet_transformer_blocks_{block}_img_mlp_net_2.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.txt_ff.mlp_in",
                possible_up_patterns=[
                    "transformer_blocks.{block}.txt_mlp.net.0.proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.0.proj.lora_up.weight",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.0.proj.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mlp.net.0.proj.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mlp.net.0.proj.lora_B.weight",
                    "transformer_blocks.{block}.txt_mlp.net.0.proj.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_txt_mlp_net_0_proj.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.txt_mlp.net.0.proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.0.proj.lora_down.weight",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.0.proj.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mlp.net.0.proj.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mlp.net.0.proj.lora_A.weight",
                    "transformer_blocks.{block}.txt_mlp.net.0.proj.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_txt_mlp_net_0_proj.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.txt_mlp.net.0.proj.alpha",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.0.proj.alpha",
                    "lora_unet_transformer_blocks_{block}_txt_mlp_net_0_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.txt_ff.mlp_out",
                possible_up_patterns=[
                    "transformer_blocks.{block}.txt_mlp.net.2.lora_up.weight",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.2.lora_up.weight",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.2.lora_B.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mlp.net.2.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mlp.net.2.lora_B.weight",
                    "transformer_blocks.{block}.txt_mlp.net.2.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_txt_mlp_net_2.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.txt_mlp.net.2.lora_down.weight",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.2.lora_down.weight",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.2.lora_A.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mlp.net.2.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mlp.net.2.lora_A.weight",
                    "transformer_blocks.{block}.txt_mlp.net.2.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_txt_mlp_net_2.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.txt_mlp.net.2.alpha",
                    "transformer.transformer_blocks.{block}.txt_mlp.net.2.alpha",
                    "lora_unet_transformer_blocks_{block}_txt_mlp_net_2.alpha",
                ],
            ),
        ]
        QwenLoRAMapping._add_comfy_direct_delta_patterns(targets)
        targets.extend(QwenLoRAMapping._get_top_level_mapping())
        targets.extend(QwenLoRAMapping._get_attention_norm_delta_mapping())
        return targets

    @staticmethod
    def get_modulation_mapping() -> List[LoRATarget]:
        targets = [
            LoRATarget(
                model_path="transformer_blocks.{block}.img_mod_linear",
                possible_up_patterns=[
                    "transformer_blocks.{block}.img_mod.1.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mod.1.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mod.1.lora_B.weight",
                    "transformer_blocks.{block}.img_mod.1.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_img_mod_1.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.img_mod.1.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mod.1.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.img_mod.1.lora_A.weight",
                    "transformer_blocks.{block}.img_mod.1.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_img_mod_1.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.img_mod.1.alpha",
                    "lora_unet_transformer_blocks_{block}_img_mod_1.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.txt_mod_linear",
                possible_up_patterns=[
                    "transformer_blocks.{block}.txt_mod.1.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mod.1.lora_up.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mod.1.lora_B.weight",
                    "transformer_blocks.{block}.txt_mod.1.lora_B.default.weight",
                    "lora_unet_transformer_blocks_{block}_txt_mod_1.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer_blocks.{block}.txt_mod.1.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mod.1.lora_down.weight",
                    "diffusion_model.transformer_blocks.{block}.txt_mod.1.lora_A.weight",
                    "transformer_blocks.{block}.txt_mod.1.lora_A.default.weight",
                    "lora_unet_transformer_blocks_{block}_txt_mod_1.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer_blocks.{block}.txt_mod.1.alpha",
                    "lora_unet_transformer_blocks_{block}_txt_mod_1.alpha",
                ],
            ),
        ]
        QwenLoRAMapping._add_comfy_direct_delta_patterns(targets)
        return targets

    @staticmethod
    def _add_comfy_direct_delta_patterns(targets: List[LoRATarget]) -> None:
        for target in targets:
            base_patterns: list[str] = []
            for pattern in target.possible_up_patterns + target.possible_down_patterns:
                base_pattern = QwenLoRAMapping._strip_lora_matrix_suffix(pattern)
                if base_pattern and base_pattern not in base_patterns:
                    base_patterns.append(base_pattern)

            target.possible_diff_patterns.extend(
                pattern for pattern in (f"{base}.diff" for base in base_patterns)
                if pattern not in target.possible_diff_patterns
            )
            target.possible_diff_b_patterns.extend(
                pattern for pattern in (f"{base}.diff_b" for base in base_patterns)
                if pattern not in target.possible_diff_b_patterns
            )

    @staticmethod
    def _strip_lora_matrix_suffix(pattern: str) -> str | None:
        for suffix in (
            ".lora_up.weight",
            ".lora_down.weight",
            ".lora.up.weight",
            ".lora.down.weight",
            ".lora_A.weight",
            ".lora_B.weight",
            ".lora_A.default.weight",
            ".lora_B.default.weight",
        ):
            if pattern.endswith(suffix):
                return pattern[: -len(suffix)]
        return None

    @staticmethod
    def _get_top_level_mapping() -> List[LoRATarget]:
        return [
            QwenLoRAMapping._linear_target("img_in"),
            QwenLoRAMapping._linear_target("txt_in"),
            QwenLoRAMapping._linear_target("time_text_embed.timestep_embedder.linear_1"),
            QwenLoRAMapping._linear_target("time_text_embed.timestep_embedder.linear_2"),
            QwenLoRAMapping._linear_target("norm_out.linear"),
            QwenLoRAMapping._linear_target("proj_out"),
        ]

    @staticmethod
    def _linear_target(source_path: str, model_path: str | None = None) -> LoRATarget:
        model_path = model_path or source_path
        lora_unet_name = source_path.replace(".", "_")
        target = LoRATarget(
            model_path=model_path,
            possible_up_patterns=[
                f"{source_path}.lora_up.weight",
                f"{source_path}.lora_B.weight",
                f"diffusion_model.{source_path}.lora_up.weight",
                f"diffusion_model.{source_path}.lora_B.weight",
                f"transformer.{source_path}.lora_up.weight",
                f"transformer.{source_path}.lora_B.weight",
                f"{source_path}.lora_B.default.weight",
                f"lora_unet_{lora_unet_name}.lora_up.weight",
                f"lycoris_{lora_unet_name}.lora_up.weight",
            ],
            possible_down_patterns=[
                f"{source_path}.lora_down.weight",
                f"{source_path}.lora_A.weight",
                f"diffusion_model.{source_path}.lora_down.weight",
                f"diffusion_model.{source_path}.lora_A.weight",
                f"transformer.{source_path}.lora_down.weight",
                f"transformer.{source_path}.lora_A.weight",
                f"{source_path}.lora_A.default.weight",
                f"lora_unet_{lora_unet_name}.lora_down.weight",
                f"lycoris_{lora_unet_name}.lora_down.weight",
            ],
            possible_alpha_patterns=[
                f"{source_path}.alpha",
                f"diffusion_model.{source_path}.alpha",
                f"transformer.{source_path}.alpha",
                f"lora_unet_{lora_unet_name}.alpha",
                f"lycoris_{lora_unet_name}.alpha",
            ],
        )
        QwenLoRAMapping._add_comfy_direct_delta_patterns([target])
        return target

    @staticmethod
    def _get_attention_norm_delta_mapping() -> List[LoRATarget]:
        targets = [
            LoRATarget(
                model_path="txt_norm",
                possible_diff_patterns=[
                    "txt_norm.diff",
                    "diffusion_model.txt_norm.diff",
                    "transformer.txt_norm.diff",
                    "lycoris_txt_norm.diff",
                ],
            )
        ]
        for source_name, model_name in (
            ("norm_q", "norm_q"),
            ("norm_k", "norm_k"),
            ("norm_added_q", "norm_added_q"),
            ("norm_added_k", "norm_added_k"),
        ):
            targets.append(
                LoRATarget(
                    model_path=f"transformer_blocks.{{block}}.attn.{model_name}",
                    possible_diff_patterns=[
                        f"transformer_blocks.{{block}}.attn.{source_name}.diff",
                        f"diffusion_model.transformer_blocks.{{block}}.attn.{source_name}.diff",
                        f"transformer.transformer_blocks.{{block}}.attn.{source_name}.diff",
                        f"lycoris_transformer_blocks_{{block}}_attn_{source_name}.diff",
                    ],
                )
            )
        return targets
