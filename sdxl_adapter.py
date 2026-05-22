import mlx.core as mx
from mlx.utils import tree_unflatten

from .mlx_sd.model_io import (
    map_clip_text_encoder_weights,
    map_unet_weights,
    map_vae_weights,
)


def is_mlx_array(value):
    value_type = type(value)
    return value_type.__name__ == "array" and value_type.__module__.startswith("mlx")


def to_mlx(value, dtype=mx.float16):
    if is_mlx_array(value):
        return value.astype(dtype)
    if hasattr(value, "detach"):
        if str(value.dtype) == "torch.bfloat16":
            value = value.float() if dtype == mx.float32 else value.half()
        value = value.detach().cpu().numpy()
    return mx.array(value).astype(dtype)


def apply_mapped_weights(model, weights, mapper):
    mapped = []
    for key, value in weights.items():
        mapped.extend(mapper(key, value))
    model.update(tree_unflatten(mapped))
    return len(mapped)


def checkpoint_has_any_prefix(keys, prefixes):
    return any(any(key.startswith(prefix) for prefix in prefixes) for key in keys)


def validate_sdxl_checkpoint_keys(torch_state_dict):
    keys = tuple(torch_state_dict.keys())

    if checkpoint_has_any_prefix(keys, (
        "model.diffusion_model.joint_blocks.",
        "model.diffusion_model.x_embedder.",
        "model.diffusion_model.context_embedder.",
        "model.diffusion_model.final_layer.",
        "diffusion_model.joint_blocks.",
        "diffusion_model.x_embedder.",
        "transformer.transformer_blocks.",
        "text_encoders.t5xxl.",
    )):
        raise ValueError(
            "SDMLX: This checkpoint looks like SD3/SD3.5. "
            "SD3 uses MMDiT, 16-channel latents and its own SD3 sampling path; "
            "the current SDMLX converter only supports SDXL UNet checkpoints."
        )

    if checkpoint_has_any_prefix(keys, (
        "model.diffusion_model.double_blocks.",
        "model.diffusion_model.single_blocks.",
        "diffusion_model.double_blocks.",
        "diffusion_model.single_blocks.",
    )):
        raise ValueError(
            "SDMLX: This checkpoint looks like FLUX. "
            "FLUX is not an SDXL UNet architecture and cannot be loaded by the current SDMLX converter."
        )

    if checkpoint_has_any_prefix(keys, (
        "model.diffusion_model.stage_b.",
        "model.diffusion_model.stage_c.",
        "prior.",
        "decoder.",
    )):
        raise ValueError(
            "SDMLX: This checkpoint does not look like SDXL "
            "(possibly Stable Cascade/Wuerstchen or another model type). "
            "The current SDMLX converter only supports SDXL UNet checkpoints."
        )

    required = {
        "SDXL UNet": "model.diffusion_model.input_blocks.0.0.weight",
        "SDXL VAE": "first_stage_model.encoder.conv_in.weight",
        "CLIP-L": "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        "CLIP-G": "conditioner.embedders.1.model.token_embedding.weight",
    }
    missing = [label for label, key in required.items() if key not in torch_state_dict]
    if missing:
        raise ValueError(
            "SDMLX: This checkpoint is not a complete SDXL checkpoint "
            f"or uses a currently unsupported storage format. Missing: {', '.join(missing)}. "
            "Currently supported: SDXL checkpoints with UNet, CLIP-L, CLIP-G and VAE in the classic "
            "Comfy/Stable-Diffusion-Safetensors-Layout."
        )

    conv_in = torch_state_dict[required["SDXL UNet"]]
    conv_shape = tuple(getattr(conv_in, "shape", ()))
    if len(conv_shape) >= 2 and int(conv_shape[1]) != 4:
        if int(conv_shape[1]) == 9:
            raise ValueError(
                "SDMLX: This checkpoint looks like a 9-channel SDXL inpaint UNet. "
                "The current SDMLX sampler supports normal 4-channel SDXL with mask/inpaint conditioning, "
                "but not native 9-channel inpaint checkpoints yet."
            )
        raise ValueError(
            "SDMLX: This checkpoint has an unexpected SDXL UNet input size "
            f"({conv_shape[1]} channels instead of 4). The current converter only supports 4-channel SDXL UNets."
        )


def map_vae_weights_for_apple(key, value):
    if (
        "mid_block.attentions.0.to_" in key
        and key.endswith(".weight")
        and len(value.shape) == 4
        and value.shape[2:] == (1, 1)
    ):
        value = value.squeeze()
    return map_vae_weights(key, value)


def split_sdxl_checkpoint(torch_state_dict, dtype=mx.float16):
    groups = {"unet": {}, "clip_l": {}, "clip_g": {}, "vae": {}}

    for key, value in torch_state_dict.items():
        mlx_value = to_mlx(value, dtype=dtype)

        if key.startswith("model.diffusion_model."):
            diffusers_key = ldm_unet_key_to_diffusers(key[len("model.diffusion_model."):])
            if diffusers_key is not None:
                groups["unet"][diffusers_key] = mlx_value
            continue

        if key.startswith("first_stage_model."):
            diffusers_key = ldm_vae_key_to_diffusers(key[len("first_stage_model."):])
            if diffusers_key is not None:
                groups["vae"][diffusers_key] = mlx_value
            continue

        if key.startswith("conditioner.embedders.0.transformer."):
            diffusers_key = key[len("conditioner.embedders.0.transformer."):]
            groups["clip_l"][diffusers_key] = mlx_value
            continue

        if key.startswith("conditioner.embedders.1.model."):
            open_clip_key = key[len("conditioner.embedders.1.model."):]
            if ".attn.in_proj_weight" in open_clip_key:
                base = open_clip_key_to_diffusers(open_clip_key.replace(".attn.in_proj_weight", ".attn.q_proj.weight"))
                if base is not None:
                    prefix = base[: -len("self_attn.q_proj.weight")]
                    q, k, v = mx.split(mlx_value, 3)
                    groups["clip_g"][prefix + "self_attn.q_proj.weight"] = q
                    groups["clip_g"][prefix + "self_attn.k_proj.weight"] = k
                    groups["clip_g"][prefix + "self_attn.v_proj.weight"] = v
                continue
            if ".attn.in_proj_bias" in open_clip_key:
                base = open_clip_key_to_diffusers(open_clip_key.replace(".attn.in_proj_bias", ".attn.q_proj.bias"))
                if base is not None:
                    prefix = base[: -len("self_attn.q_proj.bias")]
                    q, k, v = mx.split(mlx_value, 3)
                    groups["clip_g"][prefix + "self_attn.q_proj.bias"] = q
                    groups["clip_g"][prefix + "self_attn.k_proj.bias"] = k
                    groups["clip_g"][prefix + "self_attn.v_proj.bias"] = v
                continue

            diffusers_key = open_clip_key_to_diffusers(open_clip_key)
            if diffusers_key is not None:
                if diffusers_key == "text_projection.weight":
                    mlx_value = mlx_value.T
                groups["clip_g"][diffusers_key] = mlx_value
            continue

    return groups


def ldm_unet_key_to_diffusers(key):
    direct = {
        "time_embed.0.weight": "time_embedding.linear_1.weight",
        "time_embed.0.bias": "time_embedding.linear_1.bias",
        "time_embed.2.weight": "time_embedding.linear_2.weight",
        "time_embed.2.bias": "time_embedding.linear_2.bias",
        "input_blocks.0.0.weight": "conv_in.weight",
        "input_blocks.0.0.bias": "conv_in.bias",
        "out.0.weight": "conv_norm_out.weight",
        "out.0.bias": "conv_norm_out.bias",
        "out.2.weight": "conv_out.weight",
        "out.2.bias": "conv_out.bias",
        "label_emb.0.0.weight": "add_embedding.linear_1.weight",
        "label_emb.0.0.bias": "add_embedding.linear_1.bias",
        "label_emb.0.2.weight": "add_embedding.linear_2.weight",
        "label_emb.0.2.bias": "add_embedding.linear_2.bias",
    }
    if key in direct:
        return direct[key]
    if key.startswith("middle_block."):
        return map_unet_middle_key(key)
    if key.startswith("input_blocks."):
        return map_unet_input_key(key)
    if key.startswith("output_blocks."):
        return map_unet_output_key(key)
    return None


def map_unet_middle_key(key):
    rest = key[len("middle_block."):]
    if rest.startswith("0."):
        return "mid_block.resnets.0." + map_resnet_suffix(rest[2:])
    if rest.startswith("1."):
        return "mid_block.attentions.0." + map_attention_suffix(rest[2:])
    if rest.startswith("2."):
        return "mid_block.resnets.1." + map_resnet_suffix(rest[2:])
    return None


def map_unet_input_key(key):
    parts = key.split(".", 3)
    if len(parts) < 4:
        return None
    block_index = int(parts[1])
    layer = parts[2]
    suffix = parts[3]
    if block_index == 0:
        return None
    down_block = (block_index - 1) // 3
    layer_in_block = (block_index - 1) % 3
    if layer_in_block == 2:
        if layer == "0":
            if suffix.startswith("op."):
                suffix = suffix[len("op."):]
            return f"down_blocks.{down_block}.downsamplers.0.conv.{suffix}"
        return None
    if layer == "0":
        return f"down_blocks.{down_block}.resnets.{layer_in_block}." + map_resnet_suffix(suffix)
    if layer == "1":
        return f"down_blocks.{down_block}.attentions.{layer_in_block}." + map_attention_suffix(suffix)
    return None


def map_unet_output_key(key):
    parts = key.split(".", 3)
    if len(parts) < 4:
        return None
    block_index = int(parts[1])
    layer = parts[2]
    suffix = parts[3]
    up_block = block_index // 3
    layer_in_block = block_index % 3
    if layer == "0":
        return f"up_blocks.{up_block}.resnets.{layer_in_block}." + map_resnet_suffix(suffix)
    if layer == "1":
        return f"up_blocks.{up_block}.attentions.{layer_in_block}." + map_attention_suffix(suffix)
    if layer == "2":
        if suffix.startswith("conv."):
            suffix = suffix[len("conv."):]
        return f"up_blocks.{up_block}.upsamplers.0.conv.{suffix}"
    return None


def map_resnet_suffix(suffix):
    replacements = {
        "in_layers.0.": "norm1.",
        "in_layers.2.": "conv1.",
        "emb_layers.1.": "time_emb_proj.",
        "out_layers.0.": "norm2.",
        "out_layers.3.": "conv2.",
        "skip_connection.": "conv_shortcut.",
    }
    for old, new in replacements.items():
        if suffix.startswith(old):
            return new + suffix[len(old):]
    return suffix


def map_attention_suffix(suffix):
    return suffix


def ldm_vae_key_to_diffusers(key):
    direct = {
        "encoder.conv_in.weight": "encoder.conv_in.weight",
        "encoder.conv_in.bias": "encoder.conv_in.bias",
        "encoder.conv_out.weight": "encoder.conv_out.weight",
        "encoder.conv_out.bias": "encoder.conv_out.bias",
        "encoder.norm_out.weight": "encoder.conv_norm_out.weight",
        "encoder.norm_out.bias": "encoder.conv_norm_out.bias",
        "decoder.conv_in.weight": "decoder.conv_in.weight",
        "decoder.conv_in.bias": "decoder.conv_in.bias",
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        "decoder.conv_out.bias": "decoder.conv_out.bias",
        "decoder.norm_out.weight": "decoder.conv_norm_out.weight",
        "decoder.norm_out.bias": "decoder.conv_norm_out.bias",
        "quant_conv.weight": "quant_conv.weight",
        "quant_conv.bias": "quant_conv.bias",
        "post_quant_conv.weight": "post_quant_conv.weight",
        "post_quant_conv.bias": "post_quant_conv.bias",
    }
    if key in direct:
        return direct[key]
    if key.startswith("encoder.down."):
        return map_vae_block_key(key, "encoder.down.", "encoder.down_blocks", reverse=False)
    if key.startswith("decoder.up."):
        return map_vae_block_key(key, "decoder.up.", "decoder.up_blocks", reverse=True)
    if key.startswith("encoder.mid."):
        return map_vae_mid_key(key, "encoder")
    if key.startswith("decoder.mid."):
        return map_vae_mid_key(key, "decoder")
    return None


def map_vae_block_key(key, old_prefix, new_prefix, reverse):
    rest = key[len(old_prefix):]
    parts = rest.split(".", 2)
    if len(parts) < 3:
        return None
    block = int(parts[0])
    name = parts[1]
    suffix = parts[2]
    block_count = 4
    target_block = block_count - 1 - block if reverse else block
    if name == "block":
        res_parts = suffix.split(".", 1)
        if len(res_parts) != 2:
            return None
        return f"{new_prefix}.{target_block}.resnets.{int(res_parts[0])}." + map_vae_resnet_suffix(res_parts[1])
    if name == "downsample":
        return f"{new_prefix}.{target_block}.downsamplers.0.conv.{suffix[len('conv.'):] if suffix.startswith('conv.') else suffix}"
    if name == "upsample":
        return f"{new_prefix}.{target_block}.upsamplers.0.conv.{suffix[len('conv.'):] if suffix.startswith('conv.') else suffix}"
    return None


def map_vae_mid_key(key, side):
    rest = key[len(side + ".mid."):]
    if rest.startswith("block_1."):
        return f"{side}.mid_block.resnets.0." + map_vae_resnet_suffix(rest[len("block_1."):])
    if rest.startswith("block_2."):
        return f"{side}.mid_block.resnets.1." + map_vae_resnet_suffix(rest[len("block_2."):])
    if rest.startswith("attn_1."):
        return f"{side}.mid_block.attentions.0." + map_vae_attention_suffix(rest[len("attn_1."):])
    return None


def map_vae_resnet_suffix(suffix):
    replacements = {"nin_shortcut.": "conv_shortcut."}
    for old, new in replacements.items():
        if suffix.startswith(old):
            return new + suffix[len(old):]
    return suffix


def map_vae_attention_suffix(suffix):
    replacements = {
        "norm.": "group_norm.",
        "q.": "to_q.",
        "k.": "to_k.",
        "v.": "to_v.",
        "proj_out.": "to_out.0.",
    }
    for old, new in replacements.items():
        if suffix.startswith(old):
            return new + suffix[len(old):]
    return suffix


def open_clip_key_to_diffusers(key):
    if key == "text_projection":
        return "text_projection.weight"
    if key.startswith("token_embedding."):
        return key.replace("token_embedding.", "text_model.embeddings.token_embedding.")
    if key.startswith("positional_embedding"):
        return key.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if key.startswith("ln_final."):
        return key.replace("ln_final.", "text_model.final_layer_norm.")
    if key.startswith("transformer.resblocks."):
        rest = key[len("transformer.resblocks."):]
        layer, suffix = rest.split(".", 1)
        suffix = suffix.replace("attn.q_proj.", "self_attn.q_proj.")
        suffix = suffix.replace("attn.k_proj.", "self_attn.k_proj.")
        suffix = suffix.replace("attn.v_proj.", "self_attn.v_proj.")
        suffix = suffix.replace("ln_1.", "layer_norm1.")
        suffix = suffix.replace("ln_2.", "layer_norm2.")
        suffix = suffix.replace("attn.out_proj.", "self_attn.out_proj.")
        suffix = suffix.replace("mlp.c_fc.", "mlp.fc1.")
        suffix = suffix.replace("mlp.c_proj.", "mlp.fc2.")
        return f"text_model.encoder.layers.{layer}.{suffix}"
    return None


def map_clip_l_weights(weights):
    return _map_clip_weights(weights)


def map_clip_g_weights(weights):
    return _map_clip_weights(weights)


def _map_clip_weights(weights):
    mapped = []
    for key, value in weights.items():
        if key.endswith("position_ids") or key.endswith("position_ids.weight"):
            continue
        if key.startswith("text_model."):
            key = key[len("text_model."):]
        if key == "embeddings.position_ids" or key == "position_ids":
            continue
        mapped.extend(map_clip_text_encoder_weights(key, value))
    return mapped
