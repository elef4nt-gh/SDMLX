import math

import mlx.core as mx
import mlx.nn as nn

from .config import UNetConfig
from .model_io import map_unet_weights
from .unet import ResnetBlock2D, TimestepEmbedding, Transformer2D, UNetBlock2D


UNION_CONTROL_TYPES = {
    "pose": 0,
    "depth": 1,
    "soft edge to scribble": 2,
    "line to canny": 3,
    "normal": 4,
    "segment": 5,
    "tile": 6,
    "repaint": 7,
}


class ControlNetConditioningEmbedding(nn.Module):
    def __init__(self, conditioning_channels=3, block_out_channels=(16, 32, 96, 256), embedding_channels=320):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = []
        for index in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[index]
            channel_out = block_out_channels[index + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding=1))
        self.conv_out = nn.Conv2d(block_out_channels[-1], embedding_channels, kernel_size=3, padding=1)

    def __call__(self, conditioning):
        x = nn.silu(self.conv_in(conditioning))
        for block in self.blocks:
            x = nn.silu(block(x))
        return self.conv_out(x)


class ResidualAttentionMlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.c_fc = nn.Linear(dims, dims * 4)
        self.c_proj = nn.Linear(dims * 4, dims)

    def __call__(self, x):
        x = self.c_fc(x)
        x = x * mx.sigmoid(1.702 * x)
        return self.c_proj(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, dims, num_heads):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.norm1 = nn.LayerNorm(dims)
        self.mlp = ResidualAttentionMlp(dims)
        self.norm2 = nn.LayerNorm(dims)

    def __call__(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y)
        return x + self.mlp(self.norm2(x))


class ControlNetUnionModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or UNetConfig()
        config.block_out_channels = [320, 640, 1280]
        config.layers_per_block = [2, 2, 2]
        config.transformer_layers_per_block = [1, 2, 10]
        config.num_attention_heads = [5, 10, 20]
        config.cross_attention_dim = [2048] * 3
        config.down_block_types = [
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ]
        config.addition_embed_type = "text_time"
        config.addition_time_embed_dim = 256
        config.projection_class_embeddings_input_dim = 2816

        self.num_control_type = 8
        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
        self.timesteps = nn.SinusoidalPositionalEncoding(
            320,
            max_freq=1,
            min_freq=math.exp(-math.log(10000) + 2 * math.log(10000) / 320),
            scale=1.0,
            cos_first=True,
            full_turns=False,
        )
        self.time_embedding = TimestepEmbedding(320, 1280)
        self.add_time_proj = nn.SinusoidalPositionalEncoding(
            256,
            max_freq=1,
            min_freq=math.exp(-math.log(10000) + 2 * math.log(10000) / 256),
            scale=1.0,
            cos_first=True,
            full_turns=False,
        )
        self.add_embedding = TimestepEmbedding(2816, 1280)
        self.control_type_proj = nn.SinusoidalPositionalEncoding(
            256,
            max_freq=1,
            min_freq=math.exp(-math.log(10000) + 2 * math.log(10000) / 256),
            scale=1.0,
            cos_first=True,
            full_turns=False,
        )
        self.control_add_embedding = TimestepEmbedding(256 * self.num_control_type, 1280)

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding()
        self.task_embedding = mx.zeros((self.num_control_type, 320))
        self.transformer_layers = [ResidualAttentionBlock(320, 8)]
        self.spatial_ch_projs = nn.Linear(320, 320)

        block_channels = [config.block_out_channels[0]] + list(config.block_out_channels)
        self.down_blocks = [
            UNetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=1280,
                num_layers=config.layers_per_block[i],
                transformer_layers_per_block=config.transformer_layers_per_block[i],
                num_attention_heads=config.num_attention_heads[i],
                cross_attention_dim=config.cross_attention_dim[i],
                resnet_groups=config.norm_num_groups,
                add_downsample=(i < len(config.block_out_channels) - 1),
                add_upsample=False,
                add_cross_attention="CrossAttn" in config.down_block_types[i],
            )
            for i, (in_channels, out_channels) in enumerate(zip(block_channels, block_channels[1:]))
        ]

        self.mid_blocks = [
            ResnetBlock2D(1280, 1280, temb_channels=1280, groups=32),
            Transformer2D(1280, 1280, 2048, 20, num_layers=10),
            ResnetBlock2D(1280, 1280, temb_channels=1280, groups=32),
        ]

        control_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
        self.controlnet_down_blocks = [
            nn.Conv2d(channels, channels, kernel_size=1) for channels in control_channels
        ]
        self.controlnet_mid_block = nn.Conv2d(1280, 1280, kernel_size=1)

    def __call__(self, x, timestep, encoder_x, control_image, control_type_idx, conditioning_scale=1.0, text_time=None):
        dtype = x.dtype
        batch = x.shape[0]
        temb = self.time_embedding(self.timesteps(timestep).astype(dtype))

        if text_time is not None:
            text_emb, time_ids = text_time
            emb = self.add_time_proj(time_ids).flatten(1).astype(dtype)
            emb = mx.concatenate([text_emb, emb], axis=-1)
            temb = temb + self.add_embedding(emb)

        control_type = mx.zeros((batch, self.num_control_type), dtype=dtype)
        for control_type_id in control_type_idx:
            control_type = control_type + mx.array(
                [[1.0 if index == control_type_id else 0.0 for index in range(self.num_control_type)]],
                dtype=dtype,
            )
        control_embeds = self.control_type_proj(control_type.flatten()).reshape(batch, -1).astype(dtype)
        temb = temb + self.control_add_embedding(control_embeds)

        x = self.conv_in(x)
        condition = self.controlnet_cond_embedding(control_image.astype(dtype))
        feat_seq = mx.mean(condition, axis=(1, 2)) + self.task_embedding[control_type_idx[0]].astype(dtype)
        sample_seq = mx.mean(x, axis=(1, 2))
        seq = mx.stack([feat_seq, sample_seq], axis=1)
        for layer in self.transformer_layers:
            seq = layer(seq)
        alpha = self.spatial_ch_projs(seq[:, 0])[:, None, None, :]
        x = x + condition + alpha

        residuals = [x]
        for block in self.down_blocks:
            x, res = block(x, encoder_x=encoder_x, temb=temb)
            residuals.extend(res)

        x = self.mid_blocks[0](x, temb)
        x = self.mid_blocks[1](x, encoder_x, None, None)
        x = self.mid_blocks[2](x, temb)

        down = [
            block(residual) * conditioning_scale
            for block, residual in zip(self.controlnet_down_blocks, residuals)
        ]
        mid = self.controlnet_mid_block(x) * conditioning_scale
        return down, mid


def _map_transformer_layer(key, value):
    key = key.replace("transformer_layes.", "transformer_layers.")
    key = key.replace(".ln_1.", ".norm1.")
    key = key.replace(".ln_2.", ".norm2.")
    key = key.replace(".mlp.c_fc.", ".mlp.c_fc.")
    key = key.replace(".mlp.c_proj.", ".mlp.c_proj.")
    if ".attn.in_proj_weight" in key:
        prefix = key[: -len(".attn.in_proj_weight")]
        q, k, v = mx.split(value, 3)
        return [
            (prefix + ".attn.query_proj.weight", q),
            (prefix + ".attn.key_proj.weight", k),
            (prefix + ".attn.value_proj.weight", v),
        ]
    if ".attn.in_proj_bias" in key:
        prefix = key[: -len(".attn.in_proj_bias")]
        q, k, v = mx.split(value, 3)
        return [
            (prefix + ".attn.query_proj.bias", q),
            (prefix + ".attn.key_proj.bias", k),
            (prefix + ".attn.value_proj.bias", v),
        ]
    return [(key, value)]


def map_controlnet_union_weights(key, value):
    if key == "task_embedding":
        return [(key, value)]
    if key.startswith("transformer_layes."):
        return _map_transformer_layer(key, value)
    if key.startswith(("down_blocks.", "mid_block.")):
        return map_unet_weights(key, value)
    if key.startswith("time_embedding."):
        return [(key, value)]
    if key.startswith("add_embedding."):
        return [(key, value)]
    if key.startswith("control_add_embedding."):
        return [(key, value)]
    if len(value.shape) == 4:
        value = value.transpose(0, 2, 3, 1)
        value = value.reshape(-1).reshape(value.shape)
    return [(key, value)]
