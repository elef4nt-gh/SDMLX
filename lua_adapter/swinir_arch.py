# Modified from https://github.com/JingyunLiang/SwinIR
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
#
# Multi-head extension for latent upscaling by Aleksandr Razin & Danil Kazantsev.
# Standalone version — no external dependencies beyond PyTorch.

import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


# ---------------------------------------------------------------------------
# Inline utilities (replacing basicsr.archs.arch_util)
# ---------------------------------------------------------------------------
def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def _norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lo = _norm_cdf((a - mean) / std)
        hi = _norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1).erfinv_()
        tensor.mul_(std * math.sqrt(2.0)).add_(mean).clamp_(min=a, max=b)
        return tensor


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, use_sdpa=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_sdpa = bool(use_sdpa and hasattr(F, "scaled_dot_product_attention"))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def _rel_pos_bias(self):
        num_tokens = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(num_tokens, num_tokens, -1)
        return bias.permute(2, 0, 1).contiguous()

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            rel_bias = self._rel_pos_bias()
            attn_bias = rel_bias.unsqueeze(0)
            if mask is not None:
                nw = mask.shape[0]
                b_img = b_ // nw
                add_mask = (
                    mask.to(rel_bias.dtype).to(q.device)
                    .view(1, nw, n, n).repeat(b_img, 1, 1, 1).view(b_, 1, n, n)
                )
                attn_bias = attn_bias + add_mask
            attn_bias = attn_bias.to(dtype=q.dtype, device=q.device)

            head_dim = c // self.num_heads
            default_scale = head_dim ** -0.5
            if abs(self.scale - default_scale) > 1e-8:
                q = q * (self.scale / default_scale)

            cuda_backend = getattr(torch.backends, "cuda", None)
            cuda_cm = nullcontext()
            if (torch.cuda.is_available() and cuda_backend is not None
                    and hasattr(cuda_backend, "sdp_kernel")):
                cuda_cm = cuda_backend.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=True, enable_math=True)
            with cuda_cm:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=False,
                )
            x = attn_out.transpose(1, 2).reshape(b_, n, c)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._rel_pos_bias().unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_sdpa=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, use_sdpa=use_sdpa,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, use_sdpa=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_sdpa=use_sdpa,
            )
            for i in range(depth)
        ])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(lambda inp: blk(inp, x_size), x, use_reentrant=False)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, img_size=224, patch_size=4,
                 resi_connection="1conv", use_sdpa=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, downsample=downsample,
            use_checkpoint=use_checkpoint, use_sdpa=use_sdpa,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


# ---------------------------------------------------------------------------
# Upsampling modules
# ---------------------------------------------------------------------------
class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
        super().__init__(*m)


class _SwinIRPixelShuffleHead(nn.Module):
    """PixelShuffle-based upsampling head used by SwinIRMultiHead."""

    def __init__(self, embed_dim: int, num_feat: int, num_out_ch: int, scale: int) -> None:
        super().__init__()
        self.scale = int(scale)
        self.conv_before = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.upsample = Upsample(self.scale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_last(self.upsample(self.conv_before(x)))


# ---------------------------------------------------------------------------
# SwinIRMultiHead — the main LUA model
# ---------------------------------------------------------------------------
class SwinIRMultiHead(nn.Module):
    """SwinIR backbone with shared trunk and multiple upsampling heads.

    Each head produces an output at a specific scale (e.g., x2, x4) while sharing
    the same shallow/deep feature extraction trunk.
    """

    def __init__(
        self,
        img_size: int | Sequence[int] = 64,
        patch_size: int | Sequence[int] = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        img_range: float = 1.0,
        resi_connection: str = "1conv",
        use_sdpa: bool = True,
        heads: Sequence[Dict[str, object]] | Dict[str, Dict[str, object]] | None = None,
        primary_head: str | None = None,
        head_num_feat: int | None = None,
    ) -> None:
        super().__init__()
        num_in_ch = int(in_chans)
        self.img_range = float(img_range)
        if num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.window_size = int(window_size)
        self.ape = bool(ape)
        self.patch_norm = bool(patch_norm)
        self.use_sdpa = bool(use_sdpa)
        self.embed_dim = int(embed_dim)
        self.mlp_ratio = float(mlp_ratio)
        self.num_layers = len(depths)

        # 1) Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, self.embed_dim, 3, 1, 1)

        # 2) Deep feature extraction
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=self.embed_dim,
            embed_dim=self.embed_dim, norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=self.embed_dim,
            embed_dim=self.embed_dim, norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        if isinstance(depths, tuple):
            depths = list(depths)
        if isinstance(num_heads, tuple):
            num_heads = list(num_heads)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=self.embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=self.window_size, mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint,
                img_size=img_size, patch_size=patch_size,
                resi_connection=resi_connection, use_sdpa=self.use_sdpa,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.embed_dim)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(self.embed_dim // 4, self.embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(self.embed_dim // 4, self.embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError("resi_connection must be '1conv' or '3conv'")

        # 3) Upsampling heads
        base_head_feat = int(head_num_feat or 64)
        head_specs = self._normalize_heads(heads, default_scale=1, base_num_feat=base_head_feat, default_out_ch=num_in_ch)
        self.heads = nn.ModuleDict()
        self.head_scales: Dict[str, float] = OrderedDict()
        self.head_names: List[str] = []

        for spec in head_specs:
            name = spec["name"]
            scale = int(spec["scale"])
            num_feat_head = int(spec.get("num_feat", base_head_feat))
            out_ch = int(spec.get("out_chans", num_in_ch))
            head = _SwinIRPixelShuffleHead(self.embed_dim, num_feat_head, out_ch, scale=scale)
            self.heads[name] = head
            self.head_scales[name] = float(scale)
            self.head_names.append(name)
            if spec.get("primary", False):
                primary_head = name

        if not self.head_names:
            raise ValueError("SwinIRMultiHead requires at least one head specification.")
        if primary_head is not None and primary_head not in self.heads:
            raise ValueError(f"primary_head {primary_head!r} not found among defined heads {self.head_names}.")

        if primary_head is None:
            primary_head = max(self.head_scales, key=self.head_scales.get)
        self.default_head = primary_head
        self.upscale = int(self.head_scales[self.default_head])

        self.apply(self._init_weights)

    @staticmethod
    def _normalize_heads(heads, default_scale, base_num_feat, default_out_ch):
        entries = []
        if isinstance(heads, dict):
            for name, cfg in heads.items():
                entry = dict(cfg or {})
                entry.setdefault("name", str(name))
                entries.append(entry)
        elif isinstance(heads, (list, tuple)):
            for idx, cfg in enumerate(heads):
                if cfg is None:
                    continue
                if isinstance(cfg, dict):
                    entry = dict(cfg)
                    entry.setdefault("name", str(entry.get("name") or f"head{idx}"))
                else:
                    entry = {"name": f"head{idx}", "scale": cfg}
                entries.append(entry)
        elif heads is not None:
            raise TypeError("heads must be a list, tuple, dict, or None.")

        if not entries:
            entries.append({"name": f"x{default_scale}", "scale": default_scale})

        normalized = []
        for entry in entries:
            spec = dict(entry)
            spec.setdefault("scale", default_scale)
            spec["scale"] = int(spec["scale"])
            spec.setdefault("upsampler", "pixelshuffle")
            spec.setdefault("num_feat", base_num_feat)
            spec.setdefault("out_chans", default_out_ch)
            normalized.append(spec)
        return normalized

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        self.mean = self.mean.type_as(x)
        feats = (x - self.mean) * self.img_range
        feats = self.conv_first(feats)
        feats = self.conv_after_body(self.forward_features(feats)) + feats

        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name in self.head_names:
            head_out = self.heads[name](feats)
            outputs[name] = head_out / self.img_range + self.mean
        return outputs

    def forward_single_head(self, x: torch.Tensor, head_name: str) -> torch.Tensor:
        if head_name not in self.heads:
            raise ValueError(f"Head '{head_name}' not found. Available heads: {list(self.heads.keys())}")
        self.mean = self.mean.type_as(x)
        feats = (x - self.mean) * self.img_range
        feats = self.conv_first(feats)
        feats = self.conv_after_body(self.forward_features(feats)) + feats
        head_out = self.heads[head_name](feats)
        return head_out / self.img_range + self.mean

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h > 0 or mod_pad_w > 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x
