# -*- coding: utf-8 -*-
"""
AFUNet with Gated Map & Spatial Feature Transform (SFT) — NTIRE 2026 RAIM Track2
Team: whu-vip

Model overview:
  AFUNet is a multi-exposure HDR reconstruction network based on the Swin Transformer.
  It takes 3 LDR frames (under / normal / over exposed) stacked as a 9-channel input
  and outputs a single tone-mapped HDR image (3-channel).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# =============================================================================
# Part 1: Basic Helper Modules & Functions
# =============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

def to_3d(x):
    return x.flatten(2).transpose(1, 2)

def to_4d(x, h, w):
    B, L, C = x.shape
    return x.transpose(1, 2).reshape(B, C, h, w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, 1e-5)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape
        b, c, h, w = x.shape
        head = self.num_heads
        hd = c // head

        q = self.q(y).view(b, head, hd, h * w)  
        k = self.k(x).view(b, head, hd, h * w)
        v = self.v(x).view(b, head, hd, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        
        return out

class FeatureModulation(nn.Module):
    def __init__(self, dim):
        super(FeatureModulation, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, padding_mode='reflect')
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, padding_mode='reflect')
        
    def forward(self, x, feat_global):
        cat_feat = torch.cat([x, feat_global], dim=1)
        params = self.conv2(self.act(self.conv1(cat_feat)))
        gamma, beta = torch.chunk(params, 2, dim=1)
        gamma = torch.tanh(gamma)  
        return gamma, beta

class CrossAttentionProximalOperater(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias', drop_path=0.):
        super(CrossAttentionProximalOperater, self).__init__()

        self.attn = Mutual_Attention(dim, num_heads, bias)
        self.feature_modulation = FeatureModulation(dim) 
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, padding_mode='reflect')
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y, x_size):
        b, n, c = x.shape
        h, w = x_size
        assert n == h*w
        
        x_img = x.permute(0,2,1).view(b, c, h, w)
        y_img = y.permute(0,2,1).view(b, c, h, w)
        
        feat_global = self.attn(x_img + y_img, y_img)
        gamma, beta = self.feature_modulation(x_img, feat_global)
        
        feat_modulated = feat_global * (1 + gamma) + beta
        
        x_img = x_img + self.drop_path(feat_modulated)
        x_img = x_img + self.conv(x_img)

        x_out = to_3d(x_img)
        return x_out

class ReliabilityGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 2)
        )

    def reset_last_layer_to_zero(self):
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, ref, u, v):
        x = torch.cat([ref, u, v], dim=-1)
        raw = self.mlp(x)
        g1 = 0.5 + 0.5 * torch.tanh(raw[..., 0:1])
        g2 = 0.5 + 0.5 * torch.tanh(raw[..., 1:2])
        return g1, g2

# =============================================================================
# Part 2: Swin Transformer Components
# =============================================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x

class WindowSelfAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

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

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowCrossAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

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

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        B_, N, C = x.shape
        q = self.q(y).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = q, kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock_SelfAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
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
        self.attn = WindowSelfAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size, custom_mask=None):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.shift_size > 0:
            if custom_mask is not None:
                mask = custom_mask
            elif self.input_resolution == x_size:
                mask = self.attn_mask
            else:
                mask = self.calculate_mask(x_size).to(x.device)
        else:
            mask = None

        attn_windows = self.attn(x_windows, mask=mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class SwinTransformerBlock_CrossAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
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

        self.attn = WindowCrossAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.patch_embed = PatchEmbed(in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(in_chans=0, embed_dim=dim, norm_layer=None)
        self.mlp = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, padding_mode='reflect')

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, y, x_size, custom_mask=None):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x + y
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        x_windows = window_partition(shifted_x, self.window_size)
        y_windows = window_partition(shifted_y, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.shift_size > 0:
            if custom_mask is not None:
                mask = custom_mask
            elif self.input_resolution == x_size:
                mask = self.attn_mask
            else:
                mask = self.calculate_mask(x_size).to(x.device)
        else:
            mask = None

        attn_windows = self.attn(x_windows, y_windows, mask=mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.patch_embed(self.mlp(self.patch_unembed(x, x_size)))

        return x

class SelfAttentionGroup(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock_SelfAttention(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, 
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, 
                                 qk_scale=qk_scale,
                                 drop=drop, 
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, custom_mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, custom_mask)
            else:
                x = blk(x, x_size, custom_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class CrossAttentionGroup(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock_CrossAttention(dim=dim, 
                                input_resolution=input_resolution,
                                 num_heads=num_heads, 
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, 
                                 qk_scale=qk_scale,
                                 drop=drop, 
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, y, x_size, custom_mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, y, x_size, custom_mask)
            else:
                x = blk(x, y, x_size, custom_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SpatialAttentionAlignmentModule(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(SpatialAttentionAlignmentModule, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = CrossAttentionGroup(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, 
                                         qk_scale=qk_scale,
                                         drop=drop, 
                                         attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

    def forward(self, x, y, x_size, custom_mask=None):
        return self.residual_group(x, y, x_size, custom_mask) + x


class SpatialFeatureEnhancementModule(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(SpatialFeatureEnhancementModule, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        
        self.residual_group = SelfAttentionGroup(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=1, padding_mode='reflect')
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1, padding_mode='reflect'), 
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0, padding_mode='reflect'),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1, padding_mode='reflect'))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size, custom_mask=None):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, custom_mask), x_size))) + x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

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
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


# =============================================================================
# Part 3: Main AFUNet Class
# =============================================================================

class AFUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=1, in_chans=9, embed_dim=72,
                 depths=(5, 5, 5, 5), num_heads=(4, 4, 4, 4), window_size=8,
                 mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False, img_range=1.,
                 resi_connection='1conv', **kwargs):
        super().__init__()
        
        self.window_size = window_size
        self.mask_cache = {}

        self.num_layers = len(depths)
        
        self.gating_modules = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.gating_modules.append(ReliabilityGating(embed_dim // 3))
        
        hidden_features = int((embed_dim//3) * mlp_ratio)
        self.A2_T = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Mlp(embed_dim//3, hidden_features, embed_dim//3, act_layer=nn.GELU, drop=drop_rate)
            self.A2_T.append(layer)
            
        self.A1 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Mlp(embed_dim//3, hidden_features, embed_dim//3, act_layer=nn.GELU, drop=drop_rate)
            self.A1.append(layer)
            
        self.A3 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Mlp(embed_dim//3, hidden_features, embed_dim//3, act_layer=nn.GELU, drop=drop_rate)
            self.A3.append(layer)
            
        self.inv = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Mlp(embed_dim//3, hidden_features, embed_dim//3, act_layer=nn.GELU, drop=drop_rate)
            self.inv.append(layer)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans//3, embed_dim//3, 3, 1, 1, padding_mode='reflect'),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_chans//3, embed_dim//3, 3, 1, 1, padding_mode='reflect'),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chans//3, embed_dim//3, 3, 1, 1, padding_mode='reflect'),
        )

        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, 
                                      embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)

        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, 
                                          embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)

        self.patch_unembed_div3 = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim//3, 
                                               embed_dim=embed_dim//3, norm_layer=norm_layer if patch_norm else None)
        
        self.patch_embed_div3 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim//3, 
                                           embed_dim=embed_dim//3, norm_layer=norm_layer if patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        
        depths_aligment = [1, 1, 1, 1, 1, 1]
        dpr_aligment = [x.item() for x in torch.linspace(0, drop_path, sum(depths_aligment))]
        
        self.a_12 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SpatialAttentionAlignmentModule(dim=embed_dim // 3,
                         input_resolution=(self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]),
                         depth=depths_aligment[i_layer],
                         num_heads=num_heads[i_layer]//2,
                         window_size=window_size,
                         mlp_ratio=1,
                         qkv_bias=qkv_bias, 
                         qk_scale=qk_scale,
                         drop=drop_rate, 
                         attn_drop=attn_drop_rate,
                         drop_path=dpr_aligment[sum(depths_aligment[:i_layer]):sum(depths_aligment[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.a_12.append(layer)

        self.a_32 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SpatialAttentionAlignmentModule(dim=embed_dim // 3,
                         input_resolution=(self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]),
                         depth=depths_aligment[i_layer],
                         num_heads=num_heads[i_layer]//2,
                         window_size=window_size,
                         mlp_ratio=1,
                         qkv_bias=qkv_bias, 
                         qk_scale=qk_scale,
                         drop=drop_rate, 
                         attn_drop=attn_drop_rate,
                         drop_path=dpr_aligment[sum(depths_aligment[:i_layer]):sum(depths_aligment[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.a_32.append(layer)
        
        dpr_f = [x.item() for x in torch.linspace(0, drop_path, self.num_layers)]

        self.f_12 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = CrossAttentionProximalOperater(embed_dim//3,
                         num_heads=num_heads[i_layer]//2,
                         ffn_expansion_factor=1,
                         bias=False,
                         LayerNorm_type='WithBias',
                         drop_path=dpr_f[i_layer]) 
            self.f_12.append(layer)

        self.f_32 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = CrossAttentionProximalOperater(embed_dim//3,
                         num_heads=num_heads[i_layer]//2,
                         ffn_expansion_factor=1,
                         bias=False,
                         LayerNorm_type='WithBias',
                         drop_path=dpr_f[i_layer]) 
            self.f_32.append(layer)

        self.SFM = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SpatialFeatureEnhancementModule(dim=embed_dim,
                         input_resolution=(self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, 
                         qk_scale=qk_scale,
                         drop=drop_rate, 
                         attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.SFM.append(layer)

        self.mlps_end = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Mlp(embed_dim, int((embed_dim//3)*mlp_ratio), embed_dim//3, act_layer=nn.GELU, drop=drop_rate)
            self.mlps_end.append(layer)

        self.conv_skip = nn.Conv2d(embed_dim//3, embed_dim//3, 3, 1, 1, padding_mode='reflect')

        self.out = nn.Sequential(nn.Conv2d(embed_dim//3, 3, 3, 1, 1, padding_mode='reflect'))
        
        self.apply(self._init_weights)
        self._reset_gating_last_layers()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _reset_gating_last_layers(self):
        for gm in self.gating_modules:
            gm.reset_last_layer_to_zero()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def _get_mask(self, x_size, device):
        mask = getattr(self, "mask_cache", {}).get(x_size, None)
        if mask is not None:
            return mask
        
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.window_size // 2),
                    slice(-self.window_size // 2, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.window_size // 2),
                    slice(-self.window_size // 2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        self.mask_cache[x_size] = attn_mask
        return attn_mask

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        
        shared_mask = self._get_mask(x_size, x.device)

        x = self.patch_embed(x)
        
        W_x1, x2, W_x3 = x.chunk(3, dim=2)
        X_k = x2.clone()
        
        for a_12, a_32, f_12, f_32, SFM, mlp_end, A1, A3, A2_T, inv, gating \
        in zip(self.a_12, self.a_32, self.f_12, self.f_32, self.SFM, self.mlps_end, self.A1, self.A3, self.A2_T, self.inv, self.gating_modules):
            ## Feature Alignment ##
            A1_X = A1(X_k)
            A3_X = A3(X_k)
            
            W_x1 = a_12(A1_X, W_x1, x_size, custom_mask=shared_mask)
            W_x3 = a_32(A3_X, W_x3, x_size, custom_mask=shared_mask) 
            
            df = torch.cat([W_x1, X_k, W_x3], 2)
            df = SFM(df, x_size, custom_mask=shared_mask)
            U_deep, X_deep, V_deep = df.chunk(3, dim=2)
            
            ## Feature Fusion ##
            U_k = f_12(X_k, U_deep, x_size) + X_k
            V_k = f_32(X_k, V_deep, x_size) + X_k
            
           ## Data Consistency (Modified for Spatial Gating) ##
            ref_feat = A2_T(x2)
            g1, g2 = gating(ref_feat, U_k, V_k)
            
            s = ref_feat + g1 * U_k + g2 * V_k
            X_k = inv(s)
            
            f = torch.cat((U_k, X_k, V_k), dim=2)
            X_k = mlp_end(f) + X_deep

        x = self.patch_unembed_div3(X_k, x_size)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x1, x2, x3 = x.chunk(3, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.forward_features(x) + self.conv_skip(x2)
        x = self.out(x)
        
        return x[:, :, :H, :W]