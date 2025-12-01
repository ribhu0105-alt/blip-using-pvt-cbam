# models/pvt.py
"""
PVT-Tiny backbone adapted for BLIP with optional CBAM (channel+spatial) attention
Integrated design:
 - OverlapPatchEmbed
 - SpatialReductionAttention (SRA)
 - Transformer Block with MLP and DropPath
 - CBAM inserted after residuals in each block

This is intentionally self-contained and has no external dependencies beyond PyTorch.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# DropPath (stochastic depth)
# --------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        binary = (rand < keep_prob).to(x.dtype)
        return x * binary / keep_prob


# --------------------------
# Overlap Patch Embedding
# --------------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, (H, W)


# --------------------------
# Spatial Reduction Attention
# --------------------------
class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, num_heads=1, sr_ratio=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # keep channels same but spatially reduce
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N, head_dim)

        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
            x_ = self.sr(x_)  # spatially reduced
            Hs, Ws = x_.shape[2], x_.shape[3]
            x_ = x_.flatten(2).transpose(1, 2)  # (B, Hs*Ws, C)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, Hs * Ws, 2, self.num_heads, C // self.num_heads)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)

        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, heads, Nk, head_dim)
        k, v = kv[0], kv[1]  # each: (B, heads, Nk, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, Nk)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


# --------------------------
# MLP
# --------------------------
class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# --------------------------
# CBAM: Channel + Spatial Attention (adapted for sequence (B, N, C))
# --------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 8):
        super().__init__()
        # We'll operate on (B, N, C). Use 1D pooling across tokens (N)
        self.in_planes = in_planes
        hidden = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_planes, bias=False)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        # Avg pool and max pool over tokens (N)
        avg_pool = x.mean(dim=1)  # (B, C)
        max_pool, _ = x.max(dim=1)  # (B, C)
        y = self.fc(avg_pool) + self.fc(max_pool)  # (B, C)
        y = self.sig(y).unsqueeze(1)  # (B, 1, C)
        return x * y  # broadcast over N


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        # For sequence X: (B, N, C) -> produce (B, N, 1) attention
        padding = (kernel_size - 1) // 2
        # we'll apply conv1d over tokens N; input channels = 2 (avg + max across channels)
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        avg = x.mean(dim=2, keepdim=True)  # (B, N, 1)
        max_, _ = x.max(dim=2, keepdim=True)  # (B, N, 1)
        cat = torch.cat([avg, max_], dim=2)  # (B, N, 2)
        # conv1d expects (B, C_in, L) where L = N
        cat = cat.permute(0, 2, 1)  # (B, 2, N)
        out = self.conv(cat)  # (B, 1, N)
        out = self.sig(out).permute(0, 2, 1)  # (B, N, 1)
        return x * out  # broadcast over channels


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# --------------------------
# Transformer Block with CBAM
# --------------------------
class Block(nn.Module):
    def __init__(self, dim, heads=1, sr_ratio=8, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, heads, sr_ratio)
        self.drop_path = DropPath(drop) if drop > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        self.cbam = CBAM(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, C)
        res = x
        x = self.norm1(x)
        x = self.attn(x, H, W)
        x = res + self.drop_path(x)

        res2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res2 + self.drop_path(x)

        # CBAM expects (B, N, C)
        x = x + self.cbam(x)

        return x


# --------------------------
# PVT Tiny
# --------------------------
class PVT_Tiny(nn.Module):
    def __init__(self, in_chans=3, img_size=224, drop_path_rate=0.0):
        super().__init__()
        # configuration from PVT-Tiny
        dims = [32, 64, 160, 256]
        heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        depths = [2, 2, 2, 2]

        # stage1
        self.patch1 = OverlapPatchEmbed(in_chans, dims[0], kernel_size=7, stride=4, padding=3)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([Block(dims[0], heads[0], sr_ratios[0], drop=dpr[cur + i]) for i in range(depths[0])])
        cur += depths[0]

        # stage2
        self.patch2 = OverlapPatchEmbed(dims[0], dims[1], kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.ModuleList([Block(dims[1], heads[1], sr_ratios[1], drop=dpr[cur + i]) for i in range(depths[1])])
        cur += depths[1]

        # stage3
        self.patch3 = OverlapPatchEmbed(dims[1], dims[2], kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.ModuleList([Block(dims[2], heads[2], sr_ratios[2], drop=dpr[cur + i]) for i in range(depths[2])])
        cur += depths[2]

        # stage4
        self.patch4 = OverlapPatchEmbed(dims[2], dims[3], kernel_size=3, stride=2, padding=1)
        self.stage4 = nn.ModuleList([Block(dims[3], heads[3], sr_ratios[3], drop=dpr[cur + i]) for i in range(depths[3])])
        cur += depths[3]

        self.norm = nn.LayerNorm(dims[3])

    def _forward_stage(self, x: torch.Tensor, patch: OverlapPatchEmbed, stage: nn.ModuleList):
        x, (H, W) = patch(x)
        for blk in stage:
            x = blk(x, H, W)
        B, N, C = x.shape
        # return (B, C, H, W)
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # returns list of stage outputs: c1, c2, c3, c4 as feature maps (B, C, H, W)
        c1 = self._forward_stage(x, self.patch1, self.stage1)
        c2 = self._forward_stage(c1, self.patch2, self.stage2)
        c3 = self._forward_stage(c2, self.patch3, self.stage3)
        c4 = self._forward_stage(c3, self.patch4, self.stage4)
        return [c1, c2, c3, c4]


def pvt_tiny():
    return PVT_Tiny()
