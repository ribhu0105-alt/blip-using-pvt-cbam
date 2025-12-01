# models/attention_utils.py
"""
Attention Utility Blocks for Vision Transformers
================================================

This module centralizes reusable attention mechanisms:
 - CBAM (Channel + Spatial Attention)
 - Spatial Reduction Attention (SRA) for hierarchical transformers
 - MLP (feed-forward block)

Although your PVT implementation already embeds these in pvt.py,
this file is included to give the repo a modular & research-oriented
appearance, and to enable reuse for new architectures.

NOTE:
 The PVT implementation does not import from this file by default,
 but you can easily refactor pvt.py to do so if desired.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# MLP Feed Forward Network
# ---------------------------------------------------------------------
class MLP(nn.Module):
    """Standard 2-layer feed-forward network for transformers."""
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------
# Spatial Reduction Attention (SRA) — used in PVT
# ---------------------------------------------------------------------
class SpatialReductionAttention(nn.Module):
    """
    Implements SRA: spatial downsampling of K/V to reduce complexity.
    Used in PVT series. Can also be used in light-weight ViT variants.
    """
    def __init__(self, dim, num_heads=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        # Downsample for K/V
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        """
        x: (B, N, C)
        H, W: spatial dims of feature map
        """
        B, N, C = x.shape

        # Query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        # Spatial reduction for K/V
        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_)
            Hs, Ws = x_.shape[2], x_.shape[3]
            x_ = x_.flatten(2).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, Hs * Ws, 2, self.num_heads, C // self.num_heads)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)

        # Separate K, V
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


# ---------------------------------------------------------------------
# CBAM — Convolutional Block Attention Module
# Adapted for sequence (B, N, C) instead of (B, C, H, W)
# ---------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Channel attention for sequence data."""
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, N, C)
        """
        avg_pool = x.mean(dim=1)       # (B, C)
        max_pool, _ = x.max(dim=1)     # (B, C)
        out = self.fc(avg_pool) + self.fc(max_pool)
        out = self.sigmoid(out).unsqueeze(1)  # (B, 1, C)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention for sequence data."""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, N, C)
        """
        avg = x.mean(dim=2, keepdim=True)        # (B, N, 1)
        max_, _ = x.max(dim=2, keepdim=True)     # (B, N, 1)
        feat = torch.cat([avg, max_], dim=2)     # (B, N, 2)
        feat = feat.permute(0, 2, 1)             # (B, 2, N)
        attn = self.conv(feat)                   # (B, 1, N)
        attn = self.sigmoid(attn).permute(0, 2, 1)  # (B, N, 1)
        return x * attn


class CBAM(nn.Module):
    """Full CBAM block for sequence features (B, N, C)."""
    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
