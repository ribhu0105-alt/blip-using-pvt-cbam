# models/blip_vit.py
"""
A compact Vision Transformer (ViT) defined in the style of BLIP.
This is NOT used by your PVT+CBAM model, but included to make the
repo modular and professional for research publication.

This VIT implementation supports:
 - patch embedding
 - multi-head self-attention
 - MLP feed-forward
 - optional CLS token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Patch Embedding
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                  # (B, C, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


# -----------------------------
# MLP Feedforward
# -----------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# -----------------------------
# Attention Block
# -----------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads=12):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# -----------------------------
# Transformer Encoder Block
# -----------------------------
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# Mini-ViT (BLIP-style)
# -----------------------------
class BLIP_ViT(nn.Module):
    """
    A lightweight BLIP-like ViT encoder for professional repo appearance.
    Not used directly in your PVT+CBAM architecture,
    but available for experiments or ablations.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            image_size, patch_size, in_chans, embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)  # (B, N, C)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x  # (B, N+1, C)


def blip_vit_base():
    return BLIP_ViT(
        image_size=224, patch_size=16,
        embed_dim=768, depth=12, num_heads=12
    )
