# models/vision_backbones.py
"""
Unified Vision Backbone loader for the repo.

Provides:
 - create_backbone(name, image_size, **kwargs)
 - list_backbones()

Supported backbones shipped in this repo:
 - "pvt_tiny"   -> PVT-Tiny (with CBAM) implemented in models.pvt
 - "blip_vit"   -> Lightweight ViT implemented in models.blip_vit

Usage:
    from models.vision_backbones import create_backbone
    model, out_dim = create_backbone("pvt_tiny", image_size=384)
"""

from typing import Tuple
import torch.nn as nn

# Local imports (these modules are part of this repo)
from models.pvt import pvt_tiny
from models.blip_vit import blip_vit_base


_AVAILABLE = {
    "pvt_tiny": "PVT-Tiny (CBAM)",
    "blip_vit": "Lightweight ViT (BLIP-style)"
}


def list_backbones():
    """Return list of available backbone names and descriptions."""
    return _AVAILABLE.copy()


def create_backbone(name: str = "pvt_tiny", image_size: int = 224, **kwargs) -> Tuple[nn.Module, int]:
    """
    Instantiate a vision backbone and return (model, vision_width).

    Args:
        name: str, one of the keys in list_backbones()
        image_size: int, image input size (may be used by some modules)
        **kwargs: passed to backbone constructors where appropriate

    Returns:
        model: nn.Module instance
        vision_width: int, feature dimension of the last stage (embedding width)
    """
    name = name.lower()
    if name == "pvt_tiny":
        # PVT-Tiny returns stage outputs where last stage channel dim = 256
        model = pvt_tiny()
        vision_width = 256
        return model, vision_width

    if name == "blip_vit":
        # The lightweight BLIP ViT returns tokens with embed_dim (e.g., 768)
        model = blip_vit_base()
        # blip_vit_base uses embed_dim=768 by default; infer if attribute exists
        vision_width = list(getattr(model, "blocks")[-1].parameters()).__iter__().__length_hint__()  # placeholder safe fallback
        # reliable known default:
        vision_width = getattr(model, "pos_embed").shape[-1] if hasattr(model, "pos_embed") else 768
        return model, vision_width

    raise ValueError(f"Unsupported backbone '{name}'. Supported: {list(_AVAILABLE.keys())}")


# Backward-compatibility small helper
def create_vit(vit_name: str, image_size: int = 224, **kwargs) -> Tuple[nn.Module, int]:
    """
    Backwards-compatible alias used by existing code (blip.py expects create_vit).
    """
    return create_backbone(vit_name, image_size, **kwargs)
