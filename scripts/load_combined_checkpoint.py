#!/usr/bin/env python3
"""
Load a combined checkpoint and restore both PVT and HuggingFace BLIP models.

The combined checkpoint has keys prefixed:
  - 'visual_encoder.<key>' for PVT weights
  - 'text_decoder.<key>' for HuggingFace BLIP decoder weights
  - 'visual_proj.<key>' for optional projection layer weights

Usage:
    from scripts.load_combined_checkpoint import load_combined_model
    pvt, blip, proj = load_combined_model('checkpoints/blip_pvt_combined.pth', device='cuda')
"""

import os
import sys
import torch
import torch.nn as nn

# ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pvt import pvt_tiny
from transformers import BlipForConditionalGeneration


def load_combined_model(checkpoint_path, device='cpu', blip_model_name='Salesforce/blip-image-captioning-base'):
    """
    Load a combined checkpoint and restore PVT, BLIP, and optional projection.
    
    Args:
        checkpoint_path: Path to the combined checkpoint file
        device: Device to load models on ('cpu' or 'cuda')
        blip_model_name: HuggingFace model name for BLIP config
        
    Returns:
        pvt: PVT vision encoder model
        blip: HuggingFace BLIP decoder model
        proj_layer: Optional projection layer (None if not present)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    combined = torch.load(checkpoint_path, map_location='cpu')
    print(f"Loaded combined checkpoint: {len(combined)} keys")
    
    # 1) Extract and load PVT weights
    print("Loading PVT encoder...")
    pvt = pvt_tiny().to(device)
    pvt_keys = {k[len('visual_encoder.'):]: v for k, v in combined.items() if k.startswith('visual_encoder.')}
    if len(pvt_keys) > 0:
        res = pvt.load_state_dict(pvt_keys, strict=False)
        print(f"  Loaded {len(pvt_keys)} keys into PVT. Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")
    else:
        print("  No PVT keys found in checkpoint; using constructor weights.")
    pvt.eval()
    
    # 2) Extract and load BLIP decoder weights
    print("Loading HuggingFace BLIP decoder...")
    blip = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(device)
    blip_keys = {k[len('text_decoder.'):]: v for k, v in combined.items() if k.startswith('text_decoder.')}
    if len(blip_keys) > 0:
        res = blip.load_state_dict(blip_keys, strict=False)
        print(f"  Loaded {len(blip_keys)} keys into BLIP. Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")
    else:
        print("  No BLIP keys found in checkpoint.")
    blip.eval()
    
    # 3) Extract and load optional projection layer
    proj_layer = None
    proj_keys = {k[len('visual_proj.'):]: v for k, v in combined.items() if k.startswith('visual_proj.')}
    if len(proj_keys) > 0:
        print("Loading projection layer...")
        # Infer dimension from weight shape
        in_dim = None
        out_dim = None
        for k, v in proj_keys.items():
            if k == 'weight':  # Not 'weight' with suffix, just 'weight'
                out_dim, in_dim = v.shape
                break
        if in_dim is not None and out_dim is not None:
            proj_layer = nn.Linear(in_dim, out_dim).to(device)
            res = proj_layer.load_state_dict(proj_keys, strict=False)
            print(f"  Loaded projection layer ({in_dim} -> {out_dim}). Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")
        else:
            print("  Could not infer projection layer dimensions")
    
    return pvt, blip, proj_layer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/blip_pvt_combined.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    pvt, blip, proj = load_combined_model(args.checkpoint, device=args.device)
    print(f"\n✅ Successfully loaded:")
    print(f"  - PVT: {type(pvt).__name__}")
    print(f"  - BLIP: {type(blip).__name__}")
    print(f"  - Projection: {type(proj).__name__ if proj else 'None'}")
