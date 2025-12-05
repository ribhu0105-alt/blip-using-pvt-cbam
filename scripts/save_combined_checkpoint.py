#!/usr/bin/env python3
"""
Save a combined checkpoint containing:
 - visual encoder (PVT) weights prefixed with 'visual_encoder.'
 - text decoder (BLIP/HF) weights prefixed with 'text_decoder.'

This produces `checkpoints/blip_pvt_combined.pth` which you can distribute
as a single pretrained file that bundles your PVT weights with the HF BLIP
decoder weights.

Usage:
    python scripts/save_combined_checkpoint.py --pvt-checkpoint final_model.pth
"""
import os
import sys
import argparse
import torch

# ensure project root is on PYTHONPATH so `models` package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pvt import pvt_tiny
from transformers import BlipForConditionalGeneration


def load_pvt_visual_state(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"PVT checkpoint '{ckpt_path}' not found.")
        return {}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        raw = ckpt["model_state"]
    else:
        raw = ckpt if isinstance(ckpt, dict) else {}

    visual_state = {}
    # Common convention: keys saved as 'visual_encoder.<...>'
    for k, v in raw.items():
        if k.startswith("visual_encoder."):
            new_k = k[len("visual_encoder."):]
            visual_state[new_k] = v
    # Also allow direct keys for the PVT model
    for k, v in raw.items():
        if k not in visual_state and not k.startswith("text_decoder."):
            # Heuristic: include keys that look like pvt layers (contains 'patch' or 'pvt' or 'stages')
            if any(x in k.lower() for x in ("pvt", "patch", "stage", "cbam", "visual")):
                visual_state[k] = v

    return visual_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pvt-checkpoint", type=str, default="final_model.pth")
    parser.add_argument("--blip-model", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--out", type=str, default="checkpoints/blip_pvt_combined.pth")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) Load PVT and try to populate visual weights
    print("Loading local PVT constructor...")
    pvt = pvt_tiny()
    pvt_state = pvt.state_dict()

    visual_state = load_pvt_visual_state(args.pvt_checkpoint)
    if len(visual_state) > 0:
        print(f"Found {len(visual_state)} keys in PVT checkpoint; loading into PVT model (non-strict)")
        res = pvt.load_state_dict(visual_state, strict=False)
        print("pvt.load_state_dict result:", res)
    else:
        print("No PVT visual keys loaded; using pvt constructor weights.")

    # Refresh pvt_state
    pvt_state = pvt.state_dict()

    # 2) Load HF BLIP model (decoder)
    print("Loading HuggingFace BLIP model from:", args.blip_model)
    blip = BlipForConditionalGeneration.from_pretrained(args.blip_model)
    blip_state = blip.state_dict()

    # 3) Combine state dicts with prefixes
    combined = {}
    for k, v in pvt_state.items():
        combined[f"visual_encoder.{k}"] = v

    for k, v in blip_state.items():
        combined[f"text_decoder.{k}"] = v

    # 4) If original pvt checkpoint contained additional projection keys, include them
    raw = {}
    if os.path.exists(args.pvt_checkpoint):
        ckpt = torch.load(args.pvt_checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            raw = ckpt["model_state"]
        elif isinstance(ckpt, dict):
            raw = ckpt

    for k, v in raw.items():
        if any(k.lower().startswith(x) for x in ("proj", "visual_proj", "proj_layer", "visual_projection")):
            combined[f"visual_proj.{k}"] = v

    # 5) Save combined checkpoint
    print(f"Saving combined checkpoint to: {args.out} (total keys={len(combined)})")
    torch.save(combined, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
