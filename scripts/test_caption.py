# scripts/test_caption.py
"""
Sanity Test Script for BLIP-PVT Captioning
==========================================

This script:
 - Loads a single batch from Flickr8k dataset
 - Runs a forward pass through the BLIP+PVT model
 - Generates a caption for 1 sample
 - Prints shapes + caption output

Usage:
    python scripts/test_caption.py \
        --image_root data/Images \
        --caption_file data/captions.txt \
        --checkpoint output/final_model.pth
"""

import argparse
import torch
from PIL import Image

from data.flickr import Flickr8kDataset
from data.transforms import get_transform
from models.blip import blip_decoder
from models.vision_backbones import list_backbones


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    print("\n[Backbones available]:")
    for k, v in list_backbones().items():
        print(f" - {k}: {v}")

    # -------------------------------------------
    # Load Dataset
    # -------------------------------------------
    transform = get_transform(train=False, image_size=args.image_size)

    dataset = Flickr8kDataset(
        image_root=args.image_root,
        captions_file=args.caption_file,
        transform=transform
    )

    print(f"[Dataset] Loaded {len(dataset)} caption pairs\n")

    # Take one sample
    img, caption = dataset[0]
    print("[Sample Caption]:", caption)

    img = img.unsqueeze(0).to(device)  # (1, 3, H, W)

    # -------------------------------------------
    # Load Model
    # -------------------------------------------
    print("\n[Loading Model] ...")
    model = blip_decoder(
        image_size=args.image_size,
        vit="pvt_tiny",
        med_config=args.med_config
    ).to(device)

    # Optional: load weights
    if args.checkpoint:
        print("[Loading Checkpoint]:", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=False)

    model.eval()

    # -------------------------------------------
    # Generate Caption
    # -------------------------------------------
    print("\n[Running Inference] ...")
    with torch.no_grad():
        generated = model.generate(img, num_beams=3, max_length=30, min_length=5)

    print("\n[Generated Caption]:")
    print(" ->", generated[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--caption_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--med_config", type=str, default="configs/med_config.json")
    parser.add_argument("--image_size", type=int, default=384)

    args = parser.parse_args()
    main(args)
