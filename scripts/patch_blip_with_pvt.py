#!/usr/bin/env python3
"""Patch HuggingFace BLIP model to use PVT-Tiny(+CBAM) and generate captions.

Usage examples:

# Generate caption from URL (prints caption)
python scripts/patch_blip_with_pvt.py \
  --checkpoint checkpoints/blip_pvt_combined_finetuned.pth \
  --image_url https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg

# Generate caption from local file and optionally save a checkpoint with projection
python scripts/patch_blip_with_pvt.py \
  --checkpoint checkpoints/blip_pvt_combined_finetuned.pth \
  --image_path /path/to/image.jpg \
  --save_checkpoint checkpoints/blip_pvt_with_proj.pth
"""

import argparse
import os
import io
import urllib.request
from PIL import Image
import torch
import torch.nn as nn
import sys
# ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.load_combined_checkpoint import load_combined_model
from scripts.finetune_combined import PVTVisionEncoder
from transformers import BlipProcessor


def download_image(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        data = r.read()
    return Image.open(io.BytesIO(data)).convert('RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image_url', type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_checkpoint', type=str, default=None, help='Optional path to save a combined checkpoint with projection weights')
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--num_beams', type=int, default=5)
    args = parser.parse_args()

    if args.image_path is None and args.image_url is None:
        raise ValueError('Provide --image_path or --image_url')

    device = args.device
    print(f'Using device: {device}')

    # Load models from combined checkpoint
    print('Loading combined checkpoint...')
    pvt, blip, proj = load_combined_model(args.checkpoint, device=device)
    print('proj from checkpoint:', proj)

    # Create projection if missing
    if proj is None:
        print('No projection layer found in checkpoint; creating (256->768) and initializing')
        proj = nn.Linear(256, 768)
        nn.init.xavier_uniform_(proj.weight)
        nn.init.zeros_(proj.bias)

    # Put wrapper on model
    pvt_wrapper = PVTVisionEncoder(pvt, proj).to(device)
    blip.vision_model = pvt_wrapper
    blip = blip.to(device)
    blip.eval()

    # Load/process image
    if args.image_path:
        img = Image.open(args.image_path).convert('RGB')
    else:
        print(f'Downloading: {args.image_url}')
        img = download_image(args.image_url)
    img = img.resize((384,384))

    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    inputs = processor(images=[img], return_tensors='pt').to(device)

    # Generate
    with torch.no_grad():
        caption_ids = blip.generate(
            inputs['pixel_values'],
            num_beams=args.num_beams,
            max_length=args.max_length,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3
        )
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)

    print('\n=== Caption ===')
    print(caption)
    print('==============')

    # Optionally save combined checkpoint including projection weights
    if args.save_checkpoint:
        out_path = args.save_checkpoint
        print(f'Saving combined checkpoint with projection to: {out_path}')
        combined = {}
        for k, v in pvt.state_dict().items():
            combined[f'visual_encoder.{k}'] = v
        for k, v in blip.state_dict().items():
            if not k.startswith('vision_model.'):
                combined[f'text_decoder.{k}'] = v
        for k, v in proj.state_dict().items():
            combined[f'visual_proj.{k}'] = v
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(combined, out_path)
        print('Saved checkpoint keys:', len(combined))

if __name__ == '__main__':
    main()
