#!/usr/bin/env python3
"""Run batch inference using distilled combined checkpoint and save captions.

Usage: python scripts/batch_inference_distilled.py --checkpoint <path> [--out outputs/results.txt]
"""
import argparse
import os
import io
import urllib.request
from PIL import Image
import torch
import torch.nn as nn
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.load_combined_checkpoint import load_combined_model
from scripts.finetune_combined import PVTVisionEncoder
from transformers import BlipProcessor

DEFAULT_URLS = [
    'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg',
    'https://raw.githubusercontent.com/fastai/fastbook/master/images/cat.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/2/2c/Rotating_earth_%28large%29.gif',
    'https://raw.githubusercontent.com/zhanghang1989/ResNeXt/master/images/ILSVRC2012_val_00000023.JPEG',
]


def download_image(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    return Image.open(io.BytesIO(data)).convert('RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out', type=str, default='outputs/batch_inference_results.txt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--num_beams', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = args.device
    print(f'Using device: {device}')

    print('Loading combined checkpoint...')
    pvt, blip, proj = load_combined_model(args.checkpoint, device=device)
    if proj is None:
        print('No projection found in checkpoint; creating (256->768)')
        proj = nn.Linear(256, 768)
        nn.init.xavier_uniform_(proj.weight)
        nn.init.zeros_(proj.bias)

    pvt_wrapper = PVTVisionEncoder(pvt, proj).to(device)
    blip.vision_model = pvt_wrapper
    blip = blip.to(device)
    blip.eval()

    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')

    results = []
    for i, url in enumerate(DEFAULT_URLS):
        print(f'[{i+1}/{len(DEFAULT_URLS)}] Downloading: {url}')
        try:
            img = download_image(url)
        except Exception as e:
            print('Failed to download:', e)
            results.append({'url': url, 'error': str(e)})
            continue
        img = img.resize((384,384))
        inputs = processor(images=[img], return_tensors='pt').to(device)
        with torch.no_grad():
            caption_ids = blip.generate(
                inputs['pixel_values'],
                num_beams=args.num_beams,
                max_length=args.max_length,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
            )
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)
        print('Caption:', caption)
        results.append({'url': url, 'caption': caption})

    print(f'Writing results to: {args.out}')
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in results:
            if 'error' in r:
                f.write(f"URL: {r['url']}\nError: {r['error']}\n\n")
            else:
                f.write(f"URL: {r['url']}\nCaption: {r['caption']}\n\n")

    print('Done.')


if __name__ == '__main__':
    main()
