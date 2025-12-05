#!/usr/bin/env python3
"""
Download a sample image and run the BLIP model to generate a caption.
Usage: python run_inference_sample.py [image_path]
"""
import sys
import torch
from PIL import Image
from torchvision import transforms
from models.blip import blip_decoder

device = "cuda" if torch.cuda.is_available() else "cpu"

model = blip_decoder(image_size=384, vit="pvt_tiny").to(device)
model.eval()

img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
print(f"Using image: {img_path}")

img = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
img_t = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    caption = model.generate(
        img_t,
        num_beams=1,
        max_length=30,
        min_length=5,
    )

print("Generated caption:\n", caption[0])
