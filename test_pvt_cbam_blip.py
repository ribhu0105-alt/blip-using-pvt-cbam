#!/usr/bin/env python3
"""
Test script to demonstrate PVT+CBAM+BLIP model inference
Shows the complete architecture working end-to-end
"""
import torch
from PIL import Image
from torchvision import transforms
from models.blip import blip_decoder

print("=" * 60)
print("PVT-Tiny + CBAM + BLIP Image Captioning Demo")
print("=" * 60)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Initialize model with PVT-Tiny backbone
print("\nInitializing BLIP model with PVT-Tiny + CBAM...")
model = blip_decoder(
    image_size=384, 
    vit="pvt_tiny"  # Using PVT-Tiny with CBAM
).to(device)
model.eval()
print("✓ Model loaded successfully!")

# Architecture info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Test on demo images
import os
demo_images = sorted([
    f for f in os.listdir("demo_data/images") 
    if f.endswith(".jpg")
])

print("\n" + "=" * 60)
print("Running Inference on Demo Images")
print("=" * 60)

for img_name in demo_images:
    img_path = f"demo_data/images/{img_name}"
    
    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate caption with different beam search settings
    with torch.no_grad():
        # Greedy decoding (fast)
        caption_greedy = model.generate(
            img_tensor,
            num_beams=1,
            max_length=30,
            min_length=5,
        )[0]
        
        # Beam search (better quality)
        caption_beam = model.generate(
            img_tensor,
            num_beams=3,
            max_length=30,
            min_length=5,
        )[0]
    
    print(f"\n📷 {img_name}")
    print(f"   Greedy: {caption_greedy}")
    print(f"   Beam-3: {caption_beam}")

print("\n" + "=" * 60)
print("✓ Inference Complete!")
print("=" * 60)
print("\nNote: The model is untrained, so captions are random.")
print("After training on real data, it will generate meaningful captions.")
print("\nArchitecture Components:")
print("  ✓ PVT-Tiny: Pyramid Vision Transformer (hierarchical backbone)")
print("  ✓ CBAM: Convolutional Block Attention Module (spatial + channel)")
print("  ✓ BLIP: Bootstrapped Language-Image Pretraining (decoder)")
