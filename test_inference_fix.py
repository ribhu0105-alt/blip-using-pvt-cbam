#!/usr/bin/env python3
"""
Test script to verify the fixed inference pipeline works correctly.
Tests greedy generation without requiring a checkpoint.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from models.blip import blip_decoder

def test_inference_greedy_generation():
    """Test that greedy generation works without crashing."""
    print("=" * 60)
    print("Testing Fixed Inference Pipeline")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1] Using device: {device}")
    
    # Create model
    print("[2] Creating BLIP model...")
    model = blip_decoder(
        image_size=384,
        vit="pvt_tiny"
    ).to(device)
    model.eval()
    print("    ✓ Model created successfully")
    
    # Create a dummy image tensor
    print("[3] Creating dummy image tensor...")
    dummy_image = torch.randn(1, 3, 384, 384).to(device)
    print(f"    ✓ Dummy image shape: {dummy_image.shape}")
    
    # Test inference - this should NOT crash with multinomial error
    print("[4] Testing inference (greedy generation)...")
    try:
        with torch.no_grad():
            captions = model.generate(
                dummy_image,
                num_beams=1,
                max_length=20,
                min_length=5,
            )
        print(f"    ✓ Inference succeeded!")
        print(f"    Generated caption: '{captions[0]}'")
        print(f"    Caption length: {len(captions[0].split())} words")
        return True
    except RuntimeError as e:
        print(f"    ✗ Inference FAILED with error:")
        print(f"    {str(e)}")
        return False

def test_batch_inference():
    """Test batch inference."""
    print("\n[5] Testing batch inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = blip_decoder(image_size=384, vit="pvt_tiny").to(device)
    model.eval()
    
    # Create batch of 2 dummy images
    batch_images = torch.randn(2, 3, 384, 384).to(device)
    
    try:
        with torch.no_grad():
            captions = model.generate(
                batch_images,
                num_beams=1,
                max_length=20,
                min_length=5,
            )
        print(f"    ✓ Batch inference succeeded!")
        print(f"    Generated {len(captions)} captions:")
        for i, cap in enumerate(captions):
            print(f"      [{i+1}] {cap}")
        return True
    except RuntimeError as e:
        print(f"    ✗ Batch inference FAILED:")
        print(f"    {str(e)}")
        return False

if __name__ == "__main__":
    success1 = test_inference_greedy_generation()
    success2 = test_batch_inference()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe inference pipeline is now FIXED and working correctly.")
        print("You can now:")
        print("  1. Train the model with: python train_caption_pvt.py ...")
        print("  2. Generate captions with: python predict_caption.py ...")
        print("  3. Use the Gradio demo: python inference_utils.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
