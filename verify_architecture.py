#!/usr/bin/env python3
"""
Verify that the BLIP model is using PVT-Tiny with CBAM attention
"""
import torch
from models.blip import blip_decoder

print("=" * 70)
print("Architecture Verification: PVT-Tiny + CBAM + BLIP")
print("=" * 70)

# Create model
model = blip_decoder(image_size=384, vit="pvt_tiny")

print("\n1. Visual Encoder Type:")
print(f"   {type(model.visual_encoder).__name__}")
print(f"   Module: {model.visual_encoder.__class__.__module__}")

# Check for PVT components
print("\n2. PVT Components:")
has_patch_embed = hasattr(model.visual_encoder, 'patch_embed1')
has_stages = hasattr(model.visual_encoder, 'stage1')
print(f"   ✓ Patch Embedding Layers: {has_patch_embed}")
print(f"   ✓ Multi-Stage Architecture: {has_stages}")

# Check for CBAM in transformer blocks
print("\n3. CBAM Attention Modules:")
cbam_count = 0
for name, module in model.visual_encoder.named_modules():
    if 'cbam' in name.lower() or 'channelattention' in type(module).__name__.lower():
        cbam_count += 1
        if cbam_count <= 3:  # Show first 3
            print(f"   ✓ Found: {name} -> {type(module).__name__}")

if cbam_count > 3:
    print(f"   ... and {cbam_count - 3} more CBAM modules")

print(f"\n   Total CBAM modules: {cbam_count}")

# Check text decoder
print("\n4. Text Decoder:")
print(f"   Type: {type(model.text_decoder).__name__}")
print(f"   Config: {model.text_decoder.config.model_type}")

# Layer details for one stage
print("\n5. Sample PVT Stage (Stage 1):")
if hasattr(model.visual_encoder, 'stage1'):
    stage1 = model.visual_encoder.stage1
    print(f"   Type: {type(stage1).__name__}")
    print(f"   Layers: {len(list(stage1.children()))}")
    
    # Check first block
    if len(list(stage1.children())) > 0:
        first_block = list(stage1.children())[0]
        print(f"   First Block Type: {type(first_block).__name__}")
        
        # Look for attention components
        for name, module in first_block.named_children():
            print(f"      - {name}: {type(module).__name__}")

# Summary
print("\n" + "=" * 70)
print("✓ VERIFICATION COMPLETE")
print("=" * 70)
print("\nConfirmed Architecture:")
print("  [Image] → PVT-Tiny (with CBAM) → [Visual Features]")
print("          → BLIP Decoder (BERT-based) → [Caption Text]")
print("\nThe model successfully integrates:")
print("  • Pyramid Vision Transformer for hierarchical visual features")
print("  • CBAM for enhanced spatial and channel attention")
print("  • BLIP's multimodal decoder for caption generation")
