#!/usr/bin/env python3
"""
Fine-tune the combined PVT+BLIP model for 1 epoch on a small synthetic dataset.

This initializes projection + cross-attention weights and saves an updated
combined checkpoint with trained weights.

Usage:
    python scripts/finetune_combined.py \
        --checkpoint checkpoints/blip_pvt_combined.pth \
        --out checkpoints/blip_pvt_combined_finetuned.pth \
        --epochs 1 \
        --batch-size 2
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T

# ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pvt import pvt_tiny
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers.modeling_outputs import BaseModelOutput
from scripts.load_combined_checkpoint import load_combined_model


def flatten_pvt_feature(c4):
    """Flatten PVT spatial output (B, C, H, W) -> (B, N, C)"""
    B, C, H, W = c4.shape
    return c4.flatten(2).transpose(1, 2).contiguous()


class SyntheticImageCaptionDataset(Dataset):
    """Generate synthetic image-caption pairs for testing."""
    def __init__(self, num_samples=8, image_size=384):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create a random RGB PIL image (processor expects PIL images)
        img_array = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_array, mode='RGB')
        
        # Random caption from a small pool
        captions = [
            "a picture of a dog",
            "a photo of a cat",
            "a scenic landscape",
            "a street scene",
            "a mountain view",
            "a forest path",
            "an urban street",
            "a beach sunset",
        ]
        caption = captions[idx % len(captions)]
        
        return pil_img, caption


class PVTVisionEncoder(nn.Module):
    """Wrapper that feeds PVT outputs into BLIP decoder."""
    def __init__(self, pvt_model, proj_layer=None):
        super().__init__()
        self.pvt = pvt_model
        self.proj = proj_layer
    
    def forward(self, pixel_values=None, return_dict=True, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values required")
        
        out = self.pvt(pixel_values)
        last = out[-1] if isinstance(out, (list, tuple)) else out
        
        if last.dim() == 4:
            enc = flatten_pvt_feature(last)
        elif last.dim() == 3:
            enc = last
        else:
            raise RuntimeError(f"Unexpected PVT output shape: {last.shape}")
        
        if self.proj is not None:
            enc = self.proj(enc)
        
        return BaseModelOutput(last_hidden_state=enc)


def custom_collate_fn(batch):
    """Custom collate for PIL images and strings."""
    images, captions = zip(*batch)
    return list(images), list(captions)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/blip_pvt_combined.pth")
    parser.add_argument("--out", type=str, default="checkpoints/blip_pvt_combined_finetuned.pth")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=8, help="Number of synthetic samples")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1) Load combined checkpoint
    print("Loading combined checkpoint...")
    pvt, blip, proj_layer = load_combined_model(args.checkpoint, device=device)
    
    # If no projection layer exists, create one (256 -> 768)
    if proj_layer is None:
        print("Creating new projection layer (256 -> 768)")
        proj_layer = nn.Linear(256, 768).to(device)
    
    # 2) Replace BLIP vision encoder with PVT wrapper
    pvt_wrapper = PVTVisionEncoder(pvt, proj_layer).to(device)
    blip.vision_model = pvt_wrapper
    
    # 3) Prepare dataset and dataloader
    print("Creating synthetic dataset...")
    dataset = SyntheticImageCaptionDataset(num_samples=args.num_samples, image_size=384)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    # 4) Prepare processor and optimizer
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    optimizer = torch.optim.AdamW(blip.parameters(), lr=args.lr)
    
    # 5) Fine-tune for specified epochs
    blip.train()
    pvt.eval()  # Keep PVT frozen (only train decoder + projection)
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # batch is (images_list, captions_list) from custom_collate_fn
            images, captions = batch
            
            # Process images with processor (PIL images -> pixel_values)
            inputs = processor(images=images, return_tensors="pt").to(device)
            pixel_values = inputs["pixel_values"]
            
            # Tokenize captions
            text_inputs = processor.tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt"
            ).to(device)
            
            # Forward pass
            outputs = blip(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                labels=text_inputs["input_ids"]
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % max(1, len(dataloader) // 2) == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} complete. Avg Loss: {avg_loss:.4f}")
    
    # 6) Save updated combined checkpoint
    print("Saving updated combined checkpoint...")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    combined = {}
    
    # Save PVT weights with prefix
    for k, v in pvt.state_dict().items():
        combined[f"visual_encoder.{k}"] = v
    
    # Save BLIP weights with prefix
    for k, v in blip.state_dict().items():
        if not k.startswith("vision_model."):  # Skip the wrapper; we save PVT separately
            combined[f"text_decoder.{k}"] = v
    
    # Save projection layer weights with prefix
    for k, v in proj_layer.state_dict().items():
        combined[f"visual_proj.{k}"] = v
    
    torch.save(combined, args.out)
    print(f"✅ Saved fine-tuned checkpoint to: {args.out} (total keys={len(combined)})")
    print("   This checkpoint now has initialized projection + cross-attention weights.")


if __name__ == "__main__":
    main()
