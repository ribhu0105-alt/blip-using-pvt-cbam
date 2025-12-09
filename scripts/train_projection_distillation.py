"""
Train the projection layer to map PVT features to BLIP space using feature distillation.

This script:
1. Generates synthetic images
2. Extracts features using:
   - PVT encoder (custom vision model)
   - BLIP ViT encoder (teacher/reference)
3. Trains a projection layer to align PVT → BLIP feature space
4. Saves updated combined checkpoint

Why this works:
- BLIP's ViT is already trained on image captioning
- We learn a mapping: PVT_features → BLIP_features
- Then the cross-attention naturally works well
- Projection layer learns alignment in just a few epochs
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
from scripts.load_combined_checkpoint import load_combined_model


class SyntheticImageDataset(Dataset):
    """Generate random synthetic images for feature distillation."""
    
    def __init__(self, num_samples=500, image_size=384, seed=42):
        self.num_samples = num_samples
        self.image_size = image_size
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random RGB image
        img_array = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        return img


def custom_collate_fn(batch):
    """Collate function that returns list of PIL images."""
    return batch


def extract_pvt_features(model, images, device='cpu'):
    """Extract PVT features from images."""
    model.eval()
    with torch.no_grad():
        # PVT returns list of features at different scales
        features_list = model(images)
        # Use last (finest) level
        features = features_list[-1]
        # features shape: (B, C, H, W) → flatten to (B, N, C)
        if features.dim() == 4:
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, C)
    return features


def extract_blip_vit_features(model, images, device='cpu'):
    """Extract BLIP's ViT features (teacher features)."""
    model.vision_model.eval()
    with torch.no_grad():
        outputs = model.vision_model(images, return_dict=True)
        # ViT outputs sequence embeddings (B, N, 768)
        features = outputs.last_hidden_state
    return features


def train_projection_distillation(
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    num_samples=500,
    checkpoint_path='checkpoints/blip_pvt_combined.pth',
    output_path='checkpoints/blip_pvt_combined_distilled.pth',
    device='cpu'
):
    """
    Train projection layer using feature distillation.
    
    Args:
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        num_samples: Number of synthetic images
        checkpoint_path: Path to combined checkpoint
        output_path: Path to save distilled checkpoint
        device: Device to train on
    """
    
    print(f"Loading models from {checkpoint_path}...")
    pvt, blip, proj = load_combined_model(checkpoint_path, device=device)
    
    # Load teacher (BLIP with ViT)
    print("Loading teacher BLIP model...")
    teacher_blip = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Create projection layer if not exists
    if proj is None:
        print("Creating projection layer Linear(256 → 768)...")
        proj = nn.Linear(256, 768)
    
    proj = proj.to(device)
    
    # Freeze teacher
    for param in teacher_blip.parameters():
        param.requires_grad = False
    for param in pvt.parameters():
        param.requires_grad = False
    
    # Create optimizer for projection only
    optimizer = torch.optim.AdamW(proj.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Create dataset
    print(f"Creating synthetic dataset ({num_samples} images)...")
    dataset = SyntheticImageDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    print(f"\nTraining projection layer for {num_epochs} epochs on {device}...\n")
    
    proj.train()
    total_loss = 0
    num_batches = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_images in progress:
            # Process images
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            pixel_values = inputs['pixel_values']
            
            # Get features
            pvt_features = extract_pvt_features(pvt, pixel_values, device=device)  # (B, N_pvt, 256)
            blip_features = extract_blip_vit_features(teacher_blip, pixel_values, device=device)  # (B, N_blip, 768)
            
            # Project PVT features to BLIP space
            projected = proj(pvt_features)  # (B, N_pvt, 768)
            
            # Since PVT and ViT can have different sequence lengths (ViT may include a CLS token),
            # handle common cases first, else fall back to mean-pooled MSE to avoid broadcasting errors.
            Bp, Np, Cp = projected.shape
            Bt, Nt, Ct = blip_features.shape
            if Cp != Ct:
                # Dimension mismatch in channels - fall back to pooled MSE
                proj_mean = projected.mean(dim=1)
                blip_mean = blip_features.mean(dim=1)
                loss = F.mse_loss(proj_mean, blip_mean)
            else:
                # If teacher has 1 extra token (likely CLS), drop it
                if Nt == Np + 1:
                    blip_trim = blip_features[:, 1:, :]
                    loss = F.mse_loss(projected, blip_trim)
                elif Np == Nt + 1:
                    # Rare: projected longer by one token, drop first projected token
                    proj_trim = projected[:, 1:, :]
                    loss = F.mse_loss(proj_trim, blip_features)
                elif Np == Nt:
                    loss = F.mse_loss(projected, blip_features)
                else:
                    # As a safe fallback, match means to avoid shape errors
                    proj_mean = projected.mean(dim=1)
                    blip_mean = blip_features.mean(dim=1)
                    loss = F.mse_loss(proj_mean, blip_mean)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix({'loss': f"{loss.item():.6f}"})
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.6f}\n")
    
    print("✅ Training complete!")
    
    # Save distilled checkpoint
    print(f"\nSaving distilled checkpoint to {output_path}...")
    combined = {}
    
    # PVT weights
    for k, v in pvt.state_dict().items():
        combined[f"visual_encoder.{k}"] = v
    
    # BLIP decoder weights (use original, not teacher)
    for k, v in blip.state_dict().items():
        if not k.startswith("vision_model."):
            combined[f"text_decoder.{k}"] = v
    
    # Trained projection weights
    for k, v in proj.state_dict().items():
        combined[f"visual_proj.{k}"] = v
    
    torch.save(combined, output_path)
    print(f"✅ Saved {len(combined)} keys to {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1e6:.0f} MB")
    
    return combined


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    train_projection_distillation(
        num_epochs=10,
        batch_size=16,
        learning_rate=5e-3,
        num_samples=200,
        checkpoint_path='checkpoints/blip_pvt_combined.pth',
        output_path='checkpoints/blip_pvt_combined_distilled.pth',
        device=device
    )
    
    print("\n✅ Distillation training complete!")
    print("\nNext: Test inference with distilled checkpoint:")
    print("python predict_caption.py --checkpoint checkpoints/blip_pvt_combined_distilled.pth --image path/to/image.jpg")
