import sys, os

# ---- Disable tokenizer multiprocessing (Fix 5 - prevents deadlock in Colab) ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- Add project root automatically ----
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.abspath(PROJECT_ROOT))

print("Project root added:", PROJECT_ROOT)



import os
import argparse
import random
from PIL import Image
import json



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from models.blip import blip_decoder


# =========================
# Flickr8k Dataset
# =========================
class Flickr8kDataset(Dataset):
    def __init__(self, root, captions_file, transform=None):
        self.root = root
        self.transform = transform

        self.data = []
        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                     continue

                img = None
                cap = None

                # 1) Tab format  →  "image.jpg<TAB>caption"
                if "\t" in line:
                    img, cap = line.split("\t", 1)

                # 2) Format: "image.jpg, caption"
                elif ".jpg," in line:
                    img, cap = line.split(".jpg,", 1)
                    img = img.strip() + ".jpg"
                    cap = cap.strip()

                # 3) Format: "image.jpg caption"
                elif " " in line:
                    img, cap = line.split(" ", 1)

                else:
                    # cannot parse → skip
                    continue

                # Clean filename
                img = img.split("#")[0].strip().strip(",").strip()
                cap = cap.strip()

                # Validate extension
                if not img.lower().endswith(".jpg"):
                    continue

                self.data.append((img, cap))


        print(f"[Dataset] Loaded {len(self.data)} caption pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        image_path = os.path.join(self.root, img_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, caption


# =========================
# Learning rate warmup
# =========================
def get_warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        return min(1.0, step / warmup_steps)  # increase LR linearly
    return LambdaLR(optimizer, lr_lambda)


# =========================
# Main training routine
# =========================
def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)
    
    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- Data transform (with augmentation for 384x384)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---- Load dataset
    dataset = Flickr8kDataset(
        root=args.image_root,
        captions_file=args.caption_file,
        transform=transform
    )
    print("Loaded", len(dataset), "caption pairs")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )

    # ---- Load BLIP decoder model
    print("Loading BLIP model...")
    model = blip_decoder(
        image_size=args.image_size,
        vit="pvt_tiny"
    ).to(device)

    model.train()

    # ---- Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    total_steps = len(dataloader) * args.epochs
    scheduler = get_warmup_scheduler(optimizer, warmup_steps=int(0.1 * total_steps))
    
    # ---- Mixed precision training
    scaler = GradScaler(enabled=args.use_amp)

    # ---- Checkpoint management
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('epoch', 0)
        start_step = checkpoint.get('step', 0)
        print(f"Resumed from epoch {start_epoch}, step {start_step}")

    # ---- Training loop
    step = start_step
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)

            # ---- Forward pass with mixed precision
            with autocast(enabled=args.use_amp):
                loss = model(images, captions)
            
            # ---- Backward pass
            scaler.scale(loss).backward()
            
            # ---- Gradient clipping
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # ---- Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1

            # ---- Logging
            if step % args.log_freq == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                lr = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch:2d} Step {step:6d}] Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr:.2e}")

            # ---- Save checkpoint
            if step % args.ckpt_freq == 0 and step > 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pth")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'args': args
                }, ckpt_path)
                print(f"  → Saved checkpoint: {ckpt_path}")

            step += 1

        # ---- End-of-epoch stats
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n[Epoch {epoch} Complete] Avg Loss: {avg_epoch_loss:.4f}\n")

    # ---- Final save
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print("Training complete. Saved:", final_path)



# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BLIP caption model with PVT backbone")

    parser.add_argument("--image_root", type=str, required=True,
                        help="Folder with Flickr8k images")
    parser.add_argument("--caption_file", type=str, required=True,
                        help="Flickr8k captions.txt file")
    
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=384,
                        help="Input image size (will be resized to HxW)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay for AdamW")
    
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision (AMP)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold (0=disabled)")
    
    parser.add_argument("--log_freq", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--ckpt_freq", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    train(args)
