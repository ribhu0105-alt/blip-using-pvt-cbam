import sys, os

# ---- Add project root automatically ----
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.abspath(PROJECT_ROOT))

print("Project root added:", PROJECT_ROOT)



import os
import argparse
import random
from PIL import Image



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

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

    # ---- Data transform (simple)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
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
        num_workers=1,
        drop_last=True,
        pin_memory=True
    )

    # ---- Load BLIP + PVT decoder model
    model = blip_decoder(
        image_size=args.image_size,
        vit="pvt_tiny",
        med_config=args.med_config
    ).to(device)

    model.train()

    # ---- Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = get_warmup_scheduler(optimizer, warmup_steps=2000)

    # ---- Training loop
    step = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for images, captions in dataloader:

            images = images.to(device)

            loss = model(images, captions)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

            if step % 2000 == 0 and step > 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print("Saved", ckpt_path)

            step += 1

    # Final save
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print("Training complete. Saved:", final_path)



# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root", type=str, required=True,
                        help="Folder with Flickr8k images")
    parser.add_argument("--caption_file", type=str, required=True,
                        help="Flickr8k captions.txt file")
    parser.add_argument("--med_config", type=str, default="configs/med_config.json")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    train(args)
