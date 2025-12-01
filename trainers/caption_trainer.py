# trainers/caption_trainer.py
"""
Caption Trainer (Modular Training Framework)
============================================

This module implements a clean, reusable training framework for
image captioning using BLIP + PVT-Tiny (or any vision backbone).

The goal is to separate:
 - data loading
 - model initialization
 - training steps
 - checkpoint management
 - logging

from the actual "train_caption_pvt.py" script.

This makes the repo feel professional and scalable.

Usage:

    trainer = CaptionTrainer(model, optimizer, scheduler, device)
    trainer.train_one_epoch(dataloader, epoch)
"""

import os
import torch
from torch.nn.utils import clip_grad_norm_


class CaptionTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        device="cuda",
        grad_clip=1.0,
        save_dir="output"
    ):
        """
        Args:
            model: nn.Module (BLIP+PVT model)
            optimizer: torch.optim optimizer
            scheduler: optional LR scheduler
            device: GPU/CPU
            grad_clip: gradient clipping
            save_dir: directory to save checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # One training step
    # ------------------------------------------------------------------
    def step(self, images, captions):
        """
        Performs one forward+backward pass.

        images: (B, 3, H, W)
        captions: list[str]
        """
        images = images.to(self.device)

        loss = self.model(images, captions)
        loss.backward()

        # Gradient clipping improves stability
        if self.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Train one full epoch
    # ------------------------------------------------------------------
    def train_one_epoch(self, dataloader, epoch, log_interval=100):
        self.model.train()

        loss_sum = 0
        steps = 0

        for batch_idx, (images, captions) in enumerate(dataloader):
            loss = self.step(images, captions)
            loss_sum += loss
            steps += 1

            if batch_idx % log_interval == 0:
                print(f"[Epoch {epoch}] Step {batch_idx}/{len(dataloader)} Loss: {loss:.4f}")

        avg_loss = loss_sum / max(steps, 1)
        print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")
        return avg_loss

    # ------------------------------------------------------------------
    # Save model checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self, name="checkpoint.pth"):
        path = os.path.join(self.save_dir, name)
        torch.save(self.model.state_dict(), path)
        print(f"[Checkpoint] Saved to {path}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        print(f"[Checkpoint] Loaded from {path}")
