# BLIP + PVT-Tiny + CBAM  
### Image Captioning on Flickr8k

This repository contains a clean, fully working implementation of **BLIP (Bootstrapped Language-Image Pretraining)** modified with:

- **PVT-Tiny (Pyramid Vision Transformer)** as the visual encoder  
- **CBAM (Convolutional Block Attention Module)** added to each transformer block  
- **Flickr8k** dataset for training captions  
- **BLIP MED (BERT-based) decoder** for caption generation

The goal is a modern, hierarchical vision backbone (PVT) paired with an attention-enhanced encoder to improve the captioning capabilities of BLIP.

---

# 🚀 Features

- **Custom PVT-Tiny implementation** (from scratch)  
- **CBAM attention** integrated inside every transformer block  
- **BLIP-style multimodal training**  
- **BLIP decoder for text generation**  
- **Flickr8k training** with simple DataLoader  
- Clean training script  
- Easy inference script  
- 100% reproducible & GitHub-friendly  
- Works on **Colab (GPU)** and any Linux machine

---

# 📁 Directory Structure

