# BLIP+PVT+CBAM Colab Notebook Guide for ChatGPT

This document contains all technical information needed to generate a complete Colab notebook for training and inference.

---

## 🎯 Project Overview

**Objective:** Image captioning using BLIP decoder with PVT-Tiny (+ CBAM modifications) vision encoder.

**Architecture:**
- **Vision Encoder:** PVT-Tiny (13M params) with CBAM attention modules
  - Output dimension: 256
  - Input image size: 384x384
  - Outputs spatial features: (B, N, 256) where N = 576 patches
  
- **Projection Layer:** Linear(256 → 768)
  - Maps PVT features to BLIP decoder hidden size
  - Trained via cross-attention with captions
  
- **Text Decoder:** BERT-base-uncased from HuggingFace
  - Pretrained language model with cross-attention
  - Autoregressively generates captions
  - Prompt: "a picture of " (prepended to all captions)

---

## 📦 Repository Structure

```
blip-using-pvt-cbam/
├── models/
│   ├── blip.py              # BLIP_Decoder class (vision + text)
│   ├── pvt.py               # PVT-Tiny model definition
│   ├── vit.py               # ViT backbone (optional)
│   ├── med.py               # MED BERT utilities
│   └── utils.py             # Common utilities
│
├── scripts/
│   ├── load_combined_checkpoint.py      # Load checkpoint: (pvt, blip, proj)
│   ├── patch_blip_with_pvt.py           # Patch HF BLIP with PVT wrapper
│   ├── finetune_combined.py             # Quick 1-epoch init on synthetic data
│   ├── train_projection_distillation.py # Train projection via feature distillation
│   └── batch_inference_distilled.py     # Batch inference script
│
├── train_caption_pvt.py      # **MAIN TRAINING SCRIPT**
├── predict_caption.py        # Single image inference
├── requirements.txt          # Python dependencies
│
├── checkpoints/
│   ├── blip_pvt_combined.pth              # Base combined checkpoint
│   ├── blip_pvt_combined_distilled.pth    # After projection distillation
│   └── blip_pvt_combined_finetuned.pth    # After light finetuning
│
├── data/
│   ├── flickr.py      # Flickr dataset utils
│   └── transforms.py  # Image transforms
│
└── configs/
    └── med_config.json  # BERT config
```

---

## 🔧 Installation & Setup

### Python Version
- Python 3.8+
- Tested on Python 3.10, 3.11, 3.12

### Key Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.57.2
Pillow
numpy
tqdm
```

### Installation Steps (for Colab)
```bash
# 1. Clone repo
!git clone https://github.com/ribhu0105-alt/blip-using-pvt-cbam.git
%cd blip-using-pvt-cbam

# 2. Install dependencies
!pip install -q -r requirements.txt

# 3. Verify
!python test_import.py
```

---

## 📊 Dataset Format

**Required format:** TSV (tab-separated values)

**File structure:**
```
dataset/
├── images/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   ├── image_0003.jpg
│   └── ... (up to 30,000 images)
│
└── captions.txt
```

**captions.txt format:**
```
image_0001.jpg	a dog running in the park
image_0002.jpg	a cat sleeping on a bed
image_0003.jpg	people sitting on a bench
image_0004.jpg	a sunset over mountains
image_0005.jpg	a forest path in nature
...
```

**Rules:**
- One caption per line
- Format: `filename<TAB>caption_text`
- Image filename must include extension (.jpg, .png)
- No empty lines
- Captions should be 5-20 words for best results

---

## 🚀 Training Script: `train_caption_pvt.py`

### Purpose
Full training of the BLIP+PVT pipeline on image-caption dataset.

### Command
```bash
python train_caption_pvt.py \
    --image_root /path/to/images \
    --caption_file /path/to/captions.txt \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --epochs 8 \
    --lr 1e-5 \
    --image_size 384 \
    --use_amp \
    --num_workers 0 \
    --log_freq 100 \
    --ckpt_freq 1000
```

### Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image_root` | str | required | Path to images folder |
| `--caption_file` | str | required | Path to captions.txt |
| `--output_dir` | str | `./output` | Where to save checkpoints |
| `--batch_size` | int | 8 | Batch size (use 4 for free Colab, 8 for Pro) |
| `--epochs` | int | 1 | Number of training epochs |
| `--lr` | float | 1e-5 | Learning rate |
| `--image_size` | int | 384 | Input image size (384x384) |
| `--use_amp` | flag | True | Enable automatic mixed precision |
| `--weight_decay` | float | 0.05 | L2 regularization |
| `--grad_clip` | float | 1.0 | Gradient clipping (0=disabled) |
| `--num_workers` | int | 4 | Data loader workers (use 0 in Colab) |
| `--log_freq` | int | 100 | Print loss every N steps |
| `--ckpt_freq` | int | 1000 | Save checkpoint every N steps |
| `--resume_from` | str | None | Path to checkpoint to resume training |
| `--seed` | int | 42 | Random seed |

### Output Files

After training, checkpoint is saved to: `{output_dir}/final_model.pth`

**Checkpoint structure:**
```python
{
    'visual_encoder.layer1.conv.weight': ...,  # PVT weights (178 keys)
    'visual_encoder.layer2.conv.weight': ...,
    ...
    'text_decoder.bert.embeddings.word_embeddings.weight': ...,  # BLIP weights (324 keys)
    'text_decoder.bert.encoder.layer.0.attention.self.query.weight': ...,
    ...
    'visual_proj.weight': ...,  # Projection (2 keys)
    'visual_proj.bias': ...,
}
```

**Total keys:** ~504-506

### Training Tips

**For Colab Free Tier (T4 GPU):**
- Use `batch_size=4`
- Use `--use_amp` (enabled by default)
- Use `num_workers=0`
- Train for 5-10 epochs minimum
- Expected training time: ~1-2 hours per epoch for 30K images

**For Colab Pro (V100/A100 GPU):**
- Use `batch_size=8` or even 16
- Expected training time: ~30-45 minutes per epoch

**Dataset size recommendations:**
- Minimum: 1K images (quick testing)
- Good: 10K images (reasonable quality)
- Better: 30K images (good quality)
- Best: 100K+ images (production quality)

---

## 🎯 Inference Scripts

### Option 1: Single Image Inference
**Script:** `scripts/patch_blip_with_pvt.py`

```bash
python scripts/patch_blip_with_pvt.py \
    --checkpoint ./checkpoints/final_model.pth \
    --image_url "https://example.com/image.jpg" \
    --device cuda \
    --max_length 50 \
    --num_beams 5
```

**Output:**
```
=== Caption ===
a dog running in a grassy field
==============
```

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | required | Path to trained checkpoint |
| `--image_url` | str | optional | URL to image (download from web) |
| `--image_path` | str | optional | Local path to image file |
| `--device` | str | cuda | Device: 'cuda' or 'cpu' |
| `--max_length` | int | 50 | Maximum caption length |
| `--num_beams` | int | 5 | Beam search width (1-5) |
| `--save_checkpoint` | str | optional | Path to save checkpoint with projection |

### Option 2: Batch Inference
**Script:** `scripts/batch_inference_distilled.py`

```bash
python scripts/batch_inference_distilled.py \
    --checkpoint ./checkpoints/final_model.pth \
    --out results.txt \
    --device cuda
```

**Output file format:**
```
URL: https://example.com/image1.jpg
Caption: a dog running in a park

URL: https://example.com/image2.jpg
Caption: a cat sitting on a bench

...
```

### Decoding Parameters for Better Captions

If captions are noisy or repetitive, adjust these in inference:

```python
caption_ids = model.generate(
    pixel_values,
    num_beams=5,              # Higher = better quality, slower (3-5 recommended)
    max_length=50,            # Maximum words in caption
    min_length=8,             # Minimum words (helps avoid short captions)
    repetition_penalty=2.0,   # Higher = less repetition (1.0-2.5 recommended)
    no_repeat_ngram_size=3,   # Prevent n-gram repetition (3-4 recommended)
    length_penalty=1.0,       # Favor longer outputs if > 1.0
    early_stopping=True,      # Stop when all beams finish
)
```

---

## 📝 Model Loading & Utilities

### Load Trained Checkpoint

```python
from scripts.load_combined_checkpoint import load_combined_model
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pvt, blip, proj = load_combined_model(
    'checkpoints/final_model.pth',
    device=device
)
```

**Returns:**
- `pvt`: PVT-Tiny model (trained)
- `blip`: HuggingFace BLIP decoder (trained)
- `proj`: Linear projection layer (trained)

### Patch BLIP with PVT Wrapper

```python
from scripts.finetune_combined import PVTVisionEncoder
import torch.nn as nn

pvt_wrapper = PVTVisionEncoder(pvt, proj)
blip.vision_model = pvt_wrapper
blip.eval()

# Now blip is ready for inference
caption_ids = blip.generate(pixel_values, ...)
```

---

## 🐛 Troubleshooting

### Problem: Out of Memory (OOM)
**Solution:**
```bash
# Reduce batch size
--batch_size 2

# Or reduce num_workers
--num_workers 0

# Or enable gradient accumulation (manually in code)
```

### Problem: Training is very slow
**Solution:**
```bash
# Reduce num_workers
--num_workers 0

# Enable AMP (already default)
--use_amp

# Reduce image size (if acceptable for quality)
--image_size 256  # Instead of 384
```

### Problem: Repetitive captions during inference
**Solution:**
```bash
# Increase repetition penalty
--repetition_penalty 2.0

# Increase no_repeat_ngram_size
--no_repeat_ngram_size 4

# Reduce num_beams (paradoxically helps sometimes)
--num_beams 3
```

### Problem: Captions too short
**Solution:**
```bash
# Increase min_length
model.generate(..., min_length=10)

# Increase max_length
model.generate(..., max_length=60)

# Increase length_penalty
model.generate(..., length_penalty=1.5)
```

### Problem: Training loss not decreasing
**Solution:**
- Check dataset format (captions.txt should be valid TSV)
- Verify images load correctly
- Try reducing learning rate: `--lr 5e-6`
- Train for more epochs: `--epochs 10`

---

## 📌 Key Model Properties

### PVT-Tiny Vision Encoder
- **Architecture:** 4-stage pyramid vision transformer
- **Depth:** 4, 6, 20, 4 blocks per stage
- **Embed dims:** 64, 128, 320, 512
- **Output dimension:** 256 (from stage 4)
- **CBAM modules:** Added to each stage for channel-spatial attention
- **Parameters:** ~13M

### BERT-Base Decoder
- **Type:** BertLMHeadModel from HuggingFace
- **Vocab size:** 30522
- **Hidden size:** 768
- **Num layers:** 12
- **Attention heads:** 12
- **Cross-attention:** Enabled (attends to vision features)
- **Parameters:** ~110M

### Projection Layer
- **Type:** Linear(256 → 768)
- **Role:** Map PVT features to BLIP hidden space
- **Parameters:** ~196K

### Total Model Parameters
- **Vision:** 13M (PVT)
- **Text:** 110M (BERT)
- **Projection:** 196K
- **Total:** ~123M (reasonable for Colab)

---

## 🔗 Important File Locations

| File | Purpose | Key Function |
|------|---------|--------------|
| `models/blip.py` | BLIP decoder class | `BLIP_Decoder`, `blip_decoder()` |
| `models/pvt.py` | PVT vision encoder | `pvt_tiny()` |
| `scripts/load_combined_checkpoint.py` | Checkpoint loader | `load_combined_model()` |
| `scripts/patch_blip_with_pvt.py` | Inference wrapper | Command-line inference |
| `scripts/finetune_combined.py` | PVT wrapper class | `PVTVisionEncoder` |
| `train_caption_pvt.py` | **Main training script** | `train()` function |
| `predict_caption.py` | Simple inference | Standalone inference |

---

## ✅ Complete Colab Workflow

### Step 1: Setup (2-3 minutes)
```bash
!git clone https://github.com/ribhu0105-alt/blip-using-pvt-cbam.git
%cd blip-using-pvt-cbam
!pip install -q -r requirements.txt
!python test_import.py
```

### Step 2: Prepare Dataset (5-10 minutes)
- Mount Google Drive: `drive.mount('/content/drive')`
- Or upload ZIP file and extract
- Verify structure:
  - Images in `images/` folder
  - Captions in `captions.txt`

### Step 3: Train (30 minutes - 2 hours depending on dataset size)
```bash
!python train_caption_pvt.py \
    --image_root /content/drive/MyDrive/images \
    --caption_file /content/drive/MyDrive/captions.txt \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --epochs 8 \
    --use_amp
```

### Step 4: Inference (1-2 minutes)
```bash
!python scripts/patch_blip_with_pvt.py \
    --checkpoint ./checkpoints/final_model.pth \
    --image_url "https://example.com/image.jpg" \
    --device cuda
```

### Step 5: Download Results (1 minute)
```python
from google.colab import files
files.download('./checkpoints/final_model.pth')
```

---

## 📊 Expected Performance

### Training Metrics
- **Initial loss:** ~7-10 (batch size 4-8)
- **After 1 epoch:** ~2-4 (depends on data quality)
- **After 5 epochs:** ~0.5-1.5 (good convergence)
- **After 10 epochs:** ~0.2-0.8 (excellent)

### Caption Quality
- **Untrained projection:** Noisy, fragmented words
- **After 1 epoch:** Basic captions, some grammar issues
- **After 5 epochs (10K images):** Good quality, sensible descriptions
- **After 10 epochs (30K images):** Fluent, accurate captions
- **After 15+ epochs (100K images):** Production-ready

### Inference Speed
- **Single image (GPU):** ~200-500ms
- **Single image (CPU):** ~2-5 seconds
- **Batch of 10 (GPU):** ~1-2 seconds

---

## 🎓 References

**Models:**
- PVT: "Pyramid Vision Transformer for Image Classification" (Wang et al., 2021)
- BLIP: "BLIP: Bootstrapping Language-Image Pre-training" (Li et al., 2022)
- BERT: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)

**Datasets:**
- Flickr30K: 31K images, 158K captions
- COCO: 118K images, 591K captions
- Conceptual Captions: 3M+ images

---

## 📞 Common Questions

**Q: Can I use a GPU from Google Colab free tier?**
A: Yes, T4 GPU available. Enable in Runtime > Change runtime type > GPU.

**Q: What if I don't have a dataset?**
A: Create synthetic data (as done in scripts) or use public datasets (Flickr30K, COCO).

**Q: How do I get better captions?**
A: Train longer (more epochs), use more data (30K+), adjust generation parameters.

**Q: Can I fine-tune a pretrained checkpoint?**
A: Yes, use `--resume_from path/to/checkpoint.pth`

**Q: What if my images are not 384x384?**
A: They're automatically resized. Any size works, but 384x384 is optimal.

**Q: Can I use this for other languages?**
A: Partially. BERT supports multilingual, but training data should be in that language.

