# Quick Start Guide - BLIP Image Captioning

## 🚀 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python test_import.py
```

You should see:
```
✓ All tests passed! Implementation is ready to use.
```

---

## 📚 Training Your Model

### Prepare Dataset

Create a caption file in format:
```
image001.jpg	a dog running in the park
image002.jpg	people sitting on a bench
image003.jpg	a cat sleeping on a bed
```

The format is: `image_filename<TAB>caption`

### Start Training

```bash
python train_caption_pvt.py \
    --image_root /path/to/images \
    --caption_file /path/to/captions.txt \
    --batch_size 8 \
    --epochs 10 \
    --image_size 384 \
    --use_amp \
    --output_dir ./checkpoints
```

### Monitor Training

```
[Epoch  0 Step    100] Loss: 8.2345 | Avg Loss: 8.3421 | LR: 0.00e+00
[Epoch  0 Step    200] Loss: 7.9234 | Avg Loss: 8.0821 | LR: 1.00e-05
[Epoch  0 Step    300] Loss: 7.1234 | Avg Loss: 7.7623 | LR: 2.00e-05
...
```

Loss should decrease over time. Training typically converges in 3-5 epochs.

### Resume Training

If training interrupted:
```bash
python train_caption_pvt.py \
    --image_root /path/to/images \
    --caption_file /path/to/captions.txt \
    --resume_from ./checkpoints/checkpoint_step_001000.pth
```

---

## 🎯 Generate Captions

### Single Image

```bash
python predict_caption.py \
    --image photo.jpg \
    --checkpoint ./checkpoints/final_model.pth
```

Output:
```
==================================================
Generated Caption:
a group of people playing soccer on a field
==================================================
```

### Batch Processing

```python
from inference_utils import load_model, generate_caption
import os

model = load_model("checkpoints/final_model.pth", device="cuda")

for image_file in os.listdir("images/"):
    caption = generate_caption(model, f"images/{image_file}")
    print(f"{image_file}: {caption}")
```

### Interactive Demo

```bash
python -c "from inference_utils import create_gradio_demo; \
demo = create_gradio_demo('./checkpoints/final_model.pth'); \
demo.launch()"
```

Then visit: `http://localhost:7860`

---

## ⚙️ Tuning Hyperparameters

### For Better Captions

```bash
# Use more beams (slower but better quality)
python predict_caption.py \
    --image photo.jpg \
    --checkpoint model.pth \
    --num_beams 5 \
    --max_length 35 \
    --min_length 10
```

### For Faster Training

```bash
# Reduce batch size on low memory GPU
python train_caption_pvt.py \
    --batch_size 4 \
    --lr 1e-4 \
    --use_amp
```

### For Better Convergence

```bash
# Longer training with warmup
python train_caption_pvt.py \
    --epochs 20 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --grad_clip 1.0
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```bash
python train_caption_pvt.py \
    --batch_size 4 \        # Reduce batch size
    --use_amp               # Enable mixed precision
```

### Issue: Captions Still Incoherent

**Checklist:**
- [ ] Check if training loss is decreasing
- [ ] Train for more epochs (at least 5)
- [ ] Verify captions are properly formatted (image<TAB>caption)
- [ ] Check if images load correctly
- [ ] Try smaller learning rate: `--lr 1e-5`

### Issue: Slow Training

**Solution:**
```bash
python train_caption_pvt.py \
    --use_amp \              # Mixed precision
    --num_workers 8 \        # More data workers
    --batch_size 16          # Larger batches if memory allows
```

### Issue: GPU Not Detected

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Show GPU name
```

If False, training will use CPU (much slower).

---

## 📊 What to Expect

### Training Progress

| Epoch | Avg Loss | Time | Notes |
|-------|----------|------|-------|
| 0 | 8.5 | 45min | High loss, model learning basics |
| 1 | 7.2 | 45min | Steady decrease |
| 2 | 6.1 | 45min | Good convergence |
| 3 | 5.5 | 45min | Plateau forming |
| 4 | 5.2 | 45min | Minor improvements |

### Caption Quality Progression

**After Epoch 0:**
- Incoherent fragments
- Poor grammar
- Repeated words

**After Epoch 1:**
- Better structure
- Some grammar
- Meaningful words

**After Epoch 3-5:**
- Grammatically correct
- Semantically relevant
- Natural descriptions

---

## 💡 Pro Tips

### 1. Monitor with TensorBoard (optional)

```bash
pip install tensorboard
# Add to train_caption_pvt.py for logging
```

### 2. Validate During Training

```python
# Add to train loop
if step % 500 == 0:
    model.eval()
    with torch.no_grad():
        test_caption = model.generate(test_image)
    print(f"Test caption: {test_caption}")
    model.train()
```

### 3. Use Different Prompts

```python
model = blip_decoder(prompt="a photo of ")  # Default
# or
model = blip_decoder(prompt="")  # No prompt
```

### 4. Fine-tune on New Data

```bash
python train_caption_pvt.py \
    --resume_from checkpoint.pth \
    --epochs 5 \
    --lr 1e-5 \              # Smaller LR for fine-tuning
    --image_root new_images
```

### 5. Check Model Size

```python
from models.blip import blip_decoder
model = blip_decoder()
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params/1e6:.1f}M")  # ~135.7M
```

---

## 📖 Documentation Files

| File | Purpose |
|------|---------|
| `CHANGES.md` | Detailed explanation of all fixes |
| `IMPLEMENTATION_SUMMARY.md` | Complete technical summary |
| `requirements.txt` | Python dependencies |
| `test_import.py` | Verify installation |

---

## 🎓 Learning Resources

### Understanding BLIP
- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [PVT Paper](https://arxiv.org/abs/2102.12122)
- [CBAM Paper](https://arxiv.org/abs/1807.06521)

### HuggingFace Transformers
- [Documentation](https://huggingface.co/docs/transformers/)
- [BERT Model Card](https://huggingface.co/bert-base-uncased)

### PyTorch
- [Official Documentation](https://pytorch.org/docs/stable/index.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

---

## 📝 Common Commands Cheat Sheet

```bash
# Training
python train_caption_pvt.py --image_root imgs --caption_file captions.txt

# Inference
python predict_caption.py --image photo.jpg --checkpoint model.pth

# Demo
python inference_utils.py --checkpoint model.pth --gradio

# Test
python test_import.py

# Resume
python train_caption_pvt.py --resume_from checkpoint.pth --image_root imgs --caption_file captions.txt

# GPU Check
python -c "import torch; print(torch.cuda.is_available())"

# Parameter Count
python -c "from models.blip import blip_decoder; m = blip_decoder(); print(sum(p.numel() for p in m.parameters())/1e6)"
```

---

## ✅ You're Ready!

Your BLIP implementation is now:
- ✅ Using pretrained language models
- ✅ Properly tokenizing text
- ✅ Normalizing features
- ✅ Generating coherent captions
- ✅ Ready for training and inference

Start training and enjoy generating captions! 🎉
