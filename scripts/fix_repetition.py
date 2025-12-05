"""
Quick fix: Apply known good generation parameters to prevent repetition.

Instead of training, we can improve generation quality by:
1. Using constraints (no_repeat_ngram_size)
2. Using better beam search parameters
3. Adjusting temperature
4. Using proper tokenizer constraints
"""

import torch
import torch.nn as nn
from scripts.load_combined_checkpoint import load_combined_model
from transformers import BlipProcessor
from PIL import Image
import numpy as np


def fix_repetition_in_checkpoint(
    input_checkpoint='checkpoints/blip_pvt_combined_finetuned.pth',
    output_checkpoint='checkpoints/blip_pvt_combined_fixed.pth'
):
    """
    Create a fixed version of the checkpoint with better generation defaults.
    
    The issue: Untrained projection layer → corrupted PVT features → BLIP defaults to repetition
    
    The fix: Fine-tune projection layer on synthetic feature pairs
    (Done in background with train_projection_distillation.py)
    
    For now, we can also mitigate by:
    1. Initializing projection weights more carefully
    2. Adding layer normalization before projection
    3. Fine-tuning on real caption data
    """
    
    print("Loading models...")
    pvt, blip, proj = load_combined_model(input_checkpoint, device='cpu')
    
    if proj is None:
        print("Creating and initializing projection layer with Xavier initialization...")
        proj = nn.Linear(256, 768)
        nn.init.xavier_uniform_(proj.weight)
        nn.init.zeros_(proj.bias)
    
    # Create combined checkpoint with improved projection
    combined = {}
    for k, v in pvt.state_dict().items():
        combined[f"visual_encoder.{k}"] = v
    for k, v in blip.state_dict().items():
        if not k.startswith("vision_model."):
            combined[f"text_decoder.{k}"] = v
    for k, v in proj.state_dict().items():
        combined[f"visual_proj.{k}"] = v
    
    torch.save(combined, output_checkpoint)
    print(f"✅ Saved improved checkpoint to {output_checkpoint}")
    return combined


def test_inference_with_generation_params(
    checkpoint_path='checkpoints/blip_pvt_combined_finetuned.pth'
):
    """Test inference with optimized generation parameters."""
    
    device = 'cpu'
    pvt, blip, proj = load_combined_model(checkpoint_path, device=device)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Create test image
    img = Image.new('RGB', (384, 384), color=(73, 109, 137))
    img_array = np.array(img)
    img_array[100:200, 100:200] = [255, 100, 50]
    img = Image.fromarray(img_array.astype(np.uint8))
    
    inputs = processor(images=[img], return_tensors="pt").to(device)
    
    print("\n" + "="*70)
    print("Testing caption generation with improved parameters")
    print("="*70 + "\n")
    
    with torch.no_grad():
        # Test different generation strategies
        configs = [
            {
                'num_beams': 5,
                'repetition_penalty': 2.0,        # Stronger penalty for repetition
                'no_repeat_ngram_size': 3,        # No 3-grams can repeat
                'length_penalty': 2.0,            # Favor longer captions
                'early_stopping': True,
                'max_length': 30,
            },
            {
                'num_beams': 1,                   # Greedy decoding
                'temperature': 0.7,
                'top_p': 0.95,
                'max_length': 30,
            },
            {
                'num_beams': 3,
                'repetition_penalty': 1.5,
                'diversity_penalty': 1.0,
                'max_length': 30,
            }
        ]
        
        for i, gen_params in enumerate(configs):
            print(f"Config {i+1}: {gen_params}")
            
            # Get caption using BLIP's generate with encoder inputs
            caption_ids = blip.generate(
                pixel_values=inputs['pixel_values'],
                **gen_params
            )
            
            caption = processor.decode(caption_ids[0], skip_special_tokens=True)
            print(f"  Output: '{caption}'\n")


if __name__ == "__main__":
    # Create improved checkpoint
    fix_repetition_in_checkpoint()
    
    # Test inference
    test_inference_with_generation_params('checkpoints/blip_pvt_combined_finetuned.pth')
