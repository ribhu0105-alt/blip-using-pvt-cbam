# test_import.py
"""
Quick test to verify all imports and basic model instantiation work.
Run this to ensure all dependencies are installed correctly.

Usage:
    python test_import.py
"""

import sys
import os

# Add project root
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

print("=" * 60)
print("Testing BLIP Implementation")
print("=" * 60)

# Test 1: Import transformers
print("\n[1/6] Testing HuggingFace transformers import...")
try:
    from transformers import AutoModel, AutoTokenizer
    print("  ✓ transformers imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import transformers: {e}")
    sys.exit(1)

# Test 2: Import PyTorch
print("\n[2/6] Testing PyTorch import...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__} imported successfully")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  ✗ Failed to import torch: {e}")
    sys.exit(1)

# Test 3: Import PVT
print("\n[3/6] Testing PVT model import...")
try:
    from models.pvt import pvt_tiny
    print("  ✓ PVT model imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import PVT: {e}")
    sys.exit(1)

# Test 4: Import BLIP
print("\n[4/6] Testing BLIP model import...")
try:
    from models.blip import blip_decoder, load_model
    print("  ✓ BLIP model imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import BLIP: {e}")
    sys.exit(1)

# Test 5: Instantiate BLIP model
print("\n[5/6] Testing BLIP model instantiation...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = blip_decoder(image_size=384, vit='pvt_tiny')
    model = model.to(device)
    print(f"  ✓ BLIP model created successfully (device: {device})")
    print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
except Exception as e:
    print(f"  ✗ Failed to instantiate BLIP: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test inference utilities
print("\n[6/6] Testing inference utilities...")
try:
    from inference_utils import load_model, process_image, generate_caption
    print("  ✓ Inference utilities imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import inference utilities: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! Implementation is ready to use.")
print("=" * 60)

print("\n📝 Next steps:")
print("  1. Download Flickr30k or COCO dataset")
print("  2. Prepare captions in format: image.jpg<TAB>caption")
print("  3. Run: python train_caption_pvt.py --image_root ... --caption_file ...")
print("  4. Generate captions: python predict_caption.py --image ... --checkpoint ...")
print("  5. Launch demo: python -c \"from inference_utils import create_gradio_demo; demo = create_gradio_demo('checkpoint.pth'); demo.launch()\"")
print()
