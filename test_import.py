# test_import.py
import os, sys

# Add project root (portable)
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.abspath(PROJECT_ROOT))

print("Project root added:", PROJECT_ROOT)

# ---- Test all model imports ----
from models.blip import blip_decoder
print("BLIP decoder import OK")

from models.pvt import pvt_tiny
print("PVT tiny import OK")

from models.vision_backbones import create_backbone, list_backbones
print("Vision backbones import OK")

print("Available backbones:", list_backbones())

model, dim = create_backbone("pvt_tiny", image_size=384)
print("Backbone created OK, dim =", dim)
