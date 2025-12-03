"""
Inference utilities for BLIP image captioning.

This module provides:
 - load_model(): Load pretrained or checkpoint model
 - process_image(): Load and preprocess images
 - generate_caption(): Generate captions for images
 - create_gradio_demo(): Create Gradio interface for demo
"""

import os
import torch
from PIL import Image
from torchvision import transforms

from models.blip import blip_decoder, load_model as blip_load_model


def load_model(checkpoint_path=None, image_size=384, device="cuda"):
    """
    Load BLIP model for inference.
    
    Args:
        checkpoint_path: Path to checkpoint (optional)
        image_size: Input image size
        device: Device to load on ("cuda" or "cpu")
        
    Returns:
        model: BLIP_Decoder in eval mode
    """
    if checkpoint_path is None:
        print("[Warning] No checkpoint provided. Loading untrained model.")
    
    model = blip_load_model(
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        device=device
    )
    
    return model


def process_image(image_path, image_size=384):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to image file or PIL Image
        image_size: Target size
        
    Returns:
        Preprocessed tensor of shape (1, 3, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    elif isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    else:
        raise TypeError(f"Expected str or PIL.Image, got {type(image_path)}")
    
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor


def generate_caption(
    model,
    image,
    num_beams=3,
    max_length=30,
    min_length=8,
    device="cuda"
):
    """
    Generate caption for image.
    
    Args:
        model: BLIP_Decoder model
        image: Image tensor or path
        num_beams: Beam search width
        max_length: Max caption length
        min_length: Min caption length
        device: Device to use
        
    Returns:
        Caption string
    """
    if isinstance(image, str):
        image = process_image(image).to(device)
    elif isinstance(image, Image.Image):
        image = process_image(image).to(device)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
    
    with torch.no_grad():
        captions = model.generate(
            image,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length
        )
    
    return captions[0] if captions else ""


def create_gradio_demo(checkpoint_path, image_size=384):
    """
    Create a Gradio demo interface for image captioning.
    
    Args:
        checkpoint_path: Path to model checkpoint
        image_size: Input image size
        
    Returns:
        Gradio interface ready to launch
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio not installed. Run: pip install gradio")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, image_size=image_size, device=device)
    
    def caption_fn(image):
        """Gradio interface function."""
        if image is None:
            return "No image provided"
        
        caption = generate_caption(
            model,
            image,
            num_beams=3,
            max_length=30,
            min_length=8,
            device=device
        )
        return caption
    
    interface = gr.Interface(
        fn=caption_fn,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="BLIP Image Captioning",
        description="Generate captions for images using BLIP with PVT-Tiny backbone",
        examples=[],
    )
    
    return interface


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--gradio", action="store_true")
    parser.add_argument("--image", type=str)
    
    args = parser.parse_args()
    
    if args.gradio:
        print("Launching Gradio demo...")
        demo = create_gradio_demo(args.checkpoint, args.image_size)
        demo.launch()
    
    elif args.image:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.checkpoint, args.image_size, device)
        caption = generate_caption(model, args.image, device=device)
        print(f"Caption: {caption}")
    
    else:
        print("Use --gradio to launch interactive demo or --image <path> for single image")
