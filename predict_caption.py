import argparse
import torch
from PIL import Image
from torchvision import transforms

from models.blip import blip_decoder, load_model


def process_image(image_path, image_size=384):
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        image_size: Target image size
        
    Returns:
        Preprocessed image tensor of shape (1, 3, image_size, image_size)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return img


def generate_caption(model, image, num_beams=3, max_length=30, min_length=8):
    """
    Generate caption for a single image.
    
    Args:
        model: BLIP_Decoder model
        image: Image tensor of shape (1, 3, H, W)
        num_beams: Beam search width
        max_length: Maximum caption length
        min_length: Minimum caption length
        
    Returns:
        Caption string
    """
    with torch.no_grad():
        captions = model.generate(
            image,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length
        )
    return captions[0]


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Load model
    print("Loading model...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        device=device
    )

    # ---- Load image
    print(f"Loading image: {args.image}")
    img = process_image(args.image, args.image_size).to(device)

    # ---- Generate caption
    print("Generating caption...")
    caption = generate_caption(
        model,
        img,
        num_beams=args.num_beams,
        max_length=args.max_length,
        min_length=args.min_length
    )

    print("\n" + "="*50)
    print("Generated Caption:")
    print(caption)
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image captions using BLIP")

    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    parser.add_argument("--image_size", type=int, default=384,
                        help="Input image size")
    parser.add_argument("--num_beams", type=int, default=3,
                        help="Beam search width")
    parser.add_argument("--max_length", type=int, default=30,
                        help="Maximum caption length (tokens)")
    parser.add_argument("--min_length", type=int, default=8,
                        help="Minimum caption length (tokens)")

    args = parser.parse_args()
    main(args)
