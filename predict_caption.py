import argparse
import torch
from PIL import Image
from torchvision import transforms

from models.blip import blip_decoder


def load_image(path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return img


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Load model
    print("Loading model...")
    model = blip_decoder(
        image_size=args.image_size,
        vit="pvt_tiny",
        med_config=args.med_config
    ).to(device)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=True)

    model.eval()

    # ---- Load image
    img = load_image(args.image, args.image_size).to(device)

    # ---- Generate caption
    with torch.no_grad():
        caption = model.generate(img, num_beams=3, max_length=30, min_length=5)

    print("\nGenerated Caption:")
    print(caption[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--med_config", type=str, default="configs/med_config.json")
    parser.add_argument("--image_size", type=int, default=384)

    args = parser.parse_args()
    main(args)
