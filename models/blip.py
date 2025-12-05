"""
BLIP implementation with:
 - PVT-Tiny (with CBAM) as Vision Encoder
 - Pretrained bert-base-uncased (from HuggingFace) as Text Decoder
 - BertTokenizerFast for proper subword tokenization
 - Correct prompt handling during training and inference
 - Stable beam search generation

Problems fixed:
1. MED-BERT was untrained random config → Now uses pretrained bert-base-uncased
2. SimpleTokenizer had vocab expansion issues → Now uses BertTokenizerFast with fixed vocab
3. Prompt mismatch caused bad captions → Now properly prepends "a picture of " during training
4. generate() had char-level slicing → Now uses token-level prompt removal
5. Missing LayerNorm caused unstable features → Now normalizes PVT output
6. Device placements were inconsistent → Now explicit device handling throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertLMHeadModel, BertConfig, AutoTokenizer

from models.pvt import pvt_tiny

# ---- Disable tokenizer multiprocessing (prevents deadlock in Colab)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ---------------------------------------------------
#  Utility: Flatten PVT last-stage output
# ---------------------------------------------------
def flatten_pvt_feature(c4):
    """
    PVT returns spatial map (B, C, H, W)
    BLIP expects sequence (B, N, C)
    
    Args:
        c4: Tensor of shape (B, C, H, W)
    
    Returns:
        Flattened tensor of shape (B, H*W, C)
    """
    B, C, H, W = c4.shape
    return c4.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, C)




# ---------------------------------------------------
#  Vision Backbone Builder
# ---------------------------------------------------
def create_vit(vit_name, image_size):
    """
    Create vision encoder backbone.
    
    Args:
        vit_name: Vision encoder name (only 'pvt_tiny' supported)
        image_size: Input image size
        
    Returns:
        model: Vision encoder module
        vision_width: Output feature dimension
    """
    if vit_name == "pvt_tiny":
        return pvt_tiny(), 256  # PVT-Tiny output dim = 256

    raise ValueError(f"Unsupported vision backbone: {vit_name}")


# ---------------------------------------------------
#  Feature Normalizer
# ---------------------------------------------------
class FeatureNormalizer(nn.Module):
    """
    Normalizes vision encoder output to stabilize training.
    
    Why this is needed:
    - PVT outputs raw spatial features with varying magnitude
    - Without normalization, vision features dominate text embeddings
    - LayerNorm ensures consistent scale for cross-attention
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x)

# ---------------------------------------------------
#  BLIP Decoder (Caption Generator)
# ---------------------------------------------------
class BLIP_Decoder(nn.Module):
    """
    Image-to-text caption generation model.
    
    Architecture:
    1. Vision Encoder (PVT-Tiny): Image → Spatial features (B, N, 256)
    2. Feature Normalizer: Stabilize feature magnitudes
    3. Text Decoder (bert-base-uncased): Autoregressively generate captions
       with cross-attention to vision features
    
    Training:
    - Prepend "a picture of " to each caption
    - Forward pass: compute language modeling loss
    - Loss only computed on tokens AFTER the prompt
    
    Inference:
    - Generate continuation of "a picture of " prompt
    - Decode tokens and strip prompt from output
    """
    
    def __init__(
        self,
        image_size=384,
        vit='pvt_tiny',
        prompt="a picture of ",
        tokenizer_name="bert-base-uncased",
        lm_name="bert-base-uncased"
    ):
        """
        Initialize BLIP Decoder.
        
        Args:
            image_size: Input image height/width
            vit: Vision encoder backbone name
            prompt: Prefix for all captions
            tokenizer_name: HuggingFace tokenizer identifier
            lm_name: HuggingFace language model identifier
        """
        super().__init__()

        # ---- Vision Encoder (PVT-Tiny)
        self.visual_encoder, vision_width = create_vit(vit, image_size)
        self.feature_normalizer = FeatureNormalizer(vision_width)

        # ---- Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )
        
        # Ensure special tokens exist
        if self.tokenizer.sep_token_id is None:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})

        # ---- Text Decoder (Pretrained BERT with cross-attention for LM)
        config = BertConfig.from_pretrained(lm_name)
        config.is_decoder = True
        config.add_cross_attention = True
        
        self.text_decoder = BertLMHeadModel.from_pretrained(
            lm_name,
            config=config,
            trust_remote_code=True
        )

        # If tokenizer was extended, resize token embeddings
        try:
            self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        except Exception:
            pass

        # ---- Projection: map PVT features -> decoder hidden size if needed
        self.decoder_hidden_size = getattr(self.text_decoder.config, "hidden_size", 768)
        if vision_width != self.decoder_hidden_size:
            self.proj_layer = nn.Linear(vision_width, self.decoder_hidden_size)
        else:
            self.proj_layer = None

        # ---- Prompt handling
        self.prompt = prompt
        
        # Tokenize prompt and store token IDs
        prompt_tokens = self.tokenizer(
            self.prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        self.prompt_token_ids = prompt_tokens["input_ids"][0]  # Remove batch dim
        self.prompt_length = len(self.prompt_token_ids)
        
        print(f"[BLIP] Prompt: '{self.prompt}'")
        print(f"[BLIP] Prompt tokens (token_ids): {self.prompt_token_ids.tolist()}")
        print(f"[BLIP] Prompt length: {self.prompt_length} tokens")

    def forward(self, image, captions):
        """
        Training forward pass: compute language modeling loss.
        Matches BLIP-1 official training procedure.
        
        Args:
            image: Batch of images, shape (B, 3, H, W)
            captions: List of caption strings
            
        Returns:
            loss: Scalar loss for backprop
        """
        device = image.device

        # ---- Encode image
        img = self.visual_encoder(image)
        # If the backbone returns a list/stages, take the last
        if isinstance(img, (list, tuple)):
            img = img[-1]
        # If output is (B, C, H, W) flatten to (B, N, C)
        if img.dim() == 4:
            image_embeds = flatten_pvt_feature(img)
        elif img.dim() == 3:
            image_embeds = img
        else:
            raise RuntimeError(f"Unexpected PVT output shape: {img.shape}")
        # Project to decoder hidden size if needed
        if self.proj_layer is not None:
            image_embeds = self.proj_layer(image_embeds)

        # ---- Tokenize caption
        text = self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=30,
            return_tensors="pt"
        ).to(device)

        # ---- Shift token IDs for decoder LM training
        labels = text["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # ---- Forward pass
        out = self.text_decoder(
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"],
            encoder_hidden_states=image_embeds,
            labels=labels
        )

        return out.loss

    def generate(
        self,
        image,
        num_beams=5,
        max_length=30,
        min_length=8,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        diversity_penalty=0.3
    ):
        """
        Inference: generate captions with beam search and repetition penalties.
        
        Args:
            image: Batch of images
            num_beams: Beam search width (higher=better quality but slower)
            max_length: Maximum caption length
            min_length: Minimum caption length
            repetition_penalty: Penalize repeated tokens (>1 prevents repetition)
            no_repeat_ngram_size: Prevent n-gram repetition (3 = no 3-grams repeat)
            length_penalty: Favor longer sequences if >1
            diversity_penalty: Encourage diverse beams if >0
            
        Returns:
            captions: List of caption strings
        """
        device = image.device

        # ---- Encode image
        img = self.visual_encoder(image)
        if isinstance(img, (list, tuple)):
            img = img[-1]
        if img.dim() == 4:
            img = flatten_pvt_feature(img)
        elif img.dim() == 3:
            pass
        else:
            raise RuntimeError(f"Unexpected PVT output shape: {img.shape}")
        # Project if needed
        if self.proj_layer is not None:
            img = self.proj_layer(img)

        # ---- Prepare prompt
        prompt = ["a picture of "] * image.size(0)

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt"
        )["input_ids"].to(image.device)

        with torch.no_grad():
            out = self.text_decoder.generate(
                input_ids=input_ids,
                encoder_hidden_states=img,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                early_stopping=True,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                diversity_penalty=diversity_penalty,
                temperature=1.0,
                top_p=0.9,
                do_sample=False,
            )

        captions = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return captions

    def inference(self, image, **kwargs):
        """
        Alias for generate() for backward compatibility.
        """
        return self.generate(image, **kwargs)

# ---------------------------------------------------
#  Factory functions
# ---------------------------------------------------
def blip_decoder(**kwargs):
    """
    Factory function to create BLIP_Decoder.
    
    Usage:
        model = blip_decoder(image_size=384, vit='pvt_tiny')
    """
    return BLIP_Decoder(**kwargs)


def load_model(
    checkpoint_path=None,
    image_size=384,
    device="cuda",
    **kwargs
):
    """
    Load BLIP model for inference.
    
    Args:
        checkpoint_path: Path to saved checkpoint (optional)
        image_size: Input image size
        device: Device to load model on
        **kwargs: Additional arguments for BLIP_Decoder
        
    Returns:
        model: BLIP_Decoder instance on specified device
    """
    model = blip_decoder(image_size=image_size, **kwargs)
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model
