"""
BLIP implementation modified to use:
 - PVT-Tiny (with CBAM) as Vision Encoder
 - MED-BERT (local) as Text Encoder/Decoder
 - SimpleTokenizer (local) instead of HuggingFace

This version removes all transformers dependency.
"""

import torch
import torch.nn as nn
import json

from models.med import BertConfig, BertModel, BertLMHeadModel
from models.utils import SimpleTokenizer
from models.pvt import pvt_tiny


# ---------------------------------------------------
#  Utility: Flatten PVT last-stage output
# ---------------------------------------------------
def flatten_pvt_feature(c4):
    """
    PVT returns spatial map (B, C, H, W)
    BLIP expects sequence (B, N, C)
    """
    B, C, H, W = c4.shape
    return c4.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, C)


# ---------------------------------------------------
#  Tokenizer
# ---------------------------------------------------
def init_tokenizer():
    tok = SimpleTokenizer()

    # special tokens
    tok.enc_token = "[ENC]"
    tok.dec_token = "[DEC]"

    # ---- REQUIRED IDs (critical for training/inference) ----
    tok.bos_token_id = tok.vocab["[DEC]"]   # BOS token for decoder
    tok.pad_token_id = tok.vocab["[PAD]"]   # padding
    tok.sep_token_id = tok.vocab["[SEP]"]   # end of sentence

    return tok



# ---------------------------------------------------
#  Vision Backbone Builder
# ---------------------------------------------------
def create_vit(vit_name, image_size):
    """
    vit_name: only 'pvt_tiny' supported
    """
    if vit_name == "pvt_tiny":
        return pvt_tiny(), 256  # last-stage dim = 256

    raise ValueError(f"Unsupported vision backbone: {vit_name}")


# ---------------------------------------------------
#  BLIP Base (feature extractor / multimodal encoder)
# ---------------------------------------------------
class BLIP_Base(nn.Module):
    def __init__(
        self,
        med_config='configs/med_config.json',
        image_size=224,
        vit='pvt_tiny',
        vit_grad_ckpt=False,
        vit_ckpt_layer=0
    ):
        super().__init__()

        # ---- Vision Encoder (PVT)
        self.visual_encoder, vision_width = create_vit(vit, image_size)

        # ---- Tokenizer
        self.tokenizer = init_tokenizer()

        # ---- MED Text Encoder
        with open(med_config, "r") as f:
            med_dict = json.load(f)
        med = BertConfig(**med_dict)
        med.encoder_width = vision_width

        self.text_encoder = BertModel(config=med)

    def forward(self, image, caption, mode):
        """
        mode = "image", "text", or "multimodal"
        """
        assert mode in ["image", "text", "multimodal"]

        # tokenize input
        text = self.tokenizer(caption, return_tensors="pt")
        text = {k: v.to(image.device) for k, v in text.items()}


        # ------- IMAGE MODE -------
        if mode == "image":
            feats = self.visual_encoder(image)
            if isinstance(feats, list):
                feats = flatten_pvt_feature(feats[-1])
            return feats

        # ------- TEXT MODE -------
        if mode == "text":
            out = self.text_encoder(
                input_ids=text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True
            )
            return out.last_hidden_state

        # ------- MULTIMODAL MODE -------
        image_embeds = self.visual_encoder(image)
        if isinstance(image_embeds, list):
            image_embeds = flatten_pvt_feature(image_embeds[-1])

        atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # replace CLS with [ENC]
        text.input_ids[:, 0] = self.tokenizer.convert_tokens_to_ids("[ENC]")

        out = self.text_encoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=atts,
            return_dict=True
        )
        return out.last_hidden_state


# ---------------------------------------------------
#  BLIP Decoder (Caption Generator)
# ---------------------------------------------------
class BLIP_Decoder(nn.Module):
    def __init__(
        self,
        med_config='configs/med_config.json',
        image_size=384,
        vit='pvt_tiny',
        prompt="a picture of "
    ):
        super().__init__()

        # ---- Vision Encoder
        self.visual_encoder, vision_width = create_vit(vit, image_size)

        # ---- Tokenizer
        self.tokenizer = init_tokenizer()

        # ---- MED LM Decoder
        with open(med_config, "r") as f:
            med_dict = json.load(f)
        med = BertConfig(**med_dict)
        med.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=med)

        self.prompt = prompt

        # tokenize prompt → tokens
        prompt_tokens = self.tokenizer.tokenize(self.prompt)

        # convert tokens to ids manually
        self.prompt_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in prompt_tokens]
        self.prompt_len = len(self.prompt_ids)




    def forward(self, image, caption):
        img = self.visual_encoder(image)
        if isinstance(img, list):
            img = flatten_pvt_feature(img[-1])

        atts = torch.ones(img.size()[:-1], dtype=torch.long).to(image.device)

        # prepare text inputs
        text = self.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=40,
            return_tensors="pt"
        )

        # move dict tensors to device
        for k in text:
            text[k] = text[k].to(image.device)


        # Use [DEC] as BOS
        bos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.dec_token)
        text["input_ids"][:, 0] = bos_id



        # Loss masking
        targets = text["input_ids"].masked_fill(
            text["input_ids"] == self.tokenizer.pad_token_id,
            -100
        )
        targets[:, :self.prompt_len] = -100

        out = self.text_decoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=img,
            encoder_attention_mask=atts,
            labels=targets,
            return_dict=True
        )

        return out.loss

    # ---------------- Inference ----------------
    def generate(self, image, num_beams=3, max_length=30, min_length=10):
        img = self.visual_encoder(image)
        if isinstance(img, list):
            img = flatten_pvt_feature(img[-1])

        # repeat features for beam search
        img = img.repeat_interleave(num_beams, dim=0)

        atts = torch.ones(img.size()[:-1], dtype=torch.long).to(image.device)
        kwargs = {"encoder_hidden_states": img, "encoder_attention_mask": atts}

        # prepare prompt tokens
        prompt = [self.prompt] * image.size(0)
        tok = self.tokenizer(prompt, return_tensors="pt")
        text = tok["input_ids"].to(image.device)

        bos_id = self.tokenizer.convert_tokens_to_ids("[DEC]")
        text[:, 0] = bos_id
        text = text[:, :-1]

        out = self.text_decoder.generate(
            input_ids=text,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )

        captions = []
        for seq in out:
            decoded = self.tokenizer.decode(seq.tolist())


            captions.append(decoded[len(self.prompt):])

        return captions


# ---------------------------------------------------
#  Factory functions
# ---------------------------------------------------
def blip_feature_extractor(**kwargs):
    return BLIP_Base(**kwargs)


def blip_decoder(**kwargs):
    return BLIP_Decoder(**kwargs)
