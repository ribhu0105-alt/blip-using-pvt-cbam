# data/utils.py
"""
Utility helpers for dataset preparation and inspection.
=======================================================

This module provides:
 - list_images(): recursively collect image files
 - load_captions(): parse Flickr8k-style caption text files
 - count_images(): count images in a folder
 - remove_corrupted(): detect/remove corrupted images

These utilities are optional but make the repo feel more complete
and professionally organized.
"""

import os
from PIL import Image


# ----------------------------------------------------------
# Collect image filenames recursively
# ----------------------------------------------------------
def list_images(root, exts={".jpg", ".jpeg", ".png"}):
    files = []
    root = os.path.expanduser(root)

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in exts:
                files.append(os.path.join(dirpath, fname))

    return sorted(files)


# ----------------------------------------------------------
# Count images in directory
# ----------------------------------------------------------
def count_images(root):
    return len(list_images(root))


# ----------------------------------------------------------
# Parse Flickr8k caption file (alternative loader)
# ----------------------------------------------------------
def load_captions(captions_file):
    """
    Returns a dict:
        { filename: [caption1, caption2, caption3, ...] }

    Supports:
      - Flickr8k.token.txt format
      - simple "filename caption" format
    """
    caption_map = {}

    with open(captions_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Format: 123456789.jpg#3 caption text
            if "#" in line:
                left, caption = line.split("\t") if "\t" in line else line.split(" ", 1)
                img = left.split("#")[0]
            else:
                img, caption = line.split(" ", 1)

            caption_map.setdefault(img, []).append(caption)

    return caption_map


# ----------------------------------------------------------
# Check if image is corrupted
# ----------------------------------------------------------
def is_corrupted(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return False
    except Exception:
        return True


# ----------------------------------------------------------
# Remove corrupted images
# ----------------------------------------------------------
def remove_corrupted(root):
    images = list_images(root)
    removed = 0

    for img_path in images:
        if is_corrupted(img_path):
            print(f"[Warning] Removing corrupted: {img_path}")
            os.remove(img_path)
            removed += 1

    print(f"[Cleanup] Removed {removed} corrupted images.")
    return removed
# -------------------------------------------------------------
# SimpleTokenizer
# A minimal BERT-style tokenizer used by BLIP without HF library
# -------------------------------------------------------------

import re
import torch

class SimpleTokenizer:
    def __init__(self):
        # Basic vocabulary
        self.vocab = {
            "[PAD]": 0,
            "[ENC]": 1,
            "[DEC]": 2,
            "[UNK]": 3,
            "[SEP]": 4
        }

        # Reverse lookup
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.pad_token_id = self.vocab["[PAD]"]
        self.sep_token_id = self.vocab["[SEP]"]
        self.unk_token_id = self.vocab["[UNK]"]

    def tokenize(self, text):
        text = text.lower().strip()
        tokens = re.findall(r"[a-zA-Z0-9]+|[^\w\s]", text)
        return tokens

    def convert_tokens_to_ids(self, token_list):
        is_list = isinstance(token_list, list)
        items = token_list if is_list else [token_list]
        ids = []
        for t in items:
            if t not in self.vocab:
                # add new tokens automatically
                self.vocab[t] = len(self.vocab)
                self.inv_vocab[self.vocab[t]] = t
            ids.append(self.vocab[t])
        return ids if is_list else ids[0]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab.get(i, "[UNK]") for i in ids]

    def __call__(self, texts, return_tensors=None, padding="longest", max_length=None, truncation=False):
        if isinstance(texts, str):
            texts = [texts]

        encoded = []
        for t in texts:
            toks = self.tokenize(t)
            ids = self.convert_tokens_to_ids(toks)
            encoded.append(ids)

        # Padding
        max_len = max(len(x) for x in encoded)
        padded = []
        masks = []
        for seq in encoded:
            pad_len = max_len - len(seq)
            padded.append(seq + [self.pad_token_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)

        input_ids = torch.tensor(padded, dtype=torch.long)
        attn = torch.tensor(masks, dtype=torch.long)

        # TokenBatch behaves like both a dict and an object with attributes
        class TokenBatch(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)

            def __setattr__(self, name, value):
                self[name] = value

        return TokenBatch({"input_ids": input_ids, "attention_mask": attn})

    def decode(self, ids):
        toks = self.convert_ids_to_tokens(ids)
        return " ".join(toks)
