"""models/med.py

Provide MED-compatible interfaces used by the repo. Prefer the
HuggingFace `transformers` implementations when available; otherwise
fall back to small local stand-ins that are sufficient for smoke
running / training loops in this project (not a full BERT).

The fallback implementations provide minimal behavior expected by
`models/blip.py` and the training script: a `BertConfig` container,
a lightweight `BertModel` returning `last_hidden_state`, and a
`BertLMHeadModel` that computes a simple LM head + loss.
"""

try:
	from transformers import BertConfig, BertModel, BertLMHeadModel
	__all__ = ["BertConfig", "BertModel", "BertLMHeadModel"]
except Exception:
	# Minimal local fallbacks
	from types import SimpleNamespace
	import torch
	import torch.nn as nn

	class BertConfig:
		def __init__(self, **kwargs):
			# Store all keys as attributes for flexibility
			for k, v in kwargs.items():
				setattr(self, k, v)

			# sensible defaults if not provided
			if not hasattr(self, "vocab_size"):
				self.vocab_size = 30524
			if not hasattr(self, "hidden_size"):
				self.hidden_size = getattr(self, "decoder_width", 768)
			if not hasattr(self, "max_position_embeddings"):
				self.max_position_embeddings = 512

	class BertModel(nn.Module):
		"""Very small encoder-like module that returns token embeddings
		as `last_hidden_state`. This is NOT a BERT implementation, but
		it's enough for running forward passes in this repo.
		"""
		def __init__(self, config: BertConfig):
			super().__init__()
			self.config = config
			vocab_size = max(getattr(config, "vocab_size", 30524), 50000)
			hidden = getattr(config, "hidden_size", 768)
			self.embeddings = nn.Embedding(vocab_size, hidden)

		def forward(self, input_ids=None, attention_mask=None,
					encoder_hidden_states=None, encoder_attention_mask=None,
					return_dict=True, **kwargs):
			# input_ids: (B, L)
			emb = self.embeddings(input_ids)
			# Simple passthrough: no transformer layers
			out = SimpleNamespace(last_hidden_state=emb)
			return out

	class BertLMHeadModel(nn.Module):
		"""Lightweight LM head on top of token embeddings.

		Supports `forward(..., labels=...)` returning an object with
		a `loss` attribute, and a simple `generate` method.
		"""
		def __init__(self, config: BertConfig):
			super().__init__()
			self.config = config
			vocab_size = max(getattr(config, "vocab_size", 30524), 50000)
			hidden = getattr(config, "hidden_size", 768)
			self.embeddings = nn.Embedding(vocab_size, hidden)
			self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

		def forward(self, input_ids=None, attention_mask=None,
					encoder_hidden_states=None, encoder_attention_mask=None,
					labels=None, return_dict=True, **kwargs):
			# input_ids: (B, L)
			emb = self.embeddings(input_ids)  # (B, L, H)
			logits = self.lm_head(emb)  # (B, L, V)

			loss = None
			if labels is not None:
				# Shift not required for this simplistic head; compute CE per-token
				loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
				# reshape to (B*L, V) and (B*L)
				loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

			out = SimpleNamespace(loss=loss, logits=logits)
			return out

		@torch.no_grad()
		def generate(self, input_ids, max_length=20, eos_token_id=None,
					 pad_token_id=None, num_beams=1, min_length=0, **kwargs):
			# Simple greedy generation ignoring encoder states and beams
			B = input_ids.size(0)
			device = input_ids.device
			seqs = [list(row.tolist()) for row in input_ids]

			for _ in range(max_length - input_ids.size(1)):
				cur = torch.tensor([s[-1] for s in seqs], dtype=torch.long, device=device).unsqueeze(1)
				emb = self.embeddings(cur)  # (B,1,H)
				logits = self.lm_head(emb)  # (B,1,V)
				next_ids = logits.argmax(dim=-1).squeeze(1).tolist()
				for i, nid in enumerate(next_ids):
					seqs[i].append(int(nid))
				# stop early if all have produced eos
				if eos_token_id is not None:
					if all((s[-1] == eos_token_id) for s in seqs):
						break

			# return tensor of sequences
			max_len = max(len(s) for s in seqs)
			out = torch.full((B, max_len), pad_token_id if pad_token_id is not None else 0, dtype=torch.long, device=device)
			for i, s in enumerate(seqs):
				out[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
			return out

	__all__ = ["BertConfig", "BertModel", "BertLMHeadModel"]
