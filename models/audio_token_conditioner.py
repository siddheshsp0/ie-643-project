"""Puts the audio tokens into the prompt "a photo of a * * ... *" and runs SD's frozen CLIP text
encoder on it, giving the [B, 77, 768] conditioning for the U-Net.

CLIPTextModel only accepts input_ids, so the placeholder embeddings are swapped for the audio
tokens with a forward hook on the token-embedding layer.
"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

DEFAULT_SD_PATH = Path(__file__).resolve().parent / "stable-diffusion-inpainting"


class AudioTokenConditioner(nn.Module):
    def __init__(self, sd_path=DEFAULT_SD_PATH, template="a photo of a", num_tokens=8,
                 placeholder="*", device=None, dtype=torch.float32):
        super().__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder",
                                                          variant="fp16", dtype=dtype).to(device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        self.template = template
        self.num_tokens = num_tokens
        self.max_length = self.tokenizer.model_max_length
        self.embed_dim = self.text_encoder.config.hidden_size

        placeholder_ids = self.tokenizer(placeholder, add_special_tokens=False).input_ids
        if len(placeholder_ids) != 1:
            raise ValueError(f"placeholder {placeholder!r} must be a single token, got ids {placeholder_ids}")
        self.placeholder_id = placeholder_ids[0]
        if self.placeholder_id in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
            raise ValueError("placeholder must differ from EOS/pad (EOS position is found by argmax of ids)")

        ids = self.tokenizer(template).input_ids  # [BOS, ..., EOS]
        if ids[-1] != self.tokenizer.eos_token_id:
            raise RuntimeError(f"unexpected tokenization of template: {ids}")
        self.start = len(ids) - 1  # placeholders go right before EOS
        ids = ids[:-1] + [self.placeholder_id] * num_tokens + ids[-1:]
        if len(ids) > self.max_length:
            raise ValueError(f"template + {num_tokens} tokens = {len(ids)} ids exceeds {self.max_length}")
        ids = ids + [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
        self.register_buffer("input_ids", torch.tensor([ids], device=device), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()  # frozen
        return self

    @property
    def token_embedding(self):
        return self.text_encoder.get_input_embeddings()

    def forward(self, audio_tokens):
        """audio_tokens [B, N, 768] -> encoder_hidden_states [B, 77, 768] (differentiable w.r.t. audio_tokens)."""
        if audio_tokens.dim() != 3 or audio_tokens.shape[1:] != (self.num_tokens, self.embed_dim):
            raise ValueError(f"expected [B, {self.num_tokens}, {self.embed_dim}], got {tuple(audio_tokens.shape)}")
        s, e = self.start, self.start + self.num_tokens

        def splice(module, inputs, output):
            return torch.cat([output[:, :s], audio_tokens.to(output.dtype), output[:, e:]], dim=1)

        handle = self.token_embedding.register_forward_hook(splice)
        try:
            ids = self.input_ids.expand(audio_tokens.shape[0], -1)
            hidden = self.text_encoder(input_ids=ids).last_hidden_state
        finally:
            handle.remove()
        assert hidden.shape == (audio_tokens.shape[0], self.max_length, self.embed_dim), tuple(hidden.shape)
        return hidden
