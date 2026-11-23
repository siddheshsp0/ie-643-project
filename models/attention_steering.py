"""Training-free location control: steer the audio tokens' cross-attention toward where the sound is.

In Stable Diffusion's cross-attention every pixel spreads its attention over the 77 prompt positions, and the
sound's object forms where the audio-bearing positions get it: the 8 audio tokens and every position after
them (CLIP's causal mask lets those see the audio). For the first `active_frac` of the denoising steps, when the
layout is decided, this processor multiplies that attention at pixel x by

    w(x) = (floor + (1 - floor) heat(x)) (1 + boost heat(x))

and renormalises over the prompt, so the object forms at the heatmap instead of anywhere in the hole. Only the
conditional half of the classifier-free-guidance batch is steered; self-attention is unchanged.
"""

import math

import torch
import torch.nn.functional as F


class SteeredCrossAttnProcessor:
    def __init__(self, state):
        self.state = state  # shared: heat [64, 64] or None, active, first_key, floor, boost, cfg

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is not None and attn.norm_cross:
            context = attn.norm_encoder_hidden_states(context)
        batch, seq, _ = hidden_states.shape
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(context))
        value = attn.head_to_batch_dim(attn.to_v(context))
        probs = attn.get_attention_scores(query, key, attention_mask)  # [batch * heads, seq, keys]

        s = self.state
        if encoder_hidden_states is not None and s.get("active") and s.get("heat") is not None:
            side = math.isqrt(seq)
            w = F.interpolate(s["heat"][None, None].to(probs.dtype), size=(side, side), mode="area").flatten()
            w = (s["floor"] + (1 - s["floor"]) * w) * (1 + s["boost"] * w)
            p = probs.view(batch, attn.heads, seq, -1)
            half = slice(batch // 2, batch) if s.get("cfg", True) else slice(0, batch)
            steered = p[half].clone()
            steered[..., s["first_key"]:] *= w[None, None, :, None]
            p[half] = steered / steered.sum(-1, keepdim=True)
            probs = p.view(batch * attn.heads, seq, -1)

        out = attn.batch_to_head_dim(torch.bmm(probs, value))
        out = attn.to_out[1](attn.to_out[0](out))
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


class steer_cross_attention:
    """Context manager: route every cross-attention layer of `unet` through SteeredCrossAttnProcessor."""

    def __init__(self, unet, state):
        self.unet, self.state = unet, state

    def __enter__(self):
        self.saved = dict(self.unet.attn_processors)
        procs = {k: (SteeredCrossAttnProcessor(self.state) if k.endswith("attn2.processor") else v)
                 for k, v in self.saved.items()}
        self.unet.set_attn_processor(procs)
        return self.state

    def __exit__(self, *exc):
        self.unet.set_attn_processor(self.saved)
        return False
