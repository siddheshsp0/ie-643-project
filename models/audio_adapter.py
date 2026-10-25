"""The trainable part: maps CLAP features to N audio tokens in CLIP's word-embedding space (768-d).

Based on the AudioToken embedder. Differences: the CLAP stages are concatenated along time (they
have different lengths and widths), each source gets a learned embedding, and the pooling gives
N tokens instead of one.
"""

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """Linear map of the last axis: [B, T, D_in] -> [B, T, D_out] or [B, D_in] -> [B, D_out]."""

    def __init__(self, input_dim, output_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() not in (2, 3):
            raise ValueError(f"FeatureProjection expects [B, D] or [B, T, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"FeatureProjection expected last dim {self.input_dim}, got {tuple(x.shape)}")
        return self.proj(x)


class MultiTokenAttentionPooling(nn.Module):
    """Pool [B, T, D] into [B, N, D] with N learned queries attending over the sequence."""

    def __init__(self, embed_dim=768, num_tokens=8, num_heads=8, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.queries = nn.Parameter(torch.empty(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, return_attn=False):
        if x.dim() != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"attention pooling expects [B, T, {self.embed_dim}], got {tuple(x.shape)}")
        q = self.queries.expand(x.shape[0], -1, -1)
        out, weights = self.attn(q, x, x, need_weights=return_attn, average_attn_weights=True)
        assert out.shape == (x.shape[0], self.num_tokens, self.embed_dim), tuple(out.shape)
        return (out, weights) if return_attn else out


class AudioAdapter(nn.Module):
    """CLAP intermediates + global embedding -> [B, num_tokens, embed_dim]."""

    def __init__(self, intermediate_dims, global_dim, embed_dim=768, hidden_dim=1024,
                 num_tokens=8, num_heads=8, use_source_embeddings=True):
        super().__init__()
        intermediate_dims = [int(d) for d in intermediate_dims]
        if not intermediate_dims:
            raise ValueError("at least one intermediate representation is required")
        self.config = dict(intermediate_dims=intermediate_dims, global_dim=int(global_dim),
                           embed_dim=embed_dim, hidden_dim=hidden_dim, num_tokens=num_tokens,
                           num_heads=num_heads, use_source_embeddings=use_source_embeddings)
        self.embed_dim = embed_dim
        self.num_sources = len(intermediate_dims) + 1  # + global token

        self.intermediate_projs = nn.ModuleList(FeatureProjection(d, embed_dim) for d in intermediate_dims)
        self.global_proj = FeatureProjection(global_dim, embed_dim)
        if use_source_embeddings:
            self.source_embeddings = nn.Parameter(torch.empty(self.num_sources, embed_dim))
            nn.init.trunc_normal_(self.source_embeddings, std=0.02)
        else:
            self.register_parameter("source_embeddings", None)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))
        self.pool = MultiTokenAttentionPooling(embed_dim, num_tokens, num_heads)

    def forward(self, intermediates, global_embed, return_debug=False):
        if len(intermediates) != len(self.intermediate_projs):
            raise ValueError(f"expected {len(self.intermediate_projs)} intermediate tensors, got {len(intermediates)}")
        if global_embed.dim() != 2:
            raise ValueError(f"global embedding must be [B, D], got {tuple(global_embed.shape)}")
        B = global_embed.shape[0]

        projected = []
        for i, (x, proj) in enumerate(zip(intermediates, self.intermediate_projs)):
            if x.dim() != 3 or x.shape[0] != B:
                raise ValueError(f"intermediate {i} must be [B={B}, T, D], got {tuple(x.shape)}")
            y = proj(x)
            assert y.shape == (B, x.shape[1], self.embed_dim), f"projection {i}: {tuple(y.shape)}"
            projected.append(y)
        g = self.global_proj(global_embed).unsqueeze(1)
        assert g.shape == (B, 1, self.embed_dim), f"global projection: {tuple(g.shape)}"
        projected.append(g)

        if self.source_embeddings is not None:
            projected = [y + self.source_embeddings[i] for i, y in enumerate(projected)]

        seq = torch.cat(projected, dim=1)
        t_total = sum(x.shape[1] for x in intermediates) + 1
        assert seq.shape == (B, t_total, self.embed_dim), f"concat: {tuple(seq.shape)} != {(B, t_total, self.embed_dim)}"

        hidden = self.mlp(seq)
        assert hidden.shape == seq.shape, f"MLP changed shape {tuple(seq.shape)} -> {tuple(hidden.shape)}"

        if not return_debug:
            return self.pool(hidden)
        tokens, attn = self.pool(hidden, return_attn=True)
        debug = {
            "projected_shapes": [tuple(y.shape) for y in projected],
            "concat_shape": tuple(seq.shape),
            "mlp_shape": tuple(hidden.shape),
            "attn_weights": attn,  # [B, N, T_total]
        }
        return tokens, debug
