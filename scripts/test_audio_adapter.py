"""Shape check: prints tensor shapes for WAV -> CLAP -> AudioAdapter -> CLIP conditioning [B, 77, 768].

The adapter here is freshly initialised (no checkpoint is loaded) and no image is generated; use
sample_audio_inpaint.py for inference.

Usage (from the project root):
    python scripts/test_audio_adapter.py [wav ...]
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.audio_adapter import AudioAdapter  # noqa: E402
from models.audio_token_conditioner import AudioTokenConditioner  # noqa: E402
from models.clap_feature_extractor import CLAPFeatureExtractor  # noqa: E402

S4_WAV = ROOT / "data/avsbench_data_processed/Single-source/s4_data/audio_wav"
DEFAULT_WAVS = [S4_WAV / "test/baby_laughter/MrIfjRInMLU.wav", S4_WAV / "train/ambulance_siren/-jMGDLoDR48.wav"]


def finite(t):
    return bool(torch.isfinite(t).all())


def main():
    wavs = [Path(p) for p in sys.argv[1:]] or DEFAULT_WAVS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    print("== CLAP feature extraction")
    extractor = CLAPFeatureExtractor(device=device)
    print(f"checkpoint : {extractor.ckpt_path}")
    branch = extractor.clap.audio_branch
    print(f"audio branch: {type(branch).__name__}  depths={branch.depths}  embed_dim={branch.embed_dim}  "
          f"freq_ratio={branch.freq_ratio}  spec_size={branch.spec_size}")
    for w in wavs:
        print(f"wav        : {w}")
    out = extractor.extract([str(w) for w in wavs])
    for name, t in zip(out["names"], out["intermediates"]):
        print(f"  {name:<10} -> {tuple(t.shape)}  [B, T, D]  finite={finite(t)}")
    g = out["global_embed"]
    print(f"  {'global':<10} -> {tuple(g.shape)}  [B, D]  finite={finite(g)}  L2 norm={g.norm(dim=-1).tolist()}")

    print("\n== AudioAdapter")
    adapter = AudioAdapter(extractor.feature_dims, extractor.global_dim).to(device)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"config     : {adapter.config}")
    print(f"trainable parameters: {n_params:,}")
    tokens, dbg = adapter(out["intermediates"], out["global_embed"], return_debug=True)
    for name, s in zip(out["names"] + ["global"], dbg["projected_shapes"]):
        print(f"  projected {name:<10} -> {s}")
    print(f"  concat -> {dbg['concat_shape']}   MLP -> {dbg['mlp_shape']}   attn -> {tuple(dbg['attn_weights'].shape)}")
    print(f"  tokens -> {tuple(tokens.shape)}  finite={finite(tokens)}")

    print("\n== AudioTokenConditioner (frozen CLIP text encoder)")
    cond = AudioTokenConditioner(num_tokens=adapter.pool.num_tokens, device=device)
    ids = cond.input_ids[0].tolist()
    print(f"prompt ids : {cond.tokenizer.convert_ids_to_tokens(ids[:cond.start + cond.num_tokens + 1])} + pad")
    hidden = cond(tokens)
    print(f"  encoder_hidden_states -> {tuple(hidden.shape)}  finite={finite(hidden)}")


if __name__ == "__main__":
    main()
