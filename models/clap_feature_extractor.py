"""Frozen LAION-CLAP (HTSAT-tiny) audio encoder that also returns intermediate stage outputs.

HTSAT folds the 1024 spectrogram frames into a 256x256 "image" (4 time chunks stacked vertically),
so each hooked stage is a grid of tokens. unfold_tokens undoes the folding and averages over
frequency only, which keeps the time axis: layers.1 gives T=64, layers.2 and norm give T=32.
"""

import math
import os

import laion_clap
import librosa
import torch
import torch.nn as nn
from laion_clap.training.data import get_audio_features

CLAP_SAMPLE_RATE = 48000
CLAP_MAX_SAMPLES = 480000  # 10 s at 48 kHz
DEFAULT_CKPT = os.path.join(os.path.dirname(laion_clap.__file__), "630k-audioset-best.pt")
DEFAULT_LAYERS = ("layers.1", "layers.2", "norm")


def _int16_roundtrip(x):
    # Same quantization as laion_clap.hook (float32 -> int16 -> float32).
    x = x.clamp(-1.0, 1.0)
    return (x * 32767.0).to(torch.int16).to(torch.float32) / 32767.0


class CLAPFeatureExtractor(nn.Module):
    """extract(wav_paths) -> {"names", "intermediates": list of [B, T_i, D_i], "global_embed": [B, 512]}.

    The global embedding is CLAP's usual L2-normalized audio embedding. Nothing here is trained.
    """

    def __init__(self, ckpt_path=DEFAULT_CKPT, layers=DEFAULT_LAYERS, device=None, freq_pool="mean"):
        super().__init__()
        if freq_pool not in ("mean", "flatten"):
            raise ValueError(f"freq_pool must be 'mean' or 'flatten', got {freq_pool!r}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"CLAP checkpoint not found: {ckpt_path}")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.ckpt_path = ckpt_path
        self.freq_pool = freq_pool
        self.layer_names = tuple(layers)

        clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny", device=device)
        clap.load_ckpt(ckpt_path, verbose=False)
        self.clap = clap.model  # CLAP nn.Module (audio + text branches)
        self.audio_cfg = clap.model_cfg["audio_cfg"]
        self.clap.eval()
        self.clap.requires_grad_(False)

        branch = self.clap.audio_branch
        self.freq_ratio = branch.freq_ratio
        self._captured = {}
        self._hooks = []
        for name in self.layer_names:
            try:
                module = branch.get_submodule(name)
            except AttributeError:
                valid = [n for n, _ in branch.named_children()] + [f"layers.{i}" for i in range(len(branch.layers))]
                raise ValueError(f"'{name}' is not a submodule of the CLAP audio branch; e.g. {valid}") from None
            self._hooks.append(module.register_forward_hook(self._make_hook(name)))

        # run 10 s of silence once to get the feature sizes instead of hardcoding them
        probe = self.extract_from_waveforms([torch.zeros(CLAP_MAX_SAMPLES)])
        self.feature_dims = [t.shape[-1] for t in probe["intermediates"]]
        self.seq_lens = [t.shape[1] for t in probe["intermediates"]]
        self.global_dim = probe["global_embed"].shape[-1]

    @property
    def device(self):
        return next(self.clap.parameters()).device

    def train(self, mode=True):
        # CLAP must stay in eval mode (BatchNorm stats, SpecAugment, dropout).
        super().train(mode)
        self.clap.eval()
        return self

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # Swin stages return (x, attn); LayerNorm returns x.
            self._captured[name] = output[0] if isinstance(output, tuple) else output
        return hook

    def unfold_tokens(self, x, name="?"):
        """[B, H*W, C] folded-grid tokens -> [B, T, C] (or [B, F_bins*T, C] if freq_pool='flatten')."""
        if x.dim() != 3:
            raise ValueError(f"{name}: expected [B, L, C] tokens, got {tuple(x.shape)}")
        B, L, C = x.shape
        side = math.isqrt(L)
        if side * side != L:
            raise ValueError(f"{name}: token count {L} is not a square grid")
        if side % self.freq_ratio != 0:
            raise ValueError(f"{name}: grid height {side} not divisible by freq_ratio {self.freq_ratio}")
        grid = x.transpose(1, 2).reshape(B, C, side, side)  # row-major (height, width) token order
        f_bins = side // self.freq_ratio
        # height = chunk * f_bins + f  ->  [B, C, f_bins, chunk * width]   (chronological time)
        grid = grid.reshape(B, C, self.freq_ratio, f_bins, side).permute(0, 1, 3, 2, 4)
        grid = grid.reshape(B, C, f_bins, self.freq_ratio * side)
        if self.freq_pool == "mean":
            return grid.mean(dim=2).transpose(1, 2).contiguous()  # [B, T, C]
        return grid.flatten(2).transpose(1, 2).contiguous()  # [B, f_bins*T, C]

    @staticmethod
    def load_audio(paths):
        """Load WAVs as 48 kHz mono tensors, the same way laion_clap does."""
        waveforms = []
        for p in paths:
            y, _ = librosa.load(p, sr=CLAP_SAMPLE_RATE, mono=True)
            waveforms.append(torch.from_numpy(y).float())
        return waveforms

    def _prepare(self, waveforms):
        batch = []
        for w in waveforms:
            if w.dim() != 1:
                raise ValueError(f"expected mono 1-D waveform at 48 kHz, got shape {tuple(w.shape)}")
            # crop from the start ourselves so 'rand_trunc' below never picks a random offset
            w = _int16_roundtrip(w.detach().float().cpu()[:CLAP_MAX_SAMPLES])
            batch.append(get_audio_features({}, w, CLAP_MAX_SAMPLES, data_truncating="rand_trunc",
                                            data_filling="repeatpad", audio_cfg=self.audio_cfg))
        return batch

    @torch.no_grad()
    def extract_from_waveforms(self, waveforms):
        self._captured.clear()
        global_embed = self.clap.get_audio_embedding(self._prepare(waveforms))
        missing = [n for n in self.layer_names if n not in self._captured]
        if missing:
            raise RuntimeError(f"forward hooks did not fire for {missing}")
        intermediates = [self.unfold_tokens(self._captured[n], n) for n in self.layer_names]
        self._captured.clear()
        B = len(waveforms)
        for n, t in zip(self.layer_names, intermediates):
            assert t.shape[0] == B, f"{n}: batch {t.shape[0]} != {B}"
        assert global_embed.shape[0] == B
        return {"names": list(self.layer_names), "intermediates": intermediates, "global_embed": global_embed}

    def extract(self, paths):
        return self.extract_from_waveforms(self.load_audio(paths))

    def forward(self, paths):
        return self.extract(paths)
