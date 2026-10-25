"""Tests for the CLAP -> AudioAdapter -> CLIP/U-Net conditioning pipeline.

Run from the project root:
    python -m unittest tests.test_audio_adapter -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.audio_adapter import AudioAdapter, FeatureProjection, MultiTokenAttentionPooling  # noqa: E402
from models.audio_token_conditioner import DEFAULT_SD_PATH, AudioTokenConditioner  # noqa: E402
from models.clap_feature_extractor import CLAPFeatureExtractor  # noqa: E402

S4_WAV = ROOT / "data/avsbench_data_processed/Single-source/s4_data/audio_wav"
WAV_A = S4_WAV / "test/baby_laughter/MrIfjRInMLU.wav"
WAV_B = S4_WAV / "train/ambulance_siren/-jMGDLoDR48.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED = 768
N_TOKENS = 8


def all_finite(t):
    return bool(torch.isfinite(t).all())


class AudioPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.extractor = CLAPFeatureExtractor(device=DEVICE)
        cls.feats_1 = cls.extractor.extract([str(WAV_A)])
        cls.feats_2 = cls.extractor.extract([str(WAV_A), str(WAV_B)])
        cls.conditioner = AudioTokenConditioner(num_tokens=N_TOKENS, device=DEVICE)

    def make_adapter(self, **kw):
        return AudioAdapter(self.extractor.feature_dims, self.extractor.global_dim, **kw).to(DEVICE)

    # T1
    def test_1_feature_extraction(self):
        f = self.feats_1
        self.assertEqual(f["names"], ["layers.1", "layers.2", "norm"])
        self.assertEqual([tuple(t.shape) for t in f["intermediates"]], [(1, 64, 384), (1, 32, 768), (1, 32, 768)])
        self.assertEqual(tuple(f["global_embed"].shape), (1, 512))
        for t in f["intermediates"] + [f["global_embed"]]:
            self.assertTrue(all_finite(t))
            self.assertFalse(t.requires_grad)
        torch.testing.assert_close(f["global_embed"].norm(dim=-1), torch.ones(1, device=f["global_embed"].device))

        # Layout check: time-mean of un-folded 'norm' tokens == CLAP's own pre-projection 'embedding'.
        clap = self.extractor.clap
        with torch.no_grad():
            batch = self.extractor._prepare(self.extractor.load_audio([str(WAV_A)]))
            inp = {k: torch.stack([d[k] for d in batch]).to(self.extractor.device) for k in batch[0]}
            ref = clap.encode_audio(inp, device=self.extractor.device)["embedding"]
        torch.testing.assert_close(f["intermediates"][2].mean(dim=1), ref, rtol=1e-4, atol=1e-5)

    # T2
    def test_2_adapter_forward_real_features(self):
        adapter = self.make_adapter()
        inter, g = self.feats_1["intermediates"], self.feats_1["global_embed"]
        tokens, dbg = adapter(inter, g, return_debug=True)
        expected = [(1, t.shape[1], EMBED) for t in inter] + [(1, 1, EMBED)]
        self.assertEqual(dbg["projected_shapes"], expected)
        t_total = sum(t.shape[1] for t in inter) + 1
        self.assertEqual(t_total, 129)
        self.assertEqual(dbg["concat_shape"], (1, t_total, EMBED))
        self.assertEqual(dbg["mlp_shape"], (1, t_total, EMBED))
        self.assertEqual(tuple(dbg["attn_weights"].shape), (1, N_TOKENS, t_total))
        self.assertEqual(tuple(tokens.shape), (1, N_TOKENS, EMBED))
        self.assertTrue(all_finite(tokens))

    # T3
    def test_3_backprop(self):
        adapter = self.make_adapter()
        B = 2
        inter = [torch.randn(B, t, d, device=DEVICE) for t, d in zip(self.extractor.seq_lens, self.extractor.feature_dims)]
        g = torch.randn(B, self.extractor.global_dim, device=DEVICE)
        loss = adapter(inter, g).pow(2).mean()  # gradient-flow check only, not a training objective
        loss.backward()
        for name, p in adapter.named_parameters():
            self.assertTrue(p.requires_grad, name)
            self.assertIsNotNone(p.grad, f"{name} got no gradient")
            self.assertTrue(all_finite(p.grad), name)
            self.assertGreater(p.grad.abs().sum().item(), 0.0, f"{name} has an all-zero gradient")
        for name, p in self.extractor.clap.named_parameters():
            self.assertFalse(p.requires_grad, name)
            self.assertIsNone(p.grad, name)

    # T4
    def test_4_batch(self):
        f = self.feats_2
        for t in f["intermediates"] + [f["global_embed"]]:
            self.assertEqual(t.shape[0], 2)
        adapter = self.make_adapter().eval()
        tokens = adapter(f["intermediates"], f["global_embed"])
        self.assertEqual(tuple(tokens.shape), (2, N_TOKENS, EMBED))
        self.assertFalse(torch.allclose(tokens[0], tokens[1]), "different clips gave identical tokens")
        # Batched extraction matches single-sample extraction for the shared clip.
        for single, batched in zip(self.feats_1["intermediates"], f["intermediates"]):
            torch.testing.assert_close(batched[:1], single, rtol=1e-4, atol=1e-5)

    # T5
    def test_5_save_load(self):
        inter, g = self.feats_2["intermediates"], self.feats_2["global_embed"]
        for kw in ({}, {"use_source_embeddings": False}, {"num_tokens": 1}):
            with self.subTest(**kw):
                a = self.make_adapter(**kw).eval()
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / "adapter.pt"
                    torch.save({"config": a.config, "state_dict": a.state_dict()}, path)
                    ckpt = torch.load(path, map_location=DEVICE)
                b = AudioAdapter(**ckpt["config"]).to(DEVICE).eval()
                b.load_state_dict(ckpt["state_dict"])
                with torch.no_grad():
                    torch.testing.assert_close(a(inter, g), b(inter, g), rtol=0, atol=0)
                self.assertEqual(tuple(b(inter, g).shape), (2, kw.get("num_tokens", N_TOKENS), EMBED))
                self.assertEqual(b.source_embeddings is None, kw.get("use_source_embeddings") is False)

    def test_5b_component_validation(self):
        proj = FeatureProjection(384)
        self.assertEqual(tuple(proj(torch.randn(2, 5, 384)).shape), (2, 5, EMBED))
        self.assertEqual(tuple(proj(torch.randn(2, 384)).shape), (2, EMBED))
        with self.assertRaises(ValueError):
            proj(torch.randn(2, 5, 383))
        with self.assertRaises(ValueError):
            MultiTokenAttentionPooling(768, num_heads=7)
        pool = MultiTokenAttentionPooling(768, num_tokens=4, num_heads=8)
        self.assertEqual(tuple(pool(torch.randn(3, 11, 768)).shape), (3, 4, 768))
        with self.assertRaises(ValueError):
            self.make_adapter()(self.feats_1["intermediates"][:2], self.feats_1["global_embed"])

    # T6
    def test_6_conditioner(self):
        cond = self.conditioner
        tok = cond.tokenizer
        ids = cond.input_ids[0].tolist()
        self.assertEqual(len(ids), 77)
        self.assertEqual(ids[:cond.start], tok("a photo of a").input_ids[:-1])
        self.assertEqual(ids[cond.start:cond.start + N_TOKENS], [cond.placeholder_id] * N_TOKENS)
        self.assertEqual(ids[cond.start + N_TOKENS], tok.eos_token_id)

        # Hook sanity: splicing in the placeholder's own embeddings == plain text-encoder forward.
        with torch.no_grad():
            ref = cond.text_encoder(input_ids=cond.input_ids).last_hidden_state
            own = cond.token_embedding(cond.input_ids)[:, cond.start:cond.start + N_TOKENS]
            torch.testing.assert_close(cond(own), ref)

        # Gradients flow through the frozen CLIP text encoder into the adapter.
        adapter = self.make_adapter()
        hidden = cond(adapter(self.feats_2["intermediates"], self.feats_2["global_embed"]))
        self.assertEqual(tuple(hidden.shape), (2, 77, EMBED))
        self.assertTrue(all_finite(hidden))
        self.assertFalse(torch.allclose(hidden[0], hidden[1]))
        hidden.pow(2).mean().backward()
        for name, p in adapter.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} got no gradient through CLIP")
            self.assertGreater(p.grad.abs().sum().item(), 0.0, name)
        for name, p in cond.text_encoder.named_parameters():
            self.assertFalse(p.requires_grad, name)
            self.assertIsNone(p.grad, name)

    # T7
    @unittest.skipUnless(torch.cuda.is_available(), "U-Net forward test needs CUDA")
    def test_7_unet_compatibility(self):
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(DEFAULT_SD_PATH, subfolder="unet", variant="fp16",
                                                    torch_dtype=torch.float16).to(DEVICE).eval()
        self.assertEqual(unet.config.cross_attention_dim, EMBED)
        self.assertEqual(unet.config.in_channels, 9)
        adapter = self.make_adapter().eval()
        with torch.no_grad():
            tokens = adapter(self.feats_2["intermediates"], self.feats_2["global_embed"])
            hidden = self.conditioner(tokens).half()
            sample = torch.randn(2, 9, 64, 64, device=DEVICE, dtype=torch.float16)
            t = torch.tensor([500, 500], device=DEVICE)
            out = unet(sample, t, encoder_hidden_states=hidden).sample
            self.assertEqual(tuple(out.shape), (2, 4, 64, 64))
            self.assertTrue(all_finite(out))
            # Informational: the U-Net also accepts direct [B, N, 768] (not the chosen interface).
            direct = unet(sample, t, encoder_hidden_states=tokens.half()).sample
            self.assertEqual(tuple(direct.shape), (2, 4, 64, 64))
        del unet
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
