"""Tests for models/location_unet.py on a tiny random U-Net (CPU, no weights needed).

    python -m unittest tests.test_location_unet -v
"""

import unittest

import torch
from diffusers import UNet2DConditionModel

from models.location_unet import (LoRALinear, add_location_inputs, gaussian_heatmap, load_location_state,
                                  location_state_dict)


def tiny_unet(seed=0):
    torch.manual_seed(seed)
    return UNet2DConditionModel(sample_size=16, in_channels=9, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32,
                                attention_head_dim=8, norm_num_groups=8).eval()


class LocationUNetTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(2, 9, 16, 16, generator=g)
        self.heat = torch.rand(2, 1, 16, 16, generator=g)
        self.ctx = torch.randn(2, 7, 32, generator=g)
        self.t = torch.tensor([10, 500])

    def test_untrained_model_equals_the_original(self):
        for multiscale in (False, True):
            unet = tiny_unet()
            with torch.no_grad():
                ref = unet(self.x, self.t, encoder_hidden_states=self.ctx).sample
                params = add_location_inputs(unet, rank=4, alpha=4.0, multiscale=multiscale)
                out = unet(torch.cat([self.x, self.heat], 1), self.t, encoder_hidden_states=self.ctx).sample
            torch.testing.assert_close(out, ref)  # zero-initialised inputs and LoRA: any heatmap changes nothing
            self.assertEqual(unet.conv_in.in_channels, 10)
            self.assertEqual(hasattr(unet, "location_injectors"), multiscale)
            self.assertGreater(sum(isinstance(m, LoRALinear) for m in unet.modules()), 0)
            ids = {id(p) for p in params}
            stray = [n for n, p in unet.named_parameters() if p.requires_grad and id(p) not in ids]
            self.assertEqual(stray, [])

    def test_trained_state_round_trip(self):
        for multiscale in (False, True):
            unet = tiny_unet()
            params = add_location_inputs(unet, rank=4, alpha=4.0, multiscale=multiscale)
            with torch.no_grad():
                for p in params:
                    p.add_(0.05 * torch.randn_like(p))
                out = unet(torch.cat([self.x, self.heat], 1), self.t, encoder_hidden_states=self.ctx).sample
                no_heat = unet(torch.cat([self.x, 0 * self.heat], 1), self.t, encoder_hidden_states=self.ctx).sample
            self.assertFalse(torch.allclose(out, no_heat))  # once trained, the heatmap matters
            fresh = load_location_state(tiny_unet(), location_state_dict(unet), rank=4, alpha=4.0)
            self.assertEqual(hasattr(fresh, "location_injectors"), multiscale)
            with torch.no_grad():
                again = fresh(torch.cat([self.x, self.heat], 1), self.t, encoder_hidden_states=self.ctx).sample
            torch.testing.assert_close(again, out)

    def test_heatmap_peak(self):
        h = gaussian_heatmap(100.0, 300.0, 20.0, 40.0, 64, scale=8.0)
        y, x = divmod(int(h.argmax()), 64)
        self.assertEqual((x, y), (12, 37))  # pixel (100, 300) lies in latent cell (12, 37)
        self.assertAlmostEqual(float(h.max()), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
