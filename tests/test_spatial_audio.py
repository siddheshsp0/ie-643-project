"""Tests for models/spatial_audio.py: FOA round trips and camera geometry (CPU only, no model weights).

    python -m unittest tests.test_spatial_audio -v
"""

import unittest

import numpy as np

from models.megascenes_warp import megascenes_intrinsics, orbit_extrinsic
from models.spatial_audio import (FOA_SR, analyse_foa, direction_vector, estimate_distance, from_fuma,
                                  half_fov_deg, mic_to_camera, pixel_to_direction, project_point, synthesize_foa,
                                  yaw_to_face)


def test_signal(seconds=3.0, seed=0):
    """Broadband bursts (like barks or calls): white noise gated on for 0.15 s every 0.3 s."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FOA_SR)
    env = 0.05 + 0.95 * ((np.arange(n) / FOA_SR) % 0.3 < 0.15)
    return rng.standard_normal(n) * env


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


class FoaRoundTripTest(unittest.TestCase):
    def test_direction_recovered_in_a_reverberant_room(self):
        s = test_signal()
        for az, el in [(0, 0), (35, 10), (-70, 0), (120, -15), (-150, 25)]:
            for r in (1.0, 3.0):
                est = analyse_foa(synthesize_foa(s, az, el, distance_m=r, seed=abs(az) + 7))
                err = angle_between(direction_vector(az, el), direction_vector(est["az_deg"], est["el_deg"]))
                self.assertLess(err, 8.0, f"az {az} el {el} r {r}: estimated {est}")

    def test_distance_recovered(self):
        s = test_signal()
        est = {r: estimate_distance(analyse_foa(synthesize_foa(s, 20, 0, distance_m=r, seed=3))["diffuseness"])
               for r in (0.7, 1.5, 3.0, 5.0)}
        for r in (0.7, 1.5, 3.0):
            self.assertLess(abs(est[r] - r) / r, 0.3, f"true {r} m, estimated {est[r]:.2f} m")
        # far past the critical distance the reverberant field dominates and estimates compress,
        # but a farther source must still come out farther
        self.assertGreater(est[5.0], est[3.0])

    def test_fuma_conversion(self):
        foa = np.random.default_rng(1).standard_normal((4, 10))
        acn = from_fuma(foa)
        np.testing.assert_allclose(acn[0], foa[0] * np.sqrt(2))
        np.testing.assert_allclose(acn[1:], foa[[2, 3, 1]])


class CameraGeometryTest(unittest.TestCase):
    def setUp(self):
        self.K = megascenes_intrinsics(512, 512)

    def test_mic_axes_map_to_camera_axes(self):
        np.testing.assert_allclose(mic_to_camera(direction_vector(0, 0)), [0, 0, 1], atol=1e-12)  # ahead
        self.assertLess(mic_to_camera(direction_vector(30, 0))[0], 0)  # left of centre
        self.assertLess(mic_to_camera(direction_vector(0, 20))[1], 0)  # above centre

    def test_pixel_direction_round_trip(self):
        for az, el in [(0, 0), (25, 5), (-40, -10)]:
            u, v, _ = project_point(self.K, np.eye(4), 3.0 * mic_to_camera(direction_vector(az, el)))
            np.testing.assert_allclose(pixel_to_direction(u, v, self.K), (az, el), atol=1e-6)

    def test_yaw_brings_source_into_view(self):
        half = half_fov_deg(self.K, 512)
        self.assertEqual(yaw_to_face(10, self.K, 512), 0.0)
        for az in (70, -70, 90):
            yaw = yaw_to_face(az, self.K, 512, margin_deg=15, max_yaw_deg=30)
            self.assertEqual(np.sign(yaw), np.sign(az))
            self.assertLessEqual(abs(yaw), 30)
            E = orbit_extrinsic(yaw, tz=abs(np.sin(np.radians(yaw))))
            u, _, z = project_point(self.K, E, 3.0 * mic_to_camera(direction_vector(az, 0)))
            if abs(az) - abs(yaw) < half:  # reachable within the cap
                self.assertTrue(0 <= u <= 512 and z > 0, f"az {az}: u {u}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
