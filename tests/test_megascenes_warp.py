"""Tests for the MegaScenes-style warp (models/megascenes_warp.py) and depth normalization.

Run from the project root (rendering needs pyrender with EGL or OSMesa):
    python -m unittest tests.test_megascenes_warp -v
"""

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.depth_anything_v2 import disparity_to_depth  # noqa: E402
from models.megascenes_warp import (MEGASCENES_REPO, load_util_3dphoto, megascenes_intrinsics,  # noqa: E402
                                    orbit_extrinsic, warp_to_view)


def megascenes_pose_functions():
    """get_ccw_poses / get_cw_poses from MegaScenes' evalhelpers.py (just these two, the module imports DA v1)."""
    path = MEGASCENES_REPO / "dataloader/evalhelpers.py"
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("get_ccw_poses", "get_cw_poses")]
    namespace = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_ccw_poses"], namespace["get_cw_poses"]


def project(E, K, p):
    """Pixel (u, v) of input-camera point p seen from extrinsic E."""
    q = E[:3, :3] @ p + E[:3, 3]
    return K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]


class OrbitPoseTest(unittest.TestCase):
    def test_matches_megascenes_orbit(self):
        get_ccw_poses, get_cw_poses = megascenes_pose_functions()
        ccw = get_ccw_poses(c2w=False)  # default orbit: 10 steps, 30 deg, x: 0 -> -1
        cw = get_cw_poses(c2w=False)
        for step in range(10):
            yaw = 30.0 * step / 10
            np.testing.assert_allclose(orbit_extrinsic(yaw), ccw[step], atol=1e-12)
            np.testing.assert_allclose(orbit_extrinsic(-yaw), cw[step], atol=1e-12)

    def test_default_is_15_degrees(self):
        E = orbit_extrinsic()
        self.assertAlmostEqual(np.degrees(np.arctan2(E[0, 2], E[0, 0])), 15.0)
        np.testing.assert_allclose(E[:3, 3], [-15 / 27, 0.0, np.sin(np.radians(15))])

    def test_orbit_pivot_stays_centered(self):
        K = megascenes_intrinsics(512, 512)
        E = orbit_extrinsic(15.0)
        pivot = -E[0, 3] / np.sin(np.radians(15.0))
        u, v = project(E, K, np.array([0.0, 0.0, pivot]))
        self.assertAlmostEqual(u, 256.0, places=6)
        self.assertAlmostEqual(v, 256.0, places=6)
        # near content moves left, far content moves right (camera moves right, turns left)
        self.assertLess(project(E, K, np.array([0.0, 0.0, 1.0]))[0], 256.0)
        self.assertGreater(project(E, K, np.array([0.0, 0.0, 10.0]))[0], 256.0)

    def test_intrinsics_scale_with_resolution(self):
        np.testing.assert_allclose(megascenes_intrinsics(256, 256)[0, 0], 80.0)
        K = megascenes_intrinsics(384, 512)
        np.testing.assert_allclose([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], [160.0, 160.0, 256.0, 192.0])


class DepthNormalizationTest(unittest.TestCase):
    def test_quantile_and_clipping(self):
        disp = np.linspace(0.0, 10.0, 10001).reshape(73, 137)  # includes 0 -> clamped at 1e-3
        depth = disparity_to_depth(disp)
        self.assertAlmostEqual(float(np.quantile(depth, 0.2)), 1.0, places=4)
        self.assertLessEqual(depth.max(), 100.0)
        self.assertTrue(np.all(np.diff(depth.reshape(-1)) <= 1e-6))  # larger disparity = closer


class WarpTest(unittest.TestCase):
    H = W = 96
    F = 96.0

    @classmethod
    def setUpClass(cls):
        try:
            load_util_3dphoto()
        except Exception as e:  # no pyrender / OpenGL backend
            raise unittest.SkipTest(f"MegaScenes renderer unavailable: {e}")
        cls.K = np.array([[cls.F, 0, cls.W / 2], [0, cls.F, cls.H / 2], [0, 0, 1]])
        # red channel encodes the source column, green the source row
        cols, rows = np.meshgrid(np.arange(cls.W), np.arange(cls.H))
        cls.rgb = np.stack([cols * 2, rows * 2, np.full_like(cols, 128)], -1).astype(np.uint8)

    def box_scene(self):
        """Far wall at depth 4 with a near box at depth 1 over rows/columns 36..59."""
        depth = np.full((self.H, self.W), 4.0, np.float32)
        depth[36:60, 36:60] = 1.0
        return depth

    def floor_scene(self):
        """Ground plane 0.5 below the camera, seen at a grazing angle in the lower half (sky above)."""
        rows = np.arange(self.H, dtype=np.float32)[:, None] + 0.5 - self.H / 2
        depth = np.where(rows > 0, 0.5 * self.F / np.maximum(rows, 1e-6), 100.0)
        return np.broadcast_to(np.minimum(depth, 100.0), (self.H, self.W)).astype(np.float32)

    def test_identity_reproduces_input_without_holes(self):
        cases = [("megascenes", np.full((self.H, self.W), 2.0, np.float32)),
                 ("stretch", np.full((self.H, self.W), 2.0, np.float32)),
                 ("stretch", self.box_scene()), ("stretch", self.floor_scene())]
        for prune, depth in cases:
            with self.subTest(prune=prune, depth_range=(float(depth.min()), float(depth.max()))):
                warped, rdepth, valid = warp_to_view(self.rgb, depth, self.K, np.eye(4), prune=prune)
                self.assertTrue(valid.all())
                self.assertLessEqual(np.abs(warped.astype(int) - self.rgb).max(), 1)
                # pyrender reads depth from one 4x MSAA sub-sample, so only planes give exact depth
                if depth.min() == depth.max():
                    np.testing.assert_allclose(rdepth, depth, rtol=1e-3)

    def test_megascenes_rule_drops_grazing_surfaces(self):
        _, _, valid = warp_to_view(self.rgb, self.floor_scene(), self.K, np.eye(4), prune="megascenes")
        self.assertGreater((~valid[self.H // 2:]).mean(), 0.2)  # seen, yet missing

    def test_translation_parallax_and_border_hole(self):
        d, tx = 2.0, -0.25  # camera moves 0.25 to the right
        shift = self.F * -tx / d  # content moves left by 12 px
        E = np.eye(4)
        E[0, 3] = tx
        for prune in ("stretch", "megascenes"):
            with self.subTest(prune=prune):
                warped, _, valid = warp_to_view(self.rgb, np.full((self.H, self.W), d, np.float32), self.K, E,
                                                prune=prune)
                src_col = warped[..., 0].astype(float) / 2
                u = np.arange(self.W)[None].repeat(self.H, 0)
                self.assertAlmostEqual(float(np.median((src_col - u)[valid])), shift, delta=1.0)
                self.assertTrue(valid[:, : self.W - int(np.ceil(shift))].all())
                self.assertFalse(valid[:, self.W - int(shift) + 1:].any())

    def test_disocclusion_behind_near_object(self):
        E = np.eye(4)
        E[0, 3] = -0.3  # camera moves right: box moves left by 28.8 px, wall by 7.2 px
        for prune in ("stretch", "megascenes"):
            with self.subTest(prune=prune):
                _, _, valid = warp_to_view(self.rgb, self.box_scene(), self.K, E, prune=prune)
                hole = ~valid
                box_rows = slice(38, 58)
                # wall that was hidden behind the box's right edge: target cols ~31..52
                self.assertGreater(hole[box_rows, 33:51].mean(), 0.95)
                # left of the box nothing is revealed; away from the box only the right border band is lost
                self.assertEqual(hole[box_rows, 0:28].sum(), 0)
                self.assertEqual(hole[0:30, 0:85].sum(), 0)
                self.assertTrue(hole[:, 90:].all())
                # the box is not smeared over the gap: it ends near its expected position (col ~31)
                self.assertLess(np.argmax(hole[48, 20:]) + 20, 34)

    def test_stretch_holes_on_floor_are_exactly_out_of_frame(self):
        for yaw in (15.0, -15.0):
            with self.subTest(yaw=yaw):
                E = orbit_extrinsic(yaw)
                _, _, valid = warp_to_view(self.rgb, self.floor_scene(), self.K, E)
                # intersect each target pixel's ray with the floor (y = 0.5) and project it into the input view
                R, t = E[:3, :3], E[:3, 3]
                v, u = np.mgrid[0:self.H, 0:self.W] + 0.5
                rays = np.stack([(u - self.W / 2) / self.F, (v - self.H / 2) / self.F, np.ones_like(u)], -1) @ R
                center = -R.T @ t
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = (0.5 - center[1]) / rays[..., 1]
                    p = center + s[..., None] * rays
                    us = self.F * p[..., 0] / p[..., 2] + self.W / 2
                    vs = self.F * p[..., 1] / p[..., 2] + self.H / 2
                on_floor = (rays[..., 1] > 0) & (s > 0) & (p[..., 2] > 0) & (vs > self.H / 2 + 2)
                m = 1.5  # the mesh extends half a pixel (plus 1 px edge padding) past the frame
                seen = on_floor & (us > m) & (us < self.W - m) & (vs < self.H - m)
                unseen = on_floor & ((us < -m) | (us > self.W + m) | (vs > self.H + m))
                self.assertGreater(seen.sum(), 2000)
                self.assertGreater(unseen.sum(), 200)
                self.assertTrue(valid[seen].all())
                self.assertFalse(valid[unseen].any())


if __name__ == "__main__":
    unittest.main()
