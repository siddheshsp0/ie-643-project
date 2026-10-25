"""MegaScenes-style view warping: build a mesh from the depth map and render it from a new camera pose.

Intrinsics, the orbit poses and the mesh/render code follow MegaScenes (video_script.py, evalhelpers.py,
util_3dphoto.py). Pixels that get no geometry in the new view are the holes we inpaint. Coordinates are
OpenCV camera coordinates of the input view (x right, y down, z forward), in normalized depth units.
"""

import contextlib
import functools
import importlib.util
import io
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEGASCENES_REPO = ROOT / "third_party/megascenes_nvs"
MEGASCENES_FOCAL_256 = 80.0  # video_script.py: fx = fy = 80 for 256 px images
MEGASCENES_ORBIT_DEG_PER_UNIT = 27.0  # get_ccw_poses: x reaches -1 at theta = 30 * 9/10 deg
PRUNE_RULES = ("stretch", "megascenes")


@functools.lru_cache(maxsize=1)
def load_util_3dphoto():
    """Load MegaScenes' dataloader/util_3dphoto.py on its own.

    Importing it through its package would pull in their whole training stack; the file itself only
    needs numpy, trimesh and pyrender.
    """
    path = MEGASCENES_REPO / "dataloader/util_3dphoto.py"
    spec = importlib.util.spec_from_file_location("megascenes_util_3dphoto", path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stderr(io.StringIO()):  # it ic()-prints the PyOpenGL platform
        spec.loader.exec_module(module)
    return module


def megascenes_intrinsics(h, w, focal_at_256=MEGASCENES_FOCAL_256):
    f = focal_at_256 * max(h, w) / 256.0
    K = np.eye(3)
    K[0, 0] = K[1, 1] = f
    K[0, 2], K[1, 2] = w / 2.0, h / 2.0
    return K


def orbit_extrinsic(yaw_deg=15.0, tx=None, ty=0.0, tz=None, radius=1.0):
    """4x4 input-camera -> target-camera matrix for the MegaScenes orbit at yaw_deg.

    Defaults follow get_ccw_poses: tx = -yaw/27 and tz = sin(yaw). Positive yaw turns left and moves
    right; negative yaw also moves the camera forward, since tz goes negative.
    """
    theta = np.radians(yaw_deg)
    R = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ])
    if tx is None:
        tx = -yaw_deg / MEGASCENES_ORBIT_DEG_PER_UNIT
    if tz is None:
        tz = radius * np.sin(theta)
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = (tx, ty, tz)
    return E


def stretched_faces(vertices, faces, K, extrinsic, max_stretch=4.0, z_near=1e-3):
    """Bool [F]: faces that tear open in the target view, or end up behind the target camera.

    A face on a continuous surface keeps roughly its size (up to the zoom), while a face spanning a
    depth edge gets stretched by the parallax between its near and far vertices.
    """
    q = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z_src, z_tgt = vertices[:, 2], q[:, 2]
    behind = (z_tgt[faces] <= z_near).any(axis=1)
    z_safe = np.where(z_tgt > z_near, z_tgt, z_near)
    src_px = vertices[:, :2] / vertices[:, 2:] * K[[0, 1], [0, 1]]
    tgt_px = q[:, :2] / z_safe[:, None] * K[[0, 1], [0, 1]]

    def longest_edge(px):
        p = px[faces]  # [F, 3, 2]
        return np.linalg.norm(p - np.roll(p, 1, axis=1), axis=-1).max(axis=1)

    zoom = (z_src / z_safe)[faces].max(axis=1)
    return behind | (longest_edge(tgt_px) > max_stretch * zoom * longest_edge(src_px))


def warp_to_view(rgb, depth, K, extrinsic, prune="stretch", max_stretch=4.0, prune_angle=70.0):
    """Render the depth mesh of the input view from the target pose.

    Returns warped uint8 [H, W, 3] (holes black), rendered depth [H, W] (0 in holes) and valid [H, W].
    prune="stretch" drops only faces torn open by parallax, so the holes are what the input never saw.
    prune="megascenes" uses their 70 deg angle rule, which also drops slanted and far surfaces
    (about 20% of the image goes missing even with no camera motion).
    """
    if prune not in PRUNE_RULES:
        raise ValueError(f"prune must be one of {PRUNE_RULES}, got {prune!r}")
    util = load_util_3dphoto()
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        raise ValueError(f"rgb {rgb.shape[:2]} and depth {(h, w)} sizes differ")
    # vertices sit at pixel centers, so pad by 1 px or the identity pose leaves a 1 px hole on two borders
    rgb_p = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge").astype(np.float32) / 255.0
    depth_p = np.pad(depth, 1, mode="edge")
    K_p = K.copy()
    K_p[0, 2] += 1.0
    K_p[1, 2] += 1.0
    mesh = util.unproject_depth(None, rgb_p, depth_p, K_p, np.eye(4), scale_factor=1.0, add_faces=True,
                                prune_edge_faces=prune == "megascenes", prune_angle=prune_angle)
    if prune == "stretch":
        mesh.update_faces(~stretched_faces(np.asarray(mesh.vertices), np.asarray(mesh.faces), K, extrinsic,
                                           max_stretch))
        mesh.remove_unreferenced_vertices()
    colors, depths = util.render_multiviews(h, w, K, [np.linalg.inv(extrinsic)], mesh)
    valid = depths[0] > 0.0
    warped = (colors[0] * 255.0).round().astype(np.uint8)
    warped[~valid] = 0
    return warped, depths[0], valid
