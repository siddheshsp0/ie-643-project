"""First-order ambisonics (FOA): synthesis, direction and distance estimation, and projection into a camera.

Conventions
    channels      ACN order [W, Y, Z, X], SN3D normalisation (as in STARSS23). `from_fuma` converts FuMa files.
    direction     azimuth counter-clockwise from straight ahead (positive = left), elevation positive = up,
                  both in degrees, in the frame of the microphone: x forward, y left, z up.
    camera        the microphone sits at the input camera and faces the same way. OpenCV camera coordinates
                  (x right, y down, z forward), the frame used by models/megascenes_warp.py.

A plane wave s(t) from (az, el) encodes as  W = s,  Y = s sin(az) cos(el),  Z = s sin(el),  X = s cos(az) cos(el).

Direction comes from the active intensity vector I = Re{conj(W) [X, Y, Z]} summed over the louder
time-frequency bins. Distance comes from the diffuseness psi = 1 - |sum I| / sum E (E = (|W|^2 + |X|^2 +
|Y|^2 + |Z|^2) / 2, as in DirAC): for a direct path plus a diffuse reverberant field, psi = E_rev / (E_dir +
E_rev), so the direct-to-reverberant ratio is (1 - psi) / psi. In a room the reverberant level is roughly the
same everywhere while the direct level falls as 1 / r^2, so DRR = (r_c / r)^2 with r_c the critical distance.
`synthesize_foa` renders exactly that model, which is what `estimate_distance` inverts.
"""

import math

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, stft

FOA_SR = 48000
CRITICAL_DISTANCE_M = 2.0  # distance at which direct and reverberant energy are equal (a mid-sized room)
RT60_S = 0.5


def direction_vector(az_deg, el_deg):
    """Unit vector in the microphone frame (x forward, y left, z up)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])


def synthesize_foa(mono, az_deg, el_deg, distance_m=2.0, sr=FOA_SR, critical_distance_m=CRITICAL_DISTANCE_M,
                   rt60_s=RT60_S, noise_db=-50.0, seed=0):
    """Mono source at (az, el, distance) in a reverberant room -> FOA [4, N] (ACN/SN3D).

    Direct path: a plane wave with gain 1 / distance. Reverberation: the source convolved with an
    exponentially decaying noise tail, independently per channel (a diffuse field; X, Y, Z carry 1/3 of W's
    energy each, as SN3D prescribes), scaled so the direct-to-reverberant ratio is (critical / distance)^2.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(mono, dtype=np.float64)
    s = s / (np.sqrt(np.mean(s ** 2)) + 1e-12)
    d = direction_vector(az_deg, el_deg)
    direct = np.stack([s, s * d[1], s * d[2], s * d[0]]) / distance_m

    n_tail = int(rt60_s * sr)
    env = np.exp(-6.9 * np.arange(n_tail) / n_tail)  # -60 dB over rt60
    rev = np.empty_like(direct)
    for c, gain in enumerate([1.0, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]):
        h = rng.standard_normal(n_tail) * env
        rev[c] = gain * fftconvolve(s, h)[:len(s)]
    e_direct = np.mean(direct[0] ** 2) * 2  # W + |XYZ|^2 carry equal energy for a plane wave
    e_rev = np.mean(rev[0] ** 2) * 2
    rev *= np.sqrt(e_direct / e_rev * (distance_m / critical_distance_m) ** 2)
    foa = direct + rev
    foa += rng.standard_normal(foa.shape) * np.sqrt(np.mean(foa[0] ** 2)) * 10 ** (noise_db / 20)
    return (foa / (np.abs(foa).max() + 1e-12) * 0.9).astype(np.float32)


def from_fuma(foa):
    """FuMa [W, X, Y, Z] with W at -3 dB -> ACN/SN3D [W, Y, Z, X]."""
    w, x, y, z = foa
    return np.stack([w * np.sqrt(2), y, z, x])


def load_foa(path, fuma=False):
    """4-channel FOA file -> ([4, N] float32 ACN/SN3D, sample rate)."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if data.shape[1] != 4:
        raise ValueError(f"{path}: expected 4 FOA channels, got {data.shape[1]}")
    foa = data.T
    return (from_fuma(foa) if fuma else foa), sr


def save_foa(path, foa, sr=FOA_SR):
    sf.write(str(path), np.asarray(foa, dtype=np.float32).T, sr, subtype="FLOAT")


def analyse_foa(foa, sr=FOA_SR, fmin=100.0, fmax=8000.0, n_fft=1024, energy_quantile=0.5, direct_quantile=0.8):
    """FOA [4, N] -> dict(az_deg, el_deg, diffuseness).

    Every frequency counts equally, because real sources put most of their energy in a few low bins where a
    room's response swings most. Direction: the mean of unit intensity directions over the bins where the
    direct sound dominates (louder than the median and with |I| / E in the top 1 - direct_quantile).
    Diffuseness: 1 - |sum_t I| / sum_t E per frequency, then the median over frequencies.
    """
    f, _, spec = stft(np.asarray(foa, dtype=np.float64), fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
    band = (f >= fmin) & (f <= fmax)
    w, y, z, x = (spec[c][band] for c in range(4))
    energy = 0.5 * (np.abs(w) ** 2 + np.abs(x) ** 2 + np.abs(y) ** 2 + np.abs(z) ** 2) + 1e-20
    inten = np.stack([np.real(np.conj(w) * c) for c in (x, y, z)])  # [3, F, T]
    loud = energy >= np.quantile(energy, energy_quantile)
    aligned = np.linalg.norm(inten, axis=0) / energy
    direct = loud & (aligned >= np.quantile(aligned[loud], direct_quantile))
    unit = inten[:, direct] / (np.linalg.norm(inten[:, direct], axis=0) + 1e-20)
    ix, iy, iz = unit.sum(axis=1)
    psi_f = 1.0 - np.linalg.norm(inten.sum(axis=2), axis=0) / energy.sum(axis=1)
    return {"az_deg": math.degrees(math.atan2(iy, ix)), "el_deg": math.degrees(math.atan2(iz, math.hypot(ix, iy))),
            "diffuseness": float(np.clip(np.median(psi_f), 1e-4, 1 - 1e-4))}


def beamform(foa, az_deg, el_deg, w_weight=0.25):
    """First-order virtual microphone steered to (az, el): w_weight W + (1 - w_weight) d . [X, Y, Z].

    Unit gain for a plane wave from (az, el); 0.25 is a hypercardioid (diffuse reverberation down about 6 dB),
    0.5 a cardioid. In a reverberant room this gives CLAP a cleaner signal than the omnidirectional W.
    """
    d = direction_vector(az_deg, el_deg)
    w, y, z, x = np.asarray(foa, dtype=np.float64)
    return (w_weight * w + (1 - w_weight) * (d[0] * x + d[1] * y + d[2] * z)).astype(np.float32)


def estimate_distance(diffuseness, critical_distance_m=CRITICAL_DISTANCE_M):
    """Diffuseness -> distance in metres under the room model of `synthesize_foa` (DRR = (r_c / r)^2)."""
    drr = (1.0 - diffuseness) / diffuseness
    return critical_distance_m / math.sqrt(drr)


def mic_to_camera(v):
    """Microphone frame (x forward, y left, z up) -> OpenCV camera frame (x right, y down, z forward)."""
    return np.array([-v[1], -v[2], v[0]])


def camera_to_mic(v):
    return np.array([v[2], -v[0], -v[1]])


def project_point(K, extrinsic, p):
    """Point in input-camera coordinates -> (u, v, z) in the target view (z <= 0: behind the camera)."""
    q = extrinsic[:3, :3] @ p + extrinsic[:3, 3]
    if q[2] <= 1e-6:
        return float("nan"), float("nan"), float(q[2])
    return float(K[0, 0] * q[0] / q[2] + K[0, 2]), float(K[1, 1] * q[1] / q[2] + K[1, 2]), float(q[2])


def half_fov_deg(K, width):
    return math.degrees(math.atan2(width / 2.0, K[0, 0]))


def yaw_to_face(az_deg, K, width, margin_deg=15.0, max_yaw_deg=30.0):
    """Camera yaw (MegaScenes orbit, positive = turn left) that brings a source at az_deg into view.

    0 when the source is already at least margin_deg inside the frame; otherwise the smallest turn that puts
    it margin_deg inside the edge, capped at max_yaw_deg.
    """
    inside = half_fov_deg(K, width) - margin_deg
    if abs(az_deg) <= inside:
        return 0.0
    return math.copysign(min(max_yaw_deg, abs(az_deg) - inside), az_deg)


def pixel_to_direction(u, v, K):
    """Pixel in the input view -> (az_deg, el_deg) of the ray through it, in the microphone frame."""
    ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    m = camera_to_mic(ray / np.linalg.norm(ray))
    return math.degrees(math.atan2(m[1], m[0])), math.degrees(math.asin(np.clip(m[2], -1, 1)))
