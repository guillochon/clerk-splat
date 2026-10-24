"""Strip map: a pass's reconstruction unrolled along its own driving path.

Strip coordinates (metres): s = arc length along the smoothed camera path (increasing in the
driving direction), u = lateral offset (left of the driving direction positive), z = up.
Metric scale is provisional: camera height (~1.5 m on the survey car) over COLMAP's sparse road
points. Tried and dropped on this footage: dense MVS (almost no road surface - asphalt is too
textureless for PatchMatch) and photometric plane sweeps between frames (at 480p the shared road
region of two frames sits near the horizon, too blurred to constrain the height). The scale
between passes is therefore left to registration."""
from pathlib import Path
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.spatial import cKDTree
from ..colmap_io import read_cameras_images, read_camera_params, read_points3d, meters_per_unit

CAM_HEIGHT_M = 1.5

class StripMap:
    def __init__(self, seg, names=None):
        """names: optional subset of frames (e.g. one unfolded piece of a model)."""
        self.seg = Path(seg); sp = self.seg / "colmap" / "sparse" / "0"
        self.W, self.H, self.K, self.poses = read_cameras_images(sp)
        model, _, _, params = read_camera_params(sp)
        self.k1 = params[3] if model in (2, 3) else 0.0
        self.names = sorted(names or self.poses)
        C = np.array([np.linalg.inv(self.poses[n])[:3, 3] for n in self.names])
        down = np.array([self.poses[n][:3, :3].T @ [0, 1, 0] for n in self.names])
        fwd = np.array([self.poses[n][:3, :3].T @ [0, 0, 1] for n in self.names])
        self.up = -down.mean(0); self.up /= np.linalg.norm(self.up)
        # smoothed path through the camera centres, sampled densely
        c = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        spl = [make_smoothing_spline(c, C[:, k], lam=None) for k in range(3)]
        cc = np.linspace(0, c[-1], int(c[-1] / 1e-3) + 2)          # ~1e-3 unit steps
        P = np.stack([f(cc) for f in spl], 1)
        if (P[-1] - P[0]) @ fwd.mean(0) < 0: raise ValueError("camera order runs backwards")
        T = np.gradient(P, axis=0); T -= np.outer(T @ self.up, self.up); T /= np.linalg.norm(T, axis=1, keepdims=True)
        self._P, self._T, self._L = P, T, np.cross(self.up, T)
        self._s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]   # world units
        self._tree = cKDTree(P)
        # metric scale: camera height over COLMAP's sparse road points, one value per pass. Only
        # a provisional scale: the between-pass scale is solved by registration (see design).
        self.mpu = meters_per_unit(read_points3d(sp), self.poses, CAM_HEIGHT_M)
        self.cam_s_units = self._raw(C)[:, 0]
        self.cam_s = self.cam_s_units * self.mpu
        self.road_z_at_cam = np.full(len(self.names), -CAM_HEIGHT_M)    # the car rides at a fixed height
        self.length = self._s[-1] * self.mpu

    def _raw(self, X):
        """world -> (s, u, z) in world units."""
        _, k = self._tree.query(X); D = X - self._P[k]
        return np.c_[self._s[k] + (D * self._T[k]).sum(1), (D * self._L[k]).sum(1), D @ self.up]

    direction = 1          # own frames drive towards +s

    def to_strip(self, X):
        return self._raw(np.atleast_2d(X)) * self.mpu

    def to_world(self, s, u, z):
        s, u, z = (np.asarray(v, dtype=float) / self.mpu for v in (s, u, z))
        k = np.clip(np.searchsorted(self._s, s), 0, len(self._s) - 1)
        return self._P[k] + (s - self._s[k])[..., None] * self._T[k] + u[..., None] * self._L[k] + z[..., None] * self.up

    def road_z(self, s):
        """Strip z of the road surface at arc length s (metres)."""
        return np.interp(s, self.cam_s, self.road_z_at_cam)

    def project(self, name, Xw):
        """World points -> (pixel xy, depth) in image `name` (SIMPLE_RADIAL distortion)."""
        M = self.poses[name]; Xc = Xw @ M[:3, :3].T + M[:3, 3]
        z = Xc[..., 2]; x, y = Xc[..., 0] / z, Xc[..., 1] / z
        f = 1 + self.k1 * (x * x + y * y)
        return np.stack([self.K[0, 0] * x * f + self.K[0, 2], self.K[1, 1] * y * f + self.K[1, 2]], -1), z


class Cameras:
    """Frames of another pass, posed in a strip's world by a similarity transform
    (p_strip_world = s * R @ p_other + t), for rendering into that strip's BEV grid."""
    def __init__(self, other: StripMap, strip: StripMap, s, R, t, names=None):
        from ..colmap_io import transform_pose
        self.seg, self.W, self.H, self.K, self.k1 = other.seg, other.W, other.H, other.K, other.k1
        self.names = names or other.names
        self.poses = {n: transform_pose(other.poses[n], s, np.asarray(R), np.asarray(t)) for n in self.names}
        C = np.array([np.linalg.inv(self.poses[n])[:3, 3] for n in self.names])
        F = np.array([self.poses[n][:3, :3].T @ [0, 0, 1] for n in self.names])
        st = strip.to_strip(C); self.cam_s = st[:, 0]; self.cam_u = st[:, 1]; self.cam_z = st[:, 2]
        _, k = strip._tree.query(C)
        self.direction = 1 if (F * strip._T[k]).sum(1).mean() > 0 else -1
    project = StripMap.project
