"""Minimal readers for COLMAP binary models + road-plane scale. Numpy only, so it works both
inside the package (moving stage) and from render.py run as a plain script under WSL."""
import struct
import numpy as np

def read_cameras_images(sparse):
    """cameras.bin + images.bin -> (W, H, K, {name: world-to-camera 4x4}). Single camera."""
    with open(sparse / "cameras.bin", "rb") as f:
        f.read(8); cid, model, W, H = struct.unpack("<iiQQ", f.read(24))
        nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}[model]   # SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV
        p = struct.unpack(f"<{nparams}d", f.read(8 * nparams))
    fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if model in (0, 2, 3) else p[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    poses = {}
    with open(sparse / "images.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cam = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0": name += c
            f.read(24 * struct.unpack("<Q", f.read(8))[0])
            R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                          [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                          [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            M = np.eye(4); M[:3, :3] = R; M[:3, 3] = [tx, ty, tz]; poses[name.decode()] = M
    return W, H, K, poses

def read_points3d(sparse):
    with open(sparse / "points3D.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; xyz = np.empty((n, 3))
        for i in range(n):
            _, x, y, z = struct.unpack("<Qddd", f.read(32)); f.read(11)
            f.read(8 * struct.unpack("<Q", f.read(8))[0]); xyz[i] = x, y, z
    return xyz

def road_planes(points, poses, step=1, smooth=9):
    """Per camera: road plane (n, d) in camera coords (x right, y down, z forward), n.X + d = 0.
    The road normal is shared (mean camera 'down' axis - the camera rides level on the car);
    per camera only the height is estimated, as the histogram peak of the drop to SfM points
    ahead, in-lane and below the horizon, then median-smoothed along the path. (Free per-frame
    RANSAC planes were too wobbly: some tilted above the camera.)"""
    names = list(poses)
    down = np.mean([M[:3, :3].T @ [0, 1, 0] for M in poses.values()], 0); down /= np.linalg.norm(down)
    h = {}
    for name in names[::step]:
        M = poses[name]; Q = points @ M[:3, :3].T + M[:3, 3]
        x, y, z = Q.T; sel = (z > 2) & (y > 0.01 * z) & (np.abs(x) < 0.35 * z)
        if sel.sum() < 30: continue
        C = np.linalg.inv(M)[:3, 3]; drop = (points[sel] - C) @ down
        drop = drop[drop > 0]
        if len(drop) < 30: continue
        hist, edges = np.histogram(drop, bins=40, range=(0, np.percentile(drop, 90)))
        k = hist.argmax(); h[name] = (edges[k] + edges[k + 1]) / 2
    if not h: return {n: None for n in names}
    known = [n for n in names if n in h]; out = {}
    for i, name in enumerate(names):
        near = [h[n] for n in known if abs(names.index(n) - i) <= smooth // 2] or [np.median(list(h.values()))]
        hn = float(np.median(near)); nc = poses[name][:3, :3] @ down
        out[name] = (nc, -hn)
    return out

def meters_per_unit(points, poses, cam_height_m=1.5):
    """COLMAP scale is arbitrary: take the camera's height above the fitted road as ~1.5 m.
    Noisy per camera (+-40%), fine for sizing view offsets."""
    hs = [abs(p[1]) for p in road_planes(points, poses, step=5).values() if p is not None]
    return cam_height_m / np.median(hs)
