"""Render a Brush .ply along (or off) the COLMAP camera path with gsplat  runs in WSL/Linux
with CUDA. Brush only renders the training/eval poses; floaters show up off the path.

  python render.py SEGMENT_DIR SPLAT.ply OUT_DIR [--shift-m 0.75] [--lift-m 0.5] [--video]

Writes OUT_DIR/onpath_*.png (with the real frame underneath), OUT_DIR/offpath_*.png, and
with --video OUT_DIR/flythrough.mp4. Metric offsets use the camera height above the road
(~1.5 m on the 1970 survey car) to fix COLMAP's arbitrary scale."""
import argparse, sys
from pathlib import Path
import numpy as np, torch, imageio.v2 as imageio
from gsplat import rasterization
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras_images, read_points3d, meters_per_unit

def read_ply(path):
    with open(path, "rb") as f:
        props, n = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"): n = int(line.split()[-1])
            elif line.startswith("property"): props.append(line.split()[-1])
        a = np.frombuffer(f.read(n * 4 * len(props)), np.float32).reshape(n, len(props))
    col = lambda *k: a[:, [props.index(x) for x in k]]
    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda s: int(s[7:]))
    dc = col("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :]
    sh = np.concatenate([dc, a[:, [props.index(p) for p in rest]].reshape(n, 3, -1).transpose(0, 2, 1)], 1) if rest else dc
    deg = int(round(np.sqrt(sh.shape[1]))) - 1
    t = lambda x: torch.tensor(np.ascontiguousarray(x), device="cuda")
    return dict(means=t(col("x", "y", "z")), quats=t(col("rot_0", "rot_1", "rot_2", "rot_3")),
                scales=t(np.exp(col("scale_0", "scale_1", "scale_2"))),
                opacities=t(1 / (1 + np.exp(-a[:, props.index("opacity")]))), colors=t(sh)), deg

def render(gs, deg, viewmat, K, W, H, bg):
    img, alpha, _ = rasterization(viewmats=torch.tensor(viewmat[None], dtype=torch.float32, device="cuda"),
                                  Ks=torch.tensor(K[None], device="cuda"), width=W, height=H,
                                  sh_degree=deg, **gs)
    img = img[0] + (1 - alpha[0]) * torch.tensor(bg, device="cuda")
    return (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

def offset(M, right_m, up_m, mpu, pitch_deg=0.0):
    """Move a world-to-camera pose in its own frame (x right, y down) and pitch it down."""
    C2W = np.linalg.inv(M); C2W[:3, 3] += C2W[:3, :3] @ np.array([right_m, -up_m, 0]) / mpu
    a = np.radians(pitch_deg); P = np.eye(4); P[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return np.linalg.inv(C2W @ P)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segment"); ap.add_argument("ply"); ap.add_argument("out")
    ap.add_argument("--shift-m", type=float, default=0.75); ap.add_argument("--lift-m", type=float, default=0.5)
    ap.add_argument("--bg", default="0.62,0.72,0.85", help="sky colour where nothing is drawn")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--mpu", type=float, help="metres per COLMAP unit (default: estimate)")
    ap.add_argument("--frames", help="comma-separated frame indices to render (default: every 40th)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seg = Path(a.segment); bg = [float(x) for x in a.bg.split(",")]
    gs, deg = read_ply(a.ply)
    W, H, K, poses = read_cameras_images(seg / "colmap" / "sparse" / "0")
    names = sorted(poses)
    mpu = a.mpu or meters_per_unit(read_points3d(seg / "colmap" / "sparse" / "0"), poses)
    print(f"{len(gs['means']):,} splats, SH{deg}, {len(names)} cams, {mpu:.2f} m/unit, "
          f"path ~ {np.linalg.norm(np.diff([np.linalg.inv(poses[n])[:3, 3] for n in names], axis=0), axis=1).sum() * mpu:.0f} m")
    pick = [names[int(i)] for i in a.frames.split(",")] if a.frames else names[4::40]
    for n in pick:
        r = render(gs, deg, poses[n], K, W, H, bg)
        gt = imageio.imread(seg / "images" / n)[:, :, :3]
        imageio.imwrite(out / f"onpath_{Path(n).stem}.png", np.vstack([r, gt]))
        sides = [render(gs, deg, offset(poses[n], s * a.shift_m, a.lift_m, mpu, 8), K, W, H, bg) for s in (-1, 1)]
        imageio.imwrite(out / f"offpath_{Path(n).stem}.png", np.hstack(sides))
    if a.video:
        Wv, Hv = 960, 570; Kv = K.copy(); Kv[:2] *= Wv / W; Kv[1, 2] = Hv / 2
        frames = []
        for i in np.linspace(0, len(names) - 1.001, 360):
            j = int(i); f = i - j
            A, B = np.linalg.inv(poses[names[j]]), np.linalg.inv(poses[names[j + 1]])
            C2W = A.copy(); C2W[:3, 3] = (1 - f) * A[:3, 3] + f * B[:3, 3]   # rotations are near-identical frame to frame
            sway = np.sin(i / len(names) * 4 * np.pi) * a.shift_m
            frames.append(render(gs, deg, offset(np.linalg.inv(C2W), sway, a.lift_m, mpu, 6), Kv, Wv, Hv, bg))
        imageio.mimwrite(out / "flythrough.mp4", frames, fps=30, quality=8, macro_block_size=1)
    print("wrote", out)

if __name__ == "__main__":
    main()
