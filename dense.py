"""Dense MVS point cloud as Brush's starting points — run after `moving`, before `train`.
Brush seeds its Gaussians from the COLMAP points; the ~6k sparse ALIKED points leave most of
a facade empty, and with only grazing, widely spaced views the optimiser fills the gaps with
floaters. Seeding from COLMAP's PatchMatch MVS cloud (~100k points) gave +1.2 dB on held-out
frames for ECB EB. The SfM model is left untouched (moving/render need its tracks); this
builds segment/train_data/ = images + masks + SfM cameras/poses + dense points3D.bin."""
import shutil, struct
import numpy as np
from .common import run, need, segment_dirs, add_cuda_dlls

def read_fused(path):
    with open(path, "rb") as f:
        props, n = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"): n = int(line.split()[-1])
            elif line.startswith("property"): props.append((line.split()[1], line.split()[-1]))
        dt = np.dtype([(k, {"float": "<f4", "double": "<f8", "uchar": "u1"}[t]) for t, k in props])
        a = np.frombuffer(f.read(n * dt.itemsize), dt)
    return (np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64),
            np.stack([a["red"], a["green"], a["blue"]], 1))

def thin(xyz, rgb, cap):
    vox = 1e-3
    while True:
        _, idx = np.unique(np.floor(xyz / vox).astype(np.int64), axis=0, return_index=True)
        if len(idx) <= cap: return xyz[idx], rgb[idx]
        vox *= 1.3

def main(cfg):
    dc = cfg.get("dense", {})
    if not dc.get("enabled", False): return
    need("colmap"); add_cuda_dlls()
    for seg in segment_dirs(cfg):
        sparse = seg / "colmap" / "sparse" / "0"
        if not sparse.exists(): continue
        mvs = seg / "colmap" / "dense"; fused = mvs / "fused.ply"
        if not fused.exists():
            run(["colmap", "image_undistorter", "--image_path", seg / "images",
                 "--input_path", sparse, "--output_path", mvs, "--output_type", "COLMAP"])
            run(["colmap", "patch_match_stereo", "--workspace_path", mvs,
                 "--PatchMatchStereo.geom_consistency", "true",
                 "--PatchMatchStereo.window_radius", 4, "--PatchMatchStereo.num_iterations", 5])
            run(["colmap", "stereo_fusion", "--workspace_path", mvs, "--output_path", fused,
                 "--StereoFusion.min_num_pixels", 3])
        xyz, rgb = thin(*read_fused(fused), dc.get("max_points", 400_000))
        out = seg / "train_data"
        if out.exists(): shutil.rmtree(out)
        shutil.copytree(seg / "images", out / "images")
        if (seg / "masks").exists():
            shutil.copytree(seg / "masks", out / "masks", ignore=shutil.ignore_patterns(".*"))
        sp = out / "colmap" / "sparse" / "0"; sp.mkdir(parents=True)
        for fn in ("cameras.bin", "images.bin"):
            shutil.copy(sparse / fn, sp / fn)
        with open(sp / "points3D.bin", "wb") as f:     # xyz + colour only; Brush ignores tracks
            f.write(struct.pack("<Q", len(xyz)))
            for i, (p, c) in enumerate(zip(xyz, rgb)):
                f.write(struct.pack("<QdddBBBdQ", i + 1, *p, *c, 0.0, 0))
        print(f"  {seg.name}: train_data with {len(xyz):,} dense points")
