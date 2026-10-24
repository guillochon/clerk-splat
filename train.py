"""Train a Gaussian splat per segment with Brush (github.com/ArthurBrussee/brush).
Single binary, no CUDA toolchain; reads the COLMAP model from segment/train_data (built by
`dense`) or else the segment folder, and exports .ply. It also picks up masks/NAME.png
(black = ignore), so traffic flagged by `moving` stays out of the splat."""
import shutil
from .common import run, need, segment_dirs

def main(cfg):
    t = cfg["train"]; need(t["brush_bin"])
    brush = shutil.which(t["brush_bin"])   # absolute: Windows won't spawn a relative a/b path
    for seg in segment_dirs(cfg):
        if not (seg / "colmap" / "sparse" / "0").exists():
            print(f"  skip {seg.name}: no SfM model"); continue
        out = cfg["workspace"] / "splats" / seg.name; out.mkdir(parents=True, exist_ok=True)
        data = seg / "train_data" if (seg / "train_data").exists() else seg   # from `dense`
        run([brush, data, "--total-steps", t["max_iters"],
             "--export-every", t["export_every"], "--export-path", out,
             "--export-name", f"{seg.name}_{{iter}}.ply", *t.get("brush_args", [])])
