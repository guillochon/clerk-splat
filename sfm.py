"""Camera poses with COLMAP (+ GLOMAP global mapper when available)."""
import shutil
from collections import defaultdict
from .common import run, need, segment_dirs, add_cuda_dlls

def reel_pairs(seg, s):
    """Pair list for a segment holding several reels (frames named TAG_00000.jpg).
    Within a reel: the next `overlap` frames. Across reels: every frame against every
    `cross_stride`-th frame of the other reel (both ways), since we don't know up front
    which frames see the same facades."""
    reels = defaultdict(list)
    for p in sorted((seg / "images").glob("*.jpg")):
        reels[p.stem.rsplit("_", 1)[0]].append(p.name)
    if len(reels) < 2:
        return None
    pairs, k = set(), s.get("cross_stride", 3)
    for names in reels.values():
        for i, a in enumerate(names):
            pairs.update((a, b) for b in names[i + 1:i + 1 + s["overlap"]])
    tags = sorted(reels)
    for x, ta in enumerate(tags):
        for tb in tags[x + 1:]:
            A, B = reels[ta], reels[tb]
            pairs.update((a, b) for a in A for b in B[::k])
            pairs.update((a, b) for a in A[::k] for b in B)
    out = seg / "colmap" / "pairs.txt"
    out.write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    print(f"  {seg.name}: {len(reels)} reels, {len(pairs)} pairs")
    return out

def drop_weak_cross_pairs(db, min_inliers):
    """Road markings and horizons give every cross-reel pair ~20-60 'verified' inliers; left
    in, they glue unrelated stretches together and fold the model. Keep only strong links."""
    import sqlite3
    con = sqlite3.connect(db); M = 2147483647
    names = dict(con.execute("select image_id, name from images"))
    tag = lambda i: names[i].rsplit("_", 1)[0]
    weak = [(p,) for p, r in con.execute("select pair_id, rows from two_view_geometries")
            if tag(p // M) != tag(p % M) and r < min_inliers]
    con.executemany("delete from two_view_geometries where pair_id = ?", weak); con.commit()
    kept = sum(1 for (p,) in con.execute("select pair_id from two_view_geometries")
               if tag(p // M) != tag(p % M))
    print(f"  cross-reel pairs: dropped {len(weak)} below {min_inliers} inliers, kept {kept}")
    con.close()

def main(cfg):
    need("colmap"); add_cuda_dlls(); s = cfg["sfm"]
    use_glomap = s["mapper"] == "glomap" or (s["mapper"] == "auto" and shutil.which("glomap"))
    for seg in segment_dirs(cfg):
        sparse = seg / "colmap" / "sparse"
        if (sparse / "0").exists(): continue
        db = seg / "colmap" / "database.db"; sparse.mkdir(parents=True, exist_ok=True)
        # aliked = ALIKED + LightGlue (COLMAP >= 4); keeps tracks alive across the big
        # frame-to-frame jumps of time-lapse reels, where SIFT tracks die after 2-3 frames
        aliked = s.get("features", "sift") == "aliked"
        cmd = ["colmap", "feature_extractor", "--database_path", db,
               "--image_path", seg / "images",
               "--ImageReader.camera_model", s["camera_model"],
               "--ImageReader.single_camera", int(s["single_camera"])]
        if aliked:
            cmd += ["--FeatureExtraction.type", "ALIKED_N16ROT",
                    "--AlikedExtraction.max_num_features", 4096]
        if (seg / "masks_colmap").exists():
            cmd += ["--ImageReader.mask_path", seg / "masks_colmap"]
        run(cmd)
        pairs = reel_pairs(seg, s)
        if pairs:
            m = ["colmap", "matches_importer", "--database_path", db,
                 "--match_list_path", pairs, "--match_type", "pairs"]
        else:
            m = ["colmap", f"{s['matcher']}_matcher", "--database_path", db]
        if aliked:
            m += ["--FeatureMatching.type", "ALIKED_LIGHTGLUE"]
        if s["matcher"] == "sequential" and not pairs:
            m += ["--SequentialMatching.overlap", s["overlap"]]
            if s["vocab_tree"]:
                m += ["--SequentialMatching.loop_detection", 1,
                      "--SequentialMatching.vocab_tree_path", s["vocab_tree"]]
        run(m)
        if pairs:
            drop_weak_cross_pairs(db, s.get("cross_min_inliers", 100))
        if use_glomap:
            run(["glomap", "mapper", "--database_path", db,
                 "--image_path", seg / "images", "--output_path", sparse])
        else:
            run(["colmap", "mapper", "--database_path", db,
                 "--image_path", seg / "images", "--output_path", sparse])
        models = sorted(sparse.iterdir())
        print(f"  {seg.name}: {len(models)} model(s). >1 means the reel broke into pieces; "
              "model 0 is usually the largest.")
