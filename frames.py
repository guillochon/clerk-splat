"""Turn each reel into a clean, SfM-friendly image set."""
import cv2, numpy as np
from tqdm import tqdm
from .common import load_manifest

def detect_crop(cap, samples=40, thresh=18):
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); acc = None
    for idx in np.linspace(0, max(n - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, f = cap.read()
        if ok:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            acc = g if acc is None else np.maximum(acc, g)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows = np.where(acc.mean(1) > thresh)[0]; cols = np.where(acc.mean(0) > thresh)[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1

def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def motion(a, b, w=320):
    s = w / a.shape[1]
    ga = cv2.cvtColor(cv2.resize(a, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.median(np.linalg.norm(flow, axis=2))) / s  # in full-res px

def process(video, out_dir, fc, start=None, end=None, prefix=""):
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    step = max(1, round(src_fps / fc["fps"]))
    y0, y1, x0, x1 = detect_crop(cap) if fc["auto_crop_black"] else (0, None, 0, None)
    b = fc["crop_border_px"]; cb = fc.get("crop_bottom_px", 0)  # burned-in caption bar

    # optional sub-clip (seconds), set per reel in the manifest
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int((start or 0) * src_fps); last_idx = int(end * src_fps) if end else total
    if first: cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    # 1) candidates -> 2) sharpest per window -> 3) drop near-stationary frames
    window, kept, last, i = [], [], None, 0
    for _ in tqdm(range(last_idx - first), desc=video.stem[:40], leave=False):
        ok, f = cap.read()
        if not ok: break
        i += 1
        if i % step: continue
        f = f[y0:y1, x0:x1]; f = f[b:f.shape[0] - b - cb, b:f.shape[1] - b]
        window.append(f)
        if len(window) < fc["sharpness_window"]: continue
        best = max(window, key=sharpness); window = []
        if last is None or motion(last, best) >= fc["min_flow_px"]:
            kept.append(best); last = best
    cap.release()

    if len(kept) > fc["max_frames"]:
        kept = [kept[j] for j in np.linspace(0, len(kept) - 1, fc["max_frames"]).astype(int)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, f in enumerate(kept):
        cv2.imwrite(str(out_dir / f"{prefix}{j:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  {video.name}: {len(kept)} frames")

def main(cfg):
    raw = cfg["workspace"] / "raw"
    for item in [i for i in load_manifest(cfg) if i.get("include", True)]:
        vids = [p for p in raw.glob(f"{item['segment']}.*") if p.suffix not in (".json", ".part")]
        # Reels sharing a "group" (e.g. EB + WB passes of one stretch) go into one segment,
        # their frames told apart by a "tag" prefix: images/eb_00000.jpg, images/wb_00000.jpg
        prefix = f"{item.get('tag', item['id'])}_" if item.get("group") else ""
        out = cfg["workspace"] / "segments" / item.get("group", item["segment"]) / "images"
        if vids and not any(out.glob(f"{prefix}*.jpg")):
            process(vids[0], out, cfg["frames"], item.get("start"), item.get("end"), prefix)
