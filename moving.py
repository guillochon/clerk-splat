"""Unmask the static objects in masks/ (Brush) once camera poses exist — run after `sfm`.
`mask` blanks every car and person, which SfM wants; but parked cars and people waiting at a
bus stop are part of the 1970 street, and masking them leaves dark smears along the curb.
Test per detection: put its ground-contact point on the road plane, predict where a *static*
object would land in the neighbouring frames, and look for a detection there. Parked cars
pass; traffic driving with or against the survey car fails and stays masked."""
import cv2, numpy as np
from tqdm import tqdm
from .common import segment_dirs
from .colmap_io import read_cameras_images, read_points3d, road_planes

GROUPS = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle",
          "bicycle": "vehicle", "person": "person"}

def detect(model, img, cls_ids):
    r = model.predict(img, classes=cls_ids, verbose=False, retina_masks=True)[0]
    if r.masks is None: return []
    h, w = img.shape[:2]; out = []
    for box, c, m in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.masks.data.cpu().numpy()):
        out.append(dict(box=box, group=GROUPS[model.names[int(c)]],
                        mask=cv2.resize((m > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)))
    return out

def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

def is_static(det, name, names, poses, planes, dets, K, W, H, reach, min_iou):
    plane = planes.get(name)
    if plane is None: return False
    n, d = plane; x0, y0, x1, y1 = det["box"]
    ray = np.linalg.inv(K) @ np.array([(x0 + x1) / 2, y1, 1.0])
    if n @ ray <= 1e-6: return False                    # foot point above the horizon
    Xc = ray * (-d / (n @ ray)); C2W = np.linalg.inv(poses[name])
    Xw = C2W[:3, :3] @ Xc + C2W[:3, 3]
    i = names.index(name); votes = tries = 0
    for j in range(max(0, i - reach), min(len(names), i + reach + 1)):
        if j == i: continue
        M = poses[names[j]]; Xs = M[:3, :3] @ Xw + M[:3, 3]
        if Xs[2] <= Xc[2] * 0.2: continue
        u, v, _ = K @ (Xs / Xs[2])
        s = Xc[2] / Xs[2]; bw, bh = (x1 - x0) * s, (y1 - y0) * s
        full = (u - bw / 2, v - bh, u + bw / 2, v)
        pred = (max(full[0], 0), max(full[1], 0), min(full[2], W), min(full[3], H))   # compare what's on screen
        if pred[2] - pred[0] < 8 or pred[3] - pred[1] < 8: continue
        if (pred[2] - pred[0]) * (pred[3] - pred[1]) < 0.4 * bw * bh: continue    # mostly off-screen
        # uninformative: far objects barely move, so 'parked' and 'driving along' predict the
        # same box. Only frames where the two hypotheses disagree get a vote.
        if iou(pred, det["box"]) > 0.5: continue
        tries += 1
        votes += any(o["group"] == det["group"] and iou(pred, o["box"]) >= min_iou for o in dets[names[j]])
    return tries > 0 and votes / tries >= 0.5

def main(cfg):
    mc = cfg["mask"]
    if not mc["enabled"] or not mc.get("keep_static", False): return
    from ultralytics import YOLO
    model = YOLO(mc["model"]); cls_ids = [k for k, v in model.names.items() if v in mc["classes"]]
    kernel = np.ones((mc["dilate_px"],) * 2, np.uint8)
    for seg in segment_dirs(cfg):
        sparse = seg / "colmap" / "sparse" / "0"; done = seg / "masks" / ".static_done"
        if not sparse.exists() or done.exists(): continue
        W, H, K, poses = read_cameras_images(sparse)
        names = sorted(poses); planes = road_planes(read_points3d(sparse), poses)
        dets = {n: detect(model, cv2.imread(str(seg / "images" / n)), cls_ids)
                for n in tqdm(names, desc=f"{seg.name[:30]} detect")}
        kept = total = 0
        for n in names:
            img_h, img_w = H, W; keep = np.full((img_h, img_w), 255, np.uint8)
            for det in dets[n]:
                total += 1
                if is_static(det, n, names, poses, planes, dets, K, W, H,
                             mc.get("static_reach", 2), mc.get("static_iou", 0.3)):
                    kept += 1; continue
                keep[cv2.dilate(det["mask"], kernel) > 0] = 0
            if mc["mask_bottom_frac"] > 0:
                keep[int(img_h * (1 - mc["mask_bottom_frac"])):] = 0
            cv2.imwrite(str(seg / "masks" / f"{n[:-4]}.png"), keep)
        done.touch()
        print(f"  {seg.name}: {kept}/{total} detections judged static and left in the splat")
