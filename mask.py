"""Mask moving things (cars, people) so they don't become floaters in the splat.
Writes two layouts: COLMAP (masks_colmap/NAME.jpg.png, used by sfm) and masks/NAME.png
(picked up automatically by Brush in train).
White = keep, black = ignore (both tools use this convention)."""
import cv2, numpy as np
from tqdm import tqdm
from .common import segment_dirs

def main(cfg):
    mc = cfg["mask"]
    if not mc["enabled"]: return
    from ultralytics import YOLO
    model = YOLO(mc["model"])
    cls_ids = [k for k, v in model.names.items() if v in mc["classes"]]
    kernel = np.ones((mc["dilate_px"],) * 2, np.uint8)
    for seg in segment_dirs(cfg):
        imgs = sorted((seg / "images").glob("*.jpg"))
        (seg / "masks").mkdir(exist_ok=True); (seg / "masks_colmap").mkdir(exist_ok=True)
        for p in tqdm(imgs, desc=seg.name[:40]):
            img = cv2.imread(str(p)); h, w = img.shape[:2]
            keep = np.full((h, w), 255, np.uint8)
            r = model.predict(img, classes=cls_ids, verbose=False, retina_masks=True)[0]
            if r.masks is not None:
                m = (r.masks.data.cpu().numpy().max(0) > 0.5).astype(np.uint8)
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                keep[cv2.dilate(m, kernel) > 0] = 0
            if mc["mask_bottom_frac"] > 0:
                keep[int(h * (1 - mc["mask_bottom_frac"])):] = 0
            cv2.imwrite(str(seg / "masks" / f"{p.stem}.png"), keep)
            cv2.imwrite(str(seg / "masks_colmap" / f"{p.name}.png"), keep)
