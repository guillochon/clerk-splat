"""Add sky to the Brush masks (masks/NAME.png, black = ignore).
Forward-only footage can't place the sky at a depth, so Brush fills it with large splats
hanging near the cameras — the streaks you see as soon as the view leaves the path.
Masking it leaves the sky to the viewer's background colour instead.
Run after `mask`; COLMAP masks are left alone (the sky has no features anyway)."""
import cv2, numpy as np
from tqdm import tqdm
from .common import segment_dirs

def main(cfg):
    sc = cfg.get("sky", {})
    if not sc.get("enabled", False): return
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    name = sc.get("model", "nvidia/segformer-b5-finetuned-ade-640-640")
    proc = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).eval()
    sky_id = next(i for i, l in model.config.id2label.items() if l == "sky")
    kernel = np.ones((max(sc.get("dilate_px", 0), 1),) * 2, np.uint8)
    for seg in segment_dirs(cfg):
        out = seg / "masks"; out.mkdir(exist_ok=True)
        done = seg / "masks" / ".sky_done"
        if done.exists(): continue
        for p in tqdm(sorted((seg / "images").glob("*.jpg")), desc=seg.name[:40]):
            img = cv2.imread(str(p)); h, w = img.shape[:2]
            with torch.no_grad():
                logits = model(**proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors="pt")).logits
            prob = torch.nn.functional.interpolate(logits, (h, w), mode="bilinear").softmax(1)[0, sky_id].numpy()
            # washed-out 1970 film makes pale facades and signs look like sky: demand high
            # confidence and keep only regions that reach the top edge of the frame
            n, lab = cv2.connectedComponents((prob > sc.get("min_prob", 0.9)).astype(np.uint8))
            top = [l for l in np.unique(lab[0]) if l != 0]
            sky = cv2.dilate(np.isin(lab, top).astype(np.uint8), kernel) > 0
            mp = out / f"{p.stem}.png"
            keep = cv2.imread(str(mp), 0) if mp.exists() else np.full((h, w), 255, np.uint8)
            keep[sky] = 0
            cv2.imwrite(str(mp), keep)
        done.touch()
