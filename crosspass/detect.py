"""Per-frame detections for landmark building - run in a CUDA env (WSL here):
    python crosspass/detect.py <segment_dir>
Writes <segment>/crosspass/detections/<frame>.json (OCR: text, confidence, box) and
<frame>.png (semantic labels, Cityscapes ids: 5 pole, 6 traffic light, 7 traffic sign).
Models: EasyOCR (Apache-2.0); Mask2Former Swin-L Cityscapes (code MIT; Cityscapes-trained
weights - check terms before commercial use, see docs/crosspass-design.md)."""
import sys, json
from pathlib import Path
import numpy as np, cv2, torch

seg = Path(sys.argv[1]); out = seg / "crosspass" / "detections"; out.mkdir(parents=True, exist_ok=True)
frames = sorted((seg / "images").glob("*.jpg"))
import easyocr
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
name = "facebook/mask2former-swin-large-cityscapes-semantic"
proc = AutoImageProcessor.from_pretrained(name)
model = Mask2FormerForUniversalSegmentation.from_pretrained(name).cuda().eval()
for f in frames:
    if (out / (f.stem + ".png")).exists(): continue
    img = cv2.imread(str(f)); h, w = img.shape[:2]
    # OCR at 2x: signs are small at 480p
    big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    texts = [dict(text=t, conf=float(c), box=(np.array(b, float) / 2).tolist())
             for b, t, c in ocr.readtext(big[:, :, ::-1], decoder="beamsearch", paragraph=False)]
    json.dump(texts, open(out / (f.stem + ".json"), "w"))
    with torch.inference_mode():
        inp = proc(images=img[:, :, ::-1].copy(), return_tensors="pt").to("cuda")
        lab = proc.post_process_semantic_segmentation(model(**inp), target_sizes=[(h, w)])[0]
    cv2.imwrite(str(out / (f.stem + ".png")), lab.cpu().numpy().astype(np.uint8))
    print(f.name, len(texts), "texts", flush=True)
