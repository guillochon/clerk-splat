# sd-splat — Gaussian splats of 1970 San Diego from City Clerk street films

Source footage: the City Clerk's 1970 "Photolog" street videos — 35mm film shot from a
car-mounted camera driving each street, digitized by the City Clerk's Archives and posted
to YouTube and https://www.sandiego.gov/digitalarchives/film-audio/street-videos.
Forward-moving, constant-lens footage is a decent fit for structure-from-motion.

## Pipeline
| stage | does |
|---|---|
| `discover` | scrapes the archive index (fallback: YouTube search) for El Cajon Blvd reels → `manifest.json` |
| `download` | yt-dlp, video only |
| `frames` | crops scan borders + caption bar, keeps every film frame, drops frames where the car is stopped |
| `mask` | YOLO-seg masks for every car/person (SfM wants them all gone) |
| `sky` | optional, off: SegFormer sky masks (made things worse — see below) |
| `sfm` | COLMAP ALIKED + LightGlue, sequential (or cross-reel pairs), incremental mapper |
| `moving` | re-checks each detection against the camera poses: parked cars / people standing stay in the splat, traffic stays masked |
| `dense` | COLMAP PatchMatch MVS → `train_data/` with ~100k dense init points |
| `train` | Brush (SH1, 15k steps), one `.ply` per segment → `workspace/<street>/splats/<segment>/` |

`render.py` (separate, needs gsplat + CUDA under Linux/WSL) renders off-path views and a
flythrough video of a finished splat.

Each reel is its own "segment" → its own splat. El Cajon Blvd is miles long; per-reel
splats are far more tractable than one giant scene.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# COLMAP >= 4 with CUDA (ALIKED/LightGlue + PatchMatch MVS need it). Windows: unzip
#   colmap-x64-windows-cuda.zip from github.com/colmap/colmap/releases into .tools/colmap
#   and put .tools/colmap/bin on PATH. (Distro packages are often COLMAP 3.x / no CUDA.)
# Brush: unzip the brush-app release for your OS into .tools/brush (see train.brush_bin)
```
View a splat: any .ply Gaussian viewer (SuperSplat, Brush itself), or the
`viewer.html` next to the ECB EB splat (`python -m http.server` in that folder).

Offline renders / flythrough video (Linux or WSL with a CUDA toolkit; gsplat compiles its
kernels on first use):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install gsplat imageio imageio-ffmpeg packaging
python render.py workspace/<street>/segments/<seg> <splat.ply> <out_dir> --video
```

## Run
```bash
python run.py discover     # then open workspace/el_cajon_blvd/manifest.json and prune
python run.py download
python run.py frames       # inspect segments/*/images before continuing!
python run.py mask
python run.py sfm
python run.py moving
python run.py dense        # ~20-40 min per 160-frame segment on an RTX 3080
python run.py train
```
To do another street, copy `config.yaml`, change `street` and `discover.keywords`.

Optional per-reel fields in `manifest.json`: `start`/`end` (seconds) cut a sub-clip;
`group` + `tag` put several reels' frames into one segment (`images/<tag>_00000.jpg`).

## Things that will bite you (historical-footage specific)
- **Film gate weave**: each frame shifts slightly in the scanner, which COLMAP reads as
  a moving principal point. Raising `crop_border_px` helps; if SfM is still unstable, try
  `camera_model: SIMPLE_RADIAL` (fewer free parameters).
- **Forward motion only**: buildings are seen at grazing angles, so facades will be
  softer than the road. Merging eastbound + westbound reels was tried for ECB and failed
  (see below).
- **Fragmented models**: splices, dropped frames, or long stops split a reel into several
  sparse models. Check `colmap/sparse/` — split the reel or lower `min_flow_px`.
- **Compression**: YouTube re-encodes heavily. The Archives accept research requests via
  the Digital Archives site; asking for the original scans would noticeably improve results.
- **Masks**: COLMAP reads `masks_colmap/NAME.jpg.png`; Brush reads `masks/NAME.png`
  automatically when that folder sits next to `images/` (verified: an all-black mask set
  trains an empty splat).

- **YouTube says "This video is not available"**: yt-dlp needs a JS runtime now —
  `pip install deno` in the venv fixes it. Uploads top out at 480p.
- **Reels are time-lapse**: one film frame every ~5 video frames (≈6 unique frames/s),
  with big jumps between them. Keep `fps: 6` + `sharpness_window: 1` so every film frame
  survives. With SIFT, tracks die after 2–3 frames and the mapper fragments;
  `features: aliked` (ALIKED + LightGlue, COLMAP ≥ 4) fixed it on the ECB EB test.
  The Windows COLMAP build needs CUDA 12 DLLs for that — `pip install nvidia-cublas-cu12
  nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12` and put their `bin/` on PATH.
- **Burned-in caption bar** ("EL CAJON BL E/B") along the bottom: static across frames,
  so SfM latches onto it. `crop_bottom_px: 90` removes it.
- **Wide, empty intersections** can break the reconstruction (mapper re-anchors and
  overlaps earlier frames). Use the manifest's per-reel `start`/`end` (seconds) to keep
  a segment between them; check that camera positions advance monotonically.

- **Reel titles mislead**: for ECB, "El Cajon BL WB" (2:24) and the untitled "El Cajon BL"
  (1:11) together are the westbound pass (2:24 + 1:11 = EB's 3:36). The untitled one holds
  the stretch east of the overpass that EB crosses at 0:41.
- **Opposite passes don't register together**: ALIKED+LightGlue finds no EB↔WB matches
  above ~60 inliers (true neighbours: ~160), even at a shared overpass, and the weak ones
  fold the model. Multi-reel segments (manifest `group`/`tag`) still work for same-direction
  passes; `cross_min_inliers` keeps false links out.

- **Don't mask the sky** (`sky.enabled: false`): it sounds like it should stop sky floaters,
  but with sky pixels ignored nothing keeps semi-transparent splats out of that region and
  off-path views get *more* streaks. Supervised sky trains to a smooth haze instead.
- **Judge splats off the camera path**: held-out PSNR barely moves between settings
  (~18 dB, limited by the ~4 m between film frames); floaters only show when the view
  leaves the path. `render.py` (gsplat, run under WSL/Linux with CUDA) renders shifted
  views and a flythrough.

## Tuning results (ECB eastbound, 0:42-1:09, 160 frames, ~0.4-0.6 km)
Held-out PSNR on every 8th frame, excluding sky and cars, 15k steps unless noted:

| variant | PSNR | notes |
|---|---|---|
| Brush defaults (SH3) | 17.90 | first splat's settings |
| SH degree 1 | 17.95 | cleaner sky off-path, smaller file |
| SH1, 30k steps | 17.78 | more sky streaks than 15k |
| SH1 + splat cap / opacity-scale loss | 17.4-17.6 | no help |
| + sky masked | 17.97 | worse off-path (see above) |
| SH1 + moving-only masks | 17.74* | parked cars kept, passing traffic gone |
| + dense MVS init | **18.99** | the big win |
| + dense init, 1M splat cap | 18.95 | ← config default, half the file |

\* lower only because the held-out frames contain the traffic it removes; visually better.
Off-path renders: `workspace/el_cajon_blvd/splats/<segment>/before_after.jpg`.

## Rights
These are City records. Check the City Clerk's terms of use, and YouTube's terms, before
downloading at scale or publishing derived splats; crediting the City Clerk's Archives
is the courteous minimum.
