# crosspass — automatic alignment of opposite-direction street passes

Status: design, not implemented. Target: a `crosspass` package + `run.py align` stage that takes
two (or more) independently reconstructed passes of the same street — e.g. the City Clerk's
eastbound and westbound 1970 reels — and outputs one consistent set of camera poses for joint
splat training, **with no human input and a calibrated "merge / don't merge" decision**.

## 1. The problem, as measured on El Cajon Blvd

Inputs per pass: ~160–190 frames, 616×366, time-lapse (~2.5–4 m between frames), forward camera
(~40° HFOV), shot on different days (hazy vs. sunny), each pass reconstructed cleanly on its own
(COLMAP, ALIKED+LightGlue, dense MVS). What failed, and why it matters for the design:

| tried | result | lesson |
|---|---|---|
| ALIKED+LightGlue cross-pass matching | best cross pair 62 inliers vs ~160 within-pass; no structure | appearance features don't survive 180° + different day |
| RoMa dense matching + 3D consensus | ~2× chance; unstable scale | "confident" matches on road texture / vanishing point are generic |
| MASt3R + 3D consensus | 3.5× chance, stable, but 15° tilted, wrong side of the road | free 7-DoF fits lock onto the corridor's left/right symmetry |
| same, rotation pinned (gravity + heading) | no plausible scale (164 m / 695 m vs ~430 m) | with the right constraints, no real signal was left |
| dense-cloud ICP | broad optimum along the street | a street is a corridor: geometry slides along its axis |

Facts that shape the design:
1. **The two passes see different things.** Each camera sees its *own right side* up close and the
   other side small and far. Opposite cameras at the same spot share almost nothing; only
   face-to-face pairs 20–80 m apart share the stretch between them, seen from opposite ends.
2. **What *is* shared is mostly viewpoint-invariant once you stop looking at it in perspective:**
   the road surface (paint, crosswalks, stop bars, curb cuts, patches, manholes), vertical
   street furniture (poles, signals, streetlights), double-sided pole signs and their *text*,
   and the street's topology (where cross streets are).
3. **Most unknowns are already known.** Gravity (camera axes), heading (opposite), lane offset
   (westbound is north of eastbound by 1–3 lanes in US traffic). The hard unknowns are
   *along-street position* and *scale* — and since forward monocular SfM drifts, scale is not
   even one number: the mapping between passes is a smooth 1-D warp of arc length.
4. **We must know when we failed.** Today's MASt3R fit passed several sanity checks and was
   still wrong. Verification has to use evidence the solver did not use.

## 2. Core idea

Don't match images across passes. **Convert each pass into a viewpoint-invariant, metric,
street-aligned "map" and register the maps.** Each pass is unrolled along its own driving path
into a strip coordinate system `(s, u, z)` — arc length along the road, lateral offset, height —
and populated with evidence layers that look the same no matter which way you drove past:

```
pass A frames ─┐                                          ┌─ BEV road mosaic (paint/curbs)
               ├─ existing pipeline ─ poses, depth, masks ┼─ vertical landmarks (poles, signals)
pass B frames ─┘    (per pass)                            ├─ text landmarks (OCR'd sign strings)
                                                          ├─ topology events (cross streets)
                                                          └─ (optional) OpenStreetMap prior
        per-pass strip maps ──► multi-modal 1-D/2-D registration ──► global solve ──► verify
                                                                       │              │
                                          merged COLMAP model + per-pass appearance ◄─┘ (gate)
```

Registration happens in a space where the problem is ~1-D (along-street warp) with a few
nuisance terms, every modality is lighting-robust, and each modality's errors are independent —
which is what makes cross-checking meaningful.

## 3. Components

### 3.1 Pass canonicalisation (`strip.py`)
- Path: smoothing spline through camera centres → arc length `s`, tangent `t(s)`.
- Up: mean camera "down" axis (refined by the road plane fit). Strip frame: `s` along the path,
  `u` = up × t (left positive), `z` = up.
- **Metric scale** from several independent cues, robustly combined (and reported separately):
  camera height above the fitted road plane (~1.5 m on the survey car), lane width (≈3.66 m,
  from BEV paint lines), dash period (1970 California: 12 ft dash + 36 ft gap ≈ 14.6 m).
  Today the camera-height cue alone disagreed between passes by ~35%; the others exist to fix that.
- Output: `StripMap` with `to_strip(X)`, `to_world(s,u,z)` per pass.

### 3.2 Evidence layers (each with positions in strip coords + uncertainty)
**a. BEV road mosaic (`bev.py`) — the primary signal.**
- Road pixels: semantic segmentation (road/sidewalk/curb) ∩ not-masked (moving + parked cars).
- Each frame's road region 4–20 m ahead is warped through the road plane onto a 5 cm grid in
  `(s,u)`. Ground footprint at 480p: ~8 cm/px at 10 m, ~33 cm/px at 20 m — fine near-field.
- Per-cell robust fusion over the many frames that see it (median; count; viewing range).
- Day-invariant channels: (i) paint mask (locally normalised brightness + low saturation, so
  white/yellow lines survive haze vs. sun), (ii) gradient-orientation field (shadow edges are
  long, straight and inconsistent between days → down-weighted by cross-frame consistency),
  (iii) curb line and curb cuts from road/sidewalk boundary.
- Derived 1-D "barcode": transverse paint per metre of `s` (crosswalks, stop bars, turn
  arrows) — spiky, irregular, dash-period-free.

**b. Vertical landmarks (`poles.py`).** Segment pole/traffic-light/streetlight/sign-post per frame;
track within the pass; triangulate base points with the pass's known poses. Output: 2-D point
set `(s,u)` with class and height. Spacing is irregular → a fingerprint along the street.

**c. Text landmarks (`text.py`).** Text spotting (OCR) on every frame; group detections of the
same string within the pass, triangulate the sign. Pole signs ("FOOD BASKET", "CARPET
WAREHOUSE", "BEKINS") are often double-sided and readable from both directions — a match on a
long, rare string is near-certain. Fuzzy string matching (edit distance, mirrored-text guard).

**d. Topology events (`events.py`).** Cross streets: crosswalk/stop-bar clusters in the BEV
barcode, gaps in the facade layer, traffic-signal landmarks. Output: ordered list of events
along `s` with type and width.

**e. Optional geographic prior (`geo.py`).** OpenStreetMap centreline + intersection list for
the named street. Event sequence ↔ OSM intersection sequence by dynamic programming gives each
pass a coarse georeference (also the hook for the ArcGIS export).

### 3.3 Registration (`register.py`)
Unknowns between pass B and pass A, in A's strip frame:
- along-street warp `s_A = w(s_B)`, monotone *decreasing* for opposite passes, modelled as an
  affine map plus a smooth correction spline (absorbs SfM scale drift);
- lateral offset `u_A = u_B + δu(s)` and height offset `δz(s)`, both smooth;
- small yaw/roll/pitch residuals (strip frames already agree on gravity and heading).

Search, coarse → fine, every stage multi-hypothesis:
1. **Hypotheses**: scale × offset grid over a wide prior (scale 0.5–2, any offset), scored by
   (i) normalised cross-correlation of BEV barcodes, (ii) text-anchor agreement, (iii) event
   sequence alignment (DP), (iv) OSM if available. Keep the top-K per modality.
2. **Consensus**: RANSAC over anchor pairs from *all* modalities for the affine warp. Truth
   should be supported by several independent modalities; spurious corridor/dash-period fits
   usually aren't.
3. **Refinement**: 2-D BEV registration (NCC on day-invariant channels) in overlapping windows
   along `s`, plus pole point-set matching (ICP-style in 2-D, gated by the current warp) →
   dense residuals → fit the correction spline and `δu(s)`, `δz(s)` with smoothness priors.
4. **Lane check**: BEV lane lines give each pass's lane index; `δu` must agree with it.

### 3.4 Global solve (`solve.py`)
Convert accepted correspondences back to constraints and re-optimise everything together:
- Each pass gets per-keyframe Sim3 corrections (pose-graph, regularised along `s`) so the
  within-pass relative geometry stays intact.
- Cross-pass constraints: landmark positions (poles, signs) must coincide; BEV correspondences
  become 3-D road points shared between passes.
- Optional final step: write those correspondences as 2-D↔2-D image matches into one COLMAP
  database (projecting shared road/landmark points into both passes' images) and run COLMAP's
  bundle adjustment, so the merged model is a normal, self-consistent SfM model.

### 3.5 Verification & gating (`verify.py`)
Score the result only with evidence it can't have overfitted, and refuse to merge otherwise.
- **Leave-one-modality-out**: refit without each modality; the held-out modality's anchors must
  land within tolerance (e.g. text anchors within 2 m when fitted from BEV+poles).
- **Physical priors**: gravity agreement < 1°, opposite headings, lateral side and lane offset
  consistent with traffic direction, smooth scale (no folds).
- **Chance controls**: same pipeline on shuffled/reversed inputs and on deliberately
  non-overlapping segments; the real score must clear the control distribution by a margin.
- **Photometric check**: after a short joint splat fit with per-pass appearance codes, BEV
  renders of the overlap from both passes' training views must agree (gradient NCC).
- Output `confidence ∈ [0,1]`, per-check report, and diagnostic images (BEV overlays, landmark
  plots). Below threshold: emit separate per-pass splats and say why.

### 3.6 Export (`export.py`)
Merged COLMAP model (one camera per pass), merged masks, dense init cloud, and per-image
appearance groups → nerfstudio `splatfacto-w` / `splatfacto` with
`camera-optimizer.mode SO3xR3` and `use-bilateral-grid` (absorbs residual pose error and the
hazy-vs-sunny lighting). Brush remains the single-pass trainer.

## 4. Interfaces
- `python run.py align` (after `dense`): finds segments whose manifest entries share an
  `overlap` group, writes `workspace/<street>/aligned/<group>/` with `alignment.json`
  (per-pass warp, Sim3 corrections, confidence, checks), `colmap/sparse/0`, `images/`, `masks/`,
  and `diagnostics/`.
- `python run.py train` trains the aligned group when it exists and passed the gate.
- Library: `crosspass.StripMap`, `crosspass.layers.*`, `crosspass.register(a, b, prior)`,
  `crosspass.verify(result)` — usable outside this repo.

## 5. Dependencies and licensing (commercial use in mind)
Prefer permissive components; isolate the rest behind interfaces so they can be swapped.

| need | first choice | licence | note |
|---|---|---|---|
| SfM / MVS | COLMAP | BSD | already used |
| features | ALIKED + LightGlue | ALIKED: verify / LightGlue: Apache-2.0 | within-pass only |
| road/pole segmentation | Mask2Former (code MIT) | **open question** | Cityscapes/Mapillary training data have non-commercial terms that may bind the weights; SegFormer weights are NVIDIA non-commercial. Check before commercial use, or train on a permissive dataset |
| car/person masks | — | — | **Ultralytics YOLO is AGPL-3.0** → swap (e.g. Mask2Former/Detectron2, Apache) for commercial use |
| OCR | PaddleOCR | Apache-2.0 | |
| geo prior | OpenStreetMap | ODbL | attribution; share-alike only for derived databases |
| splats | nerfstudio, gsplat, Brush | Apache-2.0 | |
| MASt3R / RoMa | not used | NC / MIT | kept out of the runtime path |

## 6. Evaluation
- **Ground truth** (offline, for scoring only — never at runtime): hand tie points on 3–5
  overlap pairs (ECB EB/WB plus other City Clerk streets with both directions), ~10 points each.
- Metrics: along-street error (target median < 0.5 m, p95 < 1.5 m), lateral < 0.3 m, rotation
  < 1°; **gate precision** (merges that pass must be right — target 100%) and **recall**
  (fraction of truly-overlapping pairs merged).
- Regression set lives in `tests/crosspass/` with small cached strip maps so CI doesn't need GPUs.

## 7. Build plan (each phase ends with a measured go/no-go)
0. **Ground truth + harness** — tie points for ECB, metrics, diagnostic plots. *(~1 day)*
1. **Strip maps + BEV mosaics** — check visually that paint/crosswalks line up between passes
   after a hand-set warp; this de-risks the primary signal first. *(2–3 days)*
2. **BEV barcode + 2-D registration** — first fully automatic alignment; score vs. ground truth.
3. **Text landmarks** — OCR anchors; measure how many strings are shared per 400 m.
4. **Poles + events + gating** — multi-modal consensus, leave-one-out, controls, confidence.
5. **Global solve + joint training** — merged model, splatfacto-w, photometric check; compare
   joint splat vs. single-pass splats off-path.
6. **Scale-out** — run every both-direction street pair in the archive; report gate stats.

## 7a. Phase 0–1 findings (Oct 2026, ECB eastbound vs. untitled westbound reel)
Built: `crosspass/strip.py` (strip frame; renders another pass's cameras into it via
`Cameras`), `crosspass/bev.py` (BEV mosaic, paint layer, dash-period probe),
`crosspass/verify.py` (physical checks + paint shift test), fold detection in `sfm.py`.

1. **A folded reconstruction poisoned every earlier cross-pass test.** The westbound model looked
   complete (186/186 frames) but frames 77–185 were laid back over 0–76 at the 76→77 link.
   COLMAP's default mapper thresholds fold at the same link (so it isn't the relaxed settings;
   the link itself is misleading — possibly a splice or dropped frames). ICP, RoMa and MASt3R
   were all compared against the folded model. `sfm` now detects folds (`colmap/pieces.json`), and nothing downstream may treat a model
   as one piece without that check.
2. **Re-run per unfolded piece, MASt3R still doesn't align.** Best candidate (WB 77–185): 108
   inliers vs. 22 for the shuffled control and plausible heading/gravity (2.8°), but `verify`
   rejects it: WB cameras land 2.9 m *below* EB's road, and BEV paint correlation is ~0 at every
   shift. The verification layer works; the matcher signal is still too weak.
3. **The BEV road signal is weak at 480p.** Mosaics show the road band, curb and lane line, but
   transverse paint (crosswalks, stop bars) is sparse — a handful of short marks per 400 m, fewer
   on the sunny westbound pass. Lane lines and curbs run *along* the street, so they don't fix
   the along-street offset. Phase 1 does **not** confirm BEV paint as the primary signal.
4. **Metric scale within a pass can't be measured from the road surface here.** Dense MVS has
   almost no asphalt points; photometric plane sweeps between frames are uninformative (the
   overlap sits near the horizon, blurred); dash-period autocorrelation is too weak. Scale stays
   provisional (sparse road points) and must be solved by cross-pass registration.

Consequences for the plan: (a) promote **text landmarks** and **poles/signals** (they need far
fewer pixels than road paint) ahead of BEV registration; (b) BEV stays as a *verification*
signal (it's what rejected the last candidate) and for lateral/lane offsets; (c) the strongest
single lever is **better source footage** — the City Clerk's original scans rather than the
480p YouTube re-encodes; (d) re-test the learned matchers only on unfolded pieces, behind the gate.

## 7b. Phase 3–4 findings (text + pole landmarks, gated)
Built: `crosspass/detect.py` (EasyOCR + Mask2Former Cityscapes per frame, CUDA env),
`crosspass/landmarks.py` (fuzzy-grouped OCR strings triangulated per pass; pole/light/sign
positions by bearing voting with triangulation-angle checks), `crosspass/register.py`
(3-parameter strip registration, RANSAC, scrambled-B control), `crosspass/align.py` (the gate).

Result on ECB (EB vs. each unfolded WB piece): **rejected at rung 1**: landmark agreement equals
the scrambled control (16 vs 16; 30 vs 29), and no text string is shared between passes.

Why, and it's a property of the footage, not the code:
- **Each pass only sees its own side of the street.** The camera's ~40° HFOV is yawed ~5–6° to
  the right (towards the storefronts); the far side is a sliver at the left edge near the
  horizon. Segmentation finds poles only on the near curb (EB: south, WB: north): 90 / 75
  poles, all at u < 0. Different curbs → different poles.
- **Text likewise:** EB reads its side (BEKINS, TANG SOO, FOOD BASKET, CARPETS), WB its side
  (PAINT, GULF, CYCLE). OCR at 480p is also too garbled to read far-side signs.
- Shared content is limited to the road surface and to things visible from both sides (signals
  over intersections, tall double-sided signs) — too sparse at 480p (see 7a).

What would change the outcome, in order of expected value:
1. **Original scans** from the City Clerk (higher resolution → far-side poles/signs and road paint
   become detectable; everything above is resolution-limited).
2. **Left/right reels.** Some archive streets have `… L` / `… R` reels (e.g. "01 Ave NB L",
   "04 Ave SB R"): side-looking passes that see the far side. Where they exist, they bridge
   the two directions. ECB has none.
3. **External geometry:** georeference each pass independently (OSM centreline + cross streets,
   which each pass *can* see on its own side) and align the passes through the map. Accuracy is
   metres, not centimetres — enough for viewing both passes in one scene or GIS, not for a
   single fused splat.

## 8. Risks
- **480p road texture may be too weak** beyond ~15 m and under strong shadows → Phase 1 tests it
  first; fallbacks are poles/text, which need less resolution.
- **Repainted markings / different dates**: if passes are months apart, paint may differ →
  multi-modal consensus is the defence; the gate refuses merges that rely on one modality.
- **Dash periodicity** (~14.6 m) creates aliases → barcode uses transverse paint (crosswalks,
  stop bars), not dashes; aliases are explicit hypotheses resolved by other modalities.
- **Scale drift inside a pass** larger than the spline can absorb → solve per 100–150 m window
  and chain.
- **Occlusion by traffic** in the BEV (parked cars over curbs) → masks + multi-frame median;
  curbs are long, so partial coverage is enough.
