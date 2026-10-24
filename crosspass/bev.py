"""Bird's-eye-view (BEV) road mosaic of one pass in its strip frame.

Every frame's road region ahead (fwd_m) is resampled onto a metric (s, u) grid through the road
plane, then fused across frames (robust weighted mean: near views weigh more, outliers vs. the
first-pass mean are dropped). Cars and people are excluded with the all-objects COLMAP masks.
The paint layer (white/yellow markings) is the lighting-robust channel used for registration."""
import cv2, numpy as np

def render(strip, res=0.1, u_lim=12.0, fwd_m=(4.0, 20.0), reject=40.0, cams=None, s_range=None):
    """cams: optional Cameras (another pass's frames posed in this strip's world) to draw into
    this strip's grid instead of the strip's own frames."""
    cams = cams or strip
    sgn = cams.direction
    s0, s1 = s_range or (min(cams.cam_s) - fwd_m[1], max(cams.cam_s) + fwd_m[1])
    S = np.arange(s0, s1, res); U = np.arange(-u_lim, u_lim + 1e-9, res)
    acc = np.zeros((len(S), len(U), 3)); wsum = np.zeros((len(S), len(U))); acc2 = np.zeros((len(S), len(U)))
    first = None
    for rnd in range(2):
        acc[:] = 0; wsum[:] = 0; acc2[:] = 0
        for n, sc in zip(cams.names, cams.cam_s):
            lo, hi = sorted((sc + sgn * fwd_m[0], sc + sgn * fwd_m[1]))
            i0, i1 = np.searchsorted(S, lo), np.searchsorted(S, hi)
            if i1 <= i0: continue
            ss, uu = np.meshgrid(S[i0:i1], U, indexing="ij")
            X = strip.to_world(ss, uu, strip.road_z(ss))
            px, z = cams.project(n, X)
            img = cv2.imread(str(cams.seg / "images" / n)).astype(np.float32)
            msk = cv2.imread(str(cams.seg / "masks_colmap" / (n + ".png")), 0)
            mx, my = px[..., 0].astype(np.float32), px[..., 1].astype(np.float32)
            v = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            ok = (z > 0) & (mx >= 0) & (mx < cams.W - 1) & (my >= 0) & (my < cams.H - 1)
            if msk is not None: ok &= cv2.remap(msk, mx, my, cv2.INTER_NEAREST) > 127
            w = ok / (ss - sc) ** 2                                   # near views are sharper
            if first is not None:
                w *= np.abs(v - first[i0:i1]).max(-1) < reject       # drop transients (shadows, misses)
            acc[i0:i1] += v * w[..., None]; wsum[i0:i1] += w; acc2[i0:i1] += (v.mean(-1) ** 2) * w
        first = acc / np.maximum(wsum, 1e-12)[..., None]
    rgb = np.clip(first, 0, 255).astype(np.uint8); rgb[wsum == 0] = 0
    # cross-frame disagreement per cell (weighted std of grey level) - low when geometry is right
    var = acc2 / np.maximum(wsum, 1e-12) - first.mean(-1) ** 2
    return dict(rgb=rgb, weight=wsum, S=S, U=U, res=res, std=np.sqrt(np.maximum(var, 0)))

def paint(rgb, valid, win_m=1.5, res=0.1):
    """Road-marking likelihood: brighter than the local road (locally normalised) and either
    unsaturated (white) or yellow. Lighting-robust: compares each cell to its own neighbourhood."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]; k = int(win_m / res) | 1
    vf = valid.astype(np.float32)
    mu = cv2.blur(L * vf, (k, k)) / np.maximum(cv2.blur(vf, (k, k)), 1e-6)
    sd = np.sqrt(np.maximum(cv2.blur((L - mu) ** 2 * vf, (k, k)) / np.maximum(cv2.blur(vf, (k, k)), 1e-6), 1))
    bright = np.clip((L - mu) / (sd + 4) - 1.0, 0, 3) / 3
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    white = hsv[..., 1] < 60
    yellow = (hsv[..., 0] > 15) & (hsv[..., 0] < 35) & (hsv[..., 1] > 60)
    return bright * (white | yellow) * valid

def dash_period(p, res=0.1, lane_u=None, search=(8.0, 25.0)):
    """Dominant along-street period of paint on lane-line columns (autocorrelation).
    Returns (period_m, strength). California 1970: 12 ft dash + 36 ft gap = 14.63 m."""
    cols = p.sum(0); cols = cols / (cols.max() + 1e-9)
    lines = np.where(cols > 0.3)[0] if lane_u is None else lane_u
    sig = p[:, lines].sum(1); sig = sig - sig.mean()
    ac = np.correlate(sig, sig, "full")[len(sig) - 1:]; ac /= ac[0] + 1e-9
    lo, hi = int(search[0] / res), int(search[1] / res)
    k = lo + int(np.argmax(ac[lo:hi]))
    return k * res, float(ac[k])
