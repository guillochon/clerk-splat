"""Landmarks of one pass (or one unfolded piece), in its strip coordinates (metres).

text  - OCR'd strings (normalised A-Z0-9, >= 4 chars) seen in several frames, triangulated
        from the box centres with the pass's own poses. Pole signs are often double-sided, so
        the same string can be read driving either way.
poles - pole / traffic light / traffic sign columns from semantic segmentation. A pole is a
        vertical line, so each detection is a bearing on the ground plane; positions are where
        bearings from many frames meet (voting + least squares). Needs no road height."""
import json, re
from collections import defaultdict
import numpy as np, cv2

CLASSES = {5: "pole", 6: "light", 7: "sign"}

def _ray(strip, name, xy):
    """Unit ray (world) through pixel xy of frame `name` (undistorting SIMPLE_RADIAL)."""
    K = strip.K; n = cv2.undistortPoints(np.array([[xy]], np.float64), K, np.array([strip.k1, 0, 0, 0]))[0, 0]
    M = strip.poses[name]; d = M[:3, :3].T @ np.array([n[0], n[1], 1.0])
    return np.linalg.inv(M)[:3, 3], d / np.linalg.norm(d)

def _triangulate(rays):
    """Least-squares point closest to a set of (origin, dir) rays."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d); A += P; b += P @ o
    return np.linalg.lstsq(A, b, rcond=None)[0]

def norm_text(t):
    return re.sub(r"[^A-Z0-9]", "", t.upper())

def similar(a, b, ratio=0.75):
    """OCR at 480p garbles letters (FOOD BASKET -> FOODBUSKEF, FPODBASKET): fuzzy equality,
    or one string inside the other (PAINT within SANDIEGOGLASSPAINT)."""
    from difflib import SequenceMatcher
    if min(len(a), len(b)) >= 5 and (a in b or b in a): return True
    return SequenceMatcher(None, a, b).ratio() >= ratio

def text_landmarks(strip, min_views=2, min_conf=0.3, min_len=4):
    det = strip.seg / "crosspass" / "detections"; obs = defaultdict(list)
    for n in strip.names:
        p = det / (n[:-4] + ".json")
        if not p.exists(): continue
        for d in json.load(open(p)):
            t = norm_text(d["text"])
            if len(t) >= min_len and d["conf"] >= min_conf:
                obs[t].append((n, np.mean(d["box"], 0), d["conf"]))
    # merge garbled reads of the same sign: union strings that are similar
    keys = sorted(obs, key=lambda k: -len(obs[k])); groups = []
    for k in keys:
        for g in groups:
            if similar(k, g[0]): g.append(k); break
        else: groups.append([k])
    merged = {}
    for g in groups:
        canon = max(g, key=lambda k: (len(obs[k]), len(k)))
        merged[canon] = [x for k in g for x in obs[k]]
    out = []
    for t, seen in merged.items():
        if len({n for n, _, _ in seen}) < min_views: continue
        rays = [(n,) + _ray(strip, n, xy) for n, xy, _ in seen]
        # the same word can appear at several places: cluster pairwise triangulations
        pts = []
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                (ni, oi, di), (nj, oj, dj) = rays[i], rays[j]
                if ni == nj or np.degrees(np.arccos(np.clip(di @ dj, -1, 1))) < 0.5: continue
                X = _triangulate([(oi, di), (oj, dj)])
                if (X - oi) @ di > 0 and (X - oj) @ dj > 0: pts.append((X, i, j))
        used = set()
        for X, i, j in sorted(pts, key=lambda p: p[0][0]):
            if i in used and j in used: continue
            sup = [r for r in rays if np.linalg.norm(np.cross(X - r[1], r[2])) < 0.03 * np.linalg.norm(X - r[1])]
            if len({r[0] for r in sup}) < min_views: continue
            Xr = _triangulate([(o, d) for _, o, d in sup])
            used |= {k for k, r in enumerate(rays) if any(r is s for s in sup)}
            s, u, z = strip.to_strip(Xr)[0]
            out.append(dict(kind="text", text=t, s=s, u=u, z=z, views=len({r[0] for r in sup}), world=Xr.tolist()))
    return out

def pole_landmarks(strip, min_views=5, min_h_px=25, tol_m=0.75, rng_m=(3.0, 60.0),
                   min_bearing_deg=4.0, min_spread_deg=6.0):
    """Bearing-voting pole positions in strip (s, u). Bearings close to straight ahead barely
    move between frames and cross anywhere along the path, so they are dropped, and a pole must
    be seen over a spread of bearings (real triangulation angle)."""
    det = strip.seg / "crosspass" / "detections"; rays = []          # (frame idx, class, s0, u0, ds, du)
    for i, n in enumerate(strip.names):
        p = det / (n[:-4] + ".png")
        if not p.exists(): continue
        lab = cv2.imread(str(p), 0)
        C = np.linalg.inv(strip.poses[n])[:3, 3]; sc, uc, _ = strip.to_strip(C)[0]
        _, k = strip._tree.query(C); T, L = strip._T[k], strip._L[k]
        for cid, cname in CLASSES.items():
            num, cc, st, _ = cv2.connectedComponentsWithStats((lab == cid).astype(np.uint8))
            for c in range(1, num):
                x, y, w, h, area = st[c]
                if cname == "pole" and (h < min_h_px or h < 3 * w): continue
                if cname != "pole" and area < 30: continue
                ys = np.arange(y + h // 3, y + 2 * h // 3 + 1) if cname == "pole" else np.array([y + h // 2])
                xs = [np.median(np.where(cc[yy] == c)[0]) for yy in ys if (cc[yy] == c).any()]
                if not xs: continue
                _, d = _ray(strip, n, (float(np.median(xs)), float(np.median(ys))))
                g = np.array([d @ T, d @ L]); g /= np.linalg.norm(g)
                if abs(np.degrees(np.arctan2(g[1], g[0]))) < min_bearing_deg: continue
                rays.append((i, cname, sc, uc, g[0], g[1]))
    R = np.array([r[2:] for r in rays]); F = np.array([r[0] for r in rays]); K = np.array([r[1] for r in rays])
    out = []
    for cname in CLASSES.values():
        idx = np.where(K == cname)[0]
        if len(idx) < min_views: continue
        # pairwise intersections between frames 2..12 apart -> vote grid (0.5 m)
        votes = defaultdict(int)
        for a in idx:
            for b in idx[(F[idx] - F[a] >= 2) & (F[idx] - F[a] <= 12)]:
                (s1, u1, a1, b1), (s2, u2, a2, b2) = R[a], R[b]
                M = np.array([[a1, -a2], [b1, -b2]])
                if abs(np.linalg.det(M)) < 1e-3: continue
                t1, t2 = np.linalg.solve(M, [s2 - s1, u2 - u1])
                if not (rng_m[0] < t1 < rng_m[1] and rng_m[0] < t2 < rng_m[1]): continue
                votes[(round((s1 + t1 * a1) * 2), round((u1 + t1 * b1) * 2))] += 1
        found = []
        for (gs, gu), v in sorted(votes.items(), key=lambda kv: -kv[1]):
            P = np.array([gs / 2, gu / 2])
            if v < 3 or any(np.hypot(*(P - q[:2])) < 3.0 for q in found): continue
            d = R[idx, :2] - P; dirs = R[idx, 2:]
            along = (d * dirs).sum(1); perp = np.abs(d[:, 0] * dirs[:, 1] - d[:, 1] * dirs[:, 0])
            sup = idx[(perp < tol_m) & (along < 0)]                  # ray passes near P, pointing at it
            if len(set(F[sup])) < min_views: continue
            ang = np.degrees(np.arctan2(R[sup, 3], R[sup, 2]))
            if np.ptp(ang) < min_spread_deg: continue
            # least-squares refinement on supporting bearings
            A = np.zeros((2, 2)); bb = np.zeros(2)
            for r in R[sup]:
                n_ = np.array([-r[3], r[2]]); A += np.outer(n_, n_); bb += np.outer(n_, n_) @ r[:2]
            P = np.linalg.solve(A + 1e-9 * np.eye(2), bb)
            found.append(np.r_[P, len(set(F[sup]))])
        out += [dict(kind=cname, s=f[0], u=f[1], views=int(f[2])) for f in found]
    return out
