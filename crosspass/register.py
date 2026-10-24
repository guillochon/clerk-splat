"""Register pass B's landmarks to pass A's in strip coordinates.

Strips are already aligned with their own driving paths, so for an opposite pass the map is
    s_A = a - k * s_B,   u_A = c - k * u_B          (k: ratio of the two provisional scales)
RANSAC over landmark correspondences (identical text strings, pole<->pole, light<->light, ...),
scored by one-to-one agreement; a shuffled-B control gives the chance level."""
import numpy as np
from .landmarks import similar

WEIGHT = dict(text=5.0, light=2.0, sign=1.0, pole=1.0)

def _apply(h, s, u):
    k, a, c, sgn = h
    return a + sgn * k * s, c + sgn * k * u

def _score(h, A, B, tol):
    """One-to-one: each A landmark counts once, for its nearest same-kind B landmark."""
    total, pairs = 0.0, []
    for kind in WEIGHT:
        a = [x for x in A if x["kind"] == kind]; b = [x for x in B if x["kind"] == kind]
        if not a or not b: continue
        bs, bu = _apply(h, np.array([x["s"] for x in b]), np.array([x["u"] for x in b]))
        taken = set()
        for i, x in enumerate(a):
            d = np.hypot(bs - x["s"], bu - x["u"])
            if kind == "text": d = np.where([similar(y["text"], x["text"]) for y in b], d, np.inf)
            for j in np.argsort(d):
                if d[j] > tol: break
                if j in taken: continue
                taken.add(j); total += WEIGHT[kind]; pairs.append((kind, i, j)); break
    return total, pairs

def register(A, B, opposite=True, k_range=(0.4, 2.5), tol=2.0, iters=20000, seed=0):
    sgn = -1.0 if opposite else 1.0
    rng = np.random.default_rng(seed)
    cand = [(x, y) for x in A for y in B if x["kind"] == y["kind"] and
            (x["kind"] != "text" or similar(x["text"], y["text"]))]
    text = [p for p in cand if p[0]["kind"] == "text"]
    best = (-1, None, [])
    if len(cand) < 2: return dict(score=0, hypothesis=None, pairs=[], n_text_candidates=len(text))
    for _ in range(iters):
        # one text pair when available (strong), plus any second pair
        p1 = text[rng.integers(len(text))] if text and rng.random() < 0.7 else cand[rng.integers(len(cand))]
        p2 = cand[rng.integers(len(cand))]
        dsA, dsB = p1[0]["s"] - p2[0]["s"], p1[1]["s"] - p2[1]["s"]
        if abs(dsB) < 5: continue
        k = dsA / (sgn * dsB)
        if not (k_range[0] < k < k_range[1]): continue
        a = p1[0]["s"] - sgn * k * p1[1]["s"]
        c = np.mean([p[0]["u"] - sgn * k * p[1]["u"] for p in (p1, p2)])
        h = (k, a, c, sgn); sc, pairs = _score(h, A, B, tol)
        if sc > best[0]: best = (sc, h, pairs)
    return dict(score=best[0], hypothesis=best[1], pairs=best[2], n_text_candidates=len(text))

def control(A, B, opposite=True, runs=10, **kw):
    """Same search with B's landmark positions scrambled along the street (texts kept)."""
    rng = np.random.default_rng(1); scores = []
    s = np.array([x["s"] for x in B])
    for r in range(runs):
        perm = rng.permutation(len(B))
        Bs = [dict(x, s=float(s[perm[i]])) for i, x in enumerate(B)]
        scores.append(register(A, Bs, opposite, iters=4000, seed=r, **kw)["score"])
    return scores

def to_similarity(Astrip, Bstrip, h):
    """Turn a strip-space hypothesis into a world Sim3 (B world -> A world) by fitting
    Umeyama on points along B's path (cameras + lateral/vertical offsets)."""
    k, a, c, sgn = h
    C = np.array([np.linalg.inv(Bstrip.poses[n])[:3, 3] for n in Bstrip.names])
    sb = Bstrip.to_strip(C)
    pts = np.concatenate([sb + [0, du, dz] for du in (-6, 0, 6) for dz in (-1.5, 0, 3)])
    Pw = Bstrip.to_world(pts[:, 0], pts[:, 1], pts[:, 2])
    sA, uA = _apply(h, pts[:, 0], pts[:, 1]); zA = k * pts[:, 2]
    Qw = Astrip.to_world(sA, uA, zA)
    ms, md = Pw.mean(0), Qw.mean(0); S, D = Pw - ms, Qw - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(S)); E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: E[2, 2] = -1
    R = U @ E @ Vt; s = np.trace(np.diag(sig) @ E) / S.var(0).sum()
    resid = np.linalg.norm(Pw @ (s * R).T + md - s * R @ ms - Qw, axis=1)
    return s, R, md - s * R @ ms, float(np.median(resid)) * Astrip.mpu
