"""Gated alignment of one pass (piece) B into pass A from landmarks.
Decision ladder - every rung uses evidence the previous one didn't fit:
  1. landmark registration must beat a scrambled-B control by `margin`
  2. physical checks (gravity, opposite heading, cameras at road height)
  3. BEV paint must correlate best at zero along-street shift
Returns a report; `merge` is True only if all three pass."""
import numpy as np
from . import landmarks as lm, register as rg, verify
from .strip import Cameras

def align(A, B, margin=1.25):
    LA = lm.pole_landmarks(A) + lm.text_landmarks(A)
    LB = lm.pole_landmarks(B) + lm.text_landmarks(B)
    r = rg.register(LA, LB); ctl = rg.control(LA, LB)
    rep = dict(landmarks_A=len(LA), landmarks_B=len(LB), score=r["score"], control_max=max(ctl),
               shared_text=[(a["text"], b["text"]) for a in LA if a["kind"] == "text"
                            for b in LB if b["kind"] == "text" and lm.similar(a["text"], b["text"])],
               merge=False)
    if r["hypothesis"] is None or r["score"] <= margin * max(ctl):
        rep["reason"] = "landmark agreement not above chance (scrambled control)"
        return rep
    s, R, t, fit_m = rg.to_similarity(A, B, r["hypothesis"])
    cams = Cameras(B, A, s, R, t)
    rep.update(hypothesis=r["hypothesis"], sim3_fit_m=fit_m, physical=verify.physical_checks(A, cams))
    if not rep["physical"]["ok"]:
        rep["reason"] = "physical checks failed"; return rep
    p = verify.paint_shift_test(A, cams); p.pop("curve")
    rep["paint"] = p
    if abs(p["best_shift_m"]) > 1.0 or not p["margin"] > 0:
        rep["reason"] = "BEV paint does not peak at zero shift"; return rep
    rep.update(merge=True, sim3=dict(s=s, R=R.tolist(), t=t.tolist()), reason="all checks passed")
    return rep
