"""Checks for a candidate alignment of pass B into pass A that use evidence the alignment
itself was not fitted on. On ECB these rejected two MASt3R alignments that had passed the
obvious checks (consistent heading, plausible length): one 15 deg off level, one with the
westbound cameras 4 m under the road."""
import numpy as np
from . import bev
from .strip import CAM_HEIGHT_M

def physical_checks(A, cams):
    """A: StripMap of pass A; cams: strip.Cameras of pass B posed in A's world."""
    upB = -np.array([cams.poses[n][:3, :3].T @ [0, 1, 0] for n in cams.names]).mean(0)
    grav = np.degrees(np.arccos(np.clip(upB @ A.up / np.linalg.norm(upB), -1, 1)))
    road = A.road_z(cams.cam_s)                              # A's road under B's cameras
    return dict(
        gravity_deg=float(grav),                             # want < ~2
        opposite_heading=bool(cams.direction == -1),
        cam_height_m=float(np.median(cams.cam_z - road)),    # want ~ CAM_HEIGHT_M
        lateral_m=float(np.median(cams.cam_u)),              # westbound in US traffic: north of eastbound
        ok=bool(grav < 3 and abs(np.median(cams.cam_z - road) - CAM_HEIGHT_M) < 0.75))

def paint_shift_test(A, cams, res=0.1, u_lim=9.0, max_shift_m=20.0):
    """Correlate A's BEV paint with B's (rendered into A's grid) while sliding B along the
    street. A correct alignment peaks at zero shift, clearly above the other shifts."""
    lo = min(A.cam_s.min(), cams.cam_s.min()) - 20; hi = max(A.cam_s.max(), cams.cam_s.max()) + 20
    EA = bev.render(A, res=res, u_lim=u_lim, s_range=(lo, hi))
    EB = bev.render(A, res=res, u_lim=u_lim, cams=cams, s_range=(lo, hi))
    PA, PB = bev.paint(EA["rgb"], EA["weight"] > 0), bev.paint(EB["rgb"], EB["weight"] > 0)
    out = []
    for k in range(-int(max_shift_m / res), int(max_shift_m / res) + 1, max(1, int(0.5 / res))):
        m = (EA["weight"] > 0) & np.roll(EB["weight"] > 0, k, axis=0)
        if m.sum() < 2000: continue
        x = PA[m] - PA[m].mean(); y = np.roll(PB, k, axis=0)[m]; y = y - y.mean()
        out.append((k * res, float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9))))
    out = np.array(out); at0 = float(out[np.argmin(np.abs(out[:, 0])), 1])
    others = out[np.abs(out[:, 0]) > 2, 1]
    return dict(ncc_at_0=at0, best_shift_m=float(out[out[:, 1].argmax(), 0]),
                margin=float(at0 - np.percentile(others, 95)) if len(others) else float("nan"),
                curve=out.tolist())
