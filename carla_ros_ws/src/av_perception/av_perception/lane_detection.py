"""
lane_detection.py — ROS-free lane-offset estimation from a CARLA semantic
segmentation image.

Input format
------------
carla_ros_bridge publishes the semantic camera after converting it with
`carla.ColorConverter.CityScapesPalette`, so pixels hold *palette colours*,
not label IDs.  Road lines are (R, G, B) = (157, 234, 50) in CARLA 0.9.x.

Geometry (flat-ground inverse perspective mapping)
--------------------------------------------------
Pinhole camera at height h, pitched down by θ, zero roll.  For pixel
(u, v) let x = (u − cx)/f, y = (v − cy)/f.  The viewing ray has
    forward = cos θ − y·sin θ,   down = sin θ + y·cos θ,   right = x
and hits the ground (if down > 0) at t = h / down:
    forward distance   Z = t · forward
    lateral (right +)  X = t · x
With θ = 0 this reduces to Z = f·h/(v − cy), X = (u − cx)·h/(v − cy).
Lateral distances therefore come out in metres without any
pixels-per-metre calibration.

The ego lane's two markings are picked in the nearest rows (where they
straddle the image centre) and tracked row by row toward the horizon, so
they are followed through curves.  Each row gives the lane centre c at
forward distance Z.

c(Z) is not the offset at the vehicle: it also contains heading error
(ψ·Z) and road curvature (Z²/2R).  We fit
    c(Z) = c0 + c1·Z + c2·Z²
over the rows (robustly, with outlier rejection) and report c0, the
offset at the camera.  With too few rows / too short a span the fit falls
back to linear, then to the median.

Sign convention
---------------
    offset > 0  → ego is LEFT of the lane centre (lane centre is to the right)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ROAD_LINE_RGB = (157, 234, 50)


@dataclass
class CameraModel:
    fx: float
    cx: float
    cy: float
    height_m: float
    pitch_rad: float = 0.0          # positive = looking down

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float, height_m: float,
                 pitch_deg: float = 0.0) -> "CameraModel":
        f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
        return cls(fx=f, cx=width / 2.0, cy=height / 2.0, height_m=height_m,
                   pitch_rad=float(np.radians(pitch_deg)))

    def row_to_ground(self, v_centre: float) -> Optional[tuple[float, float]]:
        """For an image row, return (forward distance Z, metres-per-pixel scale) or None."""
        y = (v_centre - self.cy) / self.fx
        down = np.sin(self.pitch_rad) + y * np.cos(self.pitch_rad)
        if down <= 1e-6:
            return None                        # at or above the horizon
        t = self.height_m / down
        return t * (np.cos(self.pitch_rad) - y * np.sin(self.pitch_rad)), t / self.fx

    def ground_to_pixel(self, fwd: float, right: float) -> Optional[tuple[float, float]]:
        """Project a ground point (camera-relative, metres) to (u, v)."""
        c, s = np.cos(self.pitch_rad), np.sin(self.pitch_rad)
        zc = fwd * c + self.height_m * s
        yc = -fwd * s + self.height_m * c
        if zc <= 1e-3:
            return None
        return self.cx + self.fx * right / zc, self.cy + self.fx * yc / zc


@dataclass
class LaneEstimate:
    offset_m: Optional[float]                       # None = no detection
    lane_width_m: Optional[float] = None
    rows_used: int = 0
    # (v, u) pixel coordinates of the markings used, for debug drawing
    left_px: list[tuple[int, int]] = field(default_factory=list)
    right_px: list[tuple[int, int]] = field(default_factory=list)


def road_line_mask(rgb: np.ndarray, tol: int = 12) -> np.ndarray:
    """Boolean mask of pixels whose colour is within `tol` of the road-line colour."""
    ref = np.asarray(ROAD_LINE_RGB, dtype=np.int16)
    diff = np.abs(rgb[:, :, :3].astype(np.int16) - ref[None, None, :])
    return np.all(diff <= tol, axis=2)


def _row_runs(cols: np.ndarray) -> list[tuple[float, int]]:
    """Group sorted column indices into contiguous runs → [(centre_col, n_pixels)]."""
    if cols.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(cols) > 1) + 1
    return [(float(r.mean()), int(r.size)) for r in np.split(cols, breaks)]


def _predict(hist: list[tuple[float, float]], z: float) -> float:
    """Extrapolate a marking's lateral position to distance z from its recent history."""
    if len(hist) < 3:
        return hist[-1][1]
    recent = np.asarray(hist[-6:])
    if np.ptp(recent[:, 0]) < 0.5:
        return float(recent[-1, 1])
    a, b = np.polyfit(recent[:, 0], recent[:, 1], 1)
    return float(a * z + b)


def estimate_lane_offset(
    mask: np.ndarray,
    cam: CameraModel,
    max_range_m: float = 15.0,
    lane_width_m: float = 3.5,
    min_lane_width_m: float = 2.5,
    max_lane_width_m: float = 5.0,
    min_rows: int = 3,
    min_quad_span_m: float = 5.0,
    gate_m: float = 0.5,
) -> LaneEstimate:
    """
    Estimate the ego's lateral offset from the lane centre.

    Markings are found in the nearest rows (bottom of the image), where the
    ego lane's left/right markings straddle the image centre, and then
    tracked row by row toward the horizon (nearest run to the predicted
    position), so they are followed through curves even after crossing the
    image centre.

    mask         : bool image, True where a road line is visible
    max_range_m  : ignore rows whose ground distance exceeds this
    lane_width_m : assumed width when only one marking is visible
    """
    h_img, _ = mask.shape
    centres: list[float] = []
    dists: list[float] = []
    widths: list[float] = []
    left_px: list[tuple[int, int]] = []
    right_px: list[tuple[int, int]] = []
    hist = {"L": [], "R": []}          # (z, lateral) per tracked marking

    for v in range(h_img - 1, -1, -1):
        ground = cam.row_to_ground(v + 0.5)
        if ground is None:
            break                           # at / above horizon
        z, m_per_px = ground
        if z > max_range_m:
            break
        runs = _row_runs(np.flatnonzero(mask[v]))
        if not runs:
            continue
        lats = [((u + 0.5 - cam.cx) * m_per_px, u) for u, _ in runs]   # metres, right +
        width = float(np.median(widths)) if widths else lane_width_m

        found = {}
        for side, sign in (("L", -1.0), ("R", 1.0)):
            if hist[side]:
                pred = _predict(hist[side], z)
                gate = gate_m
            else:
                other = "R" if side == "L" else "L"
                if hist[other]:
                    pred, gate = _predict(hist[other], z) + sign * width, gate_m
                else:
                    # initialise: nearest run on this side of the image centre
                    cands = [lu for lu in lats if sign * lu[0] > 0 and abs(lu[0]) <= max_lane_width_m]
                    if not cands:
                        continue
                    pred, gate = min(cands, key=lambda lu: abs(lu[0]))[0], 1e-6
            best = min(lats, key=lambda lu: abs(lu[0] - pred))
            if abs(best[0] - pred) <= gate:
                found[side] = best

        if "L" in found and "R" in found and found["L"][1] == found["R"][1]:
            del found["L" if abs(found["L"][0]) > abs(found["R"][0]) else "R"]
        if "L" in found and "R" in found:
            w = found["R"][0] - found["L"][0]
            if not (min_lane_width_m <= w <= max_lane_width_m):
                del found["L" if abs(found["L"][0]) > abs(found["R"][0]) else "R"]

        for side in found:
            hist[side].append((z, found[side][0]))
        if "L" in found and "R" in found:
            centres.append((found["L"][0] + found["R"][0]) / 2.0)
            widths.append(found["R"][0] - found["L"][0])
        elif "L" in found:
            centres.append(found["L"][0] + width / 2.0)
        elif "R" in found:
            centres.append(found["R"][0] - width / 2.0)
        else:
            continue
        dists.append(z)
        if "L" in found:
            left_px.append((v, int(round(found["L"][1]))))
        if "R" in found:
            right_px.append((v, int(round(found["R"][1]))))

    if len(centres) < min_rows:
        return LaneEstimate(offset_m=None, rows_used=len(centres),
                            left_px=left_px, right_px=right_px)
    return LaneEstimate(
        offset_m=_fit_offset_at_zero(np.asarray(dists), np.asarray(centres), min_quad_span_m),
        lane_width_m=float(np.median(widths)) if widths else None,
        rows_used=len(centres),
        left_px=left_px,
        right_px=right_px,
    )


def _fit_offset_at_zero(z: np.ndarray, c: np.ndarray, min_quad_span_m: float) -> float:
    """Extrapolate the lane-centre profile c(Z) to Z = 0 (robust polynomial fit)."""
    span = float(z.max() - z.min())
    if len(z) >= 6 and span >= min_quad_span_m:
        deg = 2
    elif len(z) >= 3 and span >= 2.0:
        deg = 1
    else:
        return float(np.median(c))
    keep = np.ones(len(z), bool)
    for _ in range(2):                       # refit without gross outliers
        coef = np.polyfit(z[keep], c[keep], deg)
        resid = np.abs(np.polyval(coef, z) - c)
        thresh = max(0.15, 2.5 * float(np.median(resid[keep])))
        new_keep = resid <= thresh
        if new_keep.sum() < deg + 2 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return float(np.polyval(coef, 0.0))
