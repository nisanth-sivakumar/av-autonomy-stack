import cv2
import numpy as np
import pytest

from av_perception.lane_detection import (
    CameraModel,
    LaneOffsetFilter,
    ROAD_LINE_RGB,
    road_line_mask,
    estimate_lane_offset,
)

# Matches semantic_segmentation_front in config/objects.json
W, H = 400, 200
CAM = CameraModel.from_fov(W, H, fov_deg=90.0, height_m=2.0, pitch_deg=20.0)
ROAD_RGB = (128, 64, 128)


def render(cam, offset_m, heading_rad=0.0, radius_m=1e9, lane_width=3.5,
           draw=("left", "right"), dashed=False):
    """
    Palette image of a lane seen by `cam`.

    The camera sits `offset_m` LEFT of the lane centre with heading error
    `heading_rad` (+ = rotated left); the lane curves with radius `radius_m`
    (+ = left turn).
    """
    img = np.zeros((cam_h(cam), W, 3), np.uint8)
    img[:, :] = ROAD_RGB
    halves = {"left": lane_width / 2, "right": -lane_width / 2}
    for side in draw:
        half = halves[side]
        pts = []
        for s in np.arange(0.0, 60.0, 0.1):
            if dashed and int(s) % 6 < 3:
                pts.append(None)
                continue
            a = s / radius_m
            lx = radius_m * np.sin(a) - half * np.sin(a)
            ly = radius_m * (1 - np.cos(a)) + half * np.cos(a)
            dx, dy = lx, ly - offset_m
            fwd = dx * np.cos(heading_rad) + dy * np.sin(heading_rad)
            left = -dx * np.sin(heading_rad) + dy * np.cos(heading_rad)
            pts.append(cam.ground_to_pixel(fwd, -left) if fwd > 0.3 else None)
        for a_, b_ in zip(pts[:-1], pts[1:]):
            if a_ is None or b_ is None:
                continue
            cv2.line(img, tuple(int(round(v)) for v in a_), tuple(int(round(v)) for v in b_),
                     ROAD_LINE_RGB, 2)
    return img


def cam_h(cam):
    return int(round(cam.cy * 2))


def estimate(img, cam=CAM, **kw):
    return estimate_lane_offset(road_line_mask(img), cam, **kw)


def test_ground_projection_round_trip():
    for fwd, right in [(3.0, 0.0), (8.0, -1.75), (14.0, 2.0)]:
        u, v = CAM.ground_to_pixel(fwd, right)
        z, m_per_px = CAM.row_to_ground(v)
        assert z == pytest.approx(fwd, rel=1e-6)
        assert (u - CAM.cx) * m_per_px == pytest.approx(right, abs=1e-6)


def test_mask_matches_palette_colour_only():
    assert road_line_mask(render(CAM, 0.0)).any()
    assert not road_line_mask(np.full((H, W, 3), ROAD_RGB, np.uint8)).any()


@pytest.mark.parametrize("offset", [-0.8, -0.3, 0.0, 0.4, 0.9])
def test_offset_recovered_in_metres_with_correct_sign(offset):
    est = estimate(render(CAM, offset))
    assert est.offset_m == pytest.approx(offset, abs=0.08)
    assert est.lane_width_m == pytest.approx(3.5, abs=0.2)


@pytest.mark.parametrize("side", ["left", "right"])
def test_single_marking_uses_assumed_width(side):
    est = estimate(render(CAM, 0.5, draw=(side,)), lane_width_m=3.5)
    assert est.offset_m == pytest.approx(0.5, abs=0.1)


def test_dashed_markings_still_detected():
    assert estimate(render(CAM, 0.3, dashed=True)).offset_m == pytest.approx(0.3, abs=0.1)


@pytest.mark.parametrize("offset,heading,radius", [
    (0.0, 0.0, 30.0), (0.5, 0.0, 30.0), (-0.4, 0.05, 30.0),
    (0.3, -0.05, -40.0), (0.2, 0.08, 1e9), (-0.6, -0.08, 60.0),
])
def test_offset_at_vehicle_on_curves_and_with_heading_error(offset, heading, radius):
    est = estimate(render(CAM, offset, heading, radius))
    assert est.offset_m == pytest.approx(offset, abs=0.12)


def test_zero_pitch_camera_still_supported():
    cam = CameraModel.from_fov(W, 70, fov_deg=90.0, height_m=2.0, pitch_deg=0.0)
    est = estimate(render(cam, 0.4), cam=cam, max_range_m=25.0)
    assert est.offset_m == pytest.approx(0.4, abs=0.15)


def test_no_markings_returns_none():
    assert estimate(np.full((H, W, 3), ROAD_RGB, np.uint8)).offset_m is None


# ── Plausibility gates ───────────────────────────────────────────────────────

def add_crosswalk(img, cam, z_from=4.0, z_to=8.0, bar_w=0.5, gap=0.5, half_span=5.0):
    """Zebra bars (elongated along the road) painted across the road ahead."""
    poly_img = img.copy()
    x = -half_span
    while x < half_span:
        corners = [(z_from, x), (z_from, x + bar_w), (z_to, x + bar_w), (z_to, x)]
        px = [cam.ground_to_pixel(z, xx) for z, xx in corners]
        if all(p is not None for p in px):
            cv2.fillPoly(poly_img, [np.array([[int(round(u)), int(round(v))] for u, v in px])],
                         ROAD_LINE_RGB)
        x += bar_w + gap
    return poly_img


def test_crosswalk_alone_is_rejected():
    img = add_crosswalk(np.full((H, W, 3), ROAD_RGB, np.uint8), CAM)
    est = estimate(img)
    assert est.offset_m is None and est.reject_reason


def test_crosswalk_rows_are_skipped_when_lane_lines_exist():
    img = add_crosswalk(render(CAM, 0.4), CAM, z_from=6.0, z_to=9.0)
    est = estimate(img)
    assert est.offset_m == pytest.approx(0.4, abs=0.12)


def test_implausible_lane_width_rejected():
    # Both markings pass the loose per-row bounds (2.5-5.0 m) but a 4.85 m
    # median width is not a real lane (e.g. two lines of different lanes).
    est = estimate(render(CAM, 0.0, lane_width=4.85))
    assert est.offset_m is None and "width" in est.reject_reason


# ── Temporal filter ──────────────────────────────────────────────────────────

def test_filter_needs_confirmation_before_first_value():
    f = LaneOffsetFilter(confirm_frames=3)
    assert f.update(0.0, 0.2) is None
    assert f.update(0.1, 0.22) is None
    assert f.update(0.2, 0.21) == pytest.approx(0.21)


def test_filter_rejects_implausible_jumps_and_holds():
    f = LaneOffsetFilter(max_lateral_rate=1.5, hold_s=1.0, confirm_frames=1)
    assert f.update(0.0, 0.2) == pytest.approx(0.2)
    assert f.update(0.1, 1.9) == pytest.approx(0.2)        # 17 m/s sideways: bad frame
    assert f.update(0.2, 0.25) == pytest.approx(0.25)      # plausible: accepted
    assert f.update(0.6, None) == pytest.approx(0.25)      # within hold
    assert f.update(1.5, None) is None                     # hold expired


def test_filter_reacquires_after_garbage():
    f = LaneOffsetFilter(confirm_frames=3, hold_s=0.5)
    for t, v in [(0.0, 0.1), (0.1, 0.1), (0.2, 0.1)]:
        f.update(t, v)
    for t, v in [(0.3, 2.0), (0.4, -1.5), (0.5, 1.2), (0.9, -2.0)]:   # intersection garbage
        out = f.update(t, v)
    assert out is None                                      # hold expired, garbage not accepted
    for t, v in [(1.0, 0.3), (1.1, 0.31), (1.2, 0.3)]:
        out = f.update(t, v)
    assert out == pytest.approx(0.3)
