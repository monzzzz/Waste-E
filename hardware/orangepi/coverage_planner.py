"""
Boustrophedon (lawnmower) coverage path planner.

Adapted from navigation geometry in the self-driving-pipeline
(alpamayo_pipeline/server/navigation.py) — pure Python, no extra deps.

Usage:
    from coverage_planner import plan_coverage
    result = plan_coverage(
        polygon_latlon=[[lat, lon], ...],
        strip_width_m=0.6,
        angle_deg=0.0,
        origin_latlon=(lat, lon),   # optional, defaults to polygon centroid
    )
    # result["path"]       → [[lat, lon], ...]
    # result["area_m2"]    → float
    # result["distance_m"] → float
    # result["strips"]     → int
"""
from __future__ import annotations

import math
from typing import Optional

EARTH_RADIUS_M = 6_371_000.0


# ── geo helpers (same as geo_utils.py) ────────────────────────────────────────

def _to_local(origin_lat: float, origin_lon: float,
              lat: float, lon: float) -> tuple[float, float]:
    mean_lat = math.radians((origin_lat + lat) * 0.5)
    x = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos(mean_lat)
    y = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    return x, y


def _to_latlon(origin_lat: float, origin_lon: float,
               x: float, y: float) -> tuple[float, float]:
    lat = origin_lat + math.degrees(y / EARTH_RADIUS_M)
    mean_lat = math.radians(origin_lat + math.degrees(y / EARTH_RADIUS_M * 0.5))
    lon = origin_lon + math.degrees(x / (EARTH_RADIUS_M * math.cos(mean_lat)))
    return lat, lon


# ── 2-D geometry helpers ───────────────────────────────────────────────────────

def _rotate(points: list[tuple[float, float]], angle_rad: float) -> list[tuple[float, float]]:
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in points]


def _polygon_area(pts: list[tuple[float, float]]) -> float:
    """Shoelace formula."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) * 0.5


def _horizontal_intersections(polygon: list[tuple[float, float]],
                               y: float) -> list[float]:
    """X-coords where horizontal line at height y crosses polygon edges."""
    xs: list[float] = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if y1 == y2:
            continue
        if not (min(y1, y2) <= y < max(y1, y2)):
            continue
        t = (y - y1) / (y2 - y1)
        xs.append(x1 + t * (x2 - x1))
    xs.sort()
    return xs


def _densify(p1: tuple[float, float], p2: tuple[float, float],
             step_m: float) -> list[tuple[float, float]]:
    """Interpolate evenly spaced points between p1 and p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return [p1]
    n = max(1, int(math.ceil(dist / step_m)))
    return [(p1[0] + dx * i / n, p1[1] + dy * i / n) for i in range(n)]


# ── public API ─────────────────────────────────────────────────────────────────

def plan_coverage(
    polygon_latlon: list[list[float]],
    strip_width_m: float = 0.6,
    angle_deg: float = 0.0,
    origin_latlon: Optional[tuple[float, float]] = None,
    densify_step_m: float = 0.5,
) -> dict:
    """
    Generate a boustrophedon coverage path for a polygon.

    Args:
        polygon_latlon: Closed or open polygon as [[lat, lon], ...].
        strip_width_m:  Width of each sweep pass (≈ robot width).
        angle_deg:      Sweep direction angle (0 = E-W strips, 90 = N-S strips).
        origin_latlon:  Local-coordinate origin; defaults to polygon centroid.
        densify_step_m: Point spacing along the path.

    Returns dict with keys:
        path       [[lat, lon], ...]   ordered coverage path
        area_m2    float
        distance_m float
        strips     int
        notes      [str]
    """
    if len(polygon_latlon) < 3:
        return {"path": [], "area_m2": 0.0, "distance_m": 0.0, "strips": 0, "notes": ["Need ≥ 3 vertices."]}

    strip_width_m = max(strip_width_m, 0.05)
    notes: list[str] = []

    # ── pick origin ────────────────────────────────────────────────────────────
    if origin_latlon is None:
        mean_lat = sum(p[0] for p in polygon_latlon) / len(polygon_latlon)
        mean_lon = sum(p[1] for p in polygon_latlon) / len(polygon_latlon)
        origin_lat, origin_lon = mean_lat, mean_lon
    else:
        origin_lat, origin_lon = origin_latlon

    # ── convert polygon to local XY ────────────────────────────────────────────
    local_poly = [_to_local(origin_lat, origin_lon, p[0], p[1]) for p in polygon_latlon]
    area_m2 = _polygon_area(local_poly)
    notes.append(f"Area ≈ {area_m2:.1f} m²")

    # ── rotate so sweep direction aligns with X axis ───────────────────────────
    angle_rad = math.radians(angle_deg)
    rotated = _rotate(local_poly, -angle_rad)

    min_y = min(p[1] for p in rotated)
    max_y = max(p[1] for p in rotated)
    height = max_y - min_y
    n_strips = max(1, math.ceil(height / strip_width_m))
    notes.append(f"{n_strips} strips × {strip_width_m:.2f} m")

    # ── build sweep lines ──────────────────────────────────────────────────────
    # strip_pairs: [(start_xy, end_xy), ...] — one per sweep pass, in order
    strip_pairs: list[tuple[tuple[float,float], tuple[float,float]]] = []
    left_to_right = True

    for i in range(n_strips):
        y = min_y + (i + 0.5) * strip_width_m
        if y > max_y:
            break
        xs = _horizontal_intersections(rotated, y)
        if len(xs) < 2:
            continue

        segments = [(xs[j], xs[j + 1]) for j in range(0, len(xs) - 1, 2)]
        for seg_x_min, seg_x_max in (segments if left_to_right else reversed(segments)):
            if left_to_right:
                strip_pairs.append(((seg_x_min, y), (seg_x_max, y)))
            else:
                strip_pairs.append(((seg_x_max, y), (seg_x_min, y)))

        left_to_right = not left_to_right

    if not strip_pairs:
        return {"path": [], "strips_preview": [], "area_m2": area_m2,
                "distance_m": 0.0, "strips": 0, "notes": notes}

    # ── build ordered nav path (strip + short connector) ──────────────────────
    path_rotated: list[tuple[float, float]] = []
    for idx, (start, end) in enumerate(strip_pairs):
        if idx == 0:
            path_rotated.extend(_densify(start, end, densify_step_m))
            path_rotated.append(end)
        else:
            prev_end = strip_pairs[idx - 1][1]
            # short connector (turnaround edge) — no densification needed
            path_rotated.append(prev_end)
            path_rotated.append(start)
            path_rotated.extend(_densify(start, end, densify_step_m))
            path_rotated.append(end)

    # ── un-rotate & convert to lat/lon ────────────────────────────────────────
    def to_ll(pts: list[tuple[float,float]]) -> list[list[float]]:
        unrot = _rotate(pts, angle_rad)
        return [list(_to_latlon(origin_lat, origin_lon, x, y)) for x, y in unrot]

    path_latlon = to_ll(path_rotated)

    # strip_pairs for clean visualization: [[start_ll, end_ll], ...]
    strips_preview = [
        [list(_to_latlon(origin_lat, origin_lon, *_rotate([s], angle_rad)[0])),
         list(_to_latlon(origin_lat, origin_lon, *_rotate([e], angle_rad)[0]))]
        for s, e in strip_pairs
    ]

    # ── total nav distance ─────────────────────────────────────────────────────
    path_local = _rotate(path_rotated, angle_rad)
    distance_m = sum(
        math.hypot(path_local[i+1][0]-path_local[i][0],
                   path_local[i+1][1]-path_local[i][1])
        for i in range(len(path_local)-1)
    )
    notes.append(f"Path ≈ {distance_m:.1f} m")

    return {
        "path":           path_latlon,
        "strips_preview": strips_preview,   # [[start_ll, end_ll], ...] per strip
        "area_m2":        area_m2,
        "distance_m":     distance_m,
        "strips":         len(strip_pairs),
        "notes":          notes,
    }
