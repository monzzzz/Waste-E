from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(dlon / 2.0) ** 2)
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(max(0.0, min(a, 1.0))))


def latlon_to_local_m(
    origin_lat_deg: float,
    origin_lon_deg: float,
    lat_deg: float,
    lon_deg: float,
) -> tuple[float, float]:
    mean_lat_rad = math.radians((origin_lat_deg + lat_deg) * 0.5)
    x_m = math.radians(lon_deg - origin_lon_deg) * EARTH_RADIUS_M * math.cos(mean_lat_rad)
    y_m = math.radians(lat_deg - origin_lat_deg) * EARTH_RADIUS_M
    return x_m, y_m
