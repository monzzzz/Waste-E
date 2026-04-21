from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EgoState:
    timestamp_s: float
    x_m: float
    y_m: float
    yaw_rad: float = 0.0
    speed_mps: float = 0.0
    distance_traveled_m: float = 0.0
    angular_velocity_rad_s: float = 0.0
    lat_deg: float | None = None
    lon_deg: float | None = None
    fix_quality: int = 0
    satellites: int = 0
    hdop: float | None = None
    source: str = "unknown"
