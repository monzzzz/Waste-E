"""
Download Google Maps satellite tiles for the Victoria area
Usage: python download_map_tiles.py
"""
import math
import os
import csv
import time
from pathlib import Path

import requests
from tqdm.auto import tqdm  # ✅ progress bar


# Area bounds (Victoria, BC area)
LAT_MAX = 48.453058
LON_MIN = -123.331381
LAT_MIN = 48.450383
LON_MAX = -123.325403

ZOOM = 19
OUT_DIR = Path(f"tiles_z{ZOOM}_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1) Google Map Tiles API settings
# ----------------------------
GOOGLE_MAPS_API_KEY = "AIzaSyBxD_mG8ehYecH2mdSvoM3kkpyBQASU6rA"

CREATE_SESSION_URL = f"https://tile.googleapis.com/v1/createSession?key={GOOGLE_MAPS_API_KEY}"

SESSION_BODY = {
    "mapType": "satellite",
    "language": "en-US",
    "region": "CA"
}

REQUEST_SLEEP_SEC = 0.03

def latlon_to_tile_xy(lat: float, lon: float, z: int):
    """Convert lat/lon to tile coordinates"""
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def tile_xy_to_bounds(x: int, y: int, z: int):
    """Get lat/lon bounds for a tile"""
    n = 2.0 ** z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0

    def mercator_to_lat(ty):
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
        return math.degrees(lat_rad)

    lat_max = mercator_to_lat(y)
    lat_min = mercator_to_lat(y + 1)
    return lat_min, lon_min, lat_max, lon_max


def main():
    print("Creating Google Maps API session...")
    sess = requests.Session()
    resp = sess.post(CREATE_SESSION_URL, json=SESSION_BODY, timeout=60)
    resp.raise_for_status()
    session_token = resp.json()["session"]
    expiry = resp.json().get("expiry", None)

    print("Session token created.")
    print("Expiry:", expiry)

    # Calculate tile range
    x_a, y_a = latlon_to_tile_xy(LAT_MAX, LON_MIN, ZOOM)  # NW
    x_b, y_b = latlon_to_tile_xy(LAT_MIN, LON_MAX, ZOOM)  # SE

    x0, x1 = sorted([x_a, x_b])
    y0, y1 = sorted([y_a, y_b])

    tile_count = (x1 - x0 + 1) * (y1 - y0 + 1)
    print(f"Zoom {ZOOM} tile X: {x0}..{x1} ({x1-x0+1})")
    print(f"Zoom {ZOOM} tile Y: {y0}..{y1} ({y1-y0+1})")
    print("Total tiles:", tile_count)

    index_path = OUT_DIR / "tiles_index.csv"
    rows = []

    def download_google_tile(z, x, y, out_path: Path):
        url = f"https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}?session={session_token}&key={GOOGLE_MAPS_API_KEY}"
        r = sess.get(url, timeout=60)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"

        out_path.write_bytes(r.content)

        if out_path.stat().st_size < 2000:
            return False, "Too small (maybe error tile)"
        return True, "OK"

    ok, fail = 0, 0

    coords = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    for x, y in tqdm(coords, total=len(coords), desc=f"Downloading z{ZOOM} tiles"):
        filename = f"{x}_{y}.jpg"
        out_path = OUT_DIR / filename

        if out_path.exists() and out_path.stat().st_size > 0:
            status = "SKIP_EXISTS"
            success = True
        else:
            success, status = download_google_tile(ZOOM, x, y, out_path)
            time.sleep(REQUEST_SLEEP_SEC)

        lat_min_t, lon_min_t, lat_max_t, lon_max_t = tile_xy_to_bounds(x, y, ZOOM)

        rows.append({
            "z": ZOOM,
            "x": x,
            "y": y,
            "file": str(out_path),
            "status": status,
            "lat_min": lat_min_t,
            "lon_min": lon_min_t,
            "lat_max": lat_max_t,
            "lon_max": lon_max_t,
        })

        if success:
            ok += 1
        else:
            fail += 1

    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Done. OK={ok}, Failed={fail}")
    print("Tiles folder:", OUT_DIR.resolve())
    print("Index CSV:", index_path.resolve())


if __name__ == "__main__":
    main()
