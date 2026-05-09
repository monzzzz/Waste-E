"""
Debug visualizer — produces a frame matching the SiT dataset style:

  ┌──────────────┬──────────────┬──────────────┐
  │  Front-Left  │  Front Wide  │  Front-Right │   ← real SiT camera images
  │   (cam 2)    │   (cam 3)    │   (cam 4)    │
  └──────────────┴──────────────┴──────────────┘
  ┌──────────────────────────────────────────────┐
  │           Bird's-Eye View (BEV)              │
  │  • pedestrian past traj  ── blue             │
  │  • pedestrian future traj── red              │
  │  • robot past traj       ── yellow           │
  │  • Alpamayo prediction   ── green            │
  │  • pedestrian boxes      ── white            │
  └──────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── colours (BGR) ────────────────────────────────────────────────────────────
C_BG          = (15,  15,  15)
C_GRID        = (40,  40,  40)
C_PED_PAST    = (180, 100,  30)   # blue-ish
C_PED_FUTURE  = (30,  30, 200)    # red
C_PED_BOX     = (220, 220, 220)   # white
C_ROBOT_PAST  = (30, 200, 220)    # yellow
C_ROBOT       = (0,  220,   0)    # bright green
C_PRED        = (0,  230, 100)    # green trajectory
C_TEXT        = (230, 230, 230)
C_HUD_BG      = (0,   0,   0)

CAM_LABELS = ["Left  (cam1)", "Front  (cam4)", "Right  (cam5)"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _world_to_bev(
    wx: float, wy: float,
    robot_x: float, robot_y: float, robot_yaw: float,
    cx: int, cy: int, scale: float,
) -> Tuple[int, int]:
    """World (x,y) → BEV pixel (px, py) centred on robot."""
    dx = wx - robot_x
    dy = wy - robot_y
    # rotate into robot frame
    cos_y, sin_y = math.cos(-robot_yaw), math.sin(-robot_yaw)
    lx =  cos_y * dx - sin_y * dy   # forward
    ly =  sin_y * dx + cos_y * dy   # left
    px = int(cx - ly * scale)
    py = int(cy - lx * scale)
    return px, py


def _draw_ped_box(
    bev: np.ndarray,
    cx_bev: int, cy_bev: int,
    heading: float,
    box_half_l: float = 12, box_half_w: float = 8,
    color: Tuple = C_PED_BOX,
):
    """Draw a small oriented rectangle for a pedestrian."""
    corners_local = np.array([
        [ box_half_l,  box_half_w],
        [ box_half_l, -box_half_w],
        [-box_half_l, -box_half_w],
        [-box_half_l,  box_half_w],
    ], dtype=np.float32)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    pts = (corners_local @ rot.T).astype(int)
    pts[:, 0] += cx_bev
    pts[:, 1] += cy_bev
    cv2.polylines(bev, [pts], True, color, 1, cv2.LINE_AA)


def _draw_robot(bev: np.ndarray, cx: int, cy: int, size: int = 14):
    """Draw robot as filled green triangle pointing up."""
    tri = np.array([
        [cx,          cy - size],
        [cx - size//2, cy + size//2],
        [cx + size//2, cy + size//2],
    ])
    cv2.fillPoly(bev, [tri], C_ROBOT)
    cv2.polylines(bev, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)


# ── main API ──────────────────────────────────────────────────────────────────

def build_debug_frame(
    cam_imgs: List[Optional[np.ndarray]],       # [left, front, right] RGB (384×640)
    waypoints: Optional[np.ndarray],            # (T,2) ego-frame x=fwd, y=left
    trajectories,                               # list of PedestrianTrajectory
    sim_frame: int,
    robot_pos: np.ndarray,                      # (3,) world xyz
    robot_yaw: float,                           # radians
    robot_history: List[Tuple[float, float]],   # [(wx,wy), ...] past robot positions
    cmd_linear: float,
    cmd_angular: float,
    confidence: float,
    cam_w: int = 640,
    cam_h: int = 384,
    bev_h: int = 400,
) -> np.ndarray:
    """
    Returns a composite debug image:
      top  : camera strip  (3 × cam_w  wide, cam_h tall)
      bottom: BEV panel    (3 × cam_w  wide, bev_h tall)
    """
    total_w = cam_w * 3

    # ── 1. Camera strip ───────────────────────────────────────────────────────
    cam_strip = np.zeros((cam_h, total_w, 3), dtype=np.uint8)
    for i, img in enumerate(cam_imgs):
        x0 = i * cam_w
        if img is not None:
            tile = img if img.shape[:2] == (cam_h, cam_w) else \
                   cv2.resize(img, (cam_w, cam_h))
            cam_strip[:, x0:x0 + cam_w] = tile
        # Camera label bar
        cv2.rectangle(cam_strip, (x0, 0), (x0 + cam_w, 22), C_HUD_BG, -1)
        cv2.putText(cam_strip, CAM_LABELS[i], (x0 + 6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 255), 1, cv2.LINE_AA)
        # Divider
        if i > 0:
            cv2.line(cam_strip, (x0, 0), (x0, cam_h), (80, 80, 80), 1)

    # ── 2. BEV panel ──────────────────────────────────────────────────────────
    bev = np.full((bev_h, total_w, 3), C_BG, dtype=np.uint8)
    scale = 22.0          # pixels per metre
    cx, cy = total_w // 2, bev_h // 2 + 40   # robot origin in BEV

    # Grid (every 2 m)
    for m in range(-12, 13):
        d = int(m * 2 * scale)
        # horizontal
        y_px = cy - d
        if 0 <= y_px < bev_h:
            cv2.line(bev, (0, y_px), (total_w, y_px), C_GRID, 1)
        # vertical
        x_px = cx + d
        if 0 <= x_px < total_w:
            cv2.line(bev, (x_px, 0), (x_px, bev_h), C_GRID, 1)
    # Distance rings
    for r_m in [2, 4, 6, 8]:
        r_px = int(r_m * scale)
        cv2.circle(bev, (cx, cy), r_px, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(bev, f"{r_m}m", (cx + r_px + 2, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    rxy = (robot_pos[0], robot_pos[1])

    # Pedestrian trajectories
    past_steps  = 20
    future_steps = 30
    for traj in trajectories:
        cur_frame = sim_frame
        # Current position
        pos = traj.position_at(cur_frame)
        if pos is None:
            continue
        px_now, py_now = _world_to_bev(pos[0], pos[1], rxy[0], rxy[1], robot_yaw, cx, cy, scale)

        # Past trajectory (blue)
        past_pts = []
        for df in range(past_steps, 0, -1):
            p = traj.position_at(max(0, cur_frame - df))
            if p:
                past_pts.append(_world_to_bev(p[0], p[1], rxy[0], rxy[1], robot_yaw, cx, cy, scale))
        past_pts.append((px_now, py_now))
        if len(past_pts) >= 2:
            cv2.polylines(bev, [np.array(past_pts)], False, C_PED_PAST, 1, cv2.LINE_AA)

        # Future trajectory (red)
        future_pts = [(px_now, py_now)]
        for df in range(1, future_steps + 1):
            p = traj.position_at(cur_frame + df)
            if p:
                future_pts.append(_world_to_bev(p[0], p[1], rxy[0], rxy[1], robot_yaw, cx, cy, scale))
        if len(future_pts) >= 2:
            cv2.polylines(bev, [np.array(future_pts)], False, C_PED_FUTURE, 1, cv2.LINE_AA)

        # Box at current position
        if 0 <= px_now < total_w and 0 <= py_now < bev_h:
            _draw_ped_box(bev, px_now, py_now, traj.heading_at(cur_frame))

    # Robot past trajectory (yellow)
    if len(robot_history) >= 2:
        hist_pts = [
            _world_to_bev(wx, wy, rxy[0], rxy[1], robot_yaw, cx, cy, scale)
            for wx, wy in robot_history
        ]
        cv2.polylines(bev, [np.array(hist_pts)], False, C_ROBOT_PAST, 2, cv2.LINE_AA)

    # Alpamayo predicted trajectory (green)
    if waypoints is not None and len(waypoints) > 0:
        pred_pts = []
        for i, (fx, fy) in enumerate(waypoints):
            # ego frame: x=fwd → up in BEV, y=left → left in BEV
            px = int(cx - fy * scale)
            py = int(cy - fx * scale)
            if 0 <= px < total_w and 0 <= py < bev_h:
                pred_pts.append((px, py))
                r = max(2, 5 - i // 3)
                cv2.circle(bev, (px, py), r, C_PRED, -1, cv2.LINE_AA)
        if len(pred_pts) >= 2:
            cv2.polylines(bev, [np.array(pred_pts)], False, C_PRED, 2, cv2.LINE_AA)

    # Robot
    _draw_robot(bev, cx, cy)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = [
        (C_PED_PAST,   "Past traj. of Pedestrian"),
        (C_PED_FUTURE, "Future traj. of Pedestrian"),
        (C_PED_BOX,    "Pedestrian"),
        (C_ROBOT_PAST, "Past traj. of Robot"),
        (C_PRED,       "Alpamayo prediction"),
        (C_ROBOT,      "Robot"),
    ]
    lx, ly = 10, 18
    cv2.rectangle(bev, (6, 6), (270, 14 + len(legend) * 18), (30, 30, 30), -1)
    for color, label in legend:
        cv2.line(bev, (lx, ly), (lx + 22, ly), color, 2, cv2.LINE_AA)
        cv2.putText(bev, label, (lx + 28, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT, 1, cv2.LINE_AA)
        ly += 18

    # ── HUD bar ───────────────────────────────────────────────────────────────
    hud = (f"step {sim_frame:04d}  |  "
           f"v = {cmd_linear:+.2f} m/s  "
           f"ω = {cmd_angular:+.3f} rad/s  "
           f"conf = {confidence:.2f}")
    cv2.rectangle(bev, (0, bev_h - 22), (total_w, bev_h), C_HUD_BG, -1)
    cv2.putText(bev, hud, (10, bev_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 255, 60), 1, cv2.LINE_AA)

    # ── Stack camera strip + BEV ──────────────────────────────────────────────
    return np.vstack([cam_strip, bev])
