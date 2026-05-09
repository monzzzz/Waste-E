"""
WPedestrian SIT Dataset loader for Isaac Sim.

Expected dataset layout (one CSV per sequence):
  data/wpedestrian_sit/
    seq_001.csv
    seq_002.csv
    ...

Each CSV has columns:
  frame_id, ped_id, x, y, z, vx, vy, heading
Units: metres, metres/second, radians.

If your dataset uses a different schema, adjust COLUMN_MAP below.
"""

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

COLUMN_MAP = {
    "frame":   ["frame_id", "frame", "t", "timestep"],
    "ped_id":  ["ped_id", "agent_id", "id", "track_id"],
    "x":       ["x", "pos_x", "X"],
    "y":       ["y", "pos_y", "Y"],
    "z":       ["z", "pos_z", "Z"],
    "vx":      ["vx", "vel_x", "dx"],
    "vy":      ["vy", "vel_y", "dy"],
    "heading": ["heading", "yaw", "theta"],
}


@dataclass
class PedestrianFrame:
    frame: int
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0


@dataclass
class PedestrianTrajectory:
    ped_id: str
    frames: List[PedestrianFrame] = field(default_factory=list)

    def position_at(self, frame: int) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, z) at the given frame, interpolating if needed."""
        if not self.frames:
            return None
        frames_arr = np.array([f.frame for f in self.frames])
        idx = np.searchsorted(frames_arr, frame)
        if idx == 0:
            f = self.frames[0]
            return f.x, f.y, f.z
        if idx >= len(self.frames):
            f = self.frames[-1]
            return f.x, f.y, f.z
        f0, f1 = self.frames[idx - 1], self.frames[idx]
        t = (frame - f0.frame) / max(f1.frame - f0.frame, 1)
        return (
            f0.x + t * (f1.x - f0.x),
            f0.y + t * (f1.y - f0.y),
            f0.z + t * (f1.z - f0.z),
        )

    def heading_at(self, frame: int) -> float:
        if not self.frames:
            return 0.0
        frames_arr = np.array([f.frame for f in self.frames])
        idx = np.searchsorted(frames_arr, frame)
        if idx == 0:
            return self.frames[0].heading
        if idx >= len(self.frames):
            return self.frames[-1].heading
        f0, f1 = self.frames[idx - 1], self.frames[idx]
        t = (frame - f0.frame) / max(f1.frame - f0.frame, 1)
        return f0.heading + t * (f1.heading - f0.heading)

    @property
    def start_frame(self) -> int:
        return self.frames[0].frame if self.frames else 0

    @property
    def end_frame(self) -> int:
        return self.frames[-1].frame if self.frames else 0


class WPedestrianSITDataset:
    """Loads all sequences from a WPedestrian SIT dataset directory."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.sequences: List[Dict[str, PedestrianTrajectory]] = []
        self._load()

    # ------------------------------------------------------------------
    def _resolve_col(self, header: List[str], candidates: List[str]) -> Optional[int]:
        for name in candidates:
            if name in header:
                return header.index(name)
        return None

    def _load_csv(self, filepath: str) -> Dict[str, PedestrianTrajectory]:
        trajectories: Dict[str, PedestrianTrajectory] = {}
        with open(filepath, newline="") as f:
            reader = csv.reader(f)
            raw_header = next(reader)
            header = [h.strip() for h in raw_header]

            col = {k: self._resolve_col(header, v) for k, v in COLUMN_MAP.items()}

            # z, vx, vy, heading are optional
            for row in reader:
                if not row:
                    continue
                try:
                    frame  = int(float(row[col["frame"]]))   if col["frame"]   is not None else 0
                    ped_id = str(row[col["ped_id"]]).strip() if col["ped_id"]  is not None else "0"
                    x      = float(row[col["x"]])            if col["x"]       is not None else 0.0
                    y      = float(row[col["y"]])            if col["y"]       is not None else 0.0
                    z      = float(row[col["z"]])            if col["z"]       is not None else 0.0
                    vx     = float(row[col["vx"]])           if col["vx"]      is not None else 0.0
                    vy     = float(row[col["vy"]])           if col["vy"]      is not None else 0.0
                    hdg    = float(row[col["heading"]])      if col["heading"] is not None else 0.0
                except (ValueError, IndexError):
                    continue

                if ped_id not in trajectories:
                    trajectories[ped_id] = PedestrianTrajectory(ped_id=ped_id)
                trajectories[ped_id].frames.append(
                    PedestrianFrame(frame=frame, x=x, y=y, z=z, vx=vx, vy=vy, heading=hdg)
                )

        # sort each trajectory by frame number
        for traj in trajectories.values():
            traj.frames.sort(key=lambda f: f.frame)

        return trajectories

    def _load(self):
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        csv_files = sorted(
            f for f in os.listdir(self.dataset_path) if f.endswith(".csv")
        )
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {self.dataset_path}")

        for fname in csv_files:
            fpath = os.path.join(self.dataset_path, fname)
            seq = self._load_csv(fpath)
            self.sequences.append(seq)

        print(f"[WPedestrianSIT] Loaded {len(self.sequences)} sequence(s) "
              f"from {self.dataset_path}")

    # ------------------------------------------------------------------
    def get_sequence(self, idx: int = 0) -> Dict[str, PedestrianTrajectory]:
        return self.sequences[idx % len(self.sequences)]

    def all_trajectories(self) -> List[PedestrianTrajectory]:
        """Return all trajectories across all sequences (flattened)."""
        out = []
        for seq in self.sequences:
            out.extend(seq.values())
        return out

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)
