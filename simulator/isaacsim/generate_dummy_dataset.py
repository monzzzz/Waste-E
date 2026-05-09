"""
Generates a small synthetic WPedestrian SIT-format CSV dataset
so you can test the simulation before the real dataset is available.

Run with any Python 3.x:
    python generate_dummy_dataset.py --out data/wpedestrian_sit --peds 30 --frames 500
"""

import argparse
import csv
import math
import os
import random


def generate_sequence(
    out_dir: str,
    seq_id: int,
    num_peds: int,
    num_frames: int,
    area_size: float = 20.0,
):
    rows = []
    peds = []
    for p in range(num_peds):
        x0 = random.uniform(-area_size / 2, area_size / 2)
        y0 = random.uniform(-area_size / 2, area_size / 2)
        speed = random.uniform(0.8, 1.5)          # m/s (walking speed)
        heading = random.uniform(0, 2 * math.pi)
        # Occasionally change direction
        turn_rate = random.uniform(-0.3, 0.3)
        peds.append([x0, y0, speed, heading, turn_rate])

    for frame in range(num_frames):
        for ped_id, (x, y, speed, heading, turn_rate) in enumerate(peds):
            dt = 1.0 / 30.0  # 30 fps
            heading += turn_rate * dt
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)
            x += vx * dt
            y += vy * dt

            # Bounce inside area
            if abs(x) > area_size / 2:
                vx = -vx
                heading = math.atan2(vy, vx)
                x = max(-area_size / 2, min(area_size / 2, x))
            if abs(y) > area_size / 2:
                vy = -vy
                heading = math.atan2(vy, vx)
                y = max(-area_size / 2, min(area_size / 2, y))

            peds[ped_id] = [x, y, speed, heading, turn_rate]
            rows.append(
                {
                    "frame_id": frame,
                    "ped_id": ped_id,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "z": 0.0,
                    "vx": round(vx, 4),
                    "vy": round(vy, 4),
                    "heading": round(heading, 4),
                }
            )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"seq_{seq_id:03d}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {out_path}  ({num_peds} peds × {num_frames} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default="data/wpedestrian_sit")
    parser.add_argument("--seqs",   type=int, default=3,  help="Number of sequences")
    parser.add_argument("--peds",   type=int, default=30, help="Pedestrians per sequence")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Generating {args.seqs} sequence(s) → {args.out}")
    for i in range(args.seqs):
        generate_sequence(args.out, i, args.peds, args.frames)
    print("Done.")


if __name__ == "__main__":
    main()
