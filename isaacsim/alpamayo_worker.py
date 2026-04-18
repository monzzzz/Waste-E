"""
Alpamayo 1.5 inference worker — runs under Python 3.12.

Protocol (newline-delimited JSON on stdin/stdout):
  init   → {"model_path": "...", "device": "cuda"}
  ready  ← {"status": "ready"}

  per step → {"image_paths": ["path"|null, ...], "camera_indices": [0,1,2,...],
               "ego_history_xyz": [[[...]]],  "ego_history_rot": [[[[...]]]]}
  result   ← {"status": "ok", "waypoints": [[x,y], ...]}
           | {"status": "error", "msg": "..."}
"""

from __future__ import annotations

import json
import sys
import os
import numpy as np
import torch
from PIL import Image


def _load_image(path: str | None, target_h: int = 384, target_w: int = 640) -> torch.Tensor:
    """Return (3, H, W) uint8 tensor; black if path is None."""
    if path is None or not os.path.exists(path):
        return torch.zeros(3, target_h, target_w, dtype=torch.uint8)
    img = Image.open(path).convert("RGB").resize((target_w, target_h))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)


def _build_frames(image_paths: list[str | None], num_frames_per_cam: int = 4) -> torch.Tensor:
    """Return (N_cameras * num_frames_per_cam, 3, H, W)."""
    parts = []
    for path in image_paths:
        frame = _load_image(path)          # (3, H, W)
        parts.append(frame.unsqueeze(0).expand(num_frames_per_cam, -1, -1, -1))
    return torch.cat(parts, dim=0)         # (N*F, 3, H, W)


def main():
    # ── initialisation ──────────────────────────────────────────────────
    init = json.loads(sys.stdin.readline())
    model_path = init["model_path"]
    device = init.get("device", "cuda")

    print(json.dumps({"status": "loading"}), flush=True)

    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper

    model = Alpamayo1_5.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    print(json.dumps({"status": "ready"}), flush=True)

    # ── inference loop ──────────────────────────────────────────────────
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)

            image_paths = data["image_paths"]
            camera_indices = torch.tensor(data["camera_indices"], dtype=torch.int64)
            nav_text = data.get("nav_text", None)  # e.g. "Turn left", "Go straight"
            num_frames = 1

            frames = _build_frames(image_paths, num_frames).to(device)

            messages = helper.create_message(
                frames=frames,
                camera_indices=camera_indices,
                num_frames_per_camera=num_frames,
                nav_text=nav_text,
            )

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )

            ego_xyz = torch.tensor(
                data["ego_history_xyz"], dtype=torch.float32
            )  # (1, 1, T, 3)
            ego_rot = torch.tensor(
                data["ego_history_rot"], dtype=torch.float32
            )  # (1, 1, T, 3, 3)

            model_inputs = helper.to_device(
                {
                    "tokenized_data": inputs,
                    "ego_history_xyz": ego_xyz,
                    "ego_history_rot": ego_rot,
                },
                device,
            )

            with torch.autocast(device, dtype=torch.bfloat16):
                pred_xyz, _pred_rot, _extra = (
                    model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=0.98,
                        temperature=0.6,
                        num_traj_samples=1,
                        max_generation_length=64,
                        return_extra=True,
                    )
                )

            # pred_xyz: (batch, n_groups, n_samples, traj_len, 3)
            # Take batch=0, group=0, sample=0, all timesteps, xy
            waypoints_xy = pred_xyz[0, 0, 0, :, :2].cpu().float().numpy().tolist()

            print(json.dumps({"status": "ok", "waypoints": waypoints_xy}), flush=True)

        except Exception as exc:
            import traceback
            print(
                json.dumps({"status": "error", "msg": traceback.format_exc()}),
                flush=True,
            )


if __name__ == "__main__":
    main()
