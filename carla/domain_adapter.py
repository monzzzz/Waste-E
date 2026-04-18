"""
Sim-to-real domain adapter for Waste-E.

Uses SDXL-Turbo img2img to re-texture synthetic CARLA frames to look
photorealistic while preserving geometry and layout.

Model is downloaded once to /workspace/models/sdxl-turbo on first use.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Default model — SDXL-Turbo is fast (1-4 steps) and free
_DEFAULT_MODEL = "stabilityai/sdxl-turbo"
_MODEL_CACHE   = "/workspace/models/sdxl-turbo"

_DEFAULT_PROMPT = (
    "photorealistic urban sidewalk, city street, real photograph, "
    "DSLR camera, sharp focus, natural lighting"
)
_NEGATIVE_PROMPT = (
    "cartoon, render, video game, synthetic, blurry, painting, sketch"
)


class DomainAdapter:
    """
    Wraps SDXL-Turbo img2img.  Call adapt(images) with a list of H×W×3
    uint8 RGB arrays; returns the same list with photorealistic textures.
    """

    def __init__(
        self,
        strength: float = 0.35,
        steps: int = 4,
        prompt: str = _DEFAULT_PROMPT,
        device: str = "cuda",
        model_id: str = _DEFAULT_MODEL,
    ):
        self.strength = strength
        self.steps    = steps
        self.prompt   = prompt
        self.device   = device

        print("[DomainAdapter] Loading SDXL-Turbo (first run downloads ~7 GB)…")
        t0 = time.time()

        from diffusers import AutoPipelineForImage2Image
        import torch

        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=_MODEL_CACHE,
        ).to(device)

        # Disable safety checker to avoid false positives on street scenes
        self._pipe.safety_checker = None

        # Compile for speed on modern GPUs
        try:
            self._pipe.unet = torch.compile(
                self._pipe.unet, mode="reduce-overhead", fullgraph=True
            )
        except Exception:
            pass  # compile is optional

        print(f"[DomainAdapter] Ready in {time.time()-t0:.1f}s  "
              f"strength={strength} steps={steps}")

    # ------------------------------------------------------------------
    def adapt(
        self,
        images: list[Optional[np.ndarray]],
        strength: Optional[float] = None,
    ) -> list[Optional[np.ndarray]]:
        """
        images : list of H×W×3 uint8 RGB arrays (None entries are passed through).
        Returns a new list of the same shape with photorealistic textures.
        """
        from PIL import Image

        s = strength if strength is not None else self.strength

        # Filter out None and keep track of positions
        valid_idx = [i for i, img in enumerate(images) if img is not None]
        if not valid_idx:
            return images

        pil_imgs = [Image.fromarray(images[i]) for i in valid_idx]

        # Batch inference — all cameras in one forward pass
        results = self._pipe(
            prompt          = [self.prompt]         * len(pil_imgs),
            negative_prompt = [_NEGATIVE_PROMPT]    * len(pil_imgs),
            image           = pil_imgs,
            strength        = s,
            num_inference_steps = self.steps,
            guidance_scale  = 0.0,   # SDXL-Turbo uses CFG=0
        ).images

        out = list(images)  # copy
        for pos, result_img in zip(valid_idx, results):
            out[pos] = np.array(result_img)

        return out

    def set_strength(self, strength: float):
        self.strength = float(np.clip(strength, 0.0, 1.0))
