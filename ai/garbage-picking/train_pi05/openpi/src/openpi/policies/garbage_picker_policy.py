import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms


def make_garbage_picker_example() -> dict:
    """Creates a random input example for the garbage picker policy."""
    return {
        "state": np.ones((6,)),  # 5 joints + 1 gripper
        "images": {
            "front": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "top": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "pick up the garbage",
    }


@dataclasses.dataclass(frozen=True)
class GarbagePickerInputs(transforms.DataTransformFn):
    """Inputs for the garbage picker policy (SO-100 follower arm).

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [6] (5 joints + 1 gripper)
    - actions: [action_horizon, 6]
    """

    # The expected camera names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("front", "top", "wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that front camera always exists as base
        base_image = in_images["front"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images
        extra_image_names = {
            "top_0_rgb": "top",
            "wrist_0_rgb": "wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GarbagePickerOutputs(transforms.DataTransformFn):
    """Outputs for the garbage picker policy."""

    def __call__(self, data: dict) -> dict:
        # Return all 6 dimensions (5 joints + 1 gripper)
        actions = np.asarray(data["actions"][:, :6])
        return {"actions": actions}
