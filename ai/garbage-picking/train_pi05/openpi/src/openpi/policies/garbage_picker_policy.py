
# Based on https://github.com/Physical-Intelligence/openpi/src/openpi/policies/libero_policy.py

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GarbagePickerInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """

    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        state = np.expand_dims(state, axis=0)
        ## I have 3 cameras and it is not perfectly fit the type of cameras Pi0 uses 
        ## So I pass the front image instead of the right wrist image - it still works
        base_image = _parse_image(data["images"]["top"])
        wrist_image = _parse_image(data["images"]["wrist"])
        front_image = _parse_image(data["images"]["front"])
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": front_image,
            },
            "image_mask": {
                "base_0_rgb": np.array([True]),
                "left_wrist_0_rgb": np.array([True]),
                "right_wrist_0_rgb": np.array([True]),
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "action" in data:
            # We are padding to the model action dim.
            actions = transforms.pad_to_dim(data["action"], self.action_dim)
            inputs["action"] = actions

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GarbagePickerOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 6 actions -- since we padded actions above to fit the model action
        return {"actions": np.asarray(data["actions"][:, :6])}
