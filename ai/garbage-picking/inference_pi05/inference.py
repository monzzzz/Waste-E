from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np

config = _config.get_config("pi05_lekiwi")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "prompt": "Pick an object in front of you and place in the bin behind you.",
    "observation/state": np.zeros((9,), dtype=np.float32),  # <-- must match your robot state dim
    "observation/images/top":   np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/images/wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/images/front": np.zeros((480, 640, 3), dtype=np.uint8),
}

action_chunk = policy.infer(example)["actions"]
print(action_chunk)