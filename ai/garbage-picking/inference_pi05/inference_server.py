"""
Pi0.5 Inference Server - runs on RunPod GPU
Exposes REST API for remote inference calls
Loads checkpoint from HuggingFace Hub
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

from flask import Flask, request, jsonify
import numpy as np
import base64
import io
from PIL import Image
from huggingface_hub import snapshot_download

from openpi.training import config as train_config
from openpi.policies import policy_config

app = Flask(__name__)

# Load model at startup
print("Loading Pi0.5 model from HuggingFace...")
cfg = train_config.get_config("pi05_garbage_picker")

# Download checkpoint from HuggingFace
print("Downloading checkpoint from HuggingFace Hub...")
checkpoint_dir = snapshot_download(
    repo_id="Monzzz/pi05-garbage-picker-v1",
    repo_type="model",
    cache_dir="/workspace/hf_cache",
    token=os.environ.get("HF_TOKEN")  # Set this if repo is private
)
print(f"Checkpoint downloaded to: {checkpoint_dir}")

# Load policy
policy = policy_config.create_trained_policy(cfg, checkpoint_dir)
print("✅ Model loaded!")

def decode_image(base64_str):
    """Decode base64 image to numpy array"""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON:
    {
        "prompt": "task description",
        "state": [x, y, z, rx, ry, rz],
        "images": {
            "base_0_rgb": "base64_encoded_image",
            "left_wrist_0_rgb": "base64_encoded_image",
            "right_wrist_0_rgb": "base64_encoded_image"
        }
    }
    """
    try:
        data = request.json
        
        # Build model input
        pi_input = {
            "prompt": data["prompt"],
            "observation/state": np.array(data["state"], dtype=np.float32),
        }
        
        # Decode images
        for cam_name, base64_img in data["images"].items():
            pi_input[f"observation/images/{cam_name}"] = decode_image(base64_img)
        
        # Run inference
        output = policy.infer(pi_input)
        actions = output["actions"].tolist()
        
        return jsonify({
            "success": True,
            "actions": actions,
            "num_actions": len(actions)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "pi05_garbage_picker"})

if __name__ == '__main__':
    # Run on all interfaces so it's accessible from outside RunPod
    app.run(host='0.0.0.0', port=8000, debug=False)