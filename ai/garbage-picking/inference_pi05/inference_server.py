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
            "front": "base64_encoded_image",
            "top": "base64_encoded_image",
            "wrist": "base64_encoded_image"
        }
    }
    """
    try:
        # Check if request has JSON data
        if not request.is_json:
            raise ValueError("Request must be JSON")
        
        data = request.get_json()
        
        if data is None:
            raise ValueError("Failed to parse JSON from request body")
        
        # Debug: Print received data structure
        print("Received data:", data)
        print(f"=== Received request ===")
        print(f"Keys: {list(data.keys())}")
        print(f"State present: {'state' in data}")
        print(f"State value: {data.get('state', 'MISSING')}")
        print(f"State type: {type(data.get('state'))}")
        if 'state' in data:
            print(f"State length: {len(data.get('state', []))}")
        print(f"Images keys: {list(data.get('images', {}).keys())}")
        print(f"Prompt: {data.get('prompt', 'MISSING')[:50]}...")
        
        # Extract data with validation
        if 'state' not in data:
            raise KeyError("Missing 'state' in request payload")
        if 'images' not in data:
            raise KeyError("Missing 'images' in request payload")
        if 'prompt' not in data:
            raise KeyError("Missing 'prompt' in request payload")
        
        # Build model input
        pi_input = {
            "prompt": data["prompt"],
            "observation/state": np.array(data["state"], dtype=np.float32),
        }
        
        # Decode images - map client camera names to model expected names
        # Client sends: front, top, wrist
        # Model expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
        camera_mapping = {
            "front": "base_0_rgb",
            "top": "left_wrist_0_rgb", 
            "wrist": "right_wrist_0_rgb"
        }
        
        for client_cam, model_cam in camera_mapping.items():
            if client_cam in data["images"]:
                pi_input[f"observation/images/{model_cam}"] = decode_image(data["images"][client_cam])
                print(f"Decoded {client_cam} -> {model_cam}")
        
        print(f"Running inference with state shape: {pi_input['observation/state'].shape}")
        
        # Run inference
        output = policy.infer(pi_input)
        actions = output["actions"].tolist()
        
        print(f"Generated {len(actions)} actions")
        
        return jsonify({
            "success": True,
            "actions": actions,
            "num_actions": len(actions)
        })
        
    except KeyError as e:
        print(f"❌ KeyError: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
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