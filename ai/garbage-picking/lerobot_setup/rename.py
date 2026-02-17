from huggingface_hub import login
import subprocess
import os
import shutil

# Login
login()

repo_url = "https://huggingface.co/datasets/Monzzz/garbage-picker-v2-1"
local_dir = "/tmp/garbage-picker-v2-1"

# Step 0: Clean up if directory exists
if os.path.exists(local_dir):
    print("Step 0: Cleaning up existing directory...")
    shutil.rmtree(local_dir)

# Step 1: Clone the repository
print("Step 1: Cloning repository...")
subprocess.run([
    "git", "clone", 
    repo_url,
    local_dir
], check=True)

# Step 2: Install git-lfs if needed
print("Step 2: Setting up git-lfs...")
subprocess.run(["git", "-C", local_dir, "lfs", "install"], check=True)
subprocess.run(["git", "-C", local_dir, "lfs", "pull"], check=True)

# Step 3: Rename folders locally
print("Step 3: Swapping folders locally...")
videos_dir = os.path.join(local_dir, "videos")

subprocess.run([
    "mv", 
    os.path.join(videos_dir, "observation.images.wrist"),
    os.path.join(videos_dir, "observation.images.temp")
], check=True)

subprocess.run([
    "mv",
    os.path.join(videos_dir, "observation.images.top"), 
    os.path.join(videos_dir, "observation.images.wrist")
], check=True)

subprocess.run([
    "mv",
    os.path.join(videos_dir, "observation.images.temp"),
    os.path.join(videos_dir, "observation.images.top")
], check=True)

# Step 4: Commit and push changes
print("Step 4: Committing changes...")
subprocess.run(["git", "-C", local_dir, "add", "."], check=True)
subprocess.run([
    "git", "-C", local_dir, 
    "commit", "-m", "Swap top and wrist camera folders"
], check=True)

print("Step 5: Pushing to HuggingFace...")
subprocess.run(["git", "-C", local_dir, "push"], check=True)

# Step 6: Cleanup
print("Step 6: Cleaning up...")
shutil.rmtree(local_dir)

print("✓ Done! Top and wrist cameras swapped successfully.")