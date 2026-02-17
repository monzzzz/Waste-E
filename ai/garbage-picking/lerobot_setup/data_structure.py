from huggingface_hub import HfApi, login

# Login
login()

api = HfApi()

# Swap top and wrist folders
print("Step 1: Renaming wrist -> temp...")
api.move_repo_file(
    repo_id="Monzzz/garbage-picker-v2-1",
    repo_type="dataset",
    path_in_repo="videos/observation.images.wrist",
    new_path_in_repo="videos/observation.images.temp",
)

print("Step 2: Renaming top -> wrist...")
api.move_repo_file(
    repo_id="Monzzz/garbage-picker-v2-1",
    repo_type="dataset",
    path_in_repo="videos/observation.images.top",
    new_path_in_repo="videos/observation.images.wrist",
)

print("Step 3: Renaming temp -> top...")
api.move_repo_file(
    repo_id="Monzzz/garbage-picker-v2-1",
    repo_type="dataset",
    path_in_repo="videos/observation.images.temp",
    new_path_in_repo="videos/observation.images.top",
)

print("✓ Done! top and wrist cameras swapped.")