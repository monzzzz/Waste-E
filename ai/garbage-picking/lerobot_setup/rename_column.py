from datasets import load_dataset
from huggingface_hub import login

# Login first
login()  # This will prompt for your token

repo_id = "Monzzz/garbage-picker-v1-10"  # Use the combined dataset

# Load dataset
print(f"Loading dataset {repo_id}...")
dataset = load_dataset(repo_id)

# Rename column
dataset = dataset.rename_column("actions", "action")

# Push back to Hub (overwrite)
print(f"Pushing updated dataset to {repo_id}...")
dataset.push_to_hub(repo_id, private=True)

print("Done! Dataset updated successfully.")