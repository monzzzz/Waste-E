#!/usr/bin/env python3
"""
Merge multiple garbage-picker datasets into one combined dataset.

Usage:
    python merge_datasets.py
"""

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi, login

def merge_garbage_picker_datasets():
    """Merge all garbage-picker datasets v1-4 through v1-10."""
    
    # First, login to HuggingFace
    print("Please login to HuggingFace Hub...")
    print("Get your token from: https://huggingface.co/settings/tokens")
    login()  # This will prompt for your token
    
    dataset_versions = [4, 5, 6, 7, 8, 9, 10]
    
    print("Loading datasets...")
    datasets = []
    for version in dataset_versions:
        repo_id = f"Monzzz/garbage-picker-v1-{version}"
        print(f"  Loading {repo_id}...")
        try:
            ds = load_dataset(repo_id, split="train")
            datasets.append(ds)
            print(f"    ✓ Loaded {len(ds)} samples")
        except Exception as e:
            print(f"    ✗ Error loading {repo_id}: {e}")
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return
    
    print(f"\nConcatenating {len(datasets)} datasets...")
    combined_dataset = concatenate_datasets(datasets)
    print(f"✓ Combined dataset has {len(combined_dataset)} total samples")
    
    # Push to HuggingFace Hub
    output_repo = "Monzzz/garbage-picker-v1-combined"
    print(f"\nPushing to {output_repo}...")
    
    combined_dataset.push_to_hub(
        output_repo,
        private=False,  # Set to True if you want it private
    )
    
    print(f"✓ Successfully pushed to {output_repo}")
    
    # Create version tag required by LeRobot
    print(f"\nCreating version tag for LeRobot compatibility...")
    hub_api = HfApi()
    try:
        # Tag with v3.0 (matching your dataset's codebase_version)
        hub_api.create_tag(
            output_repo,
            tag="v3.0",
            repo_type="dataset",
            tag_message="LeRobot codebase version v3.0"
        )
        print("✓ Created version tag 'v3.0'")
    except Exception as e:
        print(f"⚠ Warning: Could not create tag (may already exist): {e}")
    
    print(f"\n✅ All done! Your dataset is ready at: {output_repo}")
    print(f"\nNow update your config to use:")
    print(f'    repo_id="{output_repo}"')

if __name__ == "__main__":
    merge_garbage_picker_datasets()
