"""Fix corrupted dataset on Mac - work with original LeRobot data"""
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import shutil

# Path to your original LeRobot dataset
dataset_path = Path("lerobot_setup/lerobot/outputs/train/garbage-picker-v1-combined")
data_dir = dataset_path / "data"

print(f"Looking for parquet files in: {data_dir}")

# Find all parquet files
parquet_files = list(data_dir.glob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files:")
for f in parquet_files:
    print(f"  - {f.name}")

# Process each parquet file
for parquet_file in parquet_files:
    print(f"\n{'='*60}")
    print(f"Processing: {parquet_file.name}")
    print(f"{'='*60}")
    
    # Read the parquet file
    table = pq.read_table(str(parquet_file))
    print(f"Original rows: {len(table)}")
    
    # Find corrupted rows
    action_column = table.column('action')
    corrupted_indices = []
    
    for i in range(len(action_column)):
        try:
            action = action_column[i].as_py()
            if action is None or len(action) == 0:
                corrupted_indices.append(i)
                print(f"  Row {i}: empty action")
            else:
                # Check each timestep in the action sequence
                for j, a in enumerate(action):
                    if a is None or len(a) != 6:
                        corrupted_indices.append(i)
                        print(f"  Row {i}, timestep {j}: invalid action (len={len(a) if a else 0})")
                        break
        except Exception as e:
            corrupted_indices.append(i)
            print(f"  Row {i}: error - {e}")
    
    # Filter if corrupted rows found
    if corrupted_indices:
        print(f"\nFound {len(corrupted_indices)} corrupted rows")
        valid_indices = [i for i in range(len(table)) if i not in corrupted_indices]
        filtered_table = table.take(valid_indices)
        
        # Backup original
        backup_file = parquet_file.with_suffix('.parquet.backup')
        shutil.copy(parquet_file, backup_file)
        print(f"Backed up to: {backup_file.name}")
        
        # Write cleaned version
        pq.write_table(filtered_table, str(parquet_file))
        print(f"✅ Fixed {parquet_file.name}: {len(table)} → {len(filtered_table)} rows")
    else:
        print(f"✅ No corrupted rows in {parquet_file.name}")

print("\n" + "="*60)
print("Dataset cleaning complete!")
print("="*60)

# Now push to HuggingFace
print("\nPushing cleaned dataset to HuggingFace...")
from huggingface_hub import HfApi
import os

api = HfApi()

# Upload all parquet files
for parquet_file in data_dir.glob("*.parquet"):
    if parquet_file.suffix == '.parquet' and not parquet_file.name.endswith('.backup'):
        print(f"Uploading {parquet_file.name}...")
        api.upload_file(
            path_or_fileobj=str(parquet_file),
            path_in_repo=f"data/{parquet_file.name}",
            repo_id="Monzzz/garbage-picker-v1-combined",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN")
        )

print("\n✅ All files uploaded to HuggingFace!")
print("\nNow on RunPod, run:")
print("  rm -rf /workspace/hf_cache/hub/datasets--Monzzz--garbage-picker-v1-combined")
print("  ./start_training.sh")