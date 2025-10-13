"""
    Scan through the h5 files source path
    remove if they are corrupted
    
    Use it with caution, as it will delete files
"""
import os
import h5py
from config import src
deleted = 0
valid = 0

for root, dirs, files in os.walk(src):
    for f in files:
        if f.endswith(".h5"):
            path = os.path.join(root, f)
            try:
                with h5py.File(path, "r"):
                    print(f"Valid: {path}")
                    valid += 1
            except:
                print(f"Deleting: {path}")
                os.remove(path)
                deleted += 1

print(f"Valid: {valid}, Deleted: {deleted}")