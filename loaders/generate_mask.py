import os
import cv2
import h5py
import numpy as np
from loaders.load_h5 import PreLoader

BASE_DIR = "/run/media/capitan/Emu/data_h5"
MASK_DIR = os.path.join(BASE_DIR, "masks")
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGET_SIZE = 256  # target size

for sub in ["blasto", "no_blasto"]:
    os.makedirs(os.path.join(MASK_DIR, sub), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, sub), exist_ok=True)

def resize_to_size(img, target_size=TARGET_SIZE):
    """Resize image to target_size x target_size"""
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def process_sample(sample):
    h5_path = sample["FullImagePath"]
    x1, y1, x2, y2 = map(int, [sample["x1"], sample["y1"], sample["x2"], sample["y2"]])
    label = sample["Label"]
    subdir = "blasto" if label == 1 else "no_blasto"
    fname = sample["Image"]
    
    img_path = os.path.join(DATA_DIR, subdir, f"{fname}.h5")
    mask_path = os.path.join(MASK_DIR, subdir, f"{fname}.h5")
    
    if os.path.exists(img_path) and os.path.exists(mask_path):
        print(f"Skipping {fname}, already exists.")
        return
    
    print(f"Processing {fname}...")
    
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        img = np.array(f[key], dtype=np.uint8)
    
    crop = img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(w, h)//2, 255, -1)
    
    crop_resized = resize_to_size(crop, TARGET_SIZE)
    mask_resized = resize_to_size(mask, TARGET_SIZE)
    
    with h5py.File(img_path, "w") as f:
        f.create_dataset("image", data=crop_resized, compression="gzip", compression_opts=6)
    
    with h5py.File(mask_path, "w") as f:
        f.create_dataset("mask", data=mask_resized, compression="gzip", compression_opts=6)
    
    print(f"Saved {fname} to {img_path} and {mask_path}")

def main():
    preloader = PreLoader()
    valid_rows, _ = preloader.split_data()
    
    for sample in valid_rows:
        process_sample(sample)

if __name__ == "__main__":
    main()
