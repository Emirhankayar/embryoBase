import os
import random 
import polars as pl
from config import src, csv

class PreLoader:
    def __init__(self):
        self.src = src
        self.csv = csv
        self.valid_rows = []
        self.not_found = 0

    def get_files_index(self):
        index = {}

        for root, dirs, files in os.walk(self.src, topdown=True):
            for f in files:
                if f.endswith((".h5")):
                    stem = os.path.splitext(f)[0]
                    index[stem] = os.path.join(root, f)

        return index

    def load_csv_data(self):
        df = pl.scan_csv(self.csv).select(["Image", "x1", "y1", "x2", "y2", "Label"])
        df = df.filter(
            pl.col("Image").str.extract(r"_([\d.]+)h$").cast(pl.Float64) <= 31.0
        ).collect()
        rows = df.to_dicts()
        return rows

    def filter_files(self, rows, h5_index):
        for r in rows:
            stem = r["Image"]

            if h5_index.get(stem):
                r["FullImagePath"] = h5_index.get(stem)
                self.valid_rows.append(r)
            else:
                self.not_found += 1

        print(f"Matched: {len(self.valid_rows)}")
        print(f"Not found: {self.not_found}")
        return self.valid_rows, self.not_found

    def split_data(self):
        h5_index = self.get_files_index()
        csv_rows = self.load_csv_data()
        self.valid_rows, self.not_found = self.filter_files(csv_rows, h5_index)
        return self.valid_rows, self.not_found

import os
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


class MaskLoader:
    def __init__(self, base_dir="/run/media/capitan/Emu/data_h5"):
        self.base_dir = Path(base_dir)
        self.mask_dir = self.base_dir / "masks"
        self.data_dir = self.base_dir / "data"
        self.blasto_files = []
        self.no_blasto_files = []

    def scan_files(self):
        # Blasto
        blasto_mask_dir = self.mask_dir / "blasto"
        blasto_data_dir = self.data_dir / "blasto"
        if blasto_mask_dir.exists() and blasto_data_dir.exists():
            mask_files = {f for f in os.listdir(blasto_mask_dir) if f.endswith(".h5")}
            data_files = {f for f in os.listdir(blasto_data_dir) if f.endswith(".h5")}
            self.blasto_files = list(mask_files.intersection(data_files))

        # No Blasto
        no_blasto_mask_dir = self.mask_dir / "no_blasto"
        no_blasto_data_dir = self.data_dir / "no_blasto"
        if no_blasto_mask_dir.exists() and no_blasto_data_dir.exists():
            mask_files = {f for f in os.listdir(no_blasto_mask_dir) if f.endswith(".h5")}
            data_files = {f for f in os.listdir(no_blasto_data_dir) if f.endswith(".h5")}
            self.no_blasto_files = list(mask_files.intersection(data_files))

    def get_all_files(self) -> List[Tuple[str, int]]:
        if not self.blasto_files and not self.no_blasto_files:
            self.scan_files()
        return [(f, 1) for f in self.blasto_files] + [(f, 0) for f in self.no_blasto_files]

    def batch_generator(self, batch_size=8, subset=None):
        if subset is None:
            subset = self.get_all_files()

        batch_images, batch_masks, batch_labels = [], [], []

        for filename, label in subset:
            subdir = "blasto" if label else "no_blasto"
            img_path = self.data_dir / subdir / filename
            mask_path = self.mask_dir / subdir / filename

            with h5py.File(img_path, "r") as f_img, h5py.File(mask_path, "r") as f_mask:
                img = np.array(f_img["image"], dtype=np.uint8)
                mask = np.array(f_mask["mask"], dtype=np.uint8)

            batch_images.append(img)
            batch_masks.append(mask)
            batch_labels.append(label)

            if len(batch_images) == batch_size:
                yield np.stack(batch_images), np.stack(batch_masks), np.array(batch_labels)
                batch_images, batch_masks, batch_labels = [], [], []

        if batch_images:
            yield np.stack(batch_images), np.stack(batch_masks), np.array(batch_labels)


    def get_stats(self) -> Dict[str, int]:
        if not self.blasto_files and not self.no_blasto_files:
            self.scan_files()
        return {
            "total_samples": len(self.blasto_files) + len(self.no_blasto_files),
            "blasto_samples": len(self.blasto_files),
            "no_blasto_samples": len(self.no_blasto_files),
        }
    
    def load_single_sample(self, filename: str, label: int) -> Tuple[np.ndarray, np.ndarray, int]:
        subdir = "blasto" if label else "no_blasto"
        img_path = self.data_dir / subdir / filename
        mask_path = self.mask_dir / subdir / filename

        with h5py.File(img_path, "r") as f_img:
            img = np.array(f_img["image"], dtype=np.uint8)
        with h5py.File(mask_path, "r") as f_mask:
            mask = np.array(f_mask["mask"], dtype=np.uint8)

        return img, mask, label
    def load_random_sample(self) -> Tuple[np.ndarray, np.ndarray, int]:
        files = self.get_all_files()
        filename, label = random.choice(files)
        return self.load_single_sample(filename, label)
