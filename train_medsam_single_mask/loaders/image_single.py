import os
import random
import numpy as np
from pathlib import Path
from PIL import Image


class ImageLoader:
    def __init__(self, base_dir="/run/media/capitan/Emu/data_single_roi"):
        self.data_dir = Path(base_dir) / "data"

    def create_mask(self, image_shape):
        h, w = image_shape[:2]
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 2 - 10

        y, x = np.ogrid[:h, :w]
        mask = ((x - center_x)**2 + (y - center_y) **
                2 <= radius**2).astype(np.uint8) * 255

        return mask

    def get_files(self):
        files = []

        # Blasto (label=1)
        blasto_dir = self.data_dir / "blasto"
        if blasto_dir.exists():
            files += [(blasto_dir / f, 1)
                      for f in os.listdir(blasto_dir) if f.endswith(".jpg")]

        # No blasto (label=0)
        no_blasto_dir = self.data_dir / "no_blasto"
        if no_blasto_dir.exists():
            files += [(no_blasto_dir / f, 0)
                      for f in os.listdir(no_blasto_dir) if f.endswith(".jpg")]

        return files

    def split_data(self, test_ratio=0.2, seed=42):
        files = self.get_files()
        random.seed(seed)
        random.shuffle(files)

        split_idx = int(len(files) * (1 - test_ratio))
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        return train_files, test_files

    def batches(self, files, batch_size=8, shuffle=False, return_hf_dataset=False):
        if shuffle:
            files = files.copy()
            random.shuffle(files)

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]

            images = []
            masks = []
            labels = []

            for path, label in batch_files:
                img = np.array(Image.open(path).convert("RGB"))
                mask = self.create_mask(img.shape)

                if mask.max() != 0:
                    images.append(img)
                    masks.append(mask)
                    labels.append(label)

            if images:
                if return_hf_dataset:
                    from datasets import Dataset
                    batch_dataset = Dataset.from_dict({
                        "image": [Image.fromarray(img) for img in images],
                        "label": [Image.fromarray(mask) for mask in masks],
                        "class": labels
                    })
                    yield batch_dataset
                else:
                    yield np.array(images), np.array(masks), np.array(labels)

    def get_bounding_box(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        return [x_min, y_min, x_max, y_max]
