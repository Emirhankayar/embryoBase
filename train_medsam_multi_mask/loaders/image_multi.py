import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from datasets import Dataset


class ImageLoader:
    def __init__(self, base_dir="/run/media/capitan/Emu/data_multi_roi/extracted_frames"):
        self.base_dir = Path(base_dir)

    def get_samples(self):
        samples = []

        for id_dir in self.base_dir.iterdir():
            if not id_dir.is_dir():
                continue
            frames = defaultdict(lambda: {"image": None, "masks": []})

            for file in id_dir.glob("*.jpg"):
                filename = file.name

                frame_num = filename.split("_frame")[-1].replace(".jpg", "")

                if id_dir.name in filename:
                    frames[frame_num]["image"] = file
                else:
                    frames[frame_num]["masks"].append(file)

            for frame_num, data in frames.items():
                if data["image"] and data["masks"]:
                    samples.append(
                        {
                            "image": data["image"],
                            "masks": data["masks"],
                            "id": id_dir.name,
                            "frame": frame_num,
                        }
                    )

        return samples

    def split_data(self, test_ratio=0.2, seed=42):
        samples = self.get_samples()
        random.seed(seed)
        random.shuffle(samples)

        split_idx = int(len(samples) * (1 - test_ratio))
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]

        return train_samples, test_samples

    def batches(self, samples, batch_size=1, shuffle=False, return_hf_dataset=False):
        if shuffle:
            samples = samples.copy()
            random.shuffle(samples)

        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i: i + batch_size]
            images = []
            masks_list = []
            metadata = []

            for sample in batch_samples:
                img = np.array(Image.open(sample["image"]).convert("RGB"))
                masks = []
                for mask_path in sample["masks"]:
                    mask = np.array(Image.open(mask_path).convert("L"))
                    if mask.max() > 0:
                        masks.append(mask)

                if masks:
                    images.append(img)
                    masks_list.append(masks)
                    metadata.append(
                        {
                            "id": sample["id"],
                            "frame": sample["frame"],
                            "num_masks": len(masks),
                        }
                    )

            if images:
                if return_hf_dataset:
                    batch_dataset = Dataset.from_dict(
                        {
                            "image": [Image.fromarray(img) for img in images],
                            "masks": [
                                [Image.fromarray(m) for m in masks]
                                for masks in masks_list
                            ],
                            "metadata": metadata,
                        }
                    )
                    yield batch_dataset
                else:
                    yield images, masks_list, metadata

    def get_bounding_box(self, mask, jitter=20):
        y_indices, x_indices = np.where(mask > 0)

        if len(y_indices) == 0:
            return None

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = mask.shape

        x_min = max(0, x_min - np.random.randint(0, jitter))
        x_max = min(W, x_max + np.random.randint(0, jitter))
        y_min = max(0, y_min - np.random.randint(0, jitter))
        y_max = min(H, y_max + np.random.randint(0, jitter))

        return [x_min, y_min, x_max, y_max]


if __name__ == "__main__":
    loader = ImageLoader()

    train_samples, test_samples = loader.split_data(test_ratio=0.2)
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    for images, masks_list, metadata in loader.batches(
        train_samples, batch_size=4, shuffle=True
    ):
        print(f"\nBatch size: {len(images)}")
        for i, (img, masks, meta) in enumerate(zip(images, masks_list, metadata)):
            print(
                f"  Sample {i}: {meta['id']}_frame{meta['frame']} - {meta['num_masks']} masks")
        break
