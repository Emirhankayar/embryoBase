from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
import os
import numpy as np
import polars as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import Adam
from monai.losses import DiceCELoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import SamProcessor, SamModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD DATA BLOCK
base_dir = "/run/media/capitan/Emu/blastodata_orig"
csv_path = "/home/capitan/Documents/Code/MedSAM/bbox2.csv"

image_paths = {}
for root, dirs, files in os.walk(base_dir):
    for f in files:
        image_paths[f.lower()] = os.path.join(root, f)

print(f"Found {len(image_paths)} images in {base_dir}")

# Load CSV
df = (
    pl.scan_csv(csv_path)
    .select(["ImagePath", "x1", "y1", "x2", "y2", "Label"])
    .with_columns(pl.col("ImagePath").str.replace(r"^[^/]*/", "").alias("ImagePath"))
    .collect()
)
rows = df.to_dicts()

valid_rows = []
for r in rows:
    name = r["ImagePath"].lower()
    if name in image_paths:
        r["FullImagePath"] = image_paths[name]
        valid_rows.append(r)
    else:
        print(f"Not found on disk: {r['ImagePath']}")

print(f"Matched {len(valid_rows)} of {len(rows)} CSV entries")
print(f"Matched {len(valid_rows)} of {len(rows)} CSV entries")

# -------------------------------
# SUBSAMPLE 10,000 IMAGES WITH BALANCED CLASSES
# -------------------------------

random.seed(42)

# Group rows by label
class_groups = defaultdict(list)
for r in valid_rows:
    class_groups[r["Label"]].append(r)

# Determine how many samples per class
n_samples_total = 10000
n_classes = len(class_groups)
samples_per_class = n_samples_total // n_classes

subsampled_rows = []
for label, rows in class_groups.items():
    n = min(samples_per_class, len(rows))
    subsampled_rows.extend(random.sample(rows, n))

print(f"Subsampled {len(subsampled_rows)} rows with balanced classes")
# -------------------------------

train_rows, test_rows = train_test_split(
    subsampled_rows,
    test_size=0.2,
    random_state=42,
    stratify=[r["Label"] for r in subsampled_rows],
)
print(f"Train samples: {len(train_rows)}, Test samples: {len(test_rows)}")

"""
FULL DATA

train_rows, test_rows = train_test_split(valid_rows, test_size=0.2, random_state=42)
print(f"Train samples: {len(train_rows)}, Test samples: {len(test_rows)}")
"""
# PREPARE DATASET


class MedSAMDataset(Dataset):
    def __init__(self, data, processor=None):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = np.array(Image.open(sample["FullImagePath"]).convert("RGB"))
        bbox = [
            float(sample["x1"]),
            float(sample["y1"]),
            float(sample["x2"]),
            float(sample["y2"]),
        ]

        # Placeholder for mask if available
        ground_truth_mask = np.array(
            sample.get(
                "label_mask", np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            )
        )

        inputs = self.processor(image, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Resize mask to match model output later
        inputs["ground_truth_mask"] = torch.tensor(
            ground_truth_mask, dtype=torch.float32
        )
        inputs["path"] = sample["FullImagePath"]
        return inputs


processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

train_dataset = MedSAMDataset(train_rows, processor)
test_dataset = MedSAMDataset(test_rows, processor)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# LOAD MODEL
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5)
seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

# TRAIN
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_loader):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        ground_truth_masks = batch["ground_truth_mask"].unsqueeze(1).to(device)

        outputs = model(
            pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False
        )
        predicted_masks = outputs.pred_masks.squeeze(1)

        # Resize ground truth mask
        ground_truth_masks_resized = torch.nn.functional.interpolate(
            ground_truth_masks, size=predicted_masks.shape[-2:], mode="nearest"
        )

        loss = seg_loss(predicted_masks, ground_truth_masks_resized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
    print(f"Epoch {epoch} mean loss: {np.mean(epoch_losses):.4f}")


# VISUALIZE
sample = test_dataset[0]
image = np.array(Image.open(sample["path"]).convert("RGB"))
ground_truth_mask = sample["ground_truth_mask"].numpy()

model.eval()
with torch.no_grad():
    inputs = {
        k: v.unsqueeze(0).to(device)
        for k, v in sample.items()
        if k in ["pixel_values", "input_boxes"]
    }
    outputs = model(**inputs, multimask_output=False)
    pred_mask = torch.sigmoid(outputs.pred_masks).squeeze(0).squeeze(0).cpu().numpy()
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

save_path = "/home/capitan/Documents/Code/MedSAM/trained_model"
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print(f"Model and processor saved to {save_path}")
# LOAD THE SAVED MODEL
"""
from transformers import SamProcessor, SamModel

model = SamModel.from_pretrained(save_path).to(device)
processor = SamProcessor.from_pretrained(save_path)

"""


def show_mask(mask, ax, color=[30 / 255, 144 / 255, 255 / 255, 0.6]):
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
show_mask(ground_truth_mask, axes[0])
axes[0].set_title("Ground Truth")
axes[0].axis("off")

axes[1].imshow(image)
show_mask(pred_mask_bin, axes[1])
axes[1].set_title("Predicted Mask")
axes[1].axis("off")
plt.show()
