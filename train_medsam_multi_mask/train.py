import os
import torch
import numpy as np
from transformers import SamModel, SamProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from loaders.image_multi import ImageLoader
from torch.nn.functional import interpolate


class MedSAMDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def get_bounding_box(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return None
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        masks = []
        boxes = []
        for mask_path in sample["masks"]:
            mask = np.array(Image.open(mask_path).convert("L"))
            if mask.max() > 0:
                masks.append(mask)
                bbox = self.get_bounding_box(mask)
                if bbox:
                    boxes.append(bbox)

        if not boxes:
            return None

        inputs = self.processor(
            image,
            input_boxes=[[boxes]],
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_masks"] = torch.tensor(
            np.stack(masks), dtype=torch.float32)

        return inputs


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return batch[0]


def train_medsam(
    base_dir="/run/media/capitan/Emu/data_multi_roi/extracted_frames",
    output_dir="./medsam_finetuned",
    epochs=10,
    lr=1e-5,
    accumulation_steps=4,
    max_samples=500
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # GPU memory optimization
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print("Loading MedSAM model...")
    model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
    processor = SamProcessor.from_pretrained(
        "flaviagiammarino/medsam-vit-base")
    model.to(device)

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    print("Loading dataset...")
    loader = ImageLoader(base_dir)
    train_samples, test_samples = loader.split_data(test_ratio=0.2)

    train_samples = train_samples[:max_samples]
    test_samples = test_samples[:max_samples//5]

    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    train_dataset = MedSAMDataset(train_samples, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].unsqueeze(0).to(device)
            input_boxes = batch["input_boxes"].unsqueeze(0).to(device)
            original_sizes = batch["original_sizes"].unsqueeze(0).to(device)
            reshaped_input_sizes = batch["reshaped_input_sizes"].unsqueeze(
                0).to(device)
            ground_truth_masks = batch["ground_truth_masks"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                original_sizes=original_sizes,
                reshaped_input_sizes=reshaped_input_sizes,
                multimask_output=False
            )

            pred_masks = outputs.pred_masks.squeeze()
            if pred_masks.dim() == 2:
                pred_masks = pred_masks.unsqueeze(0)

            pred_masks = torch.sigmoid(pred_masks)

            gt_masks_resized = interpolate(
                ground_truth_masks.unsqueeze(1).float(),
                size=pred_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            gt_masks_resized = gt_masks_resized / 255.0

            loss = torch.nn.functional.binary_cross_entropy(
                pred_masks,
                gt_masks_resized
            )
            loss = loss / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                if device == "cuda":
                    torch.cuda.empty_cache()

            total_loss += loss.item() * accumulation_steps
            num_batches += 1
            pbar.set_postfix(
                {'loss': f'{loss.item() * accumulation_steps:.4f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Training complete!")


if __name__ == "__main__":
    train_medsam(
        base_dir="/run/media/capitan/Emu/data_multi_roi/extracted_frames",
        output_dir="./medsam_finetuned",
        epochs=10,
        lr=1e-5,
        accumulation_steps=4,
        max_samples=1000
    )
