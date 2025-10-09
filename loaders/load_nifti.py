import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
import numpy as np
from skimage.transform import resize
from scan_dir import ListExperiments


class NiftiTiffLoader:
    def __init__(self, nifti_path=None, tiff_path=None):
        self.nifti_path = nifti_path
        self.tiff_path = tiff_path
        self.mask_data = None
        self.video_data = None
        self.min_frames = 0
        self.current_frame = 0
        self.fig = None
        self.im_video = None
        self.im_mask = None

    def load_nifti(self, path=None):
        if path:
            self.nifti_path = path
        if not self.nifti_path:
            raise ValueError("NIfTI path not provided.")
        nifti_img = nib.load(self.nifti_path)
        self.mask_data = nifti_img.get_fdata()
        print(f"Loaded NIfTI: {self.mask_data.shape}")
        return self.mask_data

    def load_tiff(self, path=None):
        if path:
            self.tiff_path = path
        if not self.tiff_path:
            raise ValueError("TIFF path not provided.")
        tiff = Image.open(self.tiff_path)
        frames = [
            np.array(frame.convert("L")) for frame in ImageSequence.Iterator(tiff)
        ]
        self.video_data = np.stack(frames)
        print(f"Loaded TIFF: {self.video_data.shape}")
        return self.video_data

    def align_data(self):
        if self.mask_data is None or self.video_data is None:
            raise ValueError("Load both NIfTI and TIFF before alignment.")

        if (
            self.mask_data.shape[0] == self.video_data.shape[2]
            and self.mask_data.shape[1] == self.video_data.shape[1]
        ):
            print("Transposing mask axes to match video orientation...")
            self.mask_data = np.transpose(self.mask_data, (1, 0, 2))

        # Align each frame of nifti and tiff
        mask_frames = self.mask_data.shape[2]
        video_frames = self.video_data.shape[0]
        self.min_frames = min(mask_frames, video_frames)

        if mask_frames != video_frames:
            print(f"Frame mismatch: mask={mask_frames}, video={video_frames}. Cropping to {self.min_frames}.")
            self.mask_data = self.mask_data[:, :, : self.min_frames]
            self.video_data = self.video_data[: self.min_frames]

        if self.mask_data.shape[:2] != self.video_data.shape[1:3]:
            print("Resizing mask to match video dimensions...")
            resized = np.zeros_like(self.video_data, dtype=float)
            for i in range(self.video_data.shape[0]):
                resized[i] = resize(
                    self.mask_data[:, :, i],
                    self.video_data.shape[1:3],
                    preserve_range=True,
                )
            self.mask_data = resized

        print(f"Final aligned data -> video: {self.video_data.shape}, mask: {self.mask_data.shape}")

    # === Visualization ===
    def _update_frame(self, frame_idx):
        self.im_video.set_data(self.video_data[frame_idx])
        self.im_mask.set_data(self.mask_data[:, :, frame_idx])
        self.fig.suptitle(f"Frame {frame_idx+1}/{self.min_frames}", fontsize=12)
        self.fig.canvas.draw_idle()

    def _on_key_press(self, event):
        if event.key == "right":
            self.current_frame = (self.current_frame + 1) % self.min_frames
        elif event.key == "left":
            self.current_frame = (self.current_frame - 1) % self.min_frames
        else:
            return
        self._update_frame(self.current_frame)

    def visualize(self, alpha=0.4, start_frame=0):
        if self.video_data is None or self.mask_data is None:
            raise ValueError("Data not loaded/aligned.")
        self.current_frame = start_frame

        self.fig, ax = plt.subplots(figsize=(6, 6))
        self.im_video = ax.imshow(self.video_data[self.current_frame], cmap="gray")
        self.im_mask = ax.imshow(
            self.mask_data[:, :, self.current_frame], cmap="jet", alpha=alpha
        )
        ax.axis("off")
        self.fig.suptitle(
            f"Frame {self.current_frame+1}/{self.min_frames}", fontsize=12
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        plt.show()

    def get_frame(self, idx):
        if self.video_data is None or self.mask_data is None:
            raise ValueError("Data not loaded/aligned.")
        return self.video_data[idx], self.mask_data[:, :, idx]

    def __len__(self):
        return self.min_frames if self.min_frames > 0 else 0


if __name__ == "__main__":
    root = "/run/media/capitan/Emu/BlastoData/Masks"
    finder = ListExperiments(root)
    experiments = finder.list_files()
    if experiments:
        exp = experiments[0]
        loader = NiftiTiffLoader(tiff_path=exp.tiff, nifti_path=exp.niis[5])
        loader.load_nifti()
        loader.load_tiff()
        loader.align_data()
        loader.visualize(alpha=0.3)
