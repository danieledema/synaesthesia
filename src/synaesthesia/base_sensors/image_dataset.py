from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..abstract.dataset_base import DatasetBase
from .multi_file_dataset import MultiFileDataset


class ImageDataset(MultiFileDataset):
    """
    Dataset class for Image data.
    """

    def __init__(
        self,
        folder_path: str | Path,
        extension: str,
        format: str = "RGB",
    ):
        super().__init__(folder_path, extension)

        self.format = format

    def read_data(self, file_path: Path) -> Any:
        image = Image.open(file_path)
        # Try to convert to the requested format (e.g. "RGB", "RGBA", "L").
        # If conversion fails for some reason, fall back to the original image.
        if self.format is not None:
            try:
                image = image.convert(self.format)
            except Exception:
                # keep original mode if conversion fails
                pass

        image_np = np.array(image)

        # Handle grayscale images (H, W) -> (H, W, 1)
        if image_np.ndim == 2:
            image_np = np.expand_dims(image_np, axis=2)

        # Handle images with an alpha channel (4 channels).
        # If user requested "RGB", drop the alpha channel. Otherwise keep channels as-is.
        if image_np.ndim == 3 and image_np.shape[2] == 4:
            if self.format and self.format.upper() == "RGB":
                image_np = image_np[:, :, :3]
            # else: keep the 4 channels

        # Convert channels-last (H, W, C) to channels-first (C, H, W)
        if image_np.ndim == 3:
            image_np = image_np.transpose(2, 0, 1)

        return {"RGB": image_np}

    @property
    def sensor_ids(self) -> list[str]:
        return ["RGB"]


class ImageFromVideoDataset(DatasetBase):
    def __init__(self, video_path: Path, timestamp_path: Path | None = None):
        super().__init__()

        self.video_path = video_path
        self.timestamp_path = timestamp_path
        self.cap = None

        # Probe timestamps without keeping the capture open across processes.
        # This avoids holding a VideoCapture object in the parent process which
        # can cause issues with multiprocessing workers.
        if self.timestamp_path is not None:
            # If explicit timestamps are provided, use them.
            self._timestamps = self.read_timestamps(self.timestamp_path)
        else:
            # Open a temporary capture to read fps and frame count, then release it.
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise ValueError("Video not opened")
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Validate fps
            if fps is None or fps <= 0 or np.isnan(fps):
                cap.release()
                raise ValueError("Invalid FPS in video; cannot compute timestamps")
            self._timestamps = np.arange(0, num_frames / fps, 1 / fps).tolist()
            cap.release()

    def open(self):
        """
        Open a VideoCapture for the video path. This should be used when a
        capture is required in the current process (e.g. in a DataLoader worker).
        It returns a fresh VideoCapture object which the caller is responsible for
        releasing when appropriate.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError("Video not opened")
        return cap

    @property
    def timestamps(self):
        return self._timestamps

    def __len__(self) -> int:
        return len(self.timestamps)

    def read_timestamps(self, timestamp_path: Path) -> list[int]:
        raise NotImplementedError

    def get_timestamp(self, idx) -> int:
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp) -> int:
        return self.timestamps.index(timestamp)

    def get_data(self, idx) -> dict[str, Any]:
        # Ensure a capture exists in the current process (worker-safe).
        if self.cap is None:
            # Open a capture for this worker/process.
            self.opened_in_get_data = True
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise ValueError("Could not open video in worker process")

        # Seek to the requested frame index and read it.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame at index {idx}")

        # Convert BGR (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize shape to (C, H, W)
        if frame.ndim == 2:
            frame = np.expand_dims(frame, axis=2)
        if frame.ndim == 3:
            frame = frame.transpose(2, 0, 1)

        return {"RGB": frame}

    @property
    def sensor_ids(self) -> list[str]:
        return ["RGB"]

    def __del__(self):
        self.close()

    def close(self):
        if self.cap is not None:
            self.cap.release()
