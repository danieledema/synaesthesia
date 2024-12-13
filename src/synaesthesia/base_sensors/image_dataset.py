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
        image = image.convert(self.format)
        image_np = np.array(image)

        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, None]

        if len(image_np.shape) == 3:
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

        cap, self._timestamps = self.open()
        cap.release()

    def open(self):
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened() is False:
            raise ValueError("Video not opened")

        if self.timestamp_path is not None:
            timestamps = self.read_timestamps(self.timestamp_path)
        else:
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            timestamps = np.arange(0, num_frames / fps, 1 / fps).tolist()
        return cap, timestamps

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
        # Needed for multiprocessing when using more than one worker
        if self.cap is None:
            self.opened_in_get_data = True
            self.cap, _ = self.open()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        assert ok, "Could not read frame"

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if len(frame.shape) == 3:
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
