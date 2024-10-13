from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..abstract.dataset_base import DatasetBase


class ImageDataset(DatasetBase):
    """
    Dataset class for Image data.
    """

    def __init__(
        self,
        folder_path: str | Path,
        extension: str,
        format: str = "RGB",
    ):
        super().__init__()

        self.folder_path = Path(folder_path)
        self.extension = extension
        self.format = format

        files = self.folder_path.glob(f"*.{self.extension}")
        self.files = list(files)
        self.files.sort()

        self._timestamps = [self.parse_filename(f) for f in self.files]
        self.data_dict = {t: f for t, f in zip(self._timestamps, self.files)}

    def parse_filename(self, filename) -> int:
        raise NotImplementedError

    @property
    def timestamps(self):
        return self._timestamps

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_data(self, idx) -> dict[str, Any]:
        timestamp = self.get_timestamp(idx)
        data = {self.sensor_ids[0]: self.read_data(self.data_dict[timestamp])}
        return data

    def read_data(self, file_path: Path) -> Any:
        image = Image.open(file_path)
        image = image.convert(self.format)
        image_np = np.array(image)

        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, None]

        return image_np

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp):
        try:
            return self._timestamps.index(timestamp)
        except ValueError:
            raise ValueError("Timestamp not found in dataset")

    @property
    def sensor_ids(self) -> list[str]:
        return ["RGB"]
