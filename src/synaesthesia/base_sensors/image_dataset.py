from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
