from pathlib import Path

from .magnetogram_dataset import MagnetogramDataset


class SoloMagDataset(MagnetogramDataset):
    def __init__(
        self,
        folder_path: str | Path,
        channels: list[str] = ["blos", "icnt"],
        level: int = 2,
    ):
        super().__init__(folder_path, channels, level)

    @property
    def satellite_name(self):
        return "SOLO"
