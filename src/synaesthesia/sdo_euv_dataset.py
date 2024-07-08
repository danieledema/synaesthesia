from pathlib import Path

from .euv_dataset import EuvDataset


class SdoEuvDataset(EuvDataset):
    def __init__(
        self,
        folder_path: str | Path,
        wavelengths: list[str] = ["94", "131", "171", "193", "211", "304", "335"],
        level: int = 2,
        time_threshold: int = 1,
    ):
        super().__init__(folder_path, wavelengths, level, time_threshold)

    @property
    def satellite_name(self):
        return "SDO"
