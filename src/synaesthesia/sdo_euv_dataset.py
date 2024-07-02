from .euv_dataset import EuvDataset

from pathlib import Path


class SdoEuvDataset(EuvDataset):
    def __init__(
        self,
        folder_path: str | Path,
        wavelengths: list[str] = ["94", "131", "171", "193", "211", "304", "335"],
        level: int = 2,
    ):
        super().__init__(folder_path, wavelengths, level)

    @property
    def satellite_name(self):
        return "SDO"
