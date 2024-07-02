from .euv_dataset import EuvDataset

from pathlib import Path


class SoloEuvDataset(EuvDataset):
    def __init__(
        self,
        folder_path: str | Path,
        wavelengths: list[str] = ["174", "304"],
        level: int = 2,
    ):
        super().__init__(folder_path, wavelengths, level)

    @property
    def satellite_name(self):
        return "SOLO"
