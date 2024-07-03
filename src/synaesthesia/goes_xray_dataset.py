from .xray_dataset import XRayDataset

from pathlib import Path


class GoesXRayDataset(XRayDataset):
    def __init__(self, folder_path: str | Path):
        super().__init__(folder_path)

    @property
    def satellite_name(self):
        return "GOES"
