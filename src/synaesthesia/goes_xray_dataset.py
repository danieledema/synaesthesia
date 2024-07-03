from .xray_dataset import XRayDataset

from pathlib import Path


class GoesXRayDataset(XRayDataset):
    def __init__(
        self,
        folder_path: str | Path,
        datatype: str = "flsum",
        level: int = 2,
        variables_to_include: list[str] = None,
    ):

        super().__init__(folder_path, datatype, level, variables_to_include)

    @property
    def satellite_name(self):
        return "GOES"
