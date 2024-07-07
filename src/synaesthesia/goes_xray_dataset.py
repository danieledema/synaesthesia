from pathlib import Path

from .xray_dataset import XRayDataset


class GoesXRayDataset(XRayDataset):
    def __init__(
        self,
        folder_path: str | Path,
        datatype: str = "flsum",
        goesnr: str = "16",
        level: int = 2,
        variables_to_include: list[str] = None,
    ):

        super().__init__(folder_path, datatype, goesnr, level, variables_to_include)

    @property
    def satellite_name(self):
        return "GOES"
