from pathlib import Path

from .xray_dataset import XRayDataset


class GoesXRay_Flarelabel_Dataset(XRayDataset):
    variables_to_include = ["flare_class"]
    datatype = "flsum"
    severities = ["NO_FLARE", "A", "B", "C", "M", "X"]

    def __init__(
        self,
        folder_path: str | Path,
        goesnr: str = "16",
        level: int = 2,
    ):

        super().__init__(
            folder_path, self.datatype, goesnr, level, self.variables_to_include
        )

    @property
    def satellite_name(self):
        return "GOES"

    @property
    def sensor_id(self):
        return "flarelabel"
