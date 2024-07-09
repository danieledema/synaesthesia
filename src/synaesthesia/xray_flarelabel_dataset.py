from pathlib import Path

from .xray_dataset import XRayDataset


class GoesXRayFlarelabelDataset(XRayDataset):
    variables_to_include = ["flare_class"]
    datatype = "flsum"
    severities = ["NO_FLARE", "C", "M", "X"]

    def __init__(
        self,
        folder_path: str | Path,
        goesnr: str = "16",
        level: int = 2,
    ):

        super().__init__(
            folder_path, self.datatype, goesnr, level, self.variables_to_include
        )

    def get_data(self, idx):
        data_raw = super().get_data(idx)
        for s, severity in enumerate(self.severities):
            if data_raw["flare_class"].startswith(severity):
                data_raw["flare_class"] = s
                break
        else:
            data_raw["flare_class"] = 0

        return data_raw

    @property
    def satellite_name(self):
        return "GOES"

    @property
    def sensor_ids(self):
        return ["flarelabel"]
