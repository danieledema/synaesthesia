from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .abstract_dataset import DatasetBase
from .constants import ALL_COMPONENTS, ALL_WAVELENGTHS
from .utils import convert_to_datetime


class SDOMLDataset(DatasetBase):
    def __init__(
        self,
        data_folder: Path | str,
        components: list[str] = ALL_COMPONENTS,
        wavelengths: list[str] = ALL_WAVELENGTHS,
    ):
        super().__init__()

        self.components = components
        self.wavelengths = wavelengths

        self.labels = self.components + self.wavelengths

        data_folder = Path(data_folder)
        files = list(data_folder.glob("**/*.npy"))

        self.data = OrderedDict()
        for file in files:
            filename = file.stem
            timestamp, label = filename.split("_")

            timestamp = convert_to_datetime(timestamp)
            timestamp = np.datetime64(timestamp)

            if timestamp not in self.data:
                self.data[timestamp] = {}

            if label in self.labels:
                self.data[timestamp][label] = file

        self._timestamps = list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def get_data(self, idx) -> dict[str, Any]:
        data = {l: np.load(f) for l, f in self.data[self.get_timestamp(idx)].items()}
        return data

    def get_timestamp(self, idx):
        return self._timestamps[idx]

    def get_timestamp_idx(self, timestamp):
        return self._timestamps.index(timestamp)

    @property
    def sensor_ids(self) -> list[str]:
        return ["AIA", "HMI"]

    @property
    def id(self):
        return "SDO-MLv2"

    @property
    def satellite_name(self):
        return "SDO"

    @property
    def timestamps(self):
        return self._timestamps
