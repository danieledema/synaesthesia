from pathlib import Path

import pandas

from ..abstract.conversion import convert_to_timestamp
from ..abstract.dataset_base import DatasetBase


class CsvDataset(DatasetBase):
    def __init__(self, path: str | Path, machine_name: str, cols: list[str] | str | None = None):
        super().__init__()

        self.path = Path(path)
        self.data = pandas.read_csv(self.path)
        self.data["timestamp"] = (
            self.data["timestamp"].apply(convert_to_timestamp).apply(int)
        )

        self.cols = (
            cols if isinstance(cols, list) else [cols] if cols else self.data.columns
        )
        self.cols = [col for col in self.cols if not col == "timestamp"]

        self._machine_name = machine_name
        
    def __len__(self):
        return len(self.data)

    def get_data(self, idx):
        data = {f"{col}": self.data[col].values[idx] for col in self.cols}
        return data

    @property
    def sensor_ids(self):
        return self.cols

    def get_timestamp(self, idx):
        return self.data["timestamp"].values[idx]

    def get_timestamp_idx(self, timestamp):
        return self.data[self.data["timestamp"] == timestamp].index[0]

    @property
    def timestamps(self):
        return self.data["timestamp"].values.tolist()

    @property
    def machine_name(self):
        return self._machine_name