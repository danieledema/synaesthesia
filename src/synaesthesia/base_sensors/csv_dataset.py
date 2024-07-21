from pathlib import Path

import pandas

from ..abstract.dataset_base import DatasetBase


class CsvDataset(DatasetBase):
    def __init__(self, path):
        super().__init__()

        self.path = Path(path)
        self.data = pandas.read_csv(self.path)

    def __len__(self):
        return len(self.data)

    def get_data(self, idx):
        data = {
            f"{col}": self.data[col].values[idx]
            for col in self.data.columns
            if col != "timestamp"
        }
        return data

    @property
    def sensor_ids(self):
        return [f"{self.id}-{col}" for col in self.data.columns if col != "timestamp"]

    def get_timestamp(self, idx):
        return self.data["timestamp"].values[idx]

    def get_timestamp_idx(self, timestamp):
        return self.data[self.data["timestamp"] == timestamp].index[0]

    @property
    def timestamps(self):
        return self.data["timestamp"].values.tolist()
