import random
from typing import Any

from .abstract_dataset import DatasetBase


class SequentialSensorDataset(DatasetBase):
    def __init__(
        self,
        dataset: DatasetBase,
        n_samples=2,
        skip_n=0,
        stride=1,
    ):
        self.dataset = dataset
        self.n_samples = n_samples
        self.skip_n = skip_n
        self.stride = stride

        self.idx_format = (
            (i - self.n_samples) * self.skip_n for i in range(self.n_samples)
        )

    @property
    def idxs(self) -> list[int]:
        return [i for i in range(len(self.dataset)) if i % self.stride == 0]

    def __len__(self) -> int:
        len_samples = (1 + self.skip_n) * self.n_samples + 1
        len_dataset = len(self.dataset) - len_samples
        return len_dataset // self.stride

    @property
    def timestamps(self) -> list[int]:
        return [self.dataset.get_timestamp(i) for i in self.idxs]

    def get_data(self, idx) -> dict[str, Any]:
        idx_skips = [self.skip_n for _ in range(self.n_samples - 1)]

        idxs = [0] + [
            i + j * self.skip_n for i, j in zip(idx_skips, range(self.n_samples - 1))
        ]
        idxs = [i + idx for i in idxs]

        data_list = [self.dataset.get_data(i) for i in idxs]
        data = {d: [] for d in data_list[0]}
        for d in data_list:
            for key in d:
                data[key].append(d[key])

        return data

    @property
    def sensor_id(self):
        return self.dataset.sensor_id

    @property
    def satellite_name(self):
        return self.dataset.satellite_name

    def __repr__(self) -> str:
        inner_repr = repr(self.dataset)
        lines = inner_repr.split("\n")
        inner_repr = "\n".join(["\t" + line for line in lines])
        return f"Sequential - {self.n_samples} samples\n{inner_repr}"

    def get_timestamp(self, idx):
        return self.dataset.get_timestamp(
            idx * self.stride + self.n_samples * (self.skip_n + 1) - 1
        )

    def get_timestamp_idx(self, timestamp):
        return self.timestamps.index(timestamp)
