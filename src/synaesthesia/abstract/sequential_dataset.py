from typing import Any, Literal

from .dataset_base import DatasetBase
from src.synaesthesia.abstract.filter_functions import (
    Filter,
    SkipNFilter,
    MultipleNFilter,
    ExponentialFilter,
)


class SequentialDataset(DatasetBase):
    def __init__(
        self,
        dataset: DatasetBase,
        n_samples=2,
        filter: Filter = None,
        stride: int = 1,
        timestamp_idx: Literal["first", "last"] = "first",
        return_timestamps=False,
    ):
        self.dataset = dataset
        self.n_samples = n_samples
        self.filter = filter
        self.stride = self.stride = max(stride, 1)
        self.timestamp_idx = timestamp_idx
        self.return_timestamps = return_timestamps

        # Generate idx_format using the provided filter
        if self.filter:
            self.idx_format = self.filter.get_indices(n_samples)
        else:
            # Default behavior: Generate a simple sequence [0, 1, 2, ..., n_samples-1]
            self.idx_format = list(range(n_samples))
            
    @property
    def idxs(self) -> list[int]:
        return [i * self.stride for i in range(len(self.dataset) // self.stride)]

    def __len__(self) -> int:
        len_samples = self.idx_format[-1] + 1
        return (len(self.dataset) - len_samples) // self.stride + 1

    @property
    def timestamps(self) -> list[int]:
        return [self.dataset.get_timestamp(i) for i in self.idxs]

    def get_data(self, idx) -> dict[str, Any]:
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        original_idx = idx * self.stride
        seq_idxs = [original_idx + f for f in self.idx_format]

        data_list = [self.dataset.get_data(i) for i in seq_idxs]
        data = {d: [] for d in data_list[0]}
        for d in data_list:
            for key in d:
                data[key].append(d[key])

        if self.return_timestamps:
            seq_timestamps = [self.dataset.get_timestamp(i) for i in seq_idxs]
            data["timestamps"] = seq_timestamps

        return data

    @property
    def sensor_ids(self):
        return self.dataset.sensor_ids

    @property
    def id(self):
        return self.dataset.id

    @property
    def satellite_name(self):
        return self.dataset.satellite_name

    def __repr__(self) -> str:
        inner_repr = repr(self.dataset)
        lines = inner_repr.split("\n")
        inner_repr = "\n".join(["\t" + line for line in lines])
        return f"Sequential - {len(self.idxs)} samples\n{inner_repr}"

    def get_timestamp(self, idx):
        original_idx = idx * self.stride
        if self.timestamp_idx == "first":
            return self.dataset.get_timestamp(original_idx)
        else:
            return self.dataset.get_timestamp(original_idx + self.idx_format[-1])

    def get_timestamp_idx(self, timestamp):
        return self.timestamps.index(timestamp)
