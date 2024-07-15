from .abstract_dataset import DatasetBase
from .utils import convert_to_datetime
import pandas as pd


class BoundaryFilteredDataset(DatasetBase):
    def __init__(
        self,
        dataset: DatasetBase,
        boundaries: list[tuple[str, str]],
    ):
        super().__init__()

        self.boundaries = boundaries

        indices = []
        for b in self.boundaries:
            b0 = convert_to_datetime(b[0])
            b1 = convert_to_datetime(b[1])

            func_in_b = lambda x: b0 < x < b1

            idxs = [
                i
                for i in range(len(dataset))
                if func_in_b(pd.to_datetime(dataset.get_timestamp(i)))
            ]
            indices += idxs

        self.fwd_indices = {i: idx for i, idx in enumerate(indices)}
        self.bwd_indices = {idx: i for i, idx in enumerate(indices)}

        self.dataset = dataset

    @property
    def id(self):
        return ""

    def __len__(self):
        return len(self.fwd_indices)

    def get_data(self, idx):
        return self.dataset.get_data(self.fwd_indices[idx])

    def get_timestamp(self, idx):
        return self.dataset.get_timestamp(self.fwd_indices[idx])

    def get_timestamp_idx(self, timestamp):
        return self.bwd_indices[self.dataset.get_timestamp_idx(timestamp)]

    @property
    def sensor_id(self):
        return self.dataset.sensor_id

    def __repr__(self):
        return f"BoundaryFilteredDataset({self.dataset}, {self.boundaries})\nTotal samples: {len(self)}\n"
