from .abstract_dataset import DatasetBase
from .utils import convert_to_datetime
import pandas as pd
import numpy as np

from tqdm import tqdm


def parse_custom_datetime(dt_str):
    return f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}T{dt_str[9:11]}:{dt_str[11:13]}:{dt_str[13:15]}"


class BoundaryFilteredDataset(DatasetBase):
    def __init__(
        self,
        dataset: DatasetBase,
        boundaries: list[tuple[str, str]],
    ):
        super().__init__()

        self.boundaries = boundaries

        print("Initializing BoundaryFilteredDataset.")
        print(f"Boundaries: {self.boundaries}")

        # Convert boundaries to numpy.datetime64 using the helper function
        boundaries_dt = [
            (
                np.datetime64(parse_custom_datetime(b[0])),
                np.datetime64(parse_custom_datetime(b[1])),
            )
            for b in self.boundaries
        ]

        # Use the dataset timestamps directly
        timestamps = dataset.timestamps

        indices = []
        for b0, b1 in boundaries_dt:
            # Find indices where timestamps are within the boundary
            idxs = [i for i in tqdm(range(len(timestamps))) if b0 < timestamps[i] < b1]
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
