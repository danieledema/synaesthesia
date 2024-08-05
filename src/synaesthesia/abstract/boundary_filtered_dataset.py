from tqdm import tqdm

from .conversion import convert_to_string, convert_to_timestamp
from .dataset_base import DatasetBase


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
                convert_to_timestamp(b[0]),
                convert_to_timestamp(b[1]),
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
        return self.dataset.id

    def __len__(self):
        return len(self.fwd_indices)

    def get_data(self, idx):
        return self.dataset.get_data(self.fwd_indices[idx])

    def get_timestamp(self, idx):
        return self.dataset.get_timestamp(self.fwd_indices[idx])

    def get_timestamp_idx(self, timestamp):
        return self.bwd_indices[self.dataset.get_timestamp_idx(timestamp)]

    @property
    def sensor_ids(self):
        return self.dataset.sensor_ids

    def __repr__(self):
        inner_repr = repr(self.dataset)
        lines = inner_repr.split("\n")
        inner_repr = "\n".join(["\t" + line for line in lines])

        boundaries = "\n".join(
            [
                f"\t{convert_to_string(b[0])} - {convert_to_string(b[1])}"
                for b in self.boundaries
            ]
        )

        return f"BoundaryFilteredDataset - {len(self.dataset)} samples\n{inner_repr}\n{boundaries}"
