from .abstract_dataset import SingleSignalDatasetBase


class BoundaryFilteredDataset(SingleSignalDatasetBase):
    def __init__(
        self,
        dataset: SingleSignalDatasetBase,
        boundaries: list[tuple[str, str]],
    ):
        super().__init__()

        self.boundaries = boundaries

        indices = []
        for b in self.boundaries:
            func_in_b = lambda x: b[0] < x < b[1]

            idxs = [
                i for i in range(len(dataset)) if func_in_b(dataset.get_timestamp(i))
            ]
            indices += idxs

        self.fwd_indices = {i: idx for i, idx in enumerate(indices)}
        self.bwd_indices = {idx: i for i, idx in enumerate(indices)}

        self.dataset = dataset

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
        return f"BoundaryFilteredDataset({self.dataset}, {self.boundaries})"
