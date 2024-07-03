import random

from torch.utils.data import Dataset

from .multi_signal_dataset import MultiSignalDataset


class SequentialSensorDataset(Dataset):
    def __init__(
        self, dataset: MultiSignalDataset, n_samples=2, skip_n=0, skip_random=False
    ):
        self.dataset = dataset
        self.n_samples = n_samples
        self.skip_n = skip_n
        self.skip_random = skip_random

    def __len__(self) -> int:
        return len(self.dataset) - self.n_samples * (self.skip_n + 1) + 1

    @property
    def timestamps(self) -> list[int]:
        return [self.dataset.get_timestamp(i) for i in range(len(self.dataset))]

    def get_data(self, idx) -> tuple[list[dict], list[int]]:
        if self.skip_random:
            idx_skips = [
                random.randint(0, self.skip_n) for _ in range(self.n_samples - 1)
            ]
        else:
            idx_skips = [self.skip_n for _ in range(self.n_samples - 1)]

        idxs = [0] + [
            i + j * self.skip_n for i, j in zip(idx_skips, range(self.n_samples - 1))
        ]
        idxs = [i + idx for i in idxs]

        data = [self.dataset.get_data(i) for i in idxs]
        timestamps = [self.dataset.get_timestamp(i) for i in idxs]

        return data, timestamps

    def __getitem__(self, idx):
        data, timestamps = self.get_data(idx)
        return {
            "idx": idx,
            "timestamp": timestamps,
            "data": data,
        }
