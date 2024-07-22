from typing import Any

from .abstract_dataset import DatasetBase


class SequentialDataset(DatasetBase):
    def __init__(
        self,
        dataset: DatasetBase,
        n_samples=2,
        skip_n=0,
        stride=1,
        direction="future",
        delay_start=0,
        version="1D",
        return_timestamps=False,
    ):
        self.dataset = dataset
        self.n_samples = n_samples
        self.skip_n = skip_n
        self.stride = stride
        self.direction = direction
        self.delay_start = delay_start
        self.version = version
        self.return_timestamps = return_timestamps

        idx_format = [
            self.delay_start + i * (1 + self.skip_n) for i in range(n_samples)
        ]

        if direction == "future":
            self.idx_format = tuple(idx_format)
        elif direction == "past":
            self.idx_format = tuple([-1 * i for i in idx_format[::-1]])
        else:
            raise ValueError("direction must be either 'future' or 'past'")

        self._idxs = self.make_idxs()

        if self.version == "1D":
            if self.n_samples > 1:
                print(f"1D version - Taking max of next {n_samples} samples!")

    @property
    def idxs(self) -> list[int]:
        return self._idxs

    def make_idxs(self):
        total_length = len(self.dataset)
        indices = []
        for i in range(total_length):
            if self.direction == "future":
                idxs = [i + offset for offset in self.idx_format]
            else:  # direction == "past"
                idxs = [i + offset for offset in self.idx_format]
            if all(0 <= idx < total_length for idx in idxs):
                indices.append(i)
        return indices

    def __len__(self) -> int:
        return len(self.idxs)

    @property
    def timestamps(self) -> list[int]:
        t = self.dataset.timestamps
        return [t[i] for i in self.idxs]

    def get_data(self, idx) -> dict[str, Any]:

        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        original_idx = self.idxs[idx]
        seq_idxs = [original_idx + f for f in self.idx_format]
        data_list = [self.dataset.get_data(i) for i in seq_idxs]
        data = {d: [] for d in data_list[0]}
        for d in data_list:
            for key in d:
                data[key].append(d[key])

        if self.return_timestamps:
            seq_timestamps = [self.dataset.get_timestamp(i) for i in seq_idxs]
            data["timestamps"] = seq_timestamps

        if self.version == "1D" and "flare_class" in data.keys():
            data["flare_class"] = [max(data["flare_class"])]

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
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp):
        return self.timestamps.index(timestamp)
