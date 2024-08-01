from typing import Any

from .dataset_base import DatasetBase


class TimeOffsetDataset(DatasetBase):
    def __init__(self, dataset: DatasetBase, time_offset: int):
        super().__init__()

        self.time_offset = time_offset
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_data(self, idx) -> dict[str, Any]:
        return self.dataset.get_data(idx)

    def get_timestamp(self, idx) -> int:
        return self.dataset.get_timestamp(idx) + self.time_offset

    def get_timestamp_idx(self, timestamp) -> int:
        return self.dataset.get_timestamp_idx(timestamp - self.time_offset)

    @property
    def sensor_ids(self) -> list[str]:
        return self.dataset.sensor_ids

    @property
    def id(self) -> str:
        return self.dataset.id

    @property
    def satellite_name(self) -> str:
        return self.dataset.satellite_name

    @property
    def timestamps(self) -> list[int]:
        raise NotImplementedError

    def __repr__(self) -> str:
        print_string = f"Time Offset dataset: {len(self)} samples\n"
        print_string += f"Dataset:\n"

        inner_repr = repr(self.dataset)
        lines = inner_repr.split("\n")
        inner_repr = "\n".join(["\t" + line for line in lines])

        print_string += inner_repr
        return print_string
