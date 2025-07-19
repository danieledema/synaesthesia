from functools import cached_property
from typing import Any, Dict, List

from .dataset_base import DatasetBase


class TimeOffsetDataset(DatasetBase):
    """A dataset wrapper that applies a time offset to all timestamps.

    This class wraps another dataset and adds a constant time offset to all
    timestamp operations while preserving all other functionality.

    Args:
        dataset: The underlying dataset to wrap
        time_offset: The time offset to add to all timestamps (in same units as dataset)
    """

    def __init__(self, dataset: DatasetBase, time_offset: int) -> None:
        super().__init__()

        if not isinstance(dataset, DatasetBase):
            raise TypeError("dataset must be an instance of DatasetBase")
        if not isinstance(time_offset, int):
            raise TypeError("time_offset must be an integer")

        self._dataset = dataset
        self._time_offset = time_offset

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._dataset)

    def get_data(self, idx: int) -> Dict[str, Any]:
        """Get data at the specified index."""
        return self._dataset.get_data(idx)

    def get_timestamp(self, idx: int) -> int:
        """Get timestamp at the specified index with offset applied."""
        return self._dataset.get_timestamp(idx) + self._time_offset

    def get_timestamp_idx(self, timestamp: int) -> int:
        """Get index for the given timestamp, accounting for offset."""
        return self._dataset.get_timestamp_idx(timestamp - self._time_offset)

    @property
    def sensor_ids(self) -> List[str]:
        """Get list of sensor IDs from the underlying dataset."""
        return self._dataset.sensor_ids

    @property
    def id(self) -> str:
        """Get the ID of the underlying dataset."""
        return self._dataset.id

    def get_machine_name(self) -> str:
        """Get the machine name of the underlying dataset."""
        return self._dataset.get_machine_name()

    @cached_property
    def timestamps(self) -> List[int]:
        """Get all timestamps with offset applied.

        Note: This creates a new list with offset applied to each timestamp.
        Use sparingly for large datasets as it can be memory intensive.
        """
        return [ts + self._time_offset for ts in self._dataset.timestamps]

    @property
    def time_offset(self) -> int:
        """Get the time offset value."""
        return self._time_offset

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        lines = [
            f"Time Offset dataset: {len(self)} samples",
            f"Time offset: {self._time_offset}",
            "Dataset:",
        ]

        # Indent the underlying dataset representation
        inner_repr = repr(self._dataset)
        indented_lines = [f"\t{line}" for line in inner_repr.split("\n")]
        lines.extend(indented_lines)

        return "\n".join(lines)
