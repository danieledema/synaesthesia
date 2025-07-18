from functools import lru_cache
from typing import Tuple

from .dataset_base import DatasetBase


class CustomConcatDataset(DatasetBase):
    """
    A specialisation of the ConcatDataset where the idx is returned together
    with the data.
    Used in the datamodule.
    """

    def __init__(self, datasets: list[DatasetBase]):
        super().__init__()

        if not datasets:
            raise ValueError("At least one dataset must be provided")

        self.datasets = datasets

        # Validate that all datasets have the same sensor IDs
        reference_sensor_ids = set(self.datasets[0].sensor_ids)
        for i, dataset in enumerate(self.datasets[1:], 1):
            if set(dataset.sensor_ids) != reference_sensor_ids:
                raise ValueError(
                    f"Dataset {i} has different sensor IDs: "
                    f"{dataset.sensor_ids} vs {self.datasets[0].sensor_ids}"
                )

        # Pre-compute cumulative lengths for efficient dataset lookup
        self._cumulative_lengths = self._compute_cumulative_lengths()

    def _compute_cumulative_lengths(self) -> list[int]:
        """Compute cumulative lengths for efficient dataset lookup."""
        cumulative = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            cumulative.append(total)
        return cumulative

    def _find_dataset_and_index(self, idx: int) -> Tuple[int, int]:
        """
        Find the correct dataset and local index for a given global index.

        Args:
            idx: Global index

        Returns:
            Tuple of (dataset_index, local_index)

        Raises:
            IndexError: If index is out of bounds
        """
        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self)}"
            )

        # Binary search through cumulative lengths for efficiency
        for dataset_idx, cumulative_length in enumerate(self._cumulative_lengths):
            if idx < cumulative_length:
                local_idx = idx - (
                    self._cumulative_lengths[dataset_idx - 1] if dataset_idx > 0 else 0
                )
                return dataset_idx, local_idx

        # This should never be reached due to bounds checking above
        raise IndexError(f"Index {idx} could not be mapped to any dataset")

    def get_data(self, idx: int):
        """Get data from the appropriate dataset with global index included."""
        dataset_idx, local_idx = self._find_dataset_and_index(idx)
        data = self.datasets[dataset_idx].get_data(local_idx)
        data["idx"] = idx
        return data

    @lru_cache(maxsize=1)
    def __len__(self) -> int:
        """Return total length of all datasets combined. Cached for efficiency."""
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def get_timestamp(self, idx: int) -> int:
        """Get timestamp from the appropriate dataset."""
        dataset_idx, local_idx = self._find_dataset_and_index(idx)
        return self.datasets[dataset_idx].get_timestamp(local_idx)

    def get_timestamp_idx(self, timestamp: int) -> int:
        """
        Get the global index corresponding to a timestamp.

        Args:
            timestamp: Timestamp to find

        Returns:
            Global index corresponding to the timestamp

        Raises:
            ValueError: If timestamp is not found in any dataset
        """
        global_offset = 0
        for dataset in self.datasets:
            try:
                local_idx = dataset.get_timestamp_idx(timestamp)
                return global_offset + local_idx
            except ValueError:
                global_offset += len(dataset)
                continue

        raise ValueError(f"Timestamp {timestamp} not found in any dataset")

    @property
    def timestamps(self) -> list[int]:
        """Return all timestamps from all datasets as a list."""
        all_timestamps = []
        for dataset in self.datasets:
            all_timestamps.extend(dataset.timestamps)
        return all_timestamps

    @property
    def id(self) -> str:
        """Return a combined ID from all datasets."""
        dataset_ids = [d.id for d in self.datasets if d.id]
        return "_".join(dataset_ids) if dataset_ids else "concat_dataset"

    def get_machine_name(self) -> str:
        """
        Return machine name. Warns if datasets have different machine names.
        """
        machine_names = list(set(d.machine_name for d in self.datasets))

        if len(machine_names) > 1:
            print(
                f"[WARNING] ConcatDataset contains multiple machine names: {machine_names}"
            )
            return "_".join(machine_names)

        return machine_names[0] if machine_names else "unknown_machine"

    @property
    def sensor_ids(self) -> list[str]:
        """Return sensor IDs (all datasets have the same sensor IDs)."""
        return self.datasets[0].sensor_ids

    def __repr__(self) -> str:
        """Return a detailed string representation of the concat dataset."""
        lines = [
            f"Concat dataset: {len(self)} samples",
            f"Datasets: {len(self.datasets)}",
            "",
        ]

        for i, dataset in enumerate(self.datasets):
            lines.append(f"Dataset {i}:")
            # Indent each line of the dataset representation
            dataset_repr = repr(dataset)
            indented_lines = [f"    {line}" for line in dataset_repr.split("\n")]
            lines.extend(indented_lines)
            lines.append("----")

        return "\n".join(lines).rstrip("----").strip()
