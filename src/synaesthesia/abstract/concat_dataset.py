import bisect
import warnings
from functools import lru_cache
from typing import List, Tuple

from .dataset_base import DatasetBase


class CustomConcatDataset(DatasetBase):
    """
    A specialisation of the ConcatDataset where the idx is returned together
    with the data.
    Used in the datamodule.
    """

    def __init__(self, datasets: List[DatasetBase]):
        super().__init__()

        if not datasets:
            raise ValueError("At least one dataset must be provided")

        self.datasets = datasets
        self._validate_datasets()

        # Pre-compute cumulative lengths for efficient dataset lookup
        self._cumulative_lengths = self._compute_cumulative_lengths()

    def _validate_datasets(self) -> None:
        """Validate that all datasets are compatible."""
        if not self.datasets:
            return

        reference_sensor_ids = set(self.datasets[0].sensor_ids)
        reference_machine_name = self.datasets[0].machine_name

        for i, dataset in enumerate(self.datasets[1:], 1):
            # Check sensor IDs compatibility
            if set(dataset.sensor_ids) != reference_sensor_ids:
                raise ValueError(
                    f"Dataset {i} has incompatible sensor IDs: "
                    f"{dataset.sensor_ids} vs {self.datasets[0].sensor_ids}"
                )

            # Warn about different machine names
            if dataset.machine_name != reference_machine_name:
                warnings.warn(
                    f"Dataset {i} has different machine name: "
                    f"{dataset.machine_name} vs {reference_machine_name}",
                    UserWarning,
                )

    def _compute_cumulative_lengths(self) -> List[int]:
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
        Uses binary search for O(log n) complexity.

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

        # Use binary search for O(log n) lookup instead of linear search
        dataset_idx = bisect.bisect_right(self._cumulative_lengths, idx)

        # Calculate local index
        local_idx = idx - (
            self._cumulative_lengths[dataset_idx - 1] if dataset_idx > 0 else 0
        )

        return dataset_idx, local_idx

    def get_data(self, idx: int) -> dict:
        """
        Get data from the appropriate dataset.

        Note: Does NOT include 'idx' key as this is handled by the base class.
        """
        dataset_idx, local_idx = self._find_dataset_and_index(idx)
        return self.datasets[dataset_idx].get_data(local_idx)

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
    @lru_cache(maxsize=1)
    def timestamps(self) -> List[int]:
        """Return all timestamps from all datasets as a list. Cached for efficiency."""
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
        Return machine name. Uses the first dataset's machine name.
        Validation warnings are handled in _validate_datasets.
        """
        return self.datasets[0].machine_name if self.datasets else "unknown_machine"

    @property
    def sensor_ids(self) -> List[str]:
        """Return sensor IDs (all datasets have the same sensor IDs)."""
        return self.datasets[0].sensor_ids if self.datasets else []

    def __repr__(self) -> str:
        """Return a detailed string representation of the concat dataset."""
        if not self.datasets:
            return "Empty ConcatDataset"

        lines = [
            f"ConcatDataset: {len(self)} samples from {len(self.datasets)} datasets",
            f"Machine: {self.machine_name}",
            f"Sensors: {', '.join(self.sensor_ids)}",
            "",
            "Individual Datasets:",
        ]

        for i, dataset in enumerate(self.datasets):
            lines.append(f"Dataset {i}:")
            # Indent each line of the dataset representation
            dataset_repr = repr(dataset)
            indented_lines = [f"    {line}" for line in dataset_repr.split("\n")]
            lines.extend(indented_lines)
            lines.append("----")

        return "\n".join(lines).rstrip("----").strip()
