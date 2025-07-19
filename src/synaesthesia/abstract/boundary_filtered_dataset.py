from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .conversion import convert_to_timestamp
from .dataset_base import DatasetBase


class BoundaryFilteredDataset(DatasetBase):
    """
    A dataset wrapper that filters data based on timestamp boundaries.

    This class wraps another dataset and only exposes data points that fall
    within the specified time boundaries.
    """

    def __init__(
        self,
        dataset: DatasetBase,
        boundaries: List[Tuple[str, str]],
    ):
        super().__init__()

        if not boundaries:
            raise ValueError("At least one boundary must be provided")

        self.dataset = dataset
        self.boundaries = boundaries

        print("Initializing BoundaryFilteredDataset.")
        print(f"Boundaries: {self.boundaries}")

        # Build filtered indices once during initialization
        self._build_filtered_indices()

    def _build_filtered_indices(self) -> None:
        """Build the mapping between filtered and original dataset indices."""
        # Convert boundaries to timestamps once
        boundaries_dt = [
            (convert_to_timestamp(b[0]), convert_to_timestamp(b[1]))
            for b in self.boundaries
        ]

        # Get all timestamps from the dataset
        timestamps = self.dataset.timestamps

        # Use numpy for efficient filtering if timestamps are numeric
        if timestamps and isinstance(timestamps[0], (int, float, np.number)):
            timestamps_array = np.array(timestamps)
            valid_indices = self._filter_indices_vectorized(
                timestamps_array, boundaries_dt
            )
        else:
            valid_indices = self._filter_indices_iterative(timestamps, boundaries_dt)

        # Build bidirectional index mappings
        self.fwd_indices = {i: idx for i, idx in enumerate(valid_indices)}
        self.bwd_indices = {idx: i for i, idx in enumerate(valid_indices)}

        print(
            f"Filtered dataset: {len(valid_indices)} samples from {len(timestamps)} original samples"
        )

    def _filter_indices_vectorized(
        self, timestamps: np.ndarray, boundaries_dt: List[Tuple[Any, Any]]
    ) -> List[int]:
        """Use vectorized operations for efficient filtering."""
        mask = np.zeros(len(timestamps), dtype=bool)

        for start_time, end_time in boundaries_dt:
            # Create mask for current boundary
            boundary_mask = (timestamps > start_time) & (timestamps < end_time)
            mask |= boundary_mask

        return np.where(mask)[0].tolist()

    def _filter_indices_iterative(
        self, timestamps: List[Any], boundaries_dt: List[Tuple[Any, Any]]
    ) -> List[int]:
        """Fallback iterative filtering for non-numeric timestamps."""
        valid_indices = []

        for i, timestamp in enumerate(tqdm(timestamps, desc="Filtering timestamps")):
            for start_time, end_time in boundaries_dt:
                if start_time < timestamp < end_time:
                    valid_indices.append(i)
                    break  # No need to check other boundaries for this timestamp

        return valid_indices

    @property
    def id(self) -> str:
        """Return the wrapped dataset's ID."""
        return self.dataset.id

    @property
    def sensor_ids(self) -> List[str]:
        """Return the wrapped dataset's sensor IDs."""
        return self.dataset.sensor_ids

    @property
    def timestamps(self) -> List[int]:
        """Return filtered timestamps."""
        return [
            self.dataset.get_timestamp(self.fwd_indices[i]) for i in range(len(self))
        ]

    def get_machine_name(self) -> str:
        """Return the wrapped dataset's machine name."""
        return self.dataset.get_machine_name()

    def __len__(self) -> int:
        """Return the number of filtered samples."""
        return len(self.fwd_indices)

    def get_data(self, idx: int) -> Dict[str, Any]:
        """Get data at the filtered index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        return self.dataset.get_data(self.fwd_indices[idx])

    def get_timestamp(self, idx: int) -> int:
        """Get timestamp at the filtered index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        return self.dataset.get_timestamp(self.fwd_indices[idx])

    def get_timestamp_idx(self, timestamp: int) -> int:
        """Get the filtered index for a given timestamp."""
        original_idx = self.dataset.get_timestamp_idx(timestamp)

        if original_idx not in self.bwd_indices:
            raise ValueError(f"Timestamp {timestamp} not found in filtered dataset")

        return self.bwd_indices[original_idx]

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        inner_repr = repr(self.dataset)
        lines = inner_repr.split("\n")
        indented_inner = "\n".join(["\t" + line for line in lines])

        boundaries_str = "\n".join([f"\t{b[0]} - {b[1]}" for b in self.boundaries])

        return (
            f"BoundaryFilteredDataset - {len(self)} samples "
            f"(filtered from {len(self.dataset)} samples)\n"
            f"Boundaries:\n{boundaries_str}\n"
            f"Wrapped Dataset:\n{indented_inner}"
        )
