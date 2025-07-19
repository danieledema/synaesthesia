import bisect
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

from .dataset_base import DatasetBase


class AggregationMethod(Enum):
    """Enumeration for aggregation methods."""

    ALL = "all"
    COMMON = "common"
    INDEX = "I:"  # Will be handled specially for I:<idx> pattern


class FillMethod(Enum):
    """Enumeration for fill methods."""

    NONE = "none"
    LAST = "last"
    CLOSEST = "closest"


class MultiSignalDataset(DatasetBase):
    """
    Dataset class for handling multiple signal datasets.
    """

    def __init__(
        self,
        single_signal_datasets: List[DatasetBase],
        aggregation: str = "all",
        fill: str = "none",
        time_cut: int = 60,  # in minutes
        return_indices: bool = False,
    ):
        """
        Initializes the MultiSignalDataset.

        Args:
            single_signal_datasets: List of DatasetBase objects representing single signal datasets.
            aggregation: Aggregation method for timestamps ("all", "common", "I:<idx>").
            fill: Method for filling missing timestamps ("none", "last", "closest").
            time_cut: Time cut-off in minutes for "closest" fill method.
            return_indices: Whether to return dataset indices in the output.
        """
        super().__init__()

        if not single_signal_datasets:
            raise ValueError("At least one dataset must be provided")

        self.single_signal_datasets = single_signal_datasets
        self.aggregation = aggregation
        self.fill = fill
        self.time_cut = time_cut
        self.return_indices = return_indices

        # Validate inputs
        self._validate_inputs()

        # Initialize timestamps and data mapping
        logger.info("Initializing timestamps...")
        self._timestamps = self._initialize_timestamps()

        logger.info("Initializing data dictionary...")
        self.data_dict = self._initialize_data_dict()

        # Create timestamp lookup for O(log n) access
        self._timestamp_to_idx = {ts: idx for idx, ts in enumerate(self._timestamps)}

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        # Validate aggregation method
        if self.aggregation not in [
            "all",
            "common",
        ] and not self.aggregation.startswith("I:"):
            raise ValueError(f"Invalid aggregation method: {self.aggregation}")

        if self.aggregation.startswith("I:"):
            try:
                idx = int(self.aggregation[2:])
                if idx < 0 or idx >= len(self.single_signal_datasets):
                    raise ValueError(f"Dataset index {idx} out of range")
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(
                        f"Invalid index format in aggregation: {self.aggregation}"
                    )
                raise

        # Validate fill method
        if self.fill not in ["none", "last", "closest"]:
            raise ValueError(f"Invalid fill method: {self.fill}")

    def _initialize_timestamps(self) -> List[int]:
        """Initialize timestamps based on aggregation method."""
        if self.aggregation == "all":
            return self._merge_all_timestamps()
        elif self.aggregation == "common":
            return self._find_common_timestamps()
        elif self.aggregation.startswith("I:"):
            idx = int(self.aggregation[2:])
            return self.single_signal_datasets[idx].timestamps.copy()
        else:
            raise ValueError(f"Invalid aggregation method: {self.aggregation}")

    def _merge_all_timestamps(self) -> List[int]:
        """Efficiently merge all timestamps from all datasets."""
        all_timestamps = set()

        for ds in tqdm(self.single_signal_datasets, desc="Collecting timestamps"):
            all_timestamps.update(ds.timestamps)

        return sorted(all_timestamps)

    def _find_common_timestamps(self) -> List[int]:
        """Find timestamps common to all datasets."""
        if not self.single_signal_datasets:
            return []

        # Start with first dataset's timestamps as a set
        common_timestamps = set(self.single_signal_datasets[0].timestamps)

        # Intersect with each subsequent dataset
        for ds in tqdm(
            self.single_signal_datasets[1:], desc="Finding common timestamps"
        ):
            ds_timestamps = set(ds.timestamps)
            common_timestamps &= ds_timestamps

        return sorted(common_timestamps)

    def _initialize_data_dict(self) -> Dict[int, List[Optional[int]]]:
        """Initialize data dictionary mapping timestamps to dataset indices."""
        # Initialize with None values
        data_dict = {
            timestamp: [None] * len(self.single_signal_datasets)
            for timestamp in self._timestamps
        }

        # Fill in actual indices where data exists
        for ds_idx, ds in enumerate(
            tqdm(self.single_signal_datasets, desc="Mapping data indices")
        ):
            # Create timestamp to index mapping for this dataset
            ds_timestamp_to_idx = {ts: idx for idx, ts in enumerate(ds.timestamps)}

            for timestamp in self._timestamps:
                if timestamp in ds_timestamp_to_idx:
                    data_dict[timestamp][ds_idx] = ds_timestamp_to_idx[timestamp]

        # Apply fill method
        return self._apply_fill_method(data_dict)

    def _apply_fill_method(
        self, data_dict: Dict[int, List[Optional[int]]]
    ) -> Dict[int, List[Optional[int]]]:
        """Apply the specified fill method to handle missing data."""
        if self.fill == "none":
            return data_dict
        elif self.fill == "last":
            return self._apply_last_fill(data_dict)
        elif self.fill == "closest":
            return self._apply_closest_fill(data_dict)
        else:
            raise ValueError(f"Unknown fill method: {self.fill}")

    def _apply_last_fill(
        self, data_dict: Dict[int, List[Optional[int]]]
    ) -> Dict[int, List[Optional[int]]]:
        """Apply last-value-carried-forward fill method."""
        # Find minimum common timestamp
        min_common_timestamp = max(
            ds.get_timestamp(0) for ds in self.single_signal_datasets
        )

        # Filter timestamps to start from min_common_timestamp
        valid_timestamps = [ts for ts in self._timestamps if ts >= min_common_timestamp]
        self._timestamps = valid_timestamps

        # Rebuild data_dict with filtered timestamps
        filtered_data_dict = {ts: data_dict[ts] for ts in valid_timestamps}

        # Forward fill missing values for each dataset
        for ds_idx in range(len(self.single_signal_datasets)):
            last_valid_idx = None
            for timestamp in valid_timestamps:
                if filtered_data_dict[timestamp][ds_idx] is not None:
                    last_valid_idx = filtered_data_dict[timestamp][ds_idx]
                elif last_valid_idx is not None:
                    filtered_data_dict[timestamp][ds_idx] = last_valid_idx

        return filtered_data_dict

    def _apply_closest_fill(
        self, data_dict: Dict[int, List[Optional[int]]]
    ) -> Dict[int, List[Optional[int]]]:
        """Apply closest-value fill method."""
        for ds_idx, ds in enumerate(
            tqdm(self.single_signal_datasets, desc="Applying closest fill")
        ):
            ds_timestamps = ds.timestamps

            for timestamp in self._timestamps:
                if data_dict[timestamp][ds_idx] is None:
                    # Find closest timestamp using binary search
                    closest_idx = self._find_closest_timestamp_idx(
                        timestamp, ds_timestamps
                    )

                    # Check if within time_cut threshold
                    closest_timestamp = ds_timestamps[closest_idx]
                    time_diff_minutes = abs(timestamp - closest_timestamp) / (
                        60 * 1000
                    )  # Assuming milliseconds

                    if time_diff_minutes <= self.time_cut:
                        data_dict[timestamp][ds_idx] = closest_idx

        return data_dict

    def _find_closest_timestamp_idx(
        self, target_timestamp: int, timestamps: List[int]
    ) -> int:
        """Find the index of the closest timestamp using binary search."""
        if not timestamps:
            raise ValueError("Empty timestamp list")

        # Use binary search to find insertion point
        idx = bisect.bisect_left(timestamps, target_timestamp)

        # Handle edge cases
        if idx == 0:
            return 0
        if idx == len(timestamps):
            return len(timestamps) - 1

        # Compare distances to adjacent timestamps
        left_dist = abs(target_timestamp - timestamps[idx - 1])
        right_dist = abs(target_timestamp - timestamps[idx])

        return idx - 1 if left_dist <= right_dist else idx

    @property
    def timestamps(self) -> List[int]:
        """Returns the list of timestamps."""
        return self._timestamps

    def __len__(self) -> int:
        """Returns the number of timestamps."""
        return len(self._timestamps)

    def get_data(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the data at the specified index.

        Args:
            idx: Index of the timestamp.

        Returns:
            Dictionary containing data from all datasets at the specified timestamp.
        """
        if idx < 0 or idx >= len(self._timestamps):
            raise IndexError(f"Index {idx} out of range")

        timestamp = self._timestamps[idx]
        data_indices = self.data_dict[timestamp]
        result_data = {}

        for ds_idx, ds in enumerate(self.single_signal_datasets):
            data_idx = data_indices[ds_idx]
            key_prefix = f"{ds.machine_name}_{ds.id}"

            if data_idx is None:
                # Add None values for all sensor IDs
                for sensor_id in ds.sensor_ids:
                    result_data[f"{key_prefix}-{sensor_id}"] = None
            else:
                # Get actual data
                ds_data = ds.get_data(data_idx)
                for key, value in ds_data.items():
                    result_data[f"{key_prefix}-{key}"] = value

            # Add index if requested
            if self.return_indices:
                result_data[f"{key_prefix}-index"] = data_idx

        return result_data

    def get_timestamp(self, idx: int) -> int:
        """
        Retrieves the timestamp at the specified index.

        Args:
            idx: Index of the timestamp.

        Returns:
            Timestamp at the specified index.
        """
        if idx < 0 or idx >= len(self._timestamps):
            raise IndexError(f"Index {idx} out of range")
        return self._timestamps[idx]

    def get_timestamp_idx(self, timestamp: int) -> int:
        """
        Retrieves the index of the specified timestamp.

        Args:
            timestamp: Timestamp to find the index for.

        Returns:
            Index of the specified timestamp.

        Raises:
            ValueError: If timestamp is not found.
        """
        if timestamp not in self._timestamp_to_idx:
            raise ValueError(f"Timestamp {timestamp} not found in dataset")
        return self._timestamp_to_idx[timestamp]

    def __repr__(self) -> str:
        """Returns a string representation of the MultiSignalDataset object."""
        lines = [
            f"MultiSignalDataset - {len(self)} samples",
            f"Aggregation: {self.aggregation}, Fill: {self.fill}",
            f"Datasets: {len(self.single_signal_datasets)}",
        ]

        for i, ds in enumerate(self.single_signal_datasets):
            lines.append(f"{i} ----")
            ds_repr = repr(ds)
            # Indent each line of the dataset representation
            indented_lines = [f"\t{line}" for line in ds_repr.split("\n")]
            lines.extend(indented_lines)
            lines.append("----")

        return "\n".join(lines)

    @property
    def id(self) -> str:
        """Returns the ID of the dataset."""
        return ""

    @property
    def sensor_ids(self) -> List[str]:
        """Returns a list of all sensor IDs from all datasets."""
        sensor_ids = []
        for ds in self.single_signal_datasets:
            sensor_ids.extend(ds.sensor_ids)
        return sensor_ids

    def get_machine_name(self) -> str:
        """Return a combined machine name from all datasets."""
        machine_names = [ds.machine_name for ds in self.single_signal_datasets]
        return "+".join(sorted(set(machine_names)))
