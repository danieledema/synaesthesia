from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from ..abstract.dataset_base import DatasetBase


class CsvDataset(DatasetBase):
    """
    CsvDataset handles datasets stored in CSV files with timestamps and data columns.

    Provides efficient methods for accessing data by index, extracting timestamps,
    and validating CSV file structure.
    """

    def __init__(
        self,
        path: Union[str, Path],
        cols: Union[List[str], str, None] = None,
        sep: str = ";",
    ) -> None:
        super().__init__()

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")

        try:
            self._data = pd.read_csv(self.path, sep=sep)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {self.path}: {e}")

        # Validate required timestamp column
        if "timestamp" not in self._data.columns:
            raise ValueError("CSV file must contain a 'timestamp' column")

        # Convert timestamps efficiently
        self._data["timestamp"] = (
            self._data["timestamp"].apply(self.convert_timestamp).astype(int)
        )

        # Process column selection
        self._setup_columns(cols)

        # Cache frequently accessed data
        self._timestamps = self._data["timestamp"].values

    def _setup_columns(self, cols: Union[List[str], str, None]) -> None:
        """Setup and validate column selection."""
        if cols is None:
            self.cols = [col for col in self._data.columns if col != "timestamp"]
        elif isinstance(cols, str):
            self.cols = [cols] if cols != "timestamp" else []
        elif isinstance(cols, list):
            self.cols = [col for col in cols if col != "timestamp"]
        else:
            raise TypeError("cols must be a string, list of strings, or None")

        # Validate that all specified columns exist
        missing_cols = set(self.cols) - set(self._data.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in CSV: {missing_cols}")

    @abstractmethod
    def convert_timestamp(self, timestamp: Union[str, int]) -> int:
        """Convert timestamp to integer format. Must be implemented by subclasses."""
        pass

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._data)

    def get_data(self, idx: int) -> Dict[str, Any]:
        """
        Get data at the specified index.

        Args:
            idx: Index of the data point

        Returns:
            Dictionary mapping column names to values
        """
        if not 0 <= idx < len(self._data):
            raise IndexError(f"Index {idx} out of range [0, {len(self._data)})")

        # Use iloc for more efficient access
        row = self._data.iloc[idx]
        return {col: row[col] for col in self.cols}

    @property
    def sensor_ids(self) -> List[str]:
        """Get list of sensor column names."""
        return self.cols.copy()  # Return copy to prevent external modification

    def get_timestamp(self, idx: int) -> int:
        """
        Get timestamp at the specified index.

        Args:
            idx: Index of the timestamp

        Returns:
            Timestamp as integer
        """
        if not 0 <= idx < len(self._timestamps):
            raise IndexError(f"Index {idx} out of range [0, {len(self._timestamps)})")

        return int(self._timestamps[idx])

    def get_timestamp_idx(self, timestamp: int) -> int:
        """
        Get index for the given timestamp.

        Args:
            timestamp: Timestamp to find

        Returns:
            Index of the timestamp

        Raises:
            ValueError: If timestamp not found
        """
        matches = self._data[self._data["timestamp"] == timestamp].index
        if len(matches) == 0:
            raise ValueError(f"Timestamp {timestamp} not found in dataset")
        return matches[0]

    @property
    def timestamps(self) -> List[int]:
        """Get all timestamps as a list."""
        return self._timestamps.tolist()

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return (
            f"CsvDataset(path='{self.path}', "
            f"samples={len(self)}, "
            f"sensors={len(self.cols)})"
        )
