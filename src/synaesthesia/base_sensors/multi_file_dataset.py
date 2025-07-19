from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from ..abstract.dataset_base import DatasetBase


class MultiFileDataset(DatasetBase):
    """
    Base class for datasets that read data from multiple files in a folder.

    This class handles the common functionality of discovering files with a specific
    extension, parsing filenames to extract timestamps, and organizing the data
    for efficient access.

    Methods:
    ----
    parse_filename(filename: Path) -> int:
        Abstract method to extract timestamp from filename.

    read_data(file_path: Path) -> Any:
        Abstract method to read and process data from a file.

    Properties:
    ----
    timestamps : List[int]
        Returns a list of timestamps extracted from filenames, sorted in ascending order.
    """

    def __init__(self, folder_path: str | Path, extension: str):
        """
        Initialize the MultiFileDataset.

        Args:
            folder_path: Path to the folder containing data files
            extension: File extension to filter for (without the dot)

        Raises:
            FileNotFoundError: If the folder path does not exist
            ValueError: If no files with the specified extension are found
        """
        super().__init__()

        self.folder_path = Path(folder_path)
        self.extension = extension

        # Validate folder exists
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder path does not exist: {self.folder_path}")

        # Discover and sort files
        files = list(self.folder_path.glob(f"*.{self.extension}"))
        if not files:
            raise ValueError(
                f"No files with extension '.{self.extension}' found in {self.folder_path}"
            )

        self.files = sorted(files)

        # Extract timestamps and create mapping
        self._timestamps = [self.parse_filename(f) for f in self.files]
        self.data_dict = {t: f for t, f in zip(self._timestamps, self.files)}

    @abstractmethod
    def parse_filename(self, filename: Path) -> int:
        """
        Extract timestamp from filename.

        Args:
            filename: Path object representing the file

        Returns:
            Timestamp as integer

        Raises:
            ValueError: If timestamp cannot be extracted from filename
        """
        pass

    @property
    def timestamps(self) -> List[int]:
        """
        Return a list of all timestamps in the dataset.

        Returns:
            List of timestamps sorted in ascending order
        """
        return self._timestamps

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of files/timestamps in the dataset
        """
        return len(self.timestamps)

    def get_data(self, idx: int) -> Dict[str, Any]:
        """
        Get the raw data at the specified index.

        Args:
            idx: Index of the data sample

        Returns:
            Dictionary containing the raw data from the file

        Raises:
            IndexError: If index is out of range
        """
        timestamp = self.get_timestamp(idx)
        data = self.read_data(self.data_dict[timestamp])
        return data

    def get_timestamp(self, idx: int) -> int:
        """
        Get the timestamp at the specified index.

        Args:
            idx: Index of the data sample

        Returns:
            Timestamp as integer

        Raises:
            IndexError: If index is out of range
        """
        if idx >= len(self.timestamps) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for {len(self.timestamps)} timestamps"
            )
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp: int) -> int:
        """
        Get the index corresponding to a timestamp.

        Args:
            timestamp: Timestamp to find

        Returns:
            Index corresponding to the timestamp

        Raises:
            ValueError: If timestamp is not found
        """
        try:
            return self._timestamps.index(timestamp)
        except ValueError:
            raise ValueError(f"Timestamp {timestamp} not found in dataset")

    @abstractmethod
    def read_data(self, file_path: Path) -> Any:
        """
        Read and process data from a file.

        Args:
            file_path: Path to the file to read

        Returns:
            Processed data from the file

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file cannot be read or processed
        """
        pass
