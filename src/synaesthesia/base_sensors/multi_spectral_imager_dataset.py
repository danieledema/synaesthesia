import bisect
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from tqdm import tqdm

from ..abstract.dataset_base import DatasetBase


class MultiSpectralImagerDataset(DatasetBase):
    """
    Dataset class for Multispectral Imager data.

    This dataset handles multispectral imaging data where each wavelength is treated
    as a separate sensor. It aligns data across wavelengths by timestamp and provides
    options for handling incomplete data and duplicates.

    Methods:
    ----
    collect_files() -> List[Path]:
        Abstract method to collect all relevant files from the folder.

    parse_filename(filename: Path) -> tuple[int, str]:
        Abstract method to extract timestamp and wavelength from filename.

    filename_from_timestamp(timestamp: int, wavelength: str, folder_path: Path) -> Path:
        Abstract method to construct filename from timestamp and wavelength.

    read_data(file_path: Path) -> Any:
        Abstract method to read and process data from a file.

    Properties:
    ----
    timestamps : List[int]
        Returns a list of aligned timestamps across all wavelengths.

    sensor_ids : List[str]
        Returns the list of wavelengths as sensor identifiers.
    """

    def __init__(
        self,
        folder_path: str | Path,
        wavelengths: List[str],
        time_threshold: int | None = 60,
        remove_incomplete: bool = True,
        remove_duplicates: bool = True,
        duplicate_threshold: int = 10,
        already_aligned: bool = False,
    ):
        """
        Initialize the MultiSpectralImagerDataset.

        Args:
            folder_path: Path to the folder containing multispectral data files
            wavelengths: List of wavelength identifiers to process
            time_threshold: Maximum time difference (seconds) for timestamp alignment.
                          If None, no threshold is applied
            remove_incomplete: Whether to remove timestamps that don't have all wavelengths
            remove_duplicates: Whether to remove duplicate timestamps within threshold
            duplicate_threshold: Minimum time difference (seconds) between timestamps
            already_aligned: Whether files are already aligned by timestamp

        Raises:
            ValueError: If no files are found or wavelengths list is empty
        """
        super().__init__()

        if not wavelengths:
            raise ValueError("Wavelengths list cannot be empty")

        self.folder_path = Path(folder_path)
        self.wavelengths = wavelengths
        self.time_threshold = time_threshold
        self.duplicate_threshold = duplicate_threshold

        # Collect and organize files by wavelength
        files = self.collect_files()
        if not files:
            raise ValueError(f"No files found in {folder_path}")

        files_by_wavelength = self._organize_files_by_wavelength(files)

        logger.info(f"Loading data for wavelengths {self.wavelengths}")

        # Build dataset efficiently
        if already_aligned:
            self.data_dict, self._timestamps = self._build_aligned_dataset(
                files_by_wavelength
            )
        else:
            self.data_dict, self._timestamps = self._build_unaligned_dataset(
                files_by_wavelength, remove_duplicates
            )

        # Clean up incomplete entries if requested
        if remove_incomplete:
            self._remove_incomplete_entries()

        logger.info(f"Loaded {len(self)} samples for wavelengths {self.wavelengths}")

    def _organize_files_by_wavelength(
        self, files: List[Path]
    ) -> Dict[str, List[tuple[int, Path]]]:
        """
        Organize files by wavelength with their timestamps.

        Args:
            files: List of file paths to organize

        Returns:
            Dictionary mapping wavelength to list of (timestamp, file_path) tuples
        """
        files_by_wavelength = defaultdict(list)

        for file in files:
            try:
                timestamp, wavelength = self.parse_filename(file)
                if wavelength in self.wavelengths:
                    files_by_wavelength[wavelength].append((timestamp, file))
            except Exception as e:
                logger.warning(f"Failed to parse filename {file}: {e}")
                continue

        # Sort by timestamp for each wavelength
        for wavelength in files_by_wavelength:
            files_by_wavelength[wavelength].sort(key=lambda x: x[0])

        return files_by_wavelength

    def _build_aligned_dataset(self, files_by_wavelength: Dict) -> tuple[Dict, List]:
        """
        Build dataset when files are already aligned by timestamp.

        Args:
            files_by_wavelength: Dictionary of wavelength to file mappings

        Returns:
            Tuple of (data_dict, timestamps)
        """
        # Use the first wavelength as reference
        reference_wavelength = self.wavelengths[0]
        if reference_wavelength not in files_by_wavelength:
            raise ValueError(f"Reference wavelength {reference_wavelength} not found")

        data_dict = {}
        timestamps = []

        for timestamp, file in files_by_wavelength[reference_wavelength]:
            data_dict[timestamp] = {reference_wavelength: file}
            timestamps.append(timestamp)

            # Add corresponding files for other wavelengths
            for wavelength in self.wavelengths[1:]:
                filename = self.filename_from_timestamp(
                    timestamp, wavelength, self.folder_path
                )
                if filename.exists():
                    data_dict[timestamp][wavelength] = filename

        return data_dict, timestamps

    def _build_unaligned_dataset(
        self, files_by_wavelength: Dict, remove_duplicates: bool
    ) -> tuple[Dict, List]:
        """
        Build dataset when files need timestamp alignment.

        Args:
            files_by_wavelength: Dictionary of wavelength to file mappings
            remove_duplicates: Whether to remove duplicate timestamps

        Returns:
            Tuple of (data_dict, timestamps)
        """
        # Start with reference wavelength (first one)
        reference_wavelength = self.wavelengths[0]
        if reference_wavelength not in files_by_wavelength:
            raise ValueError(f"Reference wavelength {reference_wavelength} not found")

        data_dict = {}
        timestamps = []
        last_timestamp = 0

        # Process reference wavelength first
        for timestamp, file in files_by_wavelength[reference_wavelength]:
            if (
                remove_duplicates
                and timestamp - last_timestamp < self.duplicate_threshold
            ):
                continue

            data_dict[timestamp] = {reference_wavelength: file}
            timestamps.append(timestamp)
            last_timestamp = timestamp

        # Process other wavelengths with efficient timestamp matching
        for wavelength in tqdm(self.wavelengths[1:], desc="Aligning wavelengths"):
            if wavelength not in files_by_wavelength:
                logger.warning(f"No files found for wavelength {wavelength}")
                continue

            self._align_wavelength_files(
                data_dict, timestamps, files_by_wavelength[wavelength], wavelength
            )

        return data_dict, timestamps

    def _align_wavelength_files(
        self, data_dict: Dict, timestamps: List, wavelength_files: List, wavelength: str
    ):
        """
        Efficiently align files for a specific wavelength using binary search.

        Args:
            data_dict: Dictionary to update with aligned files
            timestamps: List of reference timestamps
            wavelength_files: List of (timestamp, file) tuples for the wavelength
            wavelength: Wavelength identifier
        """
        for file_timestamp, file in wavelength_files:
            # Find closest timestamp using binary search
            closest_idx = self._find_closest_timestamp_idx(timestamps, file_timestamp)

            if closest_idx is not None:
                closest_timestamp = timestamps[closest_idx]
                time_diff = abs(file_timestamp - closest_timestamp)

                if self.time_threshold is None or time_diff < self.time_threshold:
                    data_dict[closest_timestamp][wavelength] = file
                else:
                    # Create new entry if time difference is too large
                    data_dict[file_timestamp] = {wavelength: file}
                    # Insert timestamp in sorted order
                    bisect.insort(timestamps, file_timestamp)

    def _find_closest_timestamp_idx(self, timestamps: List, target: int) -> int | None:
        """
        Find the index of the closest timestamp using binary search.

        Args:
            timestamps: Sorted list of timestamps
            target: Target timestamp to find

        Returns:
            Index of closest timestamp, or None if list is empty
        """
        if not timestamps:
            return None

        # Use binary search to find insertion point
        idx = bisect.bisect_left(timestamps, target)

        # Check boundaries and find closest
        candidates = []
        if idx > 0:
            candidates.append((abs(timestamps[idx - 1] - target), idx - 1))
        if idx < len(timestamps):
            candidates.append((abs(timestamps[idx] - target), idx))

        if candidates:
            return min(candidates)[1]
        return None

    def _remove_incomplete_entries(self):
        """Remove entries that don't have all wavelengths."""
        complete_timestamps = []
        for timestamp in list(self.data_dict.keys()):
            if len(self.data_dict[timestamp]) == len(self.wavelengths):
                complete_timestamps.append(timestamp)
            else:
                del self.data_dict[timestamp]

        self._timestamps = sorted(complete_timestamps)

    @abstractmethod
    def collect_files(self) -> List[Path]:
        """
        Collect all relevant files from the folder.

        Returns:
            List of file paths to process
        """
        pass

    def get_machine_name(self) -> str:
        """
        Extract machine name from folder path.

        Returns:
            Machine name derived from the folder name, or 'multispectral_imager' if unavailable
        """
        return self.folder_path.name or "multispectral_imager"

    @property
    def timestamps(self) -> List[int]:
        """
        Return a list of all timestamps in the dataset.

        Returns:
            List of aligned timestamps across all wavelengths
        """
        return self._timestamps

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of common timestamps available in the dataset
        """
        return len(self.timestamps)

    def get_data(self, idx: int) -> Dict[str, Any]:
        """
        Get the raw data at the specified index.

        Retrieves data corresponding to the timestamp at index `idx` in the dataset.

        Args:
            idx: Index of the timestamp to retrieve data for

        Returns:
            Dictionary containing data for each wavelength at the specified timestamp

        Raises:
            IndexError: If index is out of range
            KeyError: If wavelength data is missing for the timestamp
        """
        timestamp = self.get_timestamp(idx)
        data = {}

        for wavelength in self.wavelengths:
            if wavelength in self.data_dict[timestamp]:
                file_path = self.data_dict[timestamp][wavelength]
                data[wavelength] = self.read_data(file_path)
            else:
                logger.warning(
                    f"Missing data for wavelength {wavelength} at timestamp {timestamp}"
                )

        return data

    @abstractmethod
    def read_data(self, file_path: Path) -> Any:
        """
        Read and process data from a file.

        Args:
            file_path: Path to the file to read

        Returns:
            Processed data from the file
        """
        pass

    @abstractmethod
    def filename_from_timestamp(
        self, timestamp: int, wavelength: str, folder_path: Path
    ) -> Path:
        """
        Construct filename from timestamp and wavelength.

        Args:
            timestamp: Timestamp value
            wavelength: Wavelength identifier
            folder_path: Base folder path

        Returns:
            Constructed file path
        """
        pass

    @abstractmethod
    def parse_filename(self, filename: Path) -> tuple[int, str]:
        """
        Extract timestamp and wavelength from filename.

        Args:
            filename: Path object representing the file

        Returns:
            Tuple of (timestamp, wavelength)

        Raises:
            ValueError: If timestamp or wavelength cannot be extracted
        """
        pass

    def get_timestamp(self, idx: int) -> int:
        """
        Get the timestamp at the specified index.

        Args:
            idx: Index of the timestamp to retrieve

        Returns:
            Timestamp corresponding to the specified index

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
            timestamp: Timestamp to find the index for

        Returns:
            Index of the specified timestamp

        Raises:
            ValueError: If the timestamp is not found in the dataset
        """
        try:
            return self.timestamps.index(timestamp)
        except ValueError:
            raise ValueError(f"Timestamp {timestamp} not found in dataset")

    @property
    def id(self) -> str:
        """
        Return the dataset identifier.

        Returns:
            String identifier for the dataset type
        """
        return "multispectral_imager"

    @property
    def sensor_ids(self) -> List[str]:
        """
        Return a list of sensor IDs in the dataset.

        Returns:
            List of wavelength identifiers as sensor IDs
        """
        return self.wavelengths.copy()
