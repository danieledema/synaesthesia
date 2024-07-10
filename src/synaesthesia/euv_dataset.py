from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from astropy.io import fits
from tqdm import tqdm

from .abstract_dataset import DatasetBase
from .utils import convert_to_datetime


class EuvDataset(DatasetBase):
    """
    Dataset class for EUV (Extreme Ultraviolet) data.

    This class manages EUV data stored in FITS files organized by wavelength and timestamp.

    Attributes:
        folder_path (Path): Path to the folder containing the data files.
        wavelengths (list[str]): List of wavelengths (as strings) for which data is available.
        level (int): Level of the data (default is 2).
        files (dict): Dictionary mapping each wavelength to a sorted list of file paths.
    """

    def __init__(
        self,
        folder_path: str | Path,
        wavelengths: list[str],
        level: int = 2,
        time_threshold: int = 60,
    ):
        """
        Initializes the EUV dataset.

        Args:
            folder_path (str or Path): Path to the folder containing the data files.
            wavelengths (list of str): List of wavelengths to load data for.
            level (int, optional): Level of the data hierarchy (default is 2).
            time_threshold (int, optional): Time threshold for matching timestamps (default is 1 second).
        """
        super().__init__()

        self.folder_path = Path(folder_path)
        self.wavelengths = wavelengths
        self.level = level
        self.time_threshold = time_threshold

        # Initialize an empty dictionary to hold file paths for each wavelength
        self.files = {}

        # Populate self.files with sorted lists of file paths for each wavelength
        for wavelength in wavelengths:
            self.files[wavelength] = sorted(
                list(self.folder_path.glob(f"L{level}/**/*fsi{wavelength}*.fits"))
            )

        # Initialize an OrderedDict to store common timestamps and their corresponding data indices
        timestamps_per_wavelength = {}

        # Populate timestamps_per_wavelength with lists of timestamps for each wavelength
        for wavelength in wavelengths:
            timestamps_per_wavelength[wavelength] = [
                self.get_timestamp_from_filename(f) for f in self.files[wavelength]
            ]

        # Initialize an OrderedDict to store common timestamps and their corresponding data indices
        self.data_dict = OrderedDict()
        for i, timestamp in enumerate(timestamps_per_wavelength[wavelengths[0]]):
            self.data_dict[timestamp] = {wavelengths[0]: i}
        common_timestamps = list(self.data_dict.keys())

        # Populate self.data_dict with data indices for each wavelength
        for wavelength in tqdm(wavelengths[1:]):
            for i, timestamp in enumerate(timestamps_per_wavelength[wavelength]):
                timestamp = self.find_closest_timestamp(timestamp, common_timestamps)
                if timestamp:
                    self.data_dict[timestamp][wavelength] = i

        for timestamp in list(self.data_dict.keys()):
            if len(self.data_dict[timestamp]) != len(wavelengths):
                del self.data_dict[timestamp]

        self._timestamps = list(self.data_dict.keys())

    @property
    def timestamps(self):
        return self._timestamps

    def find_closest_timestamp(self, timestamp, timestamps):
        """
        Finds the closest timestamp to the given timestamp.

        Args:
            timestamp (str): Timestamp to find the closest match for.
            timestamps (list): List of common timestamps to search for a match.

        Returns:
            str: Closest matching timestamp.
        """
        timestamp = convert_to_datetime(timestamp)
        for t in timestamps:
            t_tmp = convert_to_datetime(t)
            if abs(t_tmp - timestamp).total_seconds() <= self.time_threshold:
                return t
        return None

    def __len__(self):
        """
        Returns the number of common timestamps available in the dataset.

        Returns:
            int: Number of common timestamps.
        """
        return len(self.timestamps)

    def get_data(self, idx):
        """
        Retrieves data corresponding to the timestamp at index `idx` in the dataset.

        Args:
            idx (int): Index of the timestamp to retrieve data for.

        Returns:
            dict: Dictionary containing data for each wavelength at the specified timestamp.
        """
        timestamp = self.get_timestamp(idx)
        data = {}
        for wavelength in self.wavelengths:
            file_idx = self.data_dict[timestamp][wavelength]
            file_path = self.files[wavelength][file_idx]
            with fits.open(file_path) as hdul:
                data[f"{wavelength}"] = hdul[1].data
        return data

    @staticmethod
    def get_timestamp_from_filename(filename):
        """
        Static method to extract timestamp from a given filename.

        Args:
            filename (Path or str): Filename from which to extract the timestamp.

        Returns:
            str: Extracted timestamp.
        """
        return filename.name.split("_")[-2]

    def get_timestamp(self, idx):
        """
        Retrieves the timestamp at index `idx` from the dataset.

        Args:
            idx (int): Index of the timestamp to retrieve.

        Returns:
            str: Timestamp corresponding to the specified index.
        """
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp):
        """
        Retrieves the index of a given `timestamp` in the dataset.

        Args:
            timestamp (str): Timestamp to find the index for.

        Returns:
            int: Index of the specified timestamp.

        Raises:
            ValueError: If the timestamp is not found in the dataset.
        """
        try:
            return self.timestamps.index(timestamp)
        except ValueError:
            raise ValueError("Timestamp not found in dataset")

    @property
    def id(self) -> str:
        return "EUV"

    @property
    def sensor_ids(self) -> list[str]:
        return [f"{self.id}-{w}" for w in self.wavelengths]
