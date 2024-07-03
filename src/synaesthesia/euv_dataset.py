from pathlib import Path

from astropy.io import fits

from .abstract_dataset import SingleSignalDatasetBase
from collections import OrderedDict


class EuvDataset(SingleSignalDatasetBase):
    """
     Dataset class for EUV (Extreme Ultraviolet) data.

    This class manages EUV data stored in FITS files organized by wavelength and timestamp.

    Attributes:
        folder_path (Path): Path to the folder containing the data files.
        wavelengths (list[str]): List of wavelengths (as strings) for which data is available.
        level (int): Level of the data (default is 2).
        files (dict): Dictionary mapping each wavelength to a sorted list of file paths.
    """

    def __init__(self, folder_path: str | Path, wavelengths: list[str], level: int = 2):
        """
        Initializes the EUV dataset.

        Args:
            folder_path (str or Path): Path to the folder containing the data files.
            wavelengths (list of str): List of wavelengths to load data for.
            level (int, optional): Level of the data hierarchy (default is 2).
        """
        super().__init__()

        self.folder_path = Path(folder_path)
        self.wavelengths = wavelengths
        self.level = level

        # Initialize an empty dictionary to hold file paths for each wavelength
        self.files = {}

        # Populate self.files with sorted lists of file paths for each wavelength
        for wavelength in wavelengths:
            self.files[wavelength] = sorted(
                list(self.folder_path.glob(f"L{level}/**/*fsi{wavelength}*.fits"))
            )

        # Initialize an OrderedDict to store common timestamps and their corresponding data indices
        timestamps_per_wavelength = {}
        print(
            "Warning: reduced resolution of timestamp in get_timestamp_from_filename to enable matches!"
        )

        # Populate timestamps_per_wavelength with lists of timestamps for each wavelength
        for wavelength in wavelengths:
            timestamps_per_wavelength[wavelength] = [
                self.get_timestamp_from_filename(f) for f in self.files[wavelength]
            ]

        # Find common timestamps across all wavelengths
        timestamp_sets = [set(i) for i in timestamps_per_wavelength.values()]
        common_timestamps = sorted(list(set.intersection(*timestamp_sets)))

        # Initialize an OrderedDict to store common timestamps and their corresponding data indices
        self.timestamps = OrderedDict()
        for timestamp in common_timestamps:
            self.timestamps[timestamp] = {}

        # Populate self.timestamps with data indices for each wavelength
        for wavelength in wavelengths:
            for i, timestamp in enumerate(timestamps_per_wavelength[wavelength]):
                if timestamp in self.timestamps:
                    self.timestamps[timestamp][wavelength] = i

    @property
    def sensor_id(self):
        """
        Property returning the sensor ID.

        Returns:
            str: Sensor ID.
        """
        return "EUV"

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
            file_idx = self.timestamps[timestamp][wavelength]
            file_path = self.files[wavelength][file_idx]
            with fits.open(file_path) as hdul:
                data[wavelength] = hdul[1].data
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
        return filename.name.split("_")[-2][:13]

    def get_timestamp(self, idx):
        """
        Retrieves the timestamp at index `idx` from the dataset.

        Args:
            idx (int): Index of the timestamp to retrieve.

        Returns:
            str: Timestamp corresponding to the specified index.
        """
        return list(self.timestamps.keys())[idx]

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
            return list(self.timestamps.keys()).index(timestamp)
        except ValueError:
            raise ValueError("Timestamp not found in dataset")
