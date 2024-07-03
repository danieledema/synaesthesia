from pathlib import Path

from astropy.io import fits

from .abstract_dataset import SingleSignalDatasetBase
from collections import OrderedDict

from scipy import ndimage

import itertools
import numpy as np


class MagnetogramDataset(SingleSignalDatasetBase):
    """
    Dataset class for Magnetogram data.

    This class manages Magnetogram data stored in FITS files organized by channel and timestamp.
    The data is stored in a dictionary with channel names as keys and numpy arrays as values.
    The images are rotated based on the CROTA header keyword.

    Attributes:
        folder_path (Path): Path to the folder containing the data files.
        channels (list[str]): List of channels (as strings) for which data is available.
        level (int): Level of the data (default is 2).
        files (dict): Dictionary mapping each channel to a sorted list of file paths.
    """

    def __init__(self, folder_path: str | Path, channels: list[str], level: int = 2):
        """
        Initializes the MAG dataset.

        Args:
            folder_path (str or Path): Path to the folder containing the data files.
            channels (list of str): List of channels to load data for.
            level (int, optional): Level of the data hierarchy (default is 2).
        """
        super().__init__()

        self.folder_path = Path(folder_path)
        self.channels = channels
        self.level = level

        # Initialize an empty dictionary to hold file paths for each channel
        self.files = {}

        # Populate self.files with sorted lists of file paths for each channel
        for channel in channels:
            self.files[channel] = sorted(
                list(self.folder_path.glob(f"L{level}/**/*fdt-{channel}*.fits"))
            )

        # Initialize an OrderedDict to store common timestamps and their corresponding data indices
        timestamps_per_channel = {}
        print(
            "Warning: reduced length of timestamp in get_timestamp_from_filename to enable matches!"
        )

        # Populate timestamps_per_channel with lists of timestamps for each channel
        for channel in channels:
            timestamps_per_channel[channel] = [
                self.get_timestamp_from_filename(f) for f in self.files[channel]
            ]

        # Find common timestamps across all channels
        timestamp_sets = [set(i) for i in timestamps_per_channel.values()]
        common_timestamps = sorted(list(set.intersection(*timestamp_sets)))

        # Initialize an OrderedDict to store common timestamps and their corresponding data indices
        self.timestamps = OrderedDict()
        for timestamp in common_timestamps:
            self.timestamps[timestamp] = {}

        # Populate self.timestamps with data indices for each channel
        for channel in channels:
            for i, timestamp in enumerate(timestamps_per_channel[channel]):
                if timestamp in self.timestamps:
                    self.timestamps[timestamp][channel] = i

    @property
    def sensor_id(self):
        """
        Property returning the sensor ID.

        Returns:
            str: Sensor ID.
        """
        return "PHI"

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
        for channel in self.channels:
            file_idx = self.timestamps[timestamp][channel]
            file_path = self.files[channel][file_idx]
            with fits.open(file_path) as hdul:
                filedata = hdul[0].data
                angle = hdul[0].header["CROTA"]
                rotated_image = ndimage.rotate(filedata, -angle, reshape=True)
                data[channel] = rotated_image  # TODO: add centering of image

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
        return filename.name.split("_")[-2][
            :13
        ]  # TO-DO: reduced length of timestamp to enable matches (down to minutes), potential issue

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
