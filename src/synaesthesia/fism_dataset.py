from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, List, OrderedDict, Tuple, Union
import numpy as np
import netCDF4 as nc

from .abstract_dataset import DatasetBase
from .utils import convert_to_datetime


class PicklableLock:
    def __init__(self):
        self._lock = Lock()

    def acquire(self):
        self._lock.acquire()

    def release(self):
        self._lock.release()

    def __reduce__(self):
        return (self.__class__, ())


class FISMDataset(DatasetBase):
    """
    Dataset class for FISM data.
    """

    def __init__(
        self,
        folder_path: Union[str, Path],
        wavelength_range: Tuple[float, float] = (0, 200),
    ):
        """
        Initializes the FISM dataset.

        Args:
            folder_path (str | Path): The path to the folder containing the data files.
            wavelength_range (tuple[float, float]): A tuple containing the start and end wavelengths.
        """
        super().__init__()

        self.folder_path = Path(folder_path)
        self.wavelength_range = wavelength_range

        # Initialize an empty dictionary to hold file paths
        self.files = sorted(list(self.folder_path.rglob(f"*.nc")))

        # Dictionary to hold timestamps for each file
        self._timestamps = OrderedDict()
        for i, f in enumerate(self.files):
            timestamps = self.generate_timestamps_from_filename(f)
            for ts in timestamps:
                ts = convert_to_datetime(ts)
                self._timestamps[ts] = i
        timestamps_list = list(self._timestamps.keys())

        # Convert datetime objects to string format
        datetime_strings = [
            dt.strftime("%Y-%m-%dT%H:%M:%S.000000000") for dt in timestamps_list
        ]

        # Create numpy array with dtype 'datetime64[ns]'
        self.datetime_array = np.array(datetime_strings, dtype="datetime64[ns]")

        # Retrieve available wavelengths from the first file
        self.available_wavelengths = self.get_available_wavelengths(self.files[0])

        self.file_connections = []
        self.lock = PicklableLock()

    def open_connection(self):
        self.lock.acquire()
        if not self.file_connections:
            self.file_connections = [nc.Dataset(file, mode="r") for file in self.files]
        self.lock.release()

    def close_connection(self):
        self.lock.acquire()
        if not self.file_connections:
            for file in self.file_connections:
                file.close()
            self.file_connections = []
        self.lock.release()

    def __del__(self):
        self.close_connection()

    @staticmethod
    def get_timestamp_from_filename(filename: Path) -> datetime:
        """
        Extracts the year and day of year from the filename and returns the corresponding datetime object.
        """
        parts = filename.name.split("_")
        year = int(parts[-3][:4])
        day_of_year = int(parts[-3][4:])
        return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

    def generate_timestamps_from_filename(self, filename: Path) -> List[str]:
        """
        Generates a list of 1-minute resolution timestamps for the given filename.
        """
        start_datetime = self.get_timestamp_from_filename(filename)
        return [
            (start_datetime + timedelta(minutes=i)).strftime("%Y%m%dT%H%M%S")
            for i in range(24 * 60)
        ]

    @property
    def timestamps(self) -> List[datetime]:
        return self.datetime_array  # self.timestamps_list

    def __len__(self):
        return len(self.timestamps)

    def get_data(self, idx: int) -> Dict[float, float]:
        """
        Loads the irradiance data for the given timestamp index for all specified wavelengths.

        Args:
            idx (int): Index of the timestamp.

        Returns:
            dict: A dictionary containing the irradiance data for the given timestamp.
        """
        # Find the corresponding timestamp and file for the given index
        timestamp = self.timestamps[idx]
        file_idx = self._timestamps[timestamp]

        self.open_connection()
        nc_in = self.file_connections[file_idx]

        utc = nc_in.variables["utc"][:]
        irradiance = nc_in.variables["irradiance"][:]
        wavelength = nc_in.variables["wavelength"][:]

        # Convert timestamp to seconds of day
        timestamp_seconds = (
            timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        ).total_seconds()

        # Convert UTC values to seconds of day
        utc_seconds = [
            (utc_value % 86400) for utc_value in utc
        ]  # Modulo 86400 to handle overflow

        # Find closest match in UTC values
        closest_index = min(
            range(len(utc_seconds)),
            key=lambda i: abs(utc_seconds[i] - timestamp_seconds),
        )

        # Extract irradiance data for the wavelengths within the specified range
        start_wavelength, end_wavelength = self.wavelength_range
        wavelength_indices = [
            i
            for i, w in enumerate(wavelength)
            if start_wavelength <= w <= end_wavelength
        ]
        irradiance_data = {
            wavelength[i]: irradiance[closest_index, i] for i in wavelength_indices
        }

        return irradiance_data

    @property
    def id(self):
        return "FISM"

    @property
    def sensor_ids(self) -> List[str]:
        """
        Generates sensor IDs for the wavelengths within the specified range.

        Returns:
            List[str]: A list of sensor IDs.
        """
        start_wavelength, end_wavelength = self.wavelength_range
        valid_wavelengths = [
            w
            for w in self.available_wavelengths
            if start_wavelength <= w <= end_wavelength
        ]
        return [f"{self.id}-{w}" for w in valid_wavelengths]

    @property
    def satellite_name(self) -> str:
        return "FISM"

    def get_timestamp(self, idx):
        """
        Retrieves the timestamp at index `idx` from the dataset.

        Args:
            idx (int): Index of the timestamp to retrieve.

        Returns:
            str: Timestamp corresponding to the specified index.
        """
        return self.timestamps[idx]

    def get_available_wavelengths(self, file: Path) -> List[float]:
        """
        Retrieves the available wavelengths from a NetCDF file.

        Args:
            file (Path): The path to the NetCDF file.

        Returns:
            List[float]: A list of available wavelengths.
        """
        with nc.Dataset(file, mode="r") as nc_in:
            wavelengths = nc_in.variables["wavelength"][:]
        return wavelengths.tolist()

    def get_timestamp_idx(self, date_str: str) -> int:
        """
        Retrieves the index for the given timestamp.

        Args:
            date_str (str): The timestamp to find the index for, in the format 'YYYYMMDDTHHMMSS'.

        Returns:
            int: The index of the given timestamp in the dataset.

        Raises:
            ValueError: If the timestamp is not found in the dataset.
        """
        date_time = convert_to_datetime(date_str)
        try:
            return self.timestamps.index(date_time)
        except ValueError as e:
            raise ValueError(
                f"Timestamp {date_str} is not found in the dataset."
            ) from e
