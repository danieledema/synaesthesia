from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from ..abstract.dataset_base import DatasetBase


class MultiSpectralImagerDataset(DatasetBase):
    """
    Dataset class for Multispectral Imager data.
    It considers each wavelenght as a different sensor to combine.
    """

    def __init__(
        self,
        folder_path: str | Path,
        wavelengths: list[str],
        time_threshold: int | None = 60,
        remove_incomplete: bool = True,
        remove_duplicates: bool = True,
        duplicate_threshold: int = 10,
        already_aligned: bool = False,
    ):
        super().__init__()

        self.folder_path = Path(folder_path)
        self.wavelengths = wavelengths

        files: list[Path] = self.collect_files()
        self.data_dict = {}
        self._timestamps = []

        logger.info(f"Loading data for wavelengths {self.wavelengths}")

        last_timestamp = 0
        for file in files:
            timestamp, wavelength = self.parse_filename(file)

            if not wavelength == self.wavelengths[0]:
                continue

            if remove_duplicates and timestamp - last_timestamp < duplicate_threshold:
                continue

            self.data_dict[timestamp] = {wavelength: file}
            self._timestamps.append(timestamp)
            last_timestamp = timestamp

        for wavelength in tqdm(
            self.wavelengths[1:],
            desc=f"Loading data for wavelengths {self.wavelengths}",
        ):
            if already_aligned:
                for timestamp in self._timestamps:
                    filename = self.filename_from_timestamp(
                        timestamp, wavelength, self.folder_path
                    )
                    self.data_dict[timestamp][wavelength] = filename

                continue

            last_timestamp_idx = 0
            for file in tqdm(files):
                timestamp, wl = self.parse_filename(file)

                if not wl == wavelength:
                    continue

                idx_prior = max(0, last_timestamp_idx - 1)
                idx = last_timestamp_idx
                idx_next = min(len(self._timestamps) - 1, last_timestamp_idx + 1)

                dt_prior = abs(timestamp - self._timestamps[idx_prior])
                dt = abs(timestamp - self._timestamps[idx])
                dt_next = abs(self._timestamps[idx_next] - timestamp)

                idx_closest = idx
                if dt_prior <= dt and dt_prior < dt_next:
                    idx_closest = idx_prior
                elif dt_next < dt and dt_next < dt_prior:
                    idx_closest = idx_next
                else:
                    idx_closest = idx

                if abs(self._timestamps[idx_closest] - timestamp) < time_threshold:
                    self.data_dict[self._timestamps[idx_closest]][wavelength] = file
                    last_timestamp_idx = idx_closest
                else:
                    self.data_dict[timestamp] = {wavelength: file}

                    while self._timestamps[last_timestamp_idx] < timestamp:
                        last_timestamp_idx += 1
                    while self._timestamps[last_timestamp_idx] > timestamp:
                        last_timestamp_idx -= 1
                    self._timestamps.insert(last_timestamp_idx, timestamp)

        if remove_incomplete:
            for timestamp in list(self.data_dict.keys()):
                if len(self.data_dict[timestamp]) != len(wavelengths):
                    del self.data_dict[timestamp]

        self._timestamps = list(self.data_dict.keys())

    def collect_files(self) -> list[Path]:
        raise NotImplementedError

    @property
    def timestamps(self):
        return self._timestamps

    def __len__(self) -> int:
        """
        Returns the number of common timestamps available in the dataset.

        Returns:
            int: Number of common timestamps.
        """
        return len(self.timestamps)

    def get_data(self, idx) -> dict[str, Any]:
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
            file_path = self.data_dict[timestamp][wavelength]
            data[f"{wavelength}"] = self.read_data(file_path)
        return data

    def read_data(self, file_path: Path) -> Any:
        raise NotImplementedError

    def filename_from_timestamp(
        self, timestamp: int, wavelength: str, folder_path: Path
    ) -> Path:
        raise NotImplementedError

    def parse_filename(self, filename) -> tuple[int, str]:
        raise NotImplementedError

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
        return [f"{w}" for w in self.wavelengths]
