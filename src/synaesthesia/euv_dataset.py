from pathlib import Path

from astropy.io import fits

from .abstract_dataset import SingleSignalDatasetBase
from collections import OrderedDict


class EuvDataset(SingleSignalDatasetBase):
    def __init__(self, folder_path: str | Path, wavelengths: list[str], level: int = 2):
        super().__init__()

        self.folder_path = Path(folder_path)
        self.wavelengths = wavelengths
        self.level = level

        self.files = {}
        for wavelength in wavelengths:
            self.files[wavelength] = sorted(
                list(self.folder_path.glob(f"L{level}/**/*fsi{wavelength}*.fits"))
            )

        timestamps_per_wavelength = {}
        print(
            "Warning: reduced resolution of timestamp in get_timestamp_from_filename to enable matches!"
        )

        for wavelength in wavelengths:
            timestamps_per_wavelength[wavelength] = [
                self.get_timestamp_from_filename(f) for f in self.files[wavelength]
            ]

        timestamp_sets = [set(i) for i in timestamps_per_wavelength.values()]
        common_timestamps = sorted(list(set.intersection(*timestamp_sets)))

        self.timestamps = OrderedDict()
        for timestamp in common_timestamps:
            self.timestamps[timestamp] = {}

        for wavelength in wavelengths:
            for i, timestamp in enumerate(timestamps_per_wavelength[wavelength]):
                if timestamp in self.timestamps:
                    self.timestamps[timestamp][wavelength] = i

    @property
    def sensor_id(self):
        return "EUV"

    def __len__(self):
        return len(self.timestamps)

    def get_data(self, idx):
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
        return filename.name.split("_")[-2][:13]

    def get_timestamp(self, idx):
        return list(self.timestamps.keys())[idx]

    def get_timestamp_idx(self, timestamp):
        try:
            return list(self.timestamps.keys()).index(timestamp)
        except ValueError:
            raise ValueError("Timestamp not found in dataset")
