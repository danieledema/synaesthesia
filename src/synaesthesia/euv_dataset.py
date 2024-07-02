from pathlib import Path

from astropy.io import fits

from .abstract_dataset import SingleSignalDatasetBase


class EuvDataset(SingleSignalDatasetBase):
    def __init__(self, folder_path: str | Path, level: int = 2):
        super().__init__()

        self.folder_path = Path(folder_path)
        self.level = level

        self.files = list(self.folder_path.glob(f"L{level}/**/*.fits"))
        self.files.sort()

    @property
    def sensor_id(self):
        return "EUV"

    @property
    def satellite_name(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def get_data(self, idx):
        with fits.open(self.files[idx]) as hdul:
            data = hdul[1].data
        return data

    def get_timestamp(self, idx):
        filename = self.files[idx].name
        return filename.split("_")[-2]

    def get_timestamp_idx(self, timestamp):
        raise NotImplementedError
