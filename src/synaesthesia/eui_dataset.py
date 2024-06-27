from pathlib import Path

from .abstract_dataset import SingleSignalDatasetBase


class EuiDataset(SingleSignalDatasetBase):
    def __init__(self, folder_path: str | Path):
        super().__init__()

        self.folder_path = Path(folder_path)

    @property
    def sensor_id(self):
        return "eui"

    def __len__(self):
        raise NotImplementedError

    def get_data(self, idx):
        raise NotImplementedError

    def get_timestamp(self, idx):
        raise NotImplementedError

    def get_timestamp_idx(self, timestamp):
        raise NotImplementedError
