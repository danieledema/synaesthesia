from pathlib import Path
from typing import Any

from ..abstract.dataset_base import DatasetBase


class MultiFileDataset(DatasetBase):
    def __init__(self, folder_path: str | Path, extension: str):
        super().__init__()

        self.folder_path = Path(folder_path)
        self.extension = extension

        files = self.folder_path.glob(f"*.{self.extension}")
        self.files = list(files)
        self.files.sort()

        self._timestamps = [self.parse_filename(f) for f in self.files]
        self.data_dict = {t: f for t, f in zip(self._timestamps, self.files)}

    def parse_filename(self, filename) -> int:
        raise NotImplementedError

    @property
    def timestamps(self):
        return self._timestamps

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_data(self, idx) -> dict[str, Any]:
        timestamp = self.get_timestamp(idx)
        data = self.read_data(self.data_dict[timestamp])
        return data

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp):
        try:
            return self._timestamps.index(timestamp)
        except ValueError:
            raise ValueError("Timestamp not found in dataset")
