from typing import Any

from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __getitem__(self, idx):
        data_sample = {
            "idx": idx,
            "timestamp": self.get_timestamp(idx),
        }
        data_sample |= (self.get_data(idx),)

    def __len__(self):
        raise NotImplementedError

    def get_data(self, idx) -> dict[str, Any]:
        raise NotImplementedError

    def get_timestamp(self, idx):
        raise NotImplementedError

    def get_timestamp_idx(self, timestamp):
        raise NotImplementedError

    def __contains__(self, t):
        try:
            _ = self.get_timestamp_idx(t)
            return True
        except (IndexError, KeyError):
            return False

    @property
    def sensor_id(self):
        raise NotImplementedError

    @property
    def satellite_name(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.satellite_name} - {self.sensor_id}: {len(self)} samples"
