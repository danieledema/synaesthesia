from torch.utils.data import Dataset


class SingleSignalDatasetBase(Dataset):
    def __getitem__(self, idx):
        return {
            "idx": idx,
            "timestamp": self.get_timestamp(idx),
            "data": self.get_data(idx),
        }

    def __len__(self):
        raise NotImplementedError

    def get_data(self, idx):
        raise NotImplementedError

    def get_timestamp(self, idx):
        raise NotImplementedError

    def get_timestamp_idx(self, timestamp):
        raise NotImplementedError

    def __contains__(self, t):
        try:
            _ = self.get_timestamp_idx(t)
            return True
        except (IndexError, KeyError, ValueError):
            return False

    @property
    def sensor_id(self):
        raise NotImplementedError

    @property
    def satellite_name(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.satellite_name} - {self.sensor_id}: {len(self)} samples"
