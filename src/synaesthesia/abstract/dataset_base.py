from typing import Any

from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __getitem__(self, idx):
        data_sample = {"idx": idx, "timestamp": self.get_timestamp(idx)}

        data = self.get_data(idx)
        for key in data:
            assert key not in data_sample, f"Duplicate key {key} in data_sample"

        if self.id:
            data = {f"{self.id}-{key}": data[key] for key in data}

        data_sample |= data
        return data_sample

    def __len__(self) -> int:
        raise NotImplementedError

    def get_data(self, idx) -> dict[str, Any]:
        raise NotImplementedError

    def get_timestamp(self, idx) -> int:
        raise NotImplementedError

    def get_timestamp_idx(self, timestamp) -> int:
        raise NotImplementedError

    def __contains__(self, t) -> bool:
        try:
            _ = self.get_timestamp_idx(t)
            return True
        except (IndexError, KeyError, ValueError):
            return False

    @property
    def sensor_ids(self) -> list[str]:
        raise NotImplementedError

    @property
    def id(self) -> str:
        raise NotImplementedError

    @property
    def satellite_name(self) -> str:
        raise NotImplementedError

    @property
    def timestamps(self) -> list[int]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.satellite_name} - {' '.join(self.sensor_ids)}: {len(self)} samples"
        )
