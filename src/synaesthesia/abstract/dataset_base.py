from typing import Any

from torch.utils.data import Dataset
from ..utils import check_camel_case_format


class DatasetBase(Dataset):
    """
    Base class for all datasets. It is a subclass of torch.utils.data.Dataset.
    It is an abstract class and should be subclassed by all datasets.

    Methods:
    --------
    get_data(idx: int) -> dict[str, Any]:
        Returns the data at the specified index as a dictionary, with the special key 'idx' containing
        the index and 'timestamp' containing the timestamp of the data.
        If the dataset has an ID, the keys in the dictionary should be prefixed with the ID.

    get_timestamp(idx: int) -> int:
        Returns the timestamp at the specified index. Raises an IndexError if the index is out of range.

    get_timestamp_idx(timestamp: int) -> int:
        Given a timestamp, returns the index corresponding to that timestamp in the dataset.
        Raises a ValueError if the timestamp is not found.

    id -> str (property):
        Returns a string combining the name of the sensor the dataset is associated with.

    machine_name -> str (property):
        Returns the machine name associated with the dataset.

    timestamps -> list[int] (property):
        Returns a list of timestamps in the dataset.

    sensor_ids -> list[str] (property):
        Returns a list containing the sensor reading in the dataset.
        For example, for a dataset containing temperature and humidity readings, this would return
        ['temperature', 'humidity'].
    """

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
    def machine_name(self) -> str:
        machine_name = self.get_machine_name()
        check_camel_case_format(machine_name)
        return machine_name

    def get_machine_name(self) -> str:
        raise NotImplementedError

    @property
    def timestamps(self) -> list[int]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.machine_name} - {' '.join(self.sensor_ids)}: {len(self)} samples"
