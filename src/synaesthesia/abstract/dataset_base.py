from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List

from torch.utils.data import Dataset

from ..utils import check_camel_case_format


class DatasetBase(Dataset, ABC):
    """
    Base class for all datasets. It is a subclass of torch.utils.data.Dataset.
    It is an abstract class and should be subclassed by all datasets.

    Methods:
    --------
    get_data(idx: int) -> Dict[str, Any]:
        Returns the data at the specified index as a dictionary, with the special key 'idx' containing
        the index and 'timestamp' containing the timestamp of the data.
        If the dataset has an ID, the keys in the dictionary should be prefixed with the ID.

    get_timestamp(idx: int) -> int:
        Returns the timestamp at the specified index. Raises an IndexError if the index is out of range.

    get_timestamp_idx(timestamp: int) -> int:
        Given a timestamp, returns the index corresponding to that timestamp in the dataset.
        Raises a ValueError if the timestamp is not found.

    Properties:
    -----------
    id : str
        Returns a string combining the name of the sensor the dataset is associated with.

    machine_name : str
        Returns the machine name associated with the dataset.

    timestamps : List[int]
        Returns a list of timestamps in the dataset.

    sensor_ids : List[str]
        Returns a list containing the sensor reading in the dataset.
        For example, for a dataset containing temperature and humidity readings, this would return
        ['temperature', 'humidity'].
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample at the specified index.

        Args:
            idx: Index of the data sample

        Returns:
            Dictionary containing the data sample with 'idx' and 'timestamp' keys,
            plus any additional data from get_data()
        """
        # Create base data sample with metadata
        data_sample = {"idx": idx, "timestamp": self.get_timestamp(idx)}

        # Get the actual data
        data = self.get_data(idx)

        # Validate no key conflicts
        conflicting_keys = set(data.keys()) & set(data_sample.keys())
        if conflicting_keys:
            raise ValueError(f"Duplicate keys found in data: {conflicting_keys}")

        # Prefix keys with dataset ID if available
        if self.id:
            data = {f"{self.id}-{key}": value for key, value in data.items()}

        # Merge data into sample
        data_sample.update(data)
        return data_sample

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def get_data(self, idx: int) -> Dict[str, Any]:
        """
        Get the raw data at the specified index.

        Args:
            idx: Index of the data sample

        Returns:
            Dictionary containing the raw data
        """
        pass

    @abstractmethod
    def get_timestamp(self, idx: int) -> int:
        """
        Get the timestamp at the specified index.

        Args:
            idx: Index of the data sample

        Returns:
            Timestamp as integer

        Raises:
            IndexError: If index is out of range
        """
        pass

    @abstractmethod
    def get_timestamp_idx(self, timestamp: int) -> int:
        """
        Get the index corresponding to a timestamp.

        Args:
            timestamp: Timestamp to find

        Returns:
            Index corresponding to the timestamp

        Raises:
            ValueError: If timestamp is not found
        """
        pass

    def __contains__(self, timestamp: int) -> bool:
        """
        Check if a timestamp exists in the dataset.

        Args:
            timestamp: Timestamp to check

        Returns:
            True if timestamp exists, False otherwise
        """
        try:
            self.get_timestamp_idx(timestamp)
            return True
        except (IndexError, KeyError, ValueError):
            return False

    @property
    @abstractmethod
    def sensor_ids(self) -> List[str]:
        """Return a list of sensor IDs in the dataset."""
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """Return the dataset identifier."""
        pass

    @property
    @lru_cache(maxsize=1)
    def machine_name(self) -> str:
        """
        Return the machine name associated with the dataset.
        Cached for efficiency since machine name shouldn't change.
        """
        machine_name = self.get_machine_name()
        check_camel_case_format(machine_name)
        return machine_name

    @abstractmethod
    def get_machine_name(self) -> str:
        """Return the raw machine name (to be validated)."""
        pass

    @property
    @abstractmethod
    def timestamps(self) -> List[int]:
        """Return a list of all timestamps in the dataset."""
        pass

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        sensor_list = " ".join(self.sensor_ids)
        return f"{self.machine_name} - {sensor_list}: {len(self)} samples"
