from abc import ABC, abstractmethod
from typing import List


class Filter(ABC):
    @abstractmethod
    def get_indices(self, n_samples: int) -> List[int]:
        """Generate the indices based on the filter strategy."""
        pass


class SkipNFilter(Filter):
    def __init__(self, skip_n: int):
        self.skip_n = skip_n

    def get_indices(self, n_samples: int) -> List[int]:
        return [i * (1 + self.skip_n) for i in range(n_samples)]


class MultipleNFilter(Filter):
    def __init__(self, multiple: int):
        self.multiple = multiple

    def get_indices(self, n_samples: int) -> List[int]:
        return [i * self.multiple for i in range(n_samples)]


class ExponentialFilter(Filter):
    def __init__(self, base: float):
        self.base = base

    def get_indices(self, n_samples: int) -> List[int]:
        return [int(self.base**i) for i in range(n_samples)]
