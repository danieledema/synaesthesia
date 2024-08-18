import pytest
from src.synaesthesia.abstract.filter_functions import (
    SkipNFilter,
    MultipleNFilter,
    ExponentialFilter,
)


# Test cases for SkipNFilter
@pytest.mark.parametrize(
    "skip_n, num_samples, expected_skip_n_indices",
    [
        (1, 5, [0, 2, 4, 6, 8]),   # Test with skip_n = 1
        (2, 5, [0, 3, 6, 9, 12]),  # Test with skip_n = 2
        (3, 4, [0, 4, 8, 12]),     # Test with skip_n = 3
        (0, 3, [0, 1, 2]),         # Test with skip_n = 0 (no skipping)
        (5, 3, [0, 6, 12]),        # Test with larger skip_n than n_samples
    ]
)
def test_skip_n_filter(skip_n, num_samples, expected_skip_n_indices):
    filter = SkipNFilter(skip_n=skip_n)
    assert filter.get_indices(n_samples=num_samples) == expected_skip_n_indices, f"Failed for skip_n={skip_n}, num_samples={num_samples}"


# Test cases for MultipleNFilter
@pytest.mark.parametrize(
    "multiple, num_samples, expected_multiple_indices",
    [
        (1, 5, [0, 1, 2, 3, 4]),   # Test with multiple = 1 (every index is included)
        (2, 5, [0, 2, 4, 6, 8]),   # Test with multiple = 2 (every second index)
        (3, 4, [0, 3, 6, 9]),      # Test with multiple = 3 (every third index)
        (5, 3, [0, 5, 10]),        # Test with multiple = 5 (every fifth index)
        (10, 3, [0, 10, 20]),      # Test with larger multiple than n_samples (larger gap between indices)
    ]
)
def test_multiple_n_filter(multiple, num_samples, expected_multiple_indices):
    filter = MultipleNFilter(multiple=multiple)
    assert filter.get_indices(n_samples=num_samples) == expected_multiple_indices, f"Failed for multiple={multiple}, num_samples={num_samples}"


# Test cases for ExponentialFilter
@pytest.mark.parametrize(
    "base, num_samples, expected_exponential_indices",
    [
        (2, 5, [1, 2, 4, 8, 16]),       # Base 2: 2^0, 2^1, 2^2, 2^3, 2^4
        (3, 4, [1, 3, 9, 27]),          # Base 3: 3^0, 3^1, 3^2, 3^3
        (1.5, 6, [1, 1, 2, 3, 5, 7]),   # Base 1.5: 1.5^0, 1.5^1, 1.5^2, 1.5^3, 1.5^4, 1.5^5
        (10, 3, [1, 10, 100]),          # Base 10: 10^0, 10^1, 10^2
        (0.5, 5, [1, 0, 0, 0, 0])      # Base 0.5: 0.5^0, 0.5^1, 0.5^2, 0.5^3, 0.5^4 (values become 0 due to integer casting)
    ]
)
def test_exponential_filter(base, num_samples, expected_exponential_indices):
    filter = ExponentialFilter(base=base)
    assert filter.get_indices(n_samples=num_samples) == expected_exponential_indices, f"Failed for base={base}, num_samples={num_samples}"
