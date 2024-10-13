import pytest
from pathlib import Path
from synaesthesia.abstract.concat_dataset import CustomConcatDataset
from .simple_csv_dataset import SimpleCsvDataset
import numpy as np


# Determine the path to the directory of the current script
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH_10 = BASE_DIR / "test_data" / "test_data_10_s.csv"
DATA_PATH_10_concat_test = BASE_DIR / "test_data" / "test_data_10_s_concat_test.csv"


# Adding datasets fixtures
@pytest.fixture
def datasets():
    """
    Fixture to load datasets for testing.
    """
    dataset1 = SimpleCsvDataset(DATA_PATH_10)
    dataset2 = SimpleCsvDataset(DATA_PATH_10_concat_test)
    return [dataset1, dataset2]


@pytest.fixture
def custom_concat_dataset(datasets):
    """
    Fixture to create a CustomConcatDataset from the loaded datasets.
    """
    return CustomConcatDataset(datasets)


# Tests
def test_custom_concat_dataset_length(datasets, custom_concat_dataset):
    # Check the total length of the concatenated dataset
    expected_length = len(datasets[0]) + len(datasets[1])
    assert (
        len(custom_concat_dataset) == expected_length
    ), f"Expected length {expected_length}, but got {len(custom_concat_dataset)}"


def test_custom_concat_dataset_get_data(datasets, custom_concat_dataset):
    # Test getting data from the concatenated dataset
    index_1 = 5
    expected_data = datasets[0].get_data(index_1)
    actual_data = custom_concat_dataset.get_data(len(datasets[0]) + index_1)
    assert np.array_equal(
        actual_data["random_integer1"], expected_data["random_integer1"]
    ), "Mismatch in random_integer1"
    assert np.isclose(
        actual_data["index_1.5"], expected_data["index_1.5"]
    ), "Mismatch in index_1.5"

    index_2 = 2
    expected_data = datasets[1].get_data(index_2)
    actual_data = custom_concat_dataset.get_data(len(datasets[0]) + index_2)
    assert np.array_equal(
        actual_data["random_integer1"], expected_data["random_integer1"]
    ), "Mismatch in random_integer1"
    assert np.isclose(
        actual_data["index_1.5"], expected_data["index_1.5"]
    ), "Mismatch in index_1.5"


def test_custom_concat_dataset_get_timestamp(datasets, custom_concat_dataset):
    # Test getting timestamp from the concatenated dataset
    index_1 = 5
    expected_timestamp = datasets[0].get_timestamp(
        index_1
    )  # Get timestamp from first dataset
    actual_timestamp = custom_concat_dataset.get_timestamp(index_1)
    assert (
        actual_timestamp == expected_timestamp
    ), f"Timestamp mismatch at index 5: expected {expected_timestamp}, got {actual_timestamp}"

    index_2 = 2
    expected_timestamp = datasets[1].get_timestamp(
        2
    )  # Get timestamp from second dataset
    actual_timestamp = custom_concat_dataset.get_timestamp(
        len(datasets[0]) + index_2
    )  # Corresponding index
    assert (
        actual_timestamp == expected_timestamp
    ), f"Timestamp mismatch at index 32: expected {expected_timestamp}, got {actual_timestamp}"


def test_custom_concat_dataset_find_right_dataset(custom_concat_dataset):
    # Test finding the right dataset and index
    i, idx = custom_concat_dataset.find_right_datset(5)
    assert i == 0 and idx == 5, "Expected to find index 5 in dataset 0"

    i, idx = custom_concat_dataset.find_right_datset(32)
    assert i == 1 and idx == 2, "Expected to find index 2 in dataset 1"


def test_custom_concat_dataset_repr(custom_concat_dataset):
    # Test the string representation of the concatenated dataset
    repr_str = repr(custom_concat_dataset)
    assert "Concat dataset: 60 samples" in repr_str
    assert "Datasets: 2" in repr_str


def test_custom_concat_dataset_negative_index(custom_concat_dataset):
    index = -1
    expected_data = custom_concat_dataset.get_data(len(custom_concat_dataset) - 1)
    actual_data = custom_concat_dataset.get_data(index)
    assert np.array_equal(
        actual_data["random_integer1"], expected_data["random_integer1"]
    ), "Mismatch in random_integer1"
