import pytest
from pathlib import Path
from src.synaesthesia.abstract.sequential_dataset import SequentialDataset
from .simple_csv_dataset import SimpleCsvDataset
from src.synaesthesia.abstract.filter_functions import SkipNFilter
from datetime import datetime

# Determine the path to the directory of the current script
BASE_DIR = Path(__file__).resolve().parent
# Construct the path to the CSV file relative to the script's directory
DATA_PATH = BASE_DIR / "test_data" / "test_data_10_s.csv"


def convert_timestamp_list_to_unix(timestamps):
    """
    Convert a list of timestamp strings in the format "YYYYMMDDTHHMMSSmmm"
    to Unix epoch time (integer format).

    :param timestamps: List of timestamp strings
    :return: List of Unix epoch timestamps (integers)
    """
    return [
        int(datetime.strptime(ts, "%Y%m%dT%H%M%S%f").timestamp()) for ts in timestamps
    ]


@pytest.fixture
def ground_truth_data():
    """
    Fixture to provide ground truth data for tests.
    """
    return [
        {
            "timestamps": convert_timestamp_list_to_unix(
                ["20220101T000000000", "20220101T000020000", "20220101T000040000"]
            ),
            "random_integer1": [5, 3, 2],
            "index_1.5": [1.5, 4.5, 7.5],
        },
        {
            "timestamps": convert_timestamp_list_to_unix(
                ["20220101T000030000", "20220101T000050000", "20220101T000110000"]
            ),
            "random_integer1": [7, 6, 4],
            "index_1.5": [6.0, 9.0, 12.0],
        },
        {
            "timestamps": convert_timestamp_list_to_unix(
                ["20220101T000100000", "20220101T000120000", "20220101T000140000"]
            ),
            "random_integer1": [8, 1, 5],
            "index_1.5": [10.5, 13.5, 16.5],
        },
    ]


@pytest.fixture
def expected_timestamps(ground_truth_data):
    """
    Fixture to provide expected timestamps for tests.
    """
    return {
        "first": [data["timestamps"][0] for data in ground_truth_data],
        "last": [data["timestamps"][-1] for data in ground_truth_data],
    }


@pytest.fixture
def common_setup():
    """
    Fixture for common setup shared by tests.
    """
    dataset = SimpleCsvDataset(DATA_PATH)
    skip_filter = SkipNFilter(skip_n=1)
    return dataset, skip_filter


def test_sequential_dataset_data_values(common_setup, ground_truth_data):
    dataset, skip_filter = common_setup

    # Initialize SequentialDataset with the SkipNFilter, stride=3, and timestamp_idx set to "first"
    sensor_dataset = SequentialDataset(
        dataset,
        n_samples=3,  # Set n_samples to 3
        filter=skip_filter,
        stride=3,
        timestamp_idx="first",
        return_timestamps=True,
    )

    # Validate the data returned by the dataset
    for idx, expected in enumerate(ground_truth_data):
        actual = sensor_dataset.get_data(idx)
        assert (
            actual == expected
        ), f"Data mismatch at index {idx}: expected {expected}, got {actual}"


def test_sequential_dataset_timestamp_indexing(common_setup, expected_timestamps):
    dataset, skip_filter = common_setup

    for timestamp_idx, expected_timestamps_for_idx in expected_timestamps.items():
        # Initialize SequentialDataset with the SkipNFilter, stride=3, and timestamp_idx
        sensor_dataset = SequentialDataset(
            dataset,
            n_samples=3,  # Set n_samples to 3
            filter=skip_filter,
            stride=3,
            timestamp_idx=timestamp_idx,
            return_timestamps=True,
        )

        # Validate that the correct timestamp is returned based on timestamp_idx
        for idx, expected_timestamp in enumerate(expected_timestamps_for_idx):
            actual_data = sensor_dataset.__getitem__(idx)
            actual_timestamp = actual_data["timestamp"]
            assert (
                actual_timestamp == expected_timestamp
            ), f"Timestamp mismatch at index {idx}: expected {expected_timestamp}, got {actual_timestamp}"


def test_sequential_dataset_length(common_setup):
    dataset, skip_filter = common_setup

    # Initialize SequentialDataset with the SkipNFilter, stride=3, and timestamp_idx
    sensor_dataset = SequentialDataset(
        dataset,
        n_samples=3,  # Set n_samples to 3
        filter=skip_filter,
        stride=3,
        timestamp_idx="first",  # Using "first" just as an example; doesn't affect length
    )

    # Validate dataset length
    assert (
        len(sensor_dataset) == 9
    ), f"Expected dataset length 9, but got {len(sensor_dataset)}"


def test_sequential_dataset_idxs(common_setup):
    dataset, skip_filter = common_setup

    # Initialize SequentialDataset with different parameters
    sensor_dataset = SequentialDataset(
        dataset,
        n_samples=3,  # Set n_samples to 3
        filter=skip_filter,
        stride=3,
        timestamp_idx="first",
        return_timestamps=True,
    )

    # Expected indices based on the dataset length and stride
    expected_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24]

    # Validate idxs property
    actual_idxs = sensor_dataset.idxs

    assert (
        actual_idxs == expected_idxs
    ), f"Expected idxs {expected_idxs}, but got {actual_idxs}"
