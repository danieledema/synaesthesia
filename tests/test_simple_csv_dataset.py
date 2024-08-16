from .simple_csv_dataset import SimpleCsvDataset
from .vigil2.data.abstract.conversion import convert_to_string, convert_to_timestamp


def test_simple_csv_dataset_length():
    path = "tests/test_data/test_data_10_s.csv"

    dataset = SimpleCsvDataset(path)

    print(f"Checking dataset length: {len(dataset)}")
    assert len(dataset) == 30


def test_simple_csv_dataset_data0():
    path = "tests/test_data/test_data_10_s.csv"

    dataset = SimpleCsvDataset(path)

    print(f"Checking dataset[0]: {dataset[0]}")
    assert dataset[0] == {
        "idx": 0,
        "timestamp": convert_to_timestamp("20220101T000000000"),
        "CSV-random_integer1": 5,
        "CSV-index_1.5": 1.5,
    }


def test_simple_csv_dataset_timestamps():
    path = "tests/test_data/test_data_10_s.csv"

    dataset = SimpleCsvDataset(path)

    print("Checking timestamps:")
    assert type(dataset.timestamps) == list
    assert len(dataset.timestamps) == 30
    assert convert_to_string(dataset.timestamps[0]) == "20220101T000000000"
    assert convert_to_string(dataset.timestamps[-1]) == "20220101T000450000"


def test_simple_csv_dataset_timestamp_idx():
    path = "tests/test_data/test_data_10_s.csv"

    dataset = SimpleCsvDataset(path)

    print("Checking timestamps idx:")
    assert dataset.get_timestamp_idx(convert_to_timestamp("20220101T000000000")) == 0
    assert dataset.get_timestamp_idx(convert_to_timestamp("20220101T000450000")) == 29
