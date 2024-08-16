from .simple_csv_dataset import SimpleCsvDataset
from .vigil2.data.abstract.conversion import convert_to_string
from .vigil2.data.abstract.sequential_dataset import SequentialDataset


def test_sequential_sensor_dataset_skip0():
    data_path = "tests/test_data/test_data_10_s.csv"
    dataset = SimpleCsvDataset(data_path)

    sensor_dataset = SequentialDataset(
        dataset, n_samples=5, skip_n=0, timestamp_idx="last"
    )

    print(f"Checking sensor_dataset length: {len(sensor_dataset)}")
    assert len(sensor_dataset) == 26

    print(f"Checking sensor_dataset[0]: {sensor_dataset[0]}")
    assert convert_to_string(sensor_dataset[0]["timestamp"]) == "20220101T000040000"
    assert sensor_dataset[0]["CSV-random_integer1"] == [5, 9, 3, 7, 2]
    assert sensor_dataset[0]["CSV-index_1.5"] == [1.5, 3.0, 4.5, 6.0, 7.5]

    print(f"Checking sensor_dataset[1]: {sensor_dataset[1]}")
    assert convert_to_string(sensor_dataset[1]["timestamp"]) == "20220101T000050000"
    assert sensor_dataset[1]["CSV-random_integer1"] == [9, 3, 7, 2, 6]
    assert sensor_dataset[1]["CSV-index_1.5"] == [3.0, 4.5, 6.0, 7.5, 9.0]


def test_sequential_sensor_dataset_skip1():
    data_path = "tests/test_data/test_data_10_s.csv"
    dataset = SimpleCsvDataset(data_path)

    sensor_dataset = SequentialDataset(
        dataset, n_samples=5, skip_n=1, timestamp_idx="last"
    )

    print(f"Checking sensor_dataset length: {len(sensor_dataset)}")
    assert len(sensor_dataset) == 22

    print(f"Checking sensor_dataset[0]: {sensor_dataset[0]}")
    assert convert_to_string(sensor_dataset[0]["timestamp"]) == "20220101T000120000"
    assert sensor_dataset[0]["CSV-random_integer1"] == [5, 3, 2, 8, 1]
    assert sensor_dataset[0]["CSV-index_1.5"] == [1.5, 4.5, 7.5, 10.5, 13.5]

    print(f"Checking sensor_dataset[1]: {sensor_dataset[1]}")
    assert convert_to_string(sensor_dataset[1]["timestamp"]) == "20220101T000130000"
    assert sensor_dataset[1]["CSV-random_integer1"] == [9, 7, 6, 4, 10]
    assert sensor_dataset[1]["CSV-index_1.5"] == [3.0, 6.0, 9.0, 12.0, 15.0]


def test_sequential_sensor_dataset_skip1_stride5():
    data_path = "tests/test_data/test_data_10_s.csv"
    dataset = SimpleCsvDataset(data_path)

    sensor_dataset = SequentialDataset(
        dataset, n_samples=5, skip_n=1, stride=5, timestamp_idx="last"
    )

    for i in sensor_dataset:
        print(i)

    print(f"Checking sensor_dataset length: {len(sensor_dataset)}")
    assert len(sensor_dataset) == 5

    print(f"Checking sensor_dataset[0]: {sensor_dataset[0]}")
    assert convert_to_string(sensor_dataset[0]["timestamp"]) == "20220101T000120000"
    assert sensor_dataset[0]["CSV-random_integer1"] == [5, 3, 2, 8, 1]
    assert sensor_dataset[0]["CSV-index_1.5"] == [1.5, 4.5, 7.5, 10.5, 13.5]

    print(f"Checking sensor_dataset[1]: {sensor_dataset[1]}")
    assert convert_to_string(sensor_dataset[1]["timestamp"]) == "20220101T000210000"
    assert sensor_dataset[1]["CSV-random_integer1"] == [6, 4, 10, 9, 7]
    assert sensor_dataset[1]["CSV-index_1.5"] == [9.0, 12.0, 15.0, 18.0, 21.0]


def test_sequential_sensor_dataset_skip0_stride5():
    data_path = "tests/test_data/test_data_10_s.csv"
    dataset = SimpleCsvDataset(data_path)

    sensor_dataset = SequentialDataset(
        dataset, n_samples=5, skip_n=0, stride=5, timestamp_idx="last"
    )

    for i in sensor_dataset:
        print(i)

    print(f"Checking sensor_dataset length: {len(sensor_dataset)}")
    assert len(sensor_dataset) == 6

    print(f"Checking sensor_dataset[0]: {sensor_dataset[0]}")
    assert convert_to_string(sensor_dataset[0]["timestamp"]) == "20220101T000040000"
    assert sensor_dataset[0]["CSV-random_integer1"] == [5, 9, 3, 7, 2]
    assert sensor_dataset[0]["CSV-index_1.5"] == [1.5, 3.0, 4.5, 6.0, 7.5]

    print(f"Checking sensor_dataset[1]: {sensor_dataset[1]}")
    assert convert_to_string(sensor_dataset[1]["timestamp"]) == "20220101T000130000"
    assert sensor_dataset[1]["CSV-random_integer1"] == [6, 8, 4, 1, 10]
    assert sensor_dataset[1]["CSV-index_1.5"] == [9.0, 10.5, 12.0, 13.5, 15.0]
