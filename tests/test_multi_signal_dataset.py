from src.synaesthesia.abstract.conversion import convert_to_string
from src.synaesthesia.abstract.multi_signal_dataset import MultiSignalDataset

from .simple_csv_dataset import SimpleCsvDataset


def test_simple_csv_dataset_both():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    assert len(dataset1) == 30
    assert len(dataset2) == 30


def test_multi_signal_dataset_all_none():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "all", "none")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 50

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000010000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000020000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000030000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 3.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == None

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == None


def test_multi_signal_dataset_common_none():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "common", "none")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 10

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000030000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000100000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000130000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 7

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 6.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == 2


def test_multi_signal_dataset_I0_none():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "I:0", "none")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 30

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000010000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000020000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000030000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 3.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == None

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == None


def test_multi_signal_dataset_I1_none():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "I:1", "none")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 30

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000030000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000100000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000130000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 7

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 6.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == 2


def test_multi_signal_dataset_common_last():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "common", "none")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 10

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000030000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000100000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000130000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 7

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 6.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == 2


def test_multi_signal_dataset_I0_last():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "I:0", "last")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 30

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000010000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000020000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000030000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 3.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == 5

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == 1


def test_multi_signal_dataset_I1_last():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "I:1", "last")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 30

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000030000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000100000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000130000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 7

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 6.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == 2


def test_multi_signal_dataset_common_closest():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "common", "closest")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 10

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000030000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000100000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000130000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 7

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 6.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == 2


def test_multi_signal_dataset_I0_closest():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "I:0", "closest")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 30

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000010000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000020000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000030000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 3.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == 5

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == 1


def test_multi_signal_dataset_I1_closest():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1, 'Left_arm')
    dataset2 = SimpleCsvDataset(data_path_2, 'Left_arm')

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "I:1", "closest")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 30

    print(f"Checking timestamps")
    assert convert_to_string(multi_dataset[0]["timestamp"]) == "20220101T000000000"
    assert convert_to_string(multi_dataset[1]["timestamp"]) == "20220101T000030000"
    assert convert_to_string(multi_dataset[2]["timestamp"]) == "20220101T000100000"
    assert convert_to_string(multi_dataset[3]["timestamp"]) == "20220101T000130000"

    print(f"Checking dataset keys {multi_dataset[0].keys()}")
    assert set(multi_dataset[0].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    print(f"Checking dataset keys {multi_dataset[1].keys()}")
    assert set(multi_dataset[1].keys()) == set(
        [
            "timestamp",
            "idx",
            "FAKE_CSV-random_integer1",
            "FAKE_CSV-index_1.5",
            "FAKE_CSV-random_integer2",
            "FAKE_CSV-index_power_2",
        ]
    )

    assert multi_dataset[0]["FAKE_CSV-random_integer1"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer1"] == 7

    assert multi_dataset[0]["FAKE_CSV-index_1.5"] == 1.5
    assert multi_dataset[1]["FAKE_CSV-index_1.5"] == 6.0

    assert multi_dataset[0]["FAKE_CSV-random_integer2"] == 5
    assert multi_dataset[1]["FAKE_CSV-random_integer2"] == 9

    assert multi_dataset[0]["FAKE_CSV-index_power_2"] == 1
    assert multi_dataset[1]["FAKE_CSV-index_power_2"] == 2
