from torch.utils.data.dataloader import DataLoader

from src.synaesthesia.abstract.multi_signal_dataset import MultiSignalDataset
from src.synaesthesia.abstract.sequential_dataset import SequentialDataset
from src.synaesthesia.collates import *

from .simple_csv_dataset import SimpleCsvDataset


def test_batch_collate():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1)
    dataset2 = SimpleCsvDataset(data_path_2)

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "common", "none")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 10

    dataloader = DataLoader(multi_dataset, batch_size=2, collate_fn=BatchCollate())
    batch = next(iter(dataloader))

    print(f"Checking batch shape: {batch}")


def test_simple_sequential_data():
    data_path = "tests/test_data/test_data_10_s.csv"
    dataset = SimpleCsvDataset(data_path)
    dataset = SequentialDataset(dataset, 3)

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=BatchCollate())
    batch = next(iter(dataloader))

    print(f"Checking batch shape: {batch}")
    assert batch["CSV-random_integer1"].shape == (2, 3)


def test_simple_sequential_data_with_strings():
    data_path = "tests/test_data/test_data_10_s.csv"
    dataset = SimpleCsvDataset(data_path)
    dataset = SequentialDataset(dataset, 3)

    data = [dataset[0], dataset[1]]
    for i in range(2):
        data[i]["idx"] = str(data[i]["idx"])

    collate = BatchCollate()
    batch = collate(data)

    print(f"Checking batch shape: {batch}")
    assert batch["CSV-random_integer1"].shape == (2, 3)


def test_batch_collate_with_none():
    data_path_1 = "tests/test_data/test_data_10_s.csv"
    data_path_2 = "tests/test_data/test_data_30_s.csv"

    dataset1 = SimpleCsvDataset(data_path_1)
    dataset2 = SimpleCsvDataset(data_path_2)

    multi_dataset = MultiSignalDataset([dataset1, dataset2], "all", "none")

    print(f"Checking length of dataset: {len(multi_dataset)}")
    assert len(multi_dataset) == 50

    dataloader = DataLoader(multi_dataset, batch_size=2, collate_fn=BatchCollate())
    batch = next(iter(dataloader))

    print(f"Checking batch shape: {batch}")


def test_batch_collate_with_none_and_images():
    data = [
        {"a": 0, "b": torch.rand(3, 10, 10), "c": "a"},
        {"a": None, "b": None, "c": None},
    ]

    collate = BatchCollate()
    batch = collate(data)

    print(f"Checking batch shape: {batch}")
    assert batch["a"].shape == (2,)
    assert batch["b"].shape == (2, 3, 10, 10)
    assert len(batch["c"]) == 2

    assert batch["a"][0] == 0 and torch.isnan(batch["a"][1])
    assert not torch.isnan(batch["b"][0]).any()
    assert torch.isnan(batch["b"][1]).all()
    assert batch["c"][0] == "a" and torch.isnan(batch["c"][1])
