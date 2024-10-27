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
