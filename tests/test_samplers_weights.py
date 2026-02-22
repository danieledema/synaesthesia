import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.synaesthesia.samplers import calculate_class_weights


class SimpleLabelDataset(Dataset):
    """
    Minimal dataset that yields a dict with 'idx' and a class label key.
    'labels' is a list of integer class labels.
    """

    def __init__(self, labels, label_key="label"):
        self.labels = list(labels)
        self.label_key = label_key

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            self.label_key: torch.tensor(self.labels[idx], dtype=torch.long),
        }


def test_calculate_class_weights_with_zero_count_class():
    # Build a dataset where class 1 is absent.
    # Labels distribution: class0 -> 3 samples, class1 -> 0 samples, class2 -> 1 sample
    labels = [0, 0, 0, 2]
    dataset = SimpleLabelDataset(labels, label_key="label")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    sample_weights, class_weights = calculate_class_weights(dataloader, class_label="label", num_classes=3)

    # Expect class counts: [3,0,1] -> nonzero max_count = 3
    # So weights should be: class0 = 3/3 = 1.0, class1 = 0.0 (absent), class2 = 3/1 = 3.0
    assert pytest.approx(class_weights[0], rel=1e-6) == 1.0
    assert pytest.approx(class_weights[1], rel=1e-6) == 0.0
    assert pytest.approx(class_weights[2], rel=1e-6) == 3.0

    # sample_weights should map each sample to its class weight
    expected_sample_weights = [class_weights[l] for l in labels]
    assert len(sample_weights) == len(labels)
    for got, expected in zip(sample_weights, expected_sample_weights):
        assert pytest.approx(got, rel=1e-6) == expected


def test_calculate_class_weights_no_samples_raises():
    # Empty dataset should raise a ValueError as implemented
    dataset = SimpleLabelDataset([], label_key="label")
    dataloader = DataLoader(dataset, batch_size=1)

    with pytest.raises(ValueError):
        calculate_class_weights(dataloader, class_label="label", num_classes=3)
