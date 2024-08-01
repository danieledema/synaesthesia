from pathlib import Path

from torch.utils.data import WeightedRandomSampler

from .abstract.dataset_base import DatasetBase


def calculate_class_weights(dataset: DatasetBase, class_label: str, num_classes: int):
    class_weights = [0] * num_classes

    sample_weights = [0 for _ in range(len(dataset))]

    for i, data in enumerate(dataset):
        class_weights[data[class_label]] += 1

        sample_weights[i] = data[class_label]

    max_class_weight = max(class_weights)
    class_weights = [max_class_weight / class_weight for class_weight in class_weights]

    return sample_weights, class_weights


class WeightedSamplerFromFile(WeightedRandomSampler):
    def __init__(self, filepath: str | Path, num_samples: int):
        sample_weights = self.read_sample_weights(filepath)

        super(WeightedSamplerFromFile, self).__init__(
            weights=sample_weights, num_samples=num_samples
        )

    def read_sample_weights(self, filepath: str | Path):
        with open(filepath, "r") as file:
            sample_weights = [float(line.strip()) for line in file]

        return sample_weights

    def write_sample_weights(self, filepath: str | Path, sample_weights):
        with open(filepath, "w") as file:
            for weight in sample_weights:
                file.write(f"{weight}\n")
