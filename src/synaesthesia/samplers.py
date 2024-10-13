from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


def calculate_class_weights(dataloader: DataLoader, class_label: str, num_classes: int):
    class_weights = [0] * num_classes

    sample_weights = [0 for _ in range(len(dataloader.dataset))]

    for data in tqdm(dataloader):
        for i, c in zip(data["idx"].tolist(), data[class_label].tolist()):
            class_weights[c] += 1
            sample_weights[i] = c

    max_class_weight = max(class_weights)
    class_weights = [max_class_weight / class_weight for class_weight in class_weights]

    for i in range(len(sample_weights)):
        sample_weights[i] = class_weights[sample_weights[i]]

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

    @staticmethod
    def write_sample_weights(filepath: str | Path, sample_weights):
        with open(filepath, "w") as file:
            for weight in sample_weights:
                file.write(f"{weight}\n")
