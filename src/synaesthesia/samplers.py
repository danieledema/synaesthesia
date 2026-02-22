from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


def calculate_class_weights(dataloader: DataLoader, class_label: str, num_classes: int):
    """
    Compute per-sample weights (for WeightedRandomSampler) and per-class weights.

    This function is robust against classes with zero samples and avoids division-by-zero.
    - If no samples are present in the dataloader, raises a ValueError.
    - For classes with zero samples, the class weight is set to 0.0 (these classes will not be sampled).
    - For present classes, weight = max_count / class_count (so rarer classes get larger weight).

    Returns:
        sample_weights: list[float] sized as the dataset (weight per sample index)
        class_weights: list[float] sized `num_classes` (weight per class id)
    """
    # Count occurrences per class and record label per sample index
    class_counts = [0] * num_classes
    sample_labels: list[int | None] = [None] * len(dataloader.dataset)

    for batch in tqdm(dataloader, desc="Computing class weights"):
        # Expect `idx` and the `class_label` in the batch
        idxs = batch["idx"].tolist()
        labels = batch[class_label].tolist()
        for i, c in zip(idxs, labels):
            # Ensure label is an int index and within bounds
            try:
                ci = int(c)
            except Exception:
                raise ValueError(
                    f"Encountered non-integer class label: {c!r} at sample idx {i}"
                )

            if ci < 0 or ci >= num_classes:
                raise ValueError(
                    f"Class label {ci} at sample idx {i} is out of expected range [0, {num_classes - 1}]"
                )

            class_counts[ci] += 1
            sample_labels[i] = ci

    # If no samples were found in the dataloader, fail early
    if sum(class_counts) == 0:
        raise ValueError("No samples found in dataloader while computing class weights")

    # Compute class weights while avoiding division by zero
    nonzero_counts = [c for c in class_counts if c > 0]
    max_count = max(nonzero_counts) if nonzero_counts else 0

    class_weights = []
    for count in class_counts:
        if count == 0:
            # No samples for this class -> weight 0 (will not be sampled)
            class_weights.append(0.0)
        else:
            # Larger weight for rarer classes
            class_weights.append(float(max_count) / float(count))

    # Build per-sample weights using computed class weights.
    sample_weights = [0.0] * len(sample_labels)
    for i, label in enumerate(sample_labels):
        if label is None:
            # If a sample had no label recorded (shouldn't happen in normal use), set weight 0
            sample_weights[i] = 0.0
        else:
            sample_weights[i] = class_weights[label]

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
