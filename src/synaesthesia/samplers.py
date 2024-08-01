from .abstract.dataset_base import DatasetBase


def calculate_class_weights(dataset: DatasetBase, class_label: str, num_classes: int):
    class_weights = [0] * num_classes

    sample_weights = [0 for _ in range(len(dataset))]

    for i, data in enumerate(dataset):
        class_weights[data[class_label]] += 1

        sample_weights[i] = data[class_label]

    sample_weights = [class_weights[label] for label in sample_weights]

    return sample_weights, class_weights
