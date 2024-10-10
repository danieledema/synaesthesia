# Synaesthesia 🧠🎨

Create general PyTorch data pipelines from simple Python, extendable to any sensors.

## Overview 🌟

Synaesthesia is a Python library that forms the foundation of a dataset stack in any PyTorch/PyTorch Lightning projects. It contains base datasets and structures that enable combination, sequencing, concatenation, and other transformations through composition mechanisms.

This library provides a flexible and modular approach to creating datasets and dataloaders for various applications. It's designed to handle different data types, including CSV and image datasets, with the ability to expand functionality through custom classes.

## Key Features 🔑

- **Modular Design** 🧩: Easily combine different dataset types and operations.
- **Multi-modal Support** 🎛️: Handle various sensor modalities and information types.
- **Flexible Combinations** 🔗: 
  - Parallel combination of datasets (MultiSignalDataset)
  - Serial concatenation of datasets (ConcatDataset)
  - Sequential data retrieval (SequentialDataset)
- **Extensibility** 🔌: Users can create custom dataset classes to extend functionality.
- **Built-in Support** 📦: Ready-to-use implementations for CSV and image datasets.

## Installation 💻

The easiest way of using Synaesthesia is to clone it as a submodule of your system:

```bash
git submodule add git@github.com:danieledema/synaesthesia.git .submodules/synaesthesia
```

Then, use `poetry` to manage the required packages by including the submodule in the installation path:

```bash
poetry add .submodules/synaesthesia
```

## Main Components 🧱

### DatasetBase 🏗️

The foundation class for all datasets in the library.

### CustomConcatDataset 🔗

Allows concatenation of multiple datasets, preserving individual dataset properties.

### MultiSignalDataset 📡

Combines multiple single-signal datasets, supporting various aggregation and fill methods.

### SequentialDataset 🔢

Enables retrieval of data sequences from a base dataset, with customizable filtering and stride options.

### Filter Classes 🔍

Provides different strategies for data filtering and selection:
- `SkipNFilter`
- `MultipleNFilter`
- `ExponentialFilter`

## Usage Examples 📚

```python
# Example 1: Creating a multi-signal dataset
csv_dataset = CSVDataset(...)
image_dataset = ImageDataset(...)
multi_dataset = MultiSignalDataset([csv_dataset, image_dataset])

# Example 2: Creating a sequential dataset
seq_dataset = SequentialDataset(csv_dataset, n_samples=5, stride=2)

# Example 3: Concatenating datasets
concat_dataset = CustomConcatDataset([dataset1, dataset2, dataset3])
```

## Extending the Library 🚀

Users can create custom dataset classes by inheriting from `DatasetBase` and implementing required methods:

```python
class MyCustomDataset(DatasetBase):
    def __init__(self, ...):
        super().__init__()
        # Custom initialization

    def get_data(self, idx):
        # Implement data retrieval logic

    # Implement other required methods
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the APACHE-2.0 License. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.
