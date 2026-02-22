# Developer & Contributor Guide — Synaesthesia

This document serves as the technical specification for the `synaesthesia` repository. Use this as the primary context when extending the library or onboarding new AI agents.

---

## Technical Requirements

- **Python:** 3.12+ (Type hints are mandatory for all new PRs).
- **Core Stack:** PyTorch, PyTorch Lightning, Kornia (for GPU-accelerated transforms), Pandas.
- **Tooling:** `pytest` for testing, `pre-commit` for linting (Black/Ruff).

---

## Core Abstractions & Logic

### 1. The `DatasetBase` Contract
Every dataset must inherit from `DatasetBase`. 
- **`get_data(idx)`**: Must return a `dict`. Keys should be prefixed with the dataset `id` to avoid collisions during fusion.
- **`machine_name`**: Must be **CamelCase**. This is enforced via `utils.check_camel_case_format` to ensure consistent logging and configuration keys.
- **Timestamps**: All synchronization relies on integer timestamps (typically Unix epoch in ms or ns).

### 2. Multi-Signal Synchronization
The `MultiSignalDataset` is the "brain" of the library. It uses three primary strategies:
- **Aggregation (`all` vs `common`)**: Determines if the master timestamp list is the Union or Intersection of all sub-datasets.
- **Fill (`none`, `last`, `closest`)**: Determines how to handle missing data when one sensor is faster than another.

### 3. Resource Management (Crucial)
When implementing file-based or video-based datasets (like `ImageFromVideoDataset`):
- **Lazy Loading**: Do not open file handles in `__init__`. Open them inside `get_data`.
- **Worker Safety**: Because PyTorch `DataLoader` uses `num_workers > 0` (multiprocessing), file handles must be unique per process. Always check if a handle exists before using it.

---

## High-level repository layout

- `synaesthesia/`
  - `docs/` — documentation (static files and guides).
  - `src/synaesthesia/` — main package code:
    - `abstract/` — core dataset abstractions:
      - `dataset_base.py` — `DatasetBase` (core API for all datasets).
      - `concat_dataset.py` — `CustomConcatDataset`.
      - `multi_signal_dataset.py` — `MultiSignalDataset`.
      - `sequential_dataset.py` — `SequentialDataset`.
      - `filter_functions.py` — filter strategies for sequence generation.
      - other helpers (time offset, boundary filtered datasets, conversions).
    - `base_sensors/` — concrete dataset implementations:
      - `csv_dataset.py` — `CsvDataset` (CSV-backed datasets).
      - `image_dataset.py` — `ImageDataset`, `ImageFromVideoDataset`.
      - `multi_file_dataset.py` — `MultiFileDataset`.
      - `multi_spectral_imager_dataset.py` — domain-specific dataset (multi-spectral).
    - `collates.py` — collate utilities and augmentations (Kornia-based transforms, tensor stacking).
    - `datamodule.py` — `ParsedDataModule` (PyTorch Lightning integration + cache save/load).
    - `samplers.py` — sampling helpers; `WeightedSamplerFromFile`.
    - `utils.py` — helpers (cache loading, camel case checker, instantiate wrappers).
  - `tests/` — unit tests (run with `pytest`).

---

## Core concepts & API

- `DatasetBase` (`src/synaesthesia/abstract/dataset_base.py`)
  - Subclasses must implement:
    - `__len__`
    - `get_data(idx) -> dict[str, Any]` — returns sensor-keyed values (no `idx` nor `timestamp`)
    - `get_timestamp(idx) -> int`
    - `get_timestamp_idx(timestamp) -> int`
    - `sensor_ids` (property) -> list[str]
    - `id` (property) -> str
    - `get_machine_name()` -> str (exposed via `machine_name` property; checked for camel-case)
    - `timestamps` (property) -> list[int]
  - `__getitem__` wraps `get_data` and `get_timestamp` and prefixes keys with `id` when present.

- `MultiFileDataset` (`src/synaesthesia/base_sensors/multi_file_dataset.py`)
  - Designed for datasets where each sample is a file in a directory.
  - Must implement `parse_filename()` and `read_data()` in subclasses.
  - Provides `timestamps`, `get_data`, and `get_timestamp_idx`.

- `CsvDataset` (`src/synaesthesia/base_sensors/csv_dataset.py`)
  - Base CSV dataset that reads a file and expects a `timestamp` column.
  - Subclasses must implement `convert_timestamp()` for converting timestamps.

- `ImageDataset` (`src/synaesthesia/base_sensors/image_dataset.py`)
  - Built on `MultiFileDataset`, implements `read_data` to load images and format them as `(C,H,W)` numpy arrays/dicts.

- `SequentialDataset` (`src/synaesthesia/abstract/sequential_dataset.py`)
  - Wraps a base `DatasetBase` to return sequences of `n_samples` according to `idx_format`.
  - Filters are provided via `filter_functions.Filter` implementations (`SkipNFilter`, `MultipleNFilter`, `ExponentialFilter`).

- `MultiSignalDataset` (`src/synaesthesia/abstract/multi_signal_dataset.py`)
  - Combines multiple `DatasetBase` instances into a synchronized view of timestamps.
  - Supports aggregation strategies: `all`, `common`, or `I:<idx>` to use timestamps from a particular dataset.
  - Supports fill strategies: `none`, `last`, `closest`.
  - Produces a `data_dict` mapping timestamp -> indices for each source dataset.

- `ParsedDataModule` (`src/synaesthesia/datamodule.py`)
  - Light wrapper around PyTorch Lightning `LightningDataModule`.
  - Supports saving / loading dataset cache via `pickle` (train/val/test datasets + config).
  - Use `ParsedDataModule.check_load_cache()` + `create_or_load_datamodule()` (in `utils.py`) to cache dataset instantiation for reproducibility.

- `collates.py`
  - Collection of `CollateBase` subclasses used to transform batches retrieved from `DataLoader`.
  - Includes tensor conversion, augmentations (Kornia), concatenation, scaling, clipping, etc.
  - Collates are composable via `ListCollate`.

---

## Development Workflow

### Adding a New Sensor
1. **Inherit** for isntance from `MultiFileDataset` (for files) or `CsvDataset` (for tabular).
2. **Implement `parse_filename`**: Logic to extract the integer timestamp from the source.
3. **Implement `read_data`**: Return a dictionary of tensors/arrays.
4. **Register `sensor_ids`**: A list of strings representing the keys returned by `read_data`.

### Testing Standards
We use `pytest`. Every new dataset **must** have a corresponding test in `tests/`.
- Use `tmp_path` fixtures to create dummy CSVs/Images.
- Assert that `get_timestamp_idx` returns the correct index for a known timestamp.
- Verify that `SequentialDataset` correctly handles boundaries (e.g., not returning a sequence that starts before index 0).

---

## Common Pitfalls & Debugging

- **Shape Mismatches**: `BatchCollate` expects consistent tensor shapes. If your sensor returns variable-sized images, you **must** add a resize/pad transform in a `Collate` class, not in the dataset itself.
- **Timestamp Drift**: If sensors are not perfectly synced, use `fill='closest'` in `MultiSignalDataset` but monitor the time delta in your logs.
- **Pickling Errors**: `ParsedDataModule` caches datasets using `pickle`. Ensure your custom classes don't hold un-picklable objects (like open database connections).

---

## Typical Task Checklist
- [ ] Implement subclass in `src/synaesthesia/base_sensors/`.
- [ ] Add type hints to all methods.
- [ ] Ensure `machine_name` is CamelCase.
- [ ] Add unit tests in `tests/`.
- [ ] Run `pre-commit run --all-files`.

---

## Contributing guidelines and style

- Use type hints consistently. Functions and public methods should have type annotations.
- Keep dataset implementations single-purpose:
  - `DatasetBase` subclasses should focus on data access logic only.
  - Avoid mixing preprocessing transforms in `get_data`; prefer `collates` or external transforms.
- Naming:
  - `machine_name` must follow camel case; `utils.check_camel_case_format` enforces this.
  - `id` should be a short descriptor of sensor (used as prefix in `__getitem__`).
- Tests:
  - Add unit tests for new dataset behaviour under `tests/`.
  - Prefer small, deterministic datasets for testing (temporary directories + minimal CSV or image files).
- Documentation:
  - Update `docs/` for new classes or behavioural changes.
  - Keep README examples up to date.
