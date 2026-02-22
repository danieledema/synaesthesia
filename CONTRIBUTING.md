# Claude Developer Guide — Synaesthesia

This document is a focused developer guide for working on the `synaesthesia` repository. It explains the project's purpose, directory layout, core abstractions, development workflows, testing, debugging tips, and suggestions for common tasks (adding datasets, collates, datamodules, etc.). Use this as a living guide while contributing or extending the library.

Note: paths, classes and functions are referenced with backticks (e.g. `DatasetBase`, `CsvDataset`) and file paths use repository-relative notation (e.g. `src/synaesthesia/...`).

---

## Quick summary

- Purpose: Provide composable dataset primitives (CSV, image, multi-file, sequential, multi-signal, concat) and utilities for building PyTorch / PyTorch Lightning data pipelines.
- Language: Python (typed hints present).
- Packaging: `pyproject.toml` present.
- Tests: `tests/` folder exists (run with `pytest`).
- Pre-commit hooks: `.pre-commit-config.yaml` exists — follow them to keep style/quality consistent.
- README: `README.md` contains overview and usage examples.

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

## How to run & test locally

1. Create a Python virtual environment and install dev dependencies.
   - Project uses `pyproject.toml`. Choose your tool:
     - `pip install -e .[dev]` or use `poetry install` or `pipx/uv` as preferred.
   - Ensure `kornia`, `pytorch`, `torchvision`, `pytorch-lightning`, `pytest`, `loguru`, `tqdm`, `pandas`, `pyinputplus`, and other listed deps are available.

2. Run tests:
   - From repo root:
     - `python -m pytest -q`
     - Or simply `pytest`

3. Pre-commit:
   - Run configured pre-commit hooks (recommended): `pre-commit run --all-files`

4. Linting/formatting:
   - Follow the style enforced by pre-commit. If a formatter (e.g. `black`) is configured, run it before committing.

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

---

## Typical development tasks

### Adding a new file-based dataset

1. Create a subclass of `MultiFileDataset` in `src/synaesthesia/base_sensors/` (e.g. `my_sensor_dataset.py`).
2. Implement:
   - `parse_filename(self, filename) -> int` — extracts numeric timestamp/index from filename.
   - `read_data(self, file_path: Path) -> Any` — read file and return dict mapping sensor keys to values.
   - `sensor_ids` property (if different from default).
3. Add unit tests in `tests/` that:
   - Create temp files with expected filenames and content.
   - Instantiate the dataset and assert `__len__`, `get_data`, `timestamps`, and `get_timestamp_idx` behave correctly.

Example skeleton:
```/dev/null/example.py#L1-40
from src.synaesthesia.base_sensors.multi_file_dataset import MultiFileDataset
from pathlib import Path

class MySensorDataset(MultiFileDataset):
    def parse_filename(self, filename) -> int:
        # Parse timestamp from filename, e.g. "12345.png" -> 12345
        return int(Path(filename).stem)

    def read_data(self, file_path: Path):
        # Load and return a dict of sensor data
        data = ...  # open file_path
        return {"value": data}

    @property
    def sensor_ids(self):
        return ["value"]
```

(Place the actual file in `src/synaesthesia/base_sensors/` and tests under `tests/`.)

### Adding a new collate or augmentation

1. Add a `CollateBase` subclass in `src/synaesthesia/collates.py`. Reuse existing patterns:
   - Implement `do_collate(self, items)` and optionally an `__init__` to configure behaviour.
2. Keep collates composable: use `ListCollate` to chain collates.
3. Add tests to verify outputs for small synthetic batches (dictionaries with consistent keys).

### Extending `CsvDataset`

- Subclass `CsvDataset` and implement `convert_timestamp` to parse your CSV timestamp format (e.g. ISO string -> epoch int).
- Use `cols` parameter to select appropriate columns.
- Example in tests: create a temporary CSV with `timestamp` column and some numeric columns, instantiate your subclass and validate outputs.

---

## Debugging tips & common pitfalls

- Multiprocessing & file handles:
  - `ImageFromVideoDataset` uses `cv2.VideoCapture`. When using DataLoader with `num_workers > 0`, make sure file handles are opened in the worker process. `get_data` checks and opens `cap` when needed, but be mindful of resource cleanup (`__del__` / `close()`).
- Timestamp alignment:
  - `MultiSignalDataset` has non-trivial logic for merging timestamps and fill strategies. When debugging synchronization mismatches, print or inspect:
    - `dataset.timestamps` for each single-signal source.
    - `multi.timestamps` and `multi.data_dict` to see mapped indices.
- Camel-case machine names:
  - `DatasetBase.machine_name` is validated by `check_camel_case_format`. If tests throw `ValueError` about camel-case, rename the machine identifier to conform to pattern `([A-Z]?[a-z]+)+[0-9]*$`.
- Collate type errors:
  - `BatchCollate.make_into_tensor` expects consistent shapes for tensors. If your dataset returns heterogeneous shapes, either:
    - Pad / normalize shapes in dataset or a collate transform.
    - Return lists for non-uniform items (the collate will return lists, not tensors).

---

## Packaging and releases

- The project uses `pyproject.toml`. Use your preferred toolchain to build and publish (e.g. `build` / `twine`, or `poetry`).
- Ensure tests pass and the README/docs are updated before publishing.
- Update the `uv.lock` / lockfile if the dependency manager expects it (the README mentions `uv add synaesthesia`).

---

## Running an example pipeline

1. Instantiate raw datasets:
   - `csv_ds = CsvSubclass(path=...)`
   - `img_ds = ImageDataset(folder_path=..., extension='png')`
2. Combine:
   - `multi = MultiSignalDataset([csv_ds, img_ds], aggregation='all', fill='closest')`
3. Wrap sequences (if needed):
   - `seq = SequentialDataset(multi, n_samples=5, stride=1)`
4. Create `ParsedDataModule` or pass dataset to `DataLoader` directly:
   - `dm = ParsedDataModule(train_dataset=seq, val_dataset=..., test_dataset=..., batch_size=32, num_workers=4)`
   - `train_loader = dm.train_dataloader()`

---

## TODOs and improvement ideas

- Add more robust timestamp types (e.g. native `datetime` / `pandas.Timestamp`) with consistent conversions.
- Improve `MultiSignalDataset` performance for very large timestamp sets (consider using numpy arrays and vectorized ops).
- Add more unit tests around edge cases for `fill='last'` and `fill='closest'`.
- Add CI (if not present) to run tests + pre-commit on PRs.
- Add examples/notebooks under `docs/` showing common dataset composition workflows.

---

If you want, I can:
- Create an initial test template for a new `MultiFileDataset` subclass.
- Add a CONTRIBUTING.md with these guidelines turned into checklist items.
- Implement any of the TODO items above.

Tell me which task to start and I will provide the code and tests.