import pickle
from pathlib import Path

import pytest

from src.synaesthesia.datamodule import ParsedDataModule


class GoodDataset:
    """A simple pickleable dataset placeholder."""

    def __init__(self, name="good"):
        self.name = name

    def __repr__(self):
        return f"GoodDataset({self.name})"


class UnpicklableDataset:
    """Dataset that raises when pickled to simulate a pickle failure."""

    def __getstate__(self):
        raise RuntimeError("Simulated pickle failure for testing")


def _list_expected_paths(root: Path):
    return [
        root / "train_dataset.pkl",
        root / "val_dataset.pkl",
        root / "test_dataset.pkl",
        root / "config.pkl",
    ]


def test_save_cleanup_on_pickle_error(tmp_path):
    """
    Ensure ParsedDataModule.save cleans up partially written files when
    a pickle error occurs while dumping datasets.
    """

    # Arrange: create a datamodule with a picklable train/test and an unpicklable val
    train = GoodDataset("train")
    val = UnpicklableDataset()
    test = GoodDataset("test")

    dm = ParsedDataModule(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        batch_size=1,
        num_workers=0,
    )

    cfg = {"some": "config"}

    root = tmp_path / "cache_dir"

    # Act / Assert: saving should raise an IOError (our implementation wraps pickle failures)
    with pytest.raises(IOError):
        dm.save(root, cfg, overwrite=True)

    # After the failure, none of the expected files should remain
    for p in _list_expected_paths(root):
        assert not p.exists(), (
            f"Expected {p} to be removed after failure but it exists."
        )


def test_save_success_writes_files(tmp_path):
    """
    Sanity check that when all datasets are pickleable, save writes all expected files.
    """

    train = GoodDataset("train")
    val = GoodDataset("val")
    test = GoodDataset("test")

    dm = ParsedDataModule(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        batch_size=1,
        num_workers=0,
    )

    cfg = {
        "train_dataset": "train_cfg",
        "val_dataset": "val_cfg",
        "test_dataset": "test_cfg",
    }

    root = tmp_path / "cache_dir_ok"
    dm.save(root, cfg, overwrite=True)

    # All expected files should exist and be loadable via pickle
    for p in _list_expected_paths(root):
        assert p.exists(), f"Expected {p} to exist after successful save."

    # Try loading them back
    with open(root / "train_dataset.pkl", "rb") as f:
        loaded_train = pickle.load(f)
    with open(root / "val_dataset.pkl", "rb") as f:
        loaded_val = pickle.load(f)
    with open(root / "test_dataset.pkl", "rb") as f:
        loaded_test = pickle.load(f)
    with open(root / "config.pkl", "rb") as f:
        loaded_cfg = pickle.load(f)

    assert isinstance(loaded_train, GoodDataset)
    assert isinstance(loaded_val, GoodDataset)
    assert isinstance(loaded_test, GoodDataset)
    assert loaded_cfg == cfg
