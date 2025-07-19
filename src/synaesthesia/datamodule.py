from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate


class CollateFn(Protocol):
    """Protocol for collate functions."""

    def __call__(self, batch: list[Any]) -> Any: ...


class ParsedDataModule(LightningDataModule):
    """A LightningDataModule that wraps pre-split datasets with caching support."""

    _CACHE_FILES = {
        "train": "train_dataset.pkl",
        "val": "val_dataset.pkl",
        "test": "test_dataset.pkl",
        "config": "config.pkl",
    }

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        test_sampler: Optional[Sampler] = None,
        train_collate_fn: Optional[CollateFn] = None,
        val_collate_fn: Optional[CollateFn] = None,
        test_collate_fn: Optional[CollateFn] = None,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler

        self.train_collate_fn = train_collate_fn or default_collate
        self.val_collate_fn = val_collate_fn or default_collate
        self.test_collate_fn = test_collate_fn or default_collate

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            collate_fn=self.train_collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            collate_fn=self.val_collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            collate_fn=self.test_collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def save(
        self, path: str | Path, current_cfg: Dict[str, Any], *, overwrite: bool = True
    ) -> None:
        """Save datasets and configuration to disk."""
        root_path = Path(path)
        root_path.mkdir(parents=True, exist_ok=True)

        if not overwrite and any(
            (root_path / fname).exists() for fname in self._CACHE_FILES.values()
        ):
            raise FileExistsError(
                f"Cache already exists at {root_path} and overwrite=False"
            )

        # Save datasets
        for split, filename in self._CACHE_FILES.items():
            if split == "config":
                continue
            dataset = getattr(self, f"{split}_dataset")
            with open(root_path / filename, "wb") as f:
                pickle.dump(dataset, f)

        # Save configuration
        with open(root_path / self._CACHE_FILES["config"], "wb") as f:
            pickle.dump(current_cfg, f)

    @classmethod
    def load(
        cls,
        root_path: str | Path,
        batch_size: int,
        num_workers: int,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        test_sampler: Optional[Sampler] = None,
        train_collate_fn: Optional[CollateFn] = None,
        val_collate_fn: Optional[CollateFn] = None,
        test_collate_fn: Optional[CollateFn] = None,
    ) -> ParsedDataModule:
        """Load datasets and configuration from disk."""
        root_path = Path(root_path)

        # Load datasets
        datasets = {}
        for split, filename in cls._CACHE_FILES.items():
            if split == "config":
                continue
            with open(root_path / filename, "rb") as f:
                datasets[split] = pickle.load(f)

        return cls(
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            batch_size=batch_size,
            num_workers=num_workers,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            test_sampler=test_sampler,
            train_collate_fn=train_collate_fn,
            val_collate_fn=val_collate_fn,
            test_collate_fn=test_collate_fn,
        )

    @classmethod
    def check_load_cache(
        cls, root_path: str | Path, current_cfg: Dict[str, Any]
    ) -> bool:
        """Check if cached datasets match the current configuration."""
        root_path = Path(root_path)

        # Check if all cache files exist
        if not all((root_path / fname).exists() for fname in cls._CACHE_FILES.values()):
            return False

        # Load and compare configuration
        try:
            with open(root_path / cls._CACHE_FILES["config"], "rb") as f:
                cached_cfg = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            return False

        # Compare relevant configuration keys
        config_keys = ["train_dataset", "val_dataset", "test_dataset"]
        return all(current_cfg.get(key) == cached_cfg.get(key) for key in config_keys)

    def __repr__(self) -> str:
        """Return a string representation of the datamodule."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  train_dataset={self.train_dataset},\n"
            f"  val_dataset={self.val_dataset},\n"
            f"  test_dataset={self.test_dataset},\n"
            f"  batch_size={self.batch_size},\n"
            f"  num_workers={self.num_workers}\n"
            f")"
        )
