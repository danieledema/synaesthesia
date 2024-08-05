import pickle
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class ParsedDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        num_workers,
        train_sampler=None,
        val_sampler=None,
        test_sampler=None,
        train_collate_fn=None,
        val_collate_fn=None,
        test_collate_fn=None,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler

        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn
        self.test_collate_fn = test_collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            sampler=self.train_sampler,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            collate_fn=self.val_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            collate_fn=self.test_collate_fn,
            pin_memory=True,
        )

    def save(self, path, current_cfg, overwrite=True):
        root_path = Path(path)
        if root_path.exists() and not overwrite:
            raise IOError(f"File {path} already exists and not overwriting")

        train_path = root_path / "train_dataset.pkl"
        val_path = root_path / "val_dataset.pkl"
        test_path = root_path / "test_dataset.pkl"

        with open(train_path, "wb") as path:
            pickle.dump(self.train_dataset, path)

        with open(val_path, "wb") as path:
            pickle.dump(self.val_dataset, path)

        with open(test_path, "wb") as path:
            pickle.dump(self.test_dataset, path)

        config_cache_path = root_path / "config.pkl"
        with open(config_cache_path, "wb") as path:
            pickle.dump(current_cfg, path)

    @staticmethod
    def load(
        root_path,
        batch_size,
        num_workers,
        train_sampler=None,
        val_sampler=None,
        test_sampler=None,
        train_collate_fn=None,
        val_collate_fn=None,
        test_collate_fn=None,
    ):
        root_path = Path(root_path)

        train_path = root_path / "train_dataset.pkl"
        val_path = root_path / "val_dataset.pkl"
        test_path = root_path / "test_dataset.pkl"

        with open(train_path, "rb") as path:
            train_dataset = pickle.load(path)

        with open(val_path, "rb") as path:
            val_dataset = pickle.load(path)

        with open(test_path, "rb") as path:
            test_dataset = pickle.load(path)

        # TODO: assert that sampler would work on the dataset

        return ParsedDataModule(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size,
            num_workers,
            train_sampler,
            val_sampler,
            test_sampler,
            train_collate_fn,
            val_collate_fn,
            test_collate_fn,
        )

    @staticmethod
    def check_load_cache(root_path, current_cfg):
        root_path = Path(root_path)
        train_path = root_path / "train_dataset.pkl"
        val_path = root_path / "val_dataset.pkl"
        test_path = root_path / "test_dataset.pkl"

        if not train_path.exists() or not val_path.exists() or not test_path.exists():
            return False

        with open(root_path / "config.pkl", "rb") as path:
            cached_cfg = pickle.load(path)

            if (
                current_cfg["train_dataset"] != cached_cfg["train_dataset"]
                or current_cfg["val_dataset"] != cached_cfg["val_dataset"]
                or current_cfg["test_dataset"] != cached_cfg["test_dataset"]
            ):
                return False

        return True

    def __repr__(self):
        return f"ParsedDataModule:\nTrain: {self.train_dataset}\nVal: {self.val_dataset}\nTest: {self.test_dataset}"
