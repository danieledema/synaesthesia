from pathlib import Path

from hydra.utils import instantiate
from pyinputplus import inputYesNo

from .datamodule import ParsedDataModule


def create_or_load_datamodule(cache_path: str | Path, cfg):
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    load_cache = False
    if ParsedDataModule.check_load_cache(cache_path, cfg):
        load_cache = (
            inputYesNo(f"Found cache at {cache_path}. Load from cache? [yes/no] ")
            == "yes"
        )

    if load_cache:
        print(f"Loading data module from {cache_path}")
        data_module = ParsedDataModule.load(
            cache_path,
            cfg["batch_size"],
            cfg["num_workers"],
        )
    else:
        data_module = instantiate(cfg)
        print(f"Saving model in {cache_path}")
        data_module.save(cache_path, cfg)

    return data_module
