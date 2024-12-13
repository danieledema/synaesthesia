from pathlib import Path

from src.synaesthesia.abstract.conversion import convert_to_timestamp
from src.synaesthesia.base_sensors.csv_dataset import CsvDataset


class SimpleCsvDataset(CsvDataset):
    def __init__(self, path: str | Path, cols: list[str] | str | None = None):
        super().__init__(path, cols, ",")

    @property
    def id(self):
        return "CSV"

    def get_machine_name(self) -> str:
        return "leftArm"

    def convert_timestamp(self, timestamp: str | int) -> int:
        return convert_to_timestamp(timestamp)
