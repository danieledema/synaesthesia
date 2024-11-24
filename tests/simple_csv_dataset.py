from src.synaesthesia.base_sensors.csv_dataset import CsvDataset
from src.synaesthesia.abstract.conversion import convert_to_timestamp


class SimpleCsvDataset(CsvDataset):
    @property
    def id(self):
        return "CSV"

    def get_machine_name(self) -> str:
        return "leftArm"

    def convert_timestamp(self, timestamp: str | int) -> int:
        return convert_to_timestamp(timestamp)
