from src.synaesthesia.base_sensors.csv_dataset import CsvDataset


class SimpleCsvDataset(CsvDataset):
    @property
    def id(self):
        return "CSV"

    def get_machine_name(self) -> str:
        return "leftArm"
