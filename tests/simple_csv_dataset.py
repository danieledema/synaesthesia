from src.vigil2.data.base_sensors.csv_dataset import CsvDataset


class SimpleCsvDataset(CsvDataset):
    @property
    def id(self):
        return "CSV"

    @property
    def satellite_name(self):
        return "FAKE"
