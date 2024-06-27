from .abstract_dataset import SingleSignalDatasetBase


class SoEuvDataset(SingleSignalDatasetBase):
    def __init__(self):
        super().__init__()

    @property
    def sensor_id(self):
        return "EUV"

    @property
    def satellite_name(self):
        return "SO"

    def __len__(self):
        raise NotImplementedError

    def get_data(self, idx):
        raise NotImplementedError

    def get_timestamp(self, idx):
        raise NotImplementedError

    def get_timestamp_idx(self, timestamp):
        raise NotImplementedError
