from .euv_dataset import EuvDataset


class SdoEuvDataset(EuvDataset):
    @property
    def satellite_name(self):
        return "SDO"
