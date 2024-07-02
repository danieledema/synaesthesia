from .euv_dataset import EuvDataset


class SoloEuvDataset(EuvDataset):
    @property
    def satellite_name(self):
        return "SOLO"
