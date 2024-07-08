from .abstract_dataset import SingleSignalDatasetBase
from .boundary_multi_signal_dataset import BoundaryMultiSignalDataset
from .multi_signal_dataset import MultiSignalDataset


class CustomConcatDataset(SingleSignalDatasetBase):
    """
    A specialisation of the ConcatDataset where the idx is returned together
    with the data.
    Used in the datamodule.
    """

    def __init__(self, datasets: list[MultiSignalDataset | BoundaryMultiSignalDataset]):
        super().__init__()

        self.datasets = datasets

        self.ssd_sensor_ids = self.datasets[0].single_signal_dataset_ids
        for ssd in self.datasets[1:]:
            assert self.ssd_sensor_ids == ssd.single_signal_dataset_ids

    def find_right_datset(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bound {len(self)}")

        while idx < 0:
            idx += len(self)

        for i, d in enumerate(self.datasets):
            if len(d) > idx:
                return i, idx
            idx -= len(d)

        return -1, -1  # It will never reach here

    def get_data_by_id(self, idx, id):
        assert id in self.ssd_sensor_ids
        i, idx2 = self.find_right_datset(idx)
        return self.datasets[i].get_data_by_id(idx2, id)

    def get_data(self, idx):
        i, idx2 = self.find_right_datset(idx)
        data = self.datasets[i].get_data(idx2)
        data["idx"] = idx
        return data

    def __len__(self):
        l = 0
        for d in self.datasets:
            l += len(d)
        return l

    def get_timestamp(self, idx):
        i, idx2 = self.find_right_datset(idx)
        return self.datasets[i].get_timestamp(idx2)

    def __repr__(self) -> str:
        print_string = f"Concat dataset: {len(self)} samples\n"
        print_string += f"Datasets: {len(self.datasets)}\n"
        for i, d in enumerate(self.datasets):
            print_string += f"{i} {d}"
            print_string += "------------------\n"
        return print_string
