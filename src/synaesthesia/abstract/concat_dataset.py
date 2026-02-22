from loguru import logger

from .dataset_base import DatasetBase


class CustomConcatDataset(DatasetBase):
    """
    A specialisation of the ConcatDataset where the idx is returned together
    with the data.
    Used in the datamodule.
    """

    def __init__(self, datasets: list[DatasetBase]):
        super().__init__()

        self.datasets = datasets

        ssd_sensor_ids = self.datasets[0].sensor_ids
        for ssd in self.datasets[1:]:
            assert set(ssd_sensor_ids) == set(ssd.sensor_ids)

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

    @property
    def timestamps(self):
        """
        Return a flat list of timestamps for the concatenated datasets.

        NOTE: return a list (not a generator) and use the datasets' `timestamps`
        property (not callable). This keeps behavior consistent with other
        DatasetBase implementations that expose a list of timestamps.
        """
        merged = []
        for d in self.datasets:
            merged.extend(d.timestamps)
        return merged

    def __repr__(self) -> str:
        print_string = f"\nConcat dataset: {len(self)} samples\n"
        print_string += f"Datasets: {len(self.datasets)}\n"

        for i, d in enumerate(self.datasets):
            inner_repr = repr(d)
            lines = inner_repr.split("\n")
            # Indent each line for better visibility
            inner_repr = "\n".join(["\t" + line for line in lines])

            print_string += f"\nDataset {i}:\n"
            print_string += f"{inner_repr}\n"
            print_string += "------------------\n"

        # This is important to ensure that the string ends with a newline for clarity
        return (
            print_string.strip()
        )  # Add an extra newline at the end for better separation

    @property
    def machine_name(self):
        """
        The concatenated dataset may represent multiple machines. Return a
        combined representation (the first name) but log a warning if there
        are multiple distinct machine names.
        """
        machine_names = list({d.machine_name for d in self.datasets})
        if len(machine_names) > 1:
            logger.warning(
                "ConcatDataset contains multiple machine names; returning the first one. Combined: %s",
                ", ".join(machine_names),
            )
        # Fall back to the first dataset's machine_name if available
        return machine_names[0] if machine_names else ""

    @property
    def sensor_ids(self) -> list[str]:
        return self.datasets[0].sensor_ids
