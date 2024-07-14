from .sdo_ml_dataset import SDOMLDataset
import pandas as pd
import numpy as np


class TimestampedSDOMLDataset(SDOMLDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        timestamps = pd.to_datetime(self.aligndata.index)
        self._timestamps = np.array(timestamps, dtype="datetime64[ns]")

    def __getitem__(self, idx):

        # sample num_frames between idx and idx - sampling_period
        items = self.aligndata.iloc[idx]
        # print(items.index, idx, pd.DataFrame([items]))
        timestamps = [
            i.strftime("%Y-%m-%d %H:%M:%S%f") for i in pd.DataFrame([items]).index
        ]

        r = {"timestamps": timestamps}

        if self.eve_data:
            image_stack, eve_data = super().__getitem__(idx)
            r["eve_data"] = eve_data
        else:
            image_stack = super().__getitem__(idx)

        r["image_stack"] = image_stack

        return r

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def satellite_name(self):
        return "SDO-MLv2"

    @property
    def sensor_ids(self):
        return ["AIA", "HMI", "EVE"]

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def get_data(self, idx):
        data = self.__getitem__(idx)
        return data

    @property
    def id(self):
        return "SDO-MLv2"
