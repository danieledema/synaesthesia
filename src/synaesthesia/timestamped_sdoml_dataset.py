from .sdo_ml_dataset import SDOMLDataset
import pandas as pd
import numpy as np


class TimestampedSDOMLDataset(SDOMLDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        # sample num_frames between idx and idx - sampling_period
        items = self.aligndata.iloc[idx]
        # print(items.index, idx, pd.DataFrame([items]))
        timestamps = [
            i.strftime("%Y-%m-%d %H:%M:%S") for i in pd.DataFrame([items]).index
        ]

        r = {"timestamps": timestamps}

        if self.eve_data:
            image_stack, eve_data = super().__getitem__(idx)
            r["eve_data"] = eve_data
        else:
            image_stack = super().__getitem__(idx)

        r["image_stack"] = image_stack

        return r
