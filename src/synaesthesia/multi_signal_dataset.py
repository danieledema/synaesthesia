from datetime import timedelta
from typing import List

from .abstract_dataset import DatasetBase
from .utils import convert_to_datetime

import pandas as pd
from tqdm import tqdm
from datetime import datetime


def convert_to_datetime(timestamp):
    # Adjust this function based on the format of your timestamps
    try:
        return datetime.strptime(timestamp, "%Y%m%dT%H%M")
    except ValueError:
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%S%f")


import numpy as np


class MultiSignalDataset(DatasetBase):
    """
    Dataset class for handling multiple signal datasets.
    """

    def __init__(
        self,
        single_signal_datasets: List[DatasetBase],
        aggregation: str = "all",
        fill: str = "none",
        time_cut: int = 60,  # in minutes
    ):
        """
        Initializes the MultiSignalDataset.

        Args:
            single_signal_datasets (list): List of DatasetBase objects representing single signal datasets.
            aggregation (str): Aggregation method for timestamps ("all", "common", "I:<idx>").
            fill (str): Method for filling missing timestamps ("none", "last", "closest").
            time_cut (int): Time cut-off in minutes for "closest" fill method.
        """
        super().__init__()

        self.single_signal_datasets = single_signal_datasets
        self.aggregation = aggregation
        self.fill = fill
        self.time_cut = time_cut

        print("Initializing MultiSignalDataset...")

        # Create a DataFrame to store timestamps and corresponding indices
        self.timestamp_df = self._initialize_timestamp_df()

        # Fill missing timestamps based on the fill method
        self._fill_missing_timestamps()

    def _initialize_timestamp_df(self) -> pd.DataFrame:
        """
        Initializes the DataFrame to store timestamps and corresponding indices.
        """
        print("Initializing timestamp DataFrame...")

        if self.aggregation == "all":
            # For 'all' aggregation, mark each timestamp present in each dataset
            print("Aggregating timestamps (method: all)...")
            # Initialize an empty DataFrame with timestamps from all datasets
            all_timestamps = pd.concat(
                [
                    pd.Series(ds.timestamps, name="timestamp")
                    for ds in self.single_signal_datasets
                ]
            ).reset_index(drop=True)

            # Remove duplicates if any
            all_timestamps = all_timestamps.drop_duplicates()

            # Initialize the DataFrame with NaNs
            timestamp_df = pd.DataFrame(
                index=all_timestamps,
                columns=[
                    f"dataset_{i}" for i in range(len(self.single_signal_datasets))
                ],
            )

        elif self.aggregation == "common":

            # For 'common' aggregation, mark timestamps common to all datasets
            print("Aggregating timestamps (method: common)...")
            common_timestamps = set.intersection(
                *[set(ds.timestamps) for ds in self.single_signal_datasets]
            )
            common_timestamps = sorted(
                common_timestamps
            )  # Convert set to a sorted list

            timestamp_df = pd.DataFrame(
                index=common_timestamps,
                columns=[
                    f"dataset_{i}" for i in range(len(self.single_signal_datasets))
                ],
            )

        elif self.aggregation.startswith("I:"):

            # For 'I:<idx>' aggregation, mark timestamps from a specific dataset
            idx = int(self.aggregation[2:])
            print(f"Aggregating timestamps (method: I:{idx})...")
            timestamp_df = pd.DataFrame(
                index=self.single_signal_datasets[idx].timestamps,
                columns=[
                    f"dataset_{i}" for i in range(len(self.single_signal_datasets))
                ],
            )

        for i, ds in enumerate(self.single_signal_datasets):
            print(f"Processing dataset {i}...")
            for j, timestamp in enumerate(tqdm(ds.timestamps)):
                if timestamp in timestamp_df.index:
                    timestamp_df.loc[timestamp, f"dataset_{i}"] = j

        return timestamp_df

    def _fill_missing_timestamps(self):
        """
        Fills missing timestamps based on the specified fill method.
        """
        print("Filling missing timestamps...")

        if self.fill == "none":
            # No filling required
            print("Fill method: none")
            # self.timestamp_df = self.timestamp_df.sort_values()

        elif self.fill == "last":
            # Fill missing timestamps with data from the last available timestamp
            print("Fill method: last")
            # self.timestamp_df = self.timestamp_df.sort_values()
            self.timestamp_df = self.timestamp_df.fillna(method="ffill")

        elif self.fill == "closest":
            # Fill missing timestamps with data from the closest available timestamp within time_cut minutes
            print("Fill method: closest")
            for i in range(len(self.timestamp_df.columns)):
                # Sort timestamps to ensure the interpolate('nearest') works correctly
                timestamp_series = self.timestamp_df.index
                limit = int(
                    self.time_cut / (timestamp_series[1] - timestamp_series[0]).seconds
                )  # Convert time_cut from minutes to seconds.
                interpolated_column = (
                    self.timestamp_df.iloc[:, i]
                    .reindex(timestamp_series)
                    .interpolate(method="nearest", limit=limit, limit_direction="both")
                )
                self.timestamp_df.iloc[:, i] = interpolated_column
                # Remove timestamps that couldn't be filled (still None after interpolation)
                self.timestamp_df = self.timestamp_df.dropna()

    @property
    def timestamps(self) -> List[pd.Timestamp]:
        """
        Returns the list of timestamps.
        """
        return self.timestamp_df.index.tolist()

    def __len__(self) -> int:
        """
        Returns the number of timestamps.
        """
        return len(self.timestamps)

    def get_data(self, idx: int) -> dict:
        """
        Retrieves the data at the specified index.

        Args:
            idx (int): Index of the timestamp.

        Returns:
            dict: Dictionary containing data from all datasets at the specified timestamp.
        """
        timestamp = self.timestamps[idx]
        data_dict = {}
        for i, ds in enumerate(self.single_signal_datasets):
            data_dict[f"dataset_{i}"] = ds.get_data(ds.get_timestamp_idx(timestamp))

        return data_dict

    def get_timestamp(self, idx: int) -> pd.Timestamp:
        """
        Retrieves the timestamp at the specified index.

        Args:
            idx (int): Index of the timestamp.

        Returns:
            pd.Timestamp: Timestamp at the specified index.
        """
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp: pd.Timestamp) -> int:
        """
        Retrieves the index of the specified timestamp.

        Args:
            timestamp (pd.Timestamp): Timestamp to find the index for.

        Returns:
            int: Index of the specified timestamp.
        """
        return self.timestamps.index(timestamp)

    def __repr__(self) -> str:
        """
        Returns a string representation of the MultiSignalDataset object.
        """
        return f"MultiSignalDataset - {len(self)} samples\nDatasets: {len(self.single_signal_datasets)}"


class MultiSignalDataset_no_pandas(DatasetBase):
    def __init__(
        self,
        single_signal_datasets: list[DatasetBase],
        aggregation="all",
        fill="none",
        time_cut=60,  # in minutes
    ):
        super().__init__()

        self.single_signal_datasets = single_signal_datasets
        n_ssds = len(self.single_signal_datasets)
        assert n_ssds > 0, "No datasets provided."

        self.timestamp_dict = {}
        if aggregation == "all":
            for ssd in self.single_signal_datasets:
                for idx in range(len(ssd)):
                    self.timestamp_dict[ssd.get_timestamp(idx)] = [None] * n_ssds

        elif aggregation == "common":
            common_timestamps = set(self.single_signal_datasets[0].timestamps)
            for ssd in self.single_signal_datasets[1:]:
                common_timestamps.intersection_update(set(ssd.timestamps))
            self.timestamp_dict = {k: [None] * n_ssds for k in common_timestamps}

        elif aggregation[:2] == "I:":
            idx = int(aggregation[2:])
            i_timestamps = set(self.single_signal_datasets[idx].timestamps)
            self.timestamp_dict = {k: [None] * n_ssds for k in i_timestamps}

        else:
            raise ValueError(f"Aggregation {aggregation} not valid.")

        if fill == "none":
            print("Filling missing timestamps: none")
            for i, ssd in enumerate(self.single_signal_datasets):
                for j, timestamp in enumerate(ssd.timestamps):
                    if timestamp in self.timestamp_dict:
                        self.timestamp_dict[timestamp][i] = j

        elif fill == "last":
            timestamps = sorted(self.timestamp_dict.keys())

            found = False
            while not found:
                found = True
                for ssd in self.single_signal_datasets:
                    if ssd.get_timestamp(0) > timestamps[0]:
                        k = timestamps.pop(0)
                        del self.timestamp_dict[k]
                        found = False
                        break

            for i, ssd in enumerate(self.single_signal_datasets):
                for idx in range(len(ssd)):
                    if ssd.get_timestamp(idx) > timestamps[0]:
                        self.timestamp_dict[timestamps[0]][i] = idx - 1
                        break

            for t0, t1 in zip(timestamps[:-1], timestamps[1:]):
                for i, ssd in enumerate(self.single_signal_datasets):
                    idx_before = self.timestamp_dict[t0][i]
                    for idx in range(idx_before, len(ssd)):
                        if ssd.get_timestamp(idx) == t1:
                            self.timestamp_dict[t1][i] = idx
                            break

                        elif ssd.get_timestamp(idx) > t1:
                            self.timestamp_dict[t1][i] = idx - 1
                            break

                    if self.timestamp_dict[t1][i] is None:
                        self.timestamp_dict[t1][i] = len(ssd) - 1

        elif fill == "closest":
            timestamps = sorted(self.timestamp_dict.keys())
            timestamps_dt = [convert_to_datetime(ts) for ts in timestamps]

            found = False
            while not found:
                found = True
                for ssd in self.single_signal_datasets:
                    if convert_to_datetime(ssd.get_timestamp(0)) > timestamps_dt[0]:
                        k = timestamps.pop(0)
                        timestamps_dt.pop(0)
                        del self.timestamp_dict[k]
                        found = False
                        break

            for i, ssd in enumerate(self.single_signal_datasets):
                for idx in range(len(ssd)):
                    current_ts = convert_to_datetime(ssd.get_timestamp(idx))
                    if current_ts >= timestamps_dt[0]:
                        if idx == 0 or abs(current_ts - timestamps_dt[0]) < abs(
                            convert_to_datetime(ssd.get_timestamp(idx - 1))
                            - timestamps_dt[0]
                        ):
                            self.timestamp_dict[timestamps[0]][i] = idx
                        else:
                            self.timestamp_dict[timestamps[0]][i] = idx - 1
                        break

            for t0, t1 in zip(timestamps_dt[:-1], timestamps_dt[1:]):
                t0_key = timestamps[timestamps_dt.index(t0)]
                t1_key = timestamps[timestamps_dt.index(t1)]
                for i, ssd in enumerate(self.single_signal_datasets):
                    idx_before = self.timestamp_dict[t0_key][i]
                    closest_idx = None
                    closest_diff = timedelta(minutes=time_cut)
                    for idx in range(idx_before, len(ssd)):
                        current_ts = convert_to_datetime(ssd.get_timestamp(idx))
                        current_diff = abs(current_ts - t1)
                        if current_diff < closest_diff:
                            closest_diff = current_diff
                            closest_idx = idx

                        if current_ts >= t1:
                            break

                    if closest_idx is not None:
                        self.timestamp_dict[t1_key][i] = closest_idx
                    else:

                        raise ValueError(
                            f"No sufficiently close timestamp found for index {i} between {t0} and {t1}."
                        )

        else:
            raise ValueError(f"Fill {fill} not valid.")

        self._timestamps = sorted(self.timestamp_dict.keys())

    @property
    def timestamps(self):
        return self._timestamps

    def __len__(self):
        return len(self.timestamps)

    def get_data(self, idx):
        timestamp = self.get_timestamp(idx)
        return_dict = {}
        for dataset, i in zip(
            self.single_signal_datasets, self.timestamp_dict[timestamp]
        ):
            if i is None:
                tmp_data = {
                    f"{dataset.satellite_name}_{sensor_id}": None
                    for sensor_id in dataset.sensor_ids
                }
            else:
                data = dataset[i]
                del data["timestamp"]
                del data["idx"]

                tmp_data = {f"{dataset.satellite_name}_{k}": v for k, v in data.items()}

            for k in tmp_data:
                assert k not in return_dict, f"Key {k} already exists."

            return_dict |= tmp_data

        return return_dict

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp):
        return self.timestamps.index(timestamp)

    @property
    def sensor_id(self):
        return "multi"

    @property
    def id(self):
        return ""

    @property
    def satellite_id(self):
        return "multi"

    def __repr__(self) -> str:
        print_string = f"MultiSignalDataset - {len(self)} samples\n"
        print_string += f"Datasets: {len(self.single_signal_datasets)}\n"

        for i, d in enumerate(self.single_signal_datasets):
            inner_repr = repr(d)
            lines = inner_repr.split("\n")
            inner_repr = "\n".join(["\t" + line for line in lines])

            print_string += f"{i} -------------\n"
            print_string += inner_repr
            print_string += "------------------\n"
        return print_string
