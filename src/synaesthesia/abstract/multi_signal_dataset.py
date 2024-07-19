from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .abstract_dataset import DatasetBase
from .utils import convert_to_datetime


def convert_to_datetime(timestamp):
    # Adjust this function based on the format of your timestamps
    try:
        return datetime.strptime(timestamp, "%Y%m%dT%H%M")
    except ValueError:
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%S%f")


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
        patch: str = "drop",
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
        self.patch = patch

        # Create a DataFrame to store timestamps and corresponding indices
        self.timestamp_df = self._initialize_timestamp_df()

        # Fill missing timestamps based on the fill method
        self._fill_missing_timestamps()

        self._timestamps = [np.datetime64(ts, "ns") for ts in self.timestamp_df.index]

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

                if self.patch == "drop":
                    print("Dropping timestamps that couldn't be filled...")
                    # Remove timestamps that couldn't be filled (still None after interpolation)
                    self.timestamp_df = self.timestamp_df.dropna()

    @property
    def timestamps(self) -> List[np.datetime64]:
        """
        Returns the list of timestamps in numpy.datetime64 format.
        """
        return self._timestamps

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
            data_dict[f"{ds.id}"] = ds.get_data(ds.get_timestamp_idx(timestamp))

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
        print_string = f"MultiSignalDataset - {len(self)} samples\nDatasets: {len(self.single_signal_datasets)}\n"
        for i, d in enumerate(self.single_signal_datasets):
            inner_repr = repr(d)
            lines = inner_repr.split("\n")
            inner_repr = "\n".join(["\t" + line for line in lines])

            print_string += f"{i} -------------\n"
            print_string += inner_repr
            print_string += "------------------\n"
        return print_string

    @property
    def id(self):
        """
        Returns the ID of the dataset.
        """
        return "MultiSignalDataset"
