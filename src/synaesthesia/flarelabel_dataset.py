from datetime import datetime
from pathlib import Path

import cftime
import netCDF4 as nc
import numpy as np
import pandas as pd
from ncflag import FlagWrap

from .abstract_dataset import DatasetBase


class FlarelabelDataset(DatasetBase):
    """
    Dataset class fl2cme.csv
    """

    def __init__(
        self,
        path: str | Path,
    ):
        """
        Initializes the Flarelabel dataset.

        Args:
            folder_path (str or Path): Path to the folder containing the data files.
        """
        super().__init__()

        self.folder_path = Path(path)
        self.datatype = datatype
        self.goesnr = goesnr

        if datatype == "avg1m":
            assert variables_to_include == [
                "xrsb_flux"
            ], "Only xrsb_flux is available for avg1m data"
        self.variables_to_include = (
            variables_to_include
            if variables_to_include is not None
            else [
                "xrsb_flux",
                "status",
                "background_flux",
                "flare_class",
                "integrated_flux",
                "flare_id",
            ]
        )

        if not self.files:
            raise ValueError("No files found for the specified parameters.")

        all_times = []
        data = {var: [] for var in self.variables_to_include}

        for file in self.files:
            nc_in = nc.Dataset(file)
            times = cftime.num2pydate(
                nc_in.variables["time"][:], nc_in.variables["time"].units
            )

            for var in self.variables_to_include:
                if var in nc_in.variables:
                    var_data = np.ma.filled(nc_in.variables[var][:], fill_value=np.nan)
                    if datatype == "avg1m":
                        xrsb_flags = FlagWrap.init_from_netcdf(
                            nc_in.variables["xrsb_flag"]
                        )
                        # set any points that are NOT good_data to nan
                        good_data = xrsb_flags.get_flag("good_data")
                        var_data[~good_data] = np.nan
                    data[var].extend(var_data)
                else:
                    raise ValueError(f"Variable '{var}' not found in file {file}")

            all_times.extend(times)
            nc_in.close()

        # Convert times to datetime objects
        self._timestamps = np.array(all_times)
        self.data = {var: np.array(data) for var, data in data.items()}
        # Apply the filter to the data
        if filtered == True:
            self._filter_data()

    def _filter_data(self):
        """
        Filter the dataset according to specified conditions.
        """
        print("Filtering data for problematic X-ray flux values...")
        # Convert data to a DataFrame for easier filtering
        df = pd.DataFrame(self.data)
        df["timestamps"] = self._timestamps

        # Apply the filter conditions
        filtered_df = df[(df["xrsb_flux"] < 1e-7) | (df["status"] == "EVENT_PEAK")]

        # Update the dataset with filtered data
        self._timestamps = filtered_df["timestamps"].values
        self.data = {var: filtered_df[var].values for var in self.variables_to_include}

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def sensor_ids(self):
        """
        Property returning the sensor ID.

        Returns:
            str: Sensor ID.
        """
        return [f"XRSB-{var}" for var in self.variables_to_include]

    def __len__(self):
        return len(self.timestamps)

    def get_data(self, idx):
        """
        Retrieves the data starting at the specified index and includes data for a number of subsequent timesteps.

        Args:
            idx (int): Index of the starting data point to retrieve.
            n_timesteps (int, optional): Number of timesteps to include from the starting index. Defaults to 1.

        Returns:
            tuple: Tuple containing the timestamps and a dictionary with arrays of data for the specified range of indices.
        """
        data = {}
        for var in self.variables_to_include:
            data[f"{var}"] = self.data[var][idx]
        return data

    def get_timestamp(self, idx):
        """
        Retrieves the timestamp at the specified index.

        Args:
            idx (int): Index of the data point to retrieve the timestamp for.

        Returns:
            datetime: Timestamp at the specified index.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        return self.timestamps[idx]

    def get_timestamp_idx(self, date_time: str):
        """
        Retrieves the index for the given timestamp.

        Args:
            timestamp (str): The timestamp to find the index for, in the format 'YYYYMMDDTHHMMSS'.

        Returns:
            int: The index of the given timestamp in the dataset.

        Raises:
            ValueError: If the timestamp is not found in the dataset.
        """
        timestamp_dt = np.array([dt for dt in self.timestamps if dt == date_time])
        if len(timestamp_dt) == 0:
            raise ValueError(
                f"No timestamp found for date {date_time.strftime("%m/%d/%Y, %H:%M:%S")}"
            )

        return np.where(self.timestamps == timestamp_dt[0])[0][0]

    @property
    def id(self):
        return "XRAY"
