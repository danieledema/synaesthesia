from pathlib import Path

import cftime
import netCDF4 as nc
import numpy as np
from ncflag import FlagWrap

from .abstract_dataset import DatasetBase


class XRayDataset(DatasetBase):
    """
    Dataset class for X-ray data.
    """

    def __init__(
        self,
        folder_path: str | Path,
        datatype: str = "flsum",
        goesnr: str = "16",
        level: int = 2,
        variables_to_include: list[str] | None = None,
    ):
        """
        Initializes the X-ray dataset.

        Args:
            folder_path (str or Path): Path to the folder containing the data files.
            datatype (str): Type of data to load (e.g., "flsum", "avg1m").
            level (int, optional): Level of the data hierarchy (default is 2).
            variables_to_include (list of str, optional): List of variable names to include in the dataset.

        """
        super().__init__()

        self.folder_path = Path(folder_path)
        self.files = sorted(
            list(self.folder_path.glob(f"*l{level}*{datatype}_g{goesnr}*.nc"))
        )
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

        if self.files == []:
            raise ValueError("No files found for the specified parameters.")

        all_times = []
        data = {var: [] for var in self.variables_to_include}

        for file in self.files:
            with nc.Dataset(file) as nc_in:
                times = cftime.num2pydate(
                    nc_in.variables["time"][:], nc_in.variables["time"].units
                )

                for var in self.variables_to_include:
                    if var in nc_in.variables:
                        var_data = np.ma.filled(
                            nc_in.variables[var][:], fill_value=np.nan
                        )
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

        # Convert times to the desired format 'YYYYMMDDTHHMMSSfff'
        self._timestamps = np.array(
            [time.strftime("%Y%m%dT%H%M%S%f") for time in all_times]
        )
        self.data = {var: np.array(data) for var, data in data.items()}

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
            data[f"XRSB-{var}"] = self.data[var][idx]
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

    def get_timestamp_idx(self, date_str: str):
        """
        Retrieves the index for the given timestamp.

        Args:
            timestamp (str): The timestamp to find the index for, in the format 'YYYYMMDDT%H%M%S%f'.

        Returns:
            int: The index of the given timestamp in the dataset.

        Raises:
            ValueError: If the timestamp is not found in the dataset.
        """
        # Convert the list of timestamps to strings if they are not already
        timestamp_strs = [str(ts) for ts in self.timestamps]

        # Find the index of the first timestamp that starts with the given date string
        for i, ts in enumerate(timestamp_strs):
            if ts.startswith(date_str):
                return i

        # If no matching timestamp is found, raise an error
        raise ValueError(f"No timestamp found for date {date_str}")
