from pathlib import Path
from .abstract_dataset import SingleSignalDatasetBase

import netCDF4 as nc
from ncflag import FlagWrap

import cftime

import numpy as np
import matplotlib.pyplot as plt


class XRayDataset(SingleSignalDatasetBase):
    """
    Dataset class for X-ray data.
    """

    def __init__(
        self,
        folder_path: str | Path,
        datatype: str = "flsum",
        goesnr: str = "16",
        level: int = 2,
        variables_to_include: list[str] = None,
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
            self.variables_to_include = []
        else:
            self.variables_to_include = (
                variables_to_include
                if variables_to_include is not None
                else [
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
        all_fluxes = []
        additional_data = {var: [] for var in self.variables_to_include}

        for file in self.files:
            with nc.Dataset(file) as nc_in:
                times = cftime.num2pydate(
                    nc_in.variables["time"][:], nc_in.variables["time"].units
                )
                xrsb_flux = np.ma.filled(
                    nc_in.variables["xrsb_flux"][:], fill_value=np.nan
                )

                if datatype == "avg1m":
                    xrsb_flags = FlagWrap.init_from_netcdf(nc_in.variables["xrsb_flag"])
                    # set any points that are NOT good_data to nan
                    good_data = xrsb_flags.get_flag("good_data")
                    xrsb_flux[~good_data] = np.nan

                for var in self.variables_to_include:
                    if var in nc_in.variables:
                        var_data = np.ma.filled(
                            nc_in.variables[var][:], fill_value=np.nan
                        )
                        additional_data[var].extend(var_data)
                    else:
                        raise ValueError(f"Variable '{var}' not found in file {file}")

                all_times.extend(times)
                all_fluxes.extend(xrsb_flux)

        # Convert times to the desired format 'YYYYMMDDTHHMMSSfff'
        self.timestamps = np.array(
            [time.strftime("%Y%m%dT%H%M%S%f") for time in all_times]
        )
        self.xrsb_flux = np.array(all_fluxes)
        self.additional_data = {
            var: np.array(data) for var, data in additional_data.items()
        }

    @property
    def sensor_id(self):
        """
        Property returning the sensor ID.

        Returns:
            str: Sensor ID.
        """
        return "XRSB"

    def __len__(self):
        return len(self.timestamps)

    def get_data(self, idx, n_timesteps=1):
        """
        Retrieves the data starting at the specified index and includes data for a number of subsequent timesteps.

        Args:
            idx (int): Index of the starting data point to retrieve.
            n_timesteps (int, optional): Number of timesteps to include from the starting index. Defaults to 1.

        Returns:
            tuple: Tuple containing the timestamps and a dictionary with arrays of data for the specified range of indices.
        """
        if idx >= len(self) or idx + n_timesteps > len(self):
            raise IndexError("Index out of range or n_timesteps exceeds data length")

        timestamps = self.timestamps[idx : idx + n_timesteps]

        # Initialize the dictionary with keys and empty lists
        data_list = {"xrsb_flux": []}
        for var in self.variables_to_include:
            data_list[var] = []

        # Populate the lists with data
        for i in range(idx, idx + n_timesteps):
            data_list["xrsb_flux"].append(self.xrsb_flux[i])
            for var in self.variables_to_include:
                data_list[var].append(self.additional_data[var][i])

        return timestamps, data_list

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

    def get_index_for_timestamp(self, date_str: str):
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

    @staticmethod
    def get_timestamp_from_filename(filename):
        raise NotImplementedError
