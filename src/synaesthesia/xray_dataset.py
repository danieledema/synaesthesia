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
        self.files = sorted(list(self.folder_path.glob(f"*l{level}*{datatype}*.nc")))
        self.datatype = datatype
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

    def get_data(self, idx):
        """
        Retrieves the data at the specified index.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: Tuple containing the time and data at the specified index.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        data = {"xrsb_flux": self.xrsb_flux[idx]}
        for var in self.variables_to_include:
            data[var] = self.additional_data[var][idx]

        return (
            self.timestamps[idx],
            data,
        )  # TO DO: return multiple timestamps and fluxes if needed

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

    @staticmethod
    def get_timestamp_from_filename(filename):
        raise NotImplementedError
