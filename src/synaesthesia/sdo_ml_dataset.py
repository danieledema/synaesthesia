# Adapted to be general from https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/src/irradiance/utilities/data_loader.py

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import ALL_COMPONENTS, ALL_IONS, ALL_WAVELENGTHS


class SDOMLDataset(Dataset):
    def __init__(
        self,
        aligndata,
        hmi_data,
        aia_data,
        eve_data,
        components,
        wavelengths,
        ions,
        freq,
        months,
        normalizations=None,
        mask=None,
        num_frames=1,
        drop_frame_dim=False,
        min_date=None,
        max_date=None,
    ):
        """
        aligndata --> aligned indexes for input-output matching
        aia_data --> zarr: aia data in zarr format
        eve_path --> zarr: eve data in zarr format
        hmi_path --> zarr: hmi data in zarr format
        components --> list: list of magnetic components for hmi (Bx, By, Bz)
        wavelengths   --> list: list of channels for aia (94, 131, 171, 193, 211, 304, 335, 1600, 1700)
        ions          --> list: list of ions for eve (MEGS A and MEGS B)
        freq          --> str: cadence used for rounding time series
        transformation: to be applied to aia in theory, but can stay None here
        use_normalizations: to use or not use normalizations, e.g. if this is test data, we don't want to use normalizations
        mask: to apply or not apply the HMI mask to AIA and HMI data
        """
        super().__init__()

        self.aligndata = aligndata

        self.aia_data = aia_data
        self.eve_data = eve_data
        self.hmi_data = hmi_data

        self.mask = mask

        # Select alls
        self.components = components
        self.wavelengths = wavelengths
        self.ions = ions

        # Loading data
        # HMI
        if self.hmi_data is not None:
            if self.components is None:
                self.components = ALL_COMPONENTS
            self.components.sort()
        # AIA
        if self.aia_data is not None:
            if self.wavelengths is None:
                self.wavelengths = ALL_WAVELENGTHS
            self.wavelengths.sort()
        # EVE
        if self.eve_data is not None:
            if self.ions is None:
                self.ions = ALL_IONS
            self.ions.sort()
        self.cadence = freq
        self.months = months

        self.normalizations = normalizations

        # get data from path
        self.aligndata = self.aligndata.loc[
            self.aligndata.index.month.isin(self.months), :
        ]

        # Determine the available date range
        data_start_date = self.aligndata.index.min()
        data_end_date = self.aligndata.index.max()

        # Validate and adjust date range if needed
        if min_date and max_date:
            if min_date < data_start_date or max_date > data_end_date:
                print(
                    f"Warning: Specified date range ({min_date} to {max_date}) is outside the available data range ({data_start_date} to {data_end_date}). Adjusting to available range."
                )
                min_date = max(min_date, data_start_date)
                max_date = min(max_date, data_end_date)

            print(f"Filtering data to date range: {min_date} to {max_date}")
            self.aligndata = self.aligndata[
                (self.aligndata.index >= min_date) & (self.aligndata.index <= max_date)
            ]

        # number of frames to return per sample
        self.num_frames = num_frames
        self.drop_frame_dim = drop_frame_dim  # for backwards compat
        if self.drop_frame_dim:
            assert self.num_frames == 1

    def __len__(self):
        # report slightly smaller such that all frame sets requested are available
        return self.aligndata.shape[0] - (self.num_frames - 1)

    def __getitem__(self, idx):

        image_stack = None
        if self.aia_data is not None:
            image_stack = self.get_aia_image(idx)

        if self.hmi_data is not None:
            image_stack = np.concatenate((image_stack, self.get_hmi_image(idx)), axis=0)

        if self.eve_data is not None:
            eve_data = self.get_eve(idx)
            return image_stack, eve_data.reshape(-1)
        else:
            return image_stack

    def get_aia_image(self, idx):
        """Get AIA image for a given index.
        Returns a numpy array of shape (num_wavelengths, num_frames, height, width).
        """
        aia_image_dict = {}
        for wavelength in self.wavelengths:
            aia_image_dict[wavelength] = []
            for frame in range(self.num_frames):
                idx_row_element = self.aligndata.iloc[idx + frame]
                idx_wavelength = idx_row_element[f"idx_{wavelength}"]
                year = str(idx_row_element.name.year)
                img = self.aia_data[year][wavelength][idx_wavelength, :, :]

                if self.mask is not None:
                    img = img * self.mask

                aia_image_dict[wavelength].append(img)

                if self.normalizations:
                    aia_image_dict[wavelength][-1] -= self.normalizations["AIA"][
                        wavelength
                    ]["mean"]
                    aia_image_dict[wavelength][-1] /= self.normalizations["AIA"][
                        wavelength
                    ]["std"]

        aia_image = np.array(list(aia_image_dict.values()))

        return aia_image[:, 0, :, :] if self.drop_frame_dim else aia_image

    def get_eve(self, idx):
        """Get EVE data for a given index.
        Returns a numpy array of shape (num_ions, num_frames, ...).
        """
        eve_ion_dict = {}
        for ion in self.ions:
            eve_ion_dict[ion] = []
            for frame in range(self.num_frames):
                idx_eve = self.aligndata.iloc[idx + frame]["idx_eve"]
                eve_ion_dict[ion].append(self.eve_data[ion][idx_eve])
                if self.normalizations:
                    eve_ion_dict[ion][-1] -= self.normalizations["EVE"][ion]["mean"]
                    eve_ion_dict[ion][-1] /= self.normalizations["EVE"][ion]["std"]

        eve_data = np.array(list(eve_ion_dict.values()), dtype=np.float32)

        return eve_data

    def get_hmi_image(self, idx):
        """Get HMI image for a given index.
        Returns a numpy array of shape (num_channels, num_frames, height, width).
        """
        hmi_image_dict = {}
        for component in self.components:
            hmi_image_dict[component] = []
            for frame in range(self.num_frames):
                idx_row_element = self.aligndata.iloc[idx + frame]
                idx_component = idx_row_element[f"idx_{self.components[0]}"]
                year = str(idx_row_element.name.year)

                img = self.hmi_data[year][component][idx_component, :, :]

                if self.mask is not None:
                    img = img * self.mask

                hmi_image_dict[component].append(img)

                if self.normalizations:
                    hmi_image_dict[component][-1] -= self.normalizations["HMI"][
                        component
                    ]["mean"]
                    hmi_image_dict[component][-1] /= self.normalizations["HMI"][
                        component
                    ]["std"]

        hmi_image = np.array(list(hmi_image_dict.values()))

        return hmi_image[:, 0, :, :] if self.drop_frame_dim else hmi_image

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output
