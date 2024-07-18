from .abstract_dataset import DatasetBase
from collections import OrderedDict
from pathlib import Path
from sunpy.map import Map
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np


class StereoEuvDataset(DatasetBase):

    def __init__(
        self,
        folder_path: str | Path,
        wavelengths: list[str] = [],
        level: int = 2,
        time_threshold: int = 60,
    ):
        super().__init__()

        self.folder_path = Path(folder_path)
        self.wavelengths = wavelengths
        self.level = level
        self.time_threshold = time_threshold

        # Initialize an empty dictionary to hold file paths for each wavelength
        self.files = {}

        # Populate the dictionary with file paths
        all_files = sorted(self.folder_path.glob("*.fts"))

        for i in range(5):
            map_obj = Map(
                all_files[i]
            )  # Specify the missing keyword with an appropriate integer value
            map_obj = map_obj.rotate(missing=np.uint16(0))
            wavelength = map_obj.meta["wavelnth"]
            print(wavelength)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1, projection=map_obj.wcs)
            map_obj.plot(axes=ax)
            plt.savefig(f"plots/image_{i}.png")

        print(map_obj.meta)
