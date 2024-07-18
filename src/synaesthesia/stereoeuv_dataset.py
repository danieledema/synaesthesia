from .abstract_dataset import DatasetBase
from collections import OrderedDict
from pathlib import Path
from sunpy.map import Map
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import astropy.units as u


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
            map_obj.plot(axes=ax, clip_interval=(50, 99.99)*u.percent)
            map_obj.draw_limb(axes=ax)
            map_obj.draw_grid(axes=ax)
            ax.set_position([0.1, 0.1, 0.8, 0.7])
            plt.savefig(f"plots/image_{i}.png")

        print(map_obj.meta)

    def rotate_image(self, image_path: str | Path, angle: float) -> Image:
        """Rotate an image by a given angle.

        Args:
            image_path (str | Path): Path to the image to rotate.
            angle (float): Angle to rotate the image.

        Returns:
            Image: The rotated image.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Open the image
        image = Image.open(image_path)

        # Rotate the image
        rotated_image = image.rotate(angle, expand=True)

        return rotated_image
    
    def crop_image(self, image_data: np.ndarray) -> np.ndarray:
        """Crop an image using specified dimensions.

        Args:
            image_data (np.ndarray): Image data as a NumPy array.

        Returns:
            np.ndarray: Cropped image data.
        """
        # Example cropping dimensions (adjust as needed)
        cropped_data = image_data[100:400, 200:500]

        return cropped_data