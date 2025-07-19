from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

from ..abstract.dataset_base import DatasetBase
from .multi_file_dataset import MultiFileDataset


class ImageDataset(MultiFileDataset):
    """
    Dataset class for Image data.

    This dataset handles image files in a folder, converting them to a specified format
    and providing them as numpy arrays with channel-first ordering (C, H, W).

    Properties:
    ----
    sensor_ids : List[str]
        Returns a list containing the image format as the sensor identifier.
    """

    def __init__(
        self,
        folder_path: str | Path,
        extension: str,
        format: str = "RGB",
    ):
        """
        Initialize the ImageDataset.

        Args:
            folder_path: Path to the folder containing image files
            extension: File extension to filter for (without the dot)
            format: Image format to convert to (e.g., 'RGB', 'RGBA', 'L')

        Raises:
            FileNotFoundError: If the folder path does not exist
            ValueError: If no image files are found or format is invalid
        """
        self.format = format
        super().__init__(folder_path, extension)

    def read_data(self, file_path: Path) -> Dict[str, Any]:
        """
        Read and process image data from a file.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary containing the processed image data with format as key

        Raises:
            ValueError: If image cannot be read or processed
        """
        try:
            image = Image.open(file_path)
            image = image.convert(self.format)
            image_np = np.array(image)

            # Handle alpha channel removal for RGBA images
            if image_np.ndim == 3 and image_np.shape[-1] == 4 and self.format != "RGBA":
                image_np = image_np[:, :, :3]  # Remove alpha channel

            # Convert to channel-first format (C, H, W)
            if image_np.ndim == 3:
                image_np = image_np.transpose(2, 0, 1)
            elif image_np.ndim == 2:
                image_np = image_np[None, :, :]  # Add channel dimension

            return {self.format: image_np}

        except Exception as e:
            raise ValueError(f"Failed to read image {file_path}: {e}")

    @property
    def id(self) -> str:
        """
        Return the dataset identifier.

        Returns:
            String identifier for the dataset type
        """
        return "image"

    @property
    def sensor_ids(self) -> List[str]:
        """
        Return a list of sensor IDs in the dataset.

        Returns:
            List containing the image format as sensor identifier
        """
        return [self.format]


class ImageFromVideoDataset(DatasetBase):
    """
    Dataset for extracting images from video files.

    This dataset treats a video file as a sequence of images, extracting frames
    at specified timestamps or generating timestamps based on video FPS.
    Timestamps are converted to integers (milliseconds) to match DatasetBase specification.

    Methods:
    ----
    read_timestamps(timestamp_path: Path) -> List[int]:
        Abstract method to read timestamps from an external file.

    Properties:
    ----
    timestamps : List[int]
        Returns a list of timestamps for video frames in milliseconds.

    sensor_ids : List[str]
        Returns ['RGB'] as the sensor identifier.
    """

    def __init__(self, video_path: Path, timestamp_path: Path | None = None):
        """
        Initialize the ImageFromVideoDataset.

        Args:
            video_path: Path to the video file
            timestamp_path: Optional path to file containing timestamps.
                          If None, timestamps are generated from video FPS

        Raises:
            ValueError: If video cannot be opened
            FileNotFoundError: If video file does not exist
        """
        super().__init__()

        self.video_path = Path(video_path)
        self.timestamp_path = timestamp_path

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Initialize timestamps
        self._timestamps = self._load_timestamps()

    def _load_timestamps(self) -> List[int]:
        """
        Load timestamps from file or generate from video properties.

        Returns:
            List of timestamps for video frames in milliseconds

        Raises:
            ValueError: If video properties cannot be read
        """
        if self.timestamp_path is not None:
            return self.read_timestamps(self.timestamp_path)
        else:
            # Generate timestamps based on video FPS
            with self._get_video_capture() as cap:
                fps = cap.get(cv2.CAP_PROP_FPS)
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if fps <= 0:
                    raise ValueError(f"Invalid FPS value: {fps}")
                if num_frames <= 0:
                    raise ValueError(f"Invalid frame count: {num_frames}")

                # Convert to milliseconds and return as integers
                return [int((i / fps) * 1000) for i in range(num_frames)]

    def _get_video_capture(self):
        """
        Context manager for video capture to ensure proper resource cleanup.

        Returns:
            Context manager that yields cv2.VideoCapture object
        """

        class VideoCapture:
            def __init__(self, video_path):
                self.video_path = video_path
                self.cap = None

            def __enter__(self):
                self.cap = cv2.VideoCapture(str(self.video_path))
                if not self.cap.isOpened():
                    raise ValueError(f"Could not open video: {self.video_path}")
                return self.cap

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.cap:
                    self.cap.release()

        return VideoCapture(self.video_path)

    @property
    def timestamps(self) -> List[int]:
        """
        Return a list of all timestamps in the dataset.

        Returns:
            List of timestamps for video frames in milliseconds
        """
        return self._timestamps

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of frames in the video
        """
        return len(self.timestamps)

    @abstractmethod
    def read_timestamps(self, timestamp_path: Path) -> List[int]:
        """
        Read timestamps from an external file.

        Args:
            timestamp_path: Path to file containing timestamps

        Returns:
            List of timestamps in milliseconds
        """
        pass

    def get_timestamp(self, idx: int) -> int:
        """
        Get the timestamp at the specified index.

        Args:
            idx: Index of the frame

        Returns:
            Timestamp as integer (milliseconds)

        Raises:
            IndexError: If index is out of range
        """
        if idx >= len(self.timestamps) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for {len(self.timestamps)} frames"
            )
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp: int) -> int:
        """
        Get the index corresponding to a timestamp.

        Args:
            timestamp: Timestamp to find (in milliseconds)

        Returns:
            Index corresponding to the timestamp

        Raises:
            ValueError: If timestamp is not found
        """
        try:
            return self.timestamps.index(timestamp)
        except ValueError:
            raise ValueError(f"Timestamp {timestamp} not found in dataset")

    def get_data(self, idx: int) -> Dict[str, Any]:
        """
        Get the raw data at the specified index.

        Extract frame at given index from video and convert to RGB format
        with channel-first ordering (C, H, W).

        Args:
            idx: Index of the frame to extract

        Returns:
            Dictionary containing the RGB frame data

        Raises:
            IndexError: If index is out of range
            ValueError: If frame cannot be read
        """
        # Use context manager for thread safety
        with self._get_video_capture() as cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                raise ValueError(f"Could not read frame at index {idx}")

            # Convert BGR to RGB and transpose to channel-first
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.ndim == 3:
                frame = frame.transpose(2, 0, 1)

            return {"RGB": frame}

    @property
    def id(self) -> str:
        """
        Return the dataset identifier.

        Returns:
            String identifier for the dataset type
        """
        return "video_image"

    @property
    def sensor_ids(self) -> List[str]:
        """
        Return a list of sensor IDs in the dataset.

        Returns:
            List containing 'RGB' as the sensor identifier
        """
        return ["RGB"]
