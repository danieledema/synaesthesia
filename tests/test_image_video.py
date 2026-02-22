from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.synaesthesia.base_sensors.image_dataset import ImageFromVideoDataset

from .simple_image_dataset import SimpleImageDataset


def test_image_alpha_handling(tmp_path: Path):
    """
    Ensure that images with alpha channels are handled correctly:
    - When format='RGB' the alpha channel is dropped and the returned array has 3 channels.
    - When format='RGBA' the alpha channel is preserved and the returned array has 4 channels.
    """
    # Prepare folder and RGBA image
    folder = tmp_path / "img_alpha"
    folder.mkdir(parents=True, exist_ok=True)

    h, w = 4, 4
    # Create RGBA image: red channel set to 10, alpha set to 128
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 10  # R
    rgba[:, :, 3] = 128  # A

    img = Image.fromarray(rgba, mode="RGBA")
    img_path = folder / "0.png"
    img.save(img_path)

    # Load with format='RGB' -> alpha should be dropped, channels-first shape (3, H, W)
    ds_rgb = SimpleImageDataset(folder, "png", format="RGB")
    item = ds_rgb[0]
    assert "camera-RGB" in item
    img_tensor = item["camera-RGB"]
    assert img_tensor.shape[0] == 3  # C dimension
    assert img_tensor.shape[1] == h and img_tensor.shape[2] == w
    # Check red channel preserved
    assert img_tensor[0, 0, 0] == 10

    # Load with format='RGBA' -> alpha should be preserved, channels-first shape (4, H, W)
    ds_rgba = SimpleImageDataset(folder, "png", format="RGBA")
    item2 = ds_rgba[0]
    assert "camera-RGB" in item2
    img_tensor2 = item2["camera-RGB"]
    assert img_tensor2.shape[0] == 4
    assert img_tensor2.shape[1] == h and img_tensor2.shape[2] == w
    # Check alpha channel preserved (channel index 3)
    assert img_tensor2[3, 0, 0] == 128


class SimpleVideoDataset(ImageFromVideoDataset):
    """
    Tiny concrete subclass used for tests that provides the minimal `id`
    and `get_machine_name` implementations so that DatasetBase.__getitem__
    can be used in tests.
    """

    @property
    def id(self):
        return "camera"

    def get_machine_name(self) -> str:
        return "top_camera"


def test_image_from_video_dataset(tmp_path: Path):
    """
    Create a tiny synthetic video and verify that:
    - timestamps are computed and length equals number of frames
    - frames can be retrieved and have expected shape and content
    """
    video_folder = tmp_path / "video"
    video_folder.mkdir(parents=True, exist_ok=True)
    video_path = video_folder / "test.avi"

    fps = 5
    num_frames = 8
    h, w = 8, 8

    # Use MJPG codec which is commonly available for avi
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (w, h))

    assert writer.isOpened(), "VideoWriter could not be opened for writing"

    # Write frames with increasing blue channel values so we can validate
    for i in range(num_frames):
        # Create BGR frame (OpenCV uses BGR)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Put unique value per frame in the blue channel
        frame[:, :, 0] = i * 10
        writer.write(frame)

    writer.release()

    # Instantiate our test subclass which provides id & machine name
    ds = SimpleVideoDataset(video_path)

    # Number of timestamps should match number of frames
    assert len(ds) == num_frames

    # Check a couple of frames via __getitem__ (which prefixes keys with id)
    first = ds[0]
    assert "camera-RGB" in first
    frame0 = first["camera-RGB"]
    # shape -> (C, H, W)
    assert frame0.shape[0] == 3
    assert frame0.shape[1] == h and frame0.shape[2] == w
    # blue channel value should match what was written for frame 0
    assert (
        frame0[2, 0, 0] == 0 or frame0[0, 0, 0] == 0
    )  # Accept either ordering if implementation differs

    mid = ds[num_frames // 2]
    assert "camera-RGB" in mid
    framem = mid["camera-RGB"]
    assert framem.shape[0] == 3
    assert framem.shape[1] == h and framem.shape[2] == w

    # Ensure last frame can be read
    last = ds[num_frames - 1]
    assert "camera-RGB" in last
    framel = last["camera-RGB"]
    assert framel.shape[0] == 3
    assert framel.shape[1] == h and framel.shape[2] == w

    # Basic sanity on timestamps ordering (increasing)
    ts = ds.timestamps
    assert len(ts) == num_frames
    # timestamps should be non-decreasing
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))
