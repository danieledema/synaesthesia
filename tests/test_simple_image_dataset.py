from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .simple_image_dataset import SimpleImageDataset

FOLDER = "./tests/test_data/image_dataset"


@pytest.fixture(scope="module")
def create_dataset_png_rgb():
    folder = Path(FOLDER) / "png_rgb"
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(10):

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :, 0] = i * 10

        image = Image.fromarray(image)
        image.save(folder / f"{i}.png")

    return folder


@pytest.fixture(scope="module")
def create_dataset_jpg_rgb():
    folder = Path(FOLDER) / "jpg_rgb"
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(10):

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :, 0] = i * 10

        image = Image.fromarray(image)
        image.save(folder / f"{i}.jpg")

    return folder


@pytest.fixture(scope="module")
def create_dataset_png_gray():
    folder = Path(FOLDER) / "png_gray"
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(10):

        image = np.zeros((4, 4), dtype=np.uint8)
        image[:, :] = i * 10

        image = Image.fromarray(image)
        image.save(folder / f"{i}.png")

    return folder


@pytest.fixture(scope="module")
def create_dataset_jpg_gray():
    folder = Path(FOLDER) / "jpg_gray"
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(10):

        image = np.zeros((4, 4), dtype=np.uint8)
        image[:, :] = i * 10

        image = Image.fromarray(image)
        image.save(folder / f"{i}.jpg")

    return folder


def test_dataset_length(create_dataset_png_rgb):
    data_folder = create_dataset_png_rgb

    dataset = SimpleImageDataset(data_folder, "png")
    assert len(dataset) == 10


def test_dataset_getitem_jpg_rgb(create_dataset_jpg_rgb):
    data_folder = create_dataset_jpg_rgb

    dataset = SimpleImageDataset(data_folder, "jpg")

    for i in range(10):
        data = dataset[i]

        assert data["idx"] == i
        assert data["timestamp"] == i
        assert "camera-RGB" in data

        image = data["camera-RGB"]
        assert image.shape == (4, 4, 3)
        # cannot check the exact value because of the compression


def test_dataset_getitem_png_rgb(create_dataset_png_rgb):
    data_folder = create_dataset_png_rgb

    dataset = SimpleImageDataset(data_folder, "png")

    for i in range(10):
        data = dataset[i]

        assert data["idx"] == i
        assert data["timestamp"] == i
        assert "camera-RGB" in data

        image = data["camera-RGB"]
        assert image.shape == (4, 4, 3)
        assert image[0, 0, 0] == i * 10
        # cannot check the exact value because of the compression


def test_dataset_getitem_jpg_gray(create_dataset_jpg_gray):
    data_folder = create_dataset_jpg_gray

    dataset = SimpleImageDataset(data_folder, "jpg", format="L")

    for i in range(10):
        data = dataset[i]

        assert data["idx"] == i
        assert data["timestamp"] == i
        assert "camera-RGB" in data

        image = data["camera-RGB"]
        assert image.shape == (4, 4, 1)


def test_dataset_getitem_png_gray(create_dataset_png_gray):
    data_folder = create_dataset_png_gray

    dataset = SimpleImageDataset(data_folder, "png", format="L")

    for i in range(10):
        data = dataset[i]

        assert data["idx"] == i
        assert data["timestamp"] == i
        assert "camera-RGB" in data

        image = data["camera-RGB"]
        assert image.shape == (4, 4, 1)
        assert image[0, 0, 0] == i * 10
