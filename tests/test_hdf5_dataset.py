import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.synaesthesia.base_sensors.hdf5_dataset import Hdf5Dataset


def test_hdf5_dataset_basic(tmp_path):
    """
    Create a small HDF5 file with timestamps and two sensor datasets (acc, gyro),
    instantiate Hdf5Dataset and verify basic API:
      - length
      - timestamps / get_timestamp
      - get_data returns expected keys and shapes
      - get_timestamp_idx finds the correct index
    Also exercise DataLoader behavior (single-worker). If DataLoader / multiprocessing
    isn't available in the environment, skip that portion.
    """
    # Prepare test HDF5 file
    file_path = tmp_path / "test_data.h5"
    N = 12
    acc_data = np.arange(N * 3, dtype=np.float32).reshape(N, 3)  # shape (N,3)
    gyro_data = (
        np.arange(N * 3, dtype=np.float32).reshape(N, 3) + 100.0
    )  # offset to distinguish

    with h5py.File(str(file_path), "w") as f:
        f.create_dataset("timestamps", data=np.arange(N, dtype=np.int64))
        f.create_dataset("acc", data=acc_data)
        f.create_dataset("gyro", data=gyro_data)

    # Instantiate dataset with sensor_map
    sensor_map = {"acc": "acc", "gyro": "gyro"}
    ds = Hdf5Dataset(
        str(file_path),
        sensor_map=sensor_map,
        timestamps_path="timestamps",
        id="imu",
        machine_name="LeftArm",
    )

    # Basic properties
    assert len(ds) == N
    assert isinstance(ds.timestamps, list)
    assert ds.get_timestamp(0) == 0
    assert ds.get_timestamp(N - 1) == N - 1

    # get_data and shapes
    idx = 5
    sample = ds.get_data(idx)
    assert "acc" in sample and "gyro" in sample
    acc_sample = sample["acc"]
    gyro_sample = sample["gyro"]
    assert isinstance(acc_sample, np.ndarray)
    assert isinstance(gyro_sample, np.ndarray)
    assert acc_sample.shape == (3,)
    assert gyro_sample.shape == (3,)

    # get_timestamp_idx
    t = ds.get_timestamp(idx)
    assert ds.get_timestamp_idx(t) == idx

    # Try num_workers=0 and attempt num_workers=1; skip worker test on failure
    for nworkers in (0, 1):
        dl = DataLoader(ds, batch_size=2, num_workers=nworkers)
        batch = next(iter(dl))

        # Expect keys to be present and tensors to have correct batch dimension
        assert "imu-acc" in batch and "imu-gyro" in batch
        acc_batch = batch["imu-acc"]
        gyro_batch = batch["imu-gyro"]
        # After default collate, numpy arrays become tensors with shape (batch, ...).
        assert isinstance(acc_batch, torch.Tensor)
        assert isinstance(gyro_batch, torch.Tensor)
        assert acc_batch.shape[0] == 2
        assert gyro_batch.shape[0] == 2

    # Close dataset resources explicitly
    ds.close()
    # Ensure close is idempotent
    ds.close()
