from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

from ..abstract.dataset_base import DatasetBase


class Hdf5Dataset(DatasetBase):
    """
    Dataset for reading samples from a single HDF5 file.

    Features
    - Worker-safe: the HDF5 file is opened lazily in the process that calls `get_data`.
      This avoids sharing an h5py.File object across processes (which is unsafe).
    - Loads timestamps into memory at construction time (fast lookups).
    - Returns sample dictionaries mapping sensor_id -> numpy array / scalar.
    - Expects each sensor dataset to be indexed along the first axis (time / sample).
      Example: a dataset stored as `/sensors/accelerometer` with shape (N, 3, 128)
      will be indexed as dataset[idx] and returned as a numpy array.

    Parameters
    - file_path: path to the hdf5 file
    - sensor_map: mapping of sensor ids (strings) to dataset paths inside the HDF5 file.
      Example: {"acc": "/sensors/acc", "rgb": "/images/rgb"}
    - timestamps_path: path inside HDF5 file to the timestamps dataset (default: "timestamps")
    - id: optional id prefix for this dataset (used by DatasetBase.__getitem__)
    - machine_name: human-readable machine name for this dataset (must be camel-case)
    """

    def __init__(
        self,
        file_path: str | Path,
        sensor_map: Dict[str, str],
        timestamps_path: str = "timestamps",
        id: str = "",
        machine_name: str = "Hdf5Machine",
    ):
        super().__init__()

        self.file_path = Path(file_path)
        self.sensor_map = dict(sensor_map)  # sensor_id -> hdf5 dataset path
        self.timestamps_path = timestamps_path
        self._id = id
        self._machine_name = machine_name

        # Internal: do not keep an h5py.File open across processes; open temporarily
        # to read timestamps into memory, then close it.
        self._h5_file: Optional[h5py.File] = None
        self._open_in_worker = False

        # Load timestamps eagerly into memory for quick lookups and indexing.
        # This opens the file briefly and then closes it.
        with h5py.File(str(self.file_path), "r") as f:
            if self.timestamps_path not in f:
                # Allow for nested groups, try to access like a path
                raise KeyError(
                    f"Timestamp dataset '{self.timestamps_path}' not found in {self.file_path}"
                )
            ts_ds = f[self.timestamps_path]
            # Read into numpy array of ints; support any dtype convertible to int64
            ts_arr = np.asarray(ts_ds)
            # Ensure integer representation for downstream code (other parts expect int)
            try:
                ts_int = ts_arr.astype("int64")
            except Exception:
                # Fallback: convert elementwise
                ts_int = np.array([int(x) for x in ts_arr], dtype="int64")

            # Expose as a Python list for consistent behavior with other datasets in the library
            self._timestamps = ts_int.tolist()

    # Worker/resource management ------------------------------------------------

    def _ensure_open_in_worker(self) -> None:
        """
        Ensure we have an open h5py.File in the current process for reads.
        This opens the file only if it's not already open.
        """
        if self._h5_file is None:
            # Open read-only; convert Path to str for h5py compatibility
            self._h5_file = h5py.File(str(self.file_path), "r")
            # Mark that this file was opened lazily (useful if you need special cleanup)
            self._open_in_worker = True

    def close(self) -> None:
        """
        Close any open h5py.File held by this object.
        Safe to call multiple times.
        """
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                # best-effort close; don't raise on interpreter shutdown
                pass
            finally:
                self._h5_file = None
                self._open_in_worker = False

    def __del__(self):
        # Try to close file if GC'd
        try:
            self.close()
        except Exception:
            pass

    # DatasetBase API ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._timestamps)

    def get_data(self, idx: int) -> Dict[str, Any]:
        """
        Return a dict mapping sensor_id -> value for the given index.

        The returned values are the raw items read from the HDF5 datasets (numpy
        scalars or numpy arrays). Key names correspond to the `sensor_map` keys.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset length {len(self)}")

        # Open the file in the current process if needed (worker-safe)
        self._ensure_open_in_worker()
        assert self._h5_file is not None  # type checker

        sample: Dict[str, Any] = {}
        for sensor_id, ds_path in self.sensor_map.items():
            if ds_path not in self._h5_file:
                # Try supporting absolute or relative paths; raise clear error otherwise
                raise KeyError(
                    f"Sensor dataset '{ds_path}' (sensor id '{sensor_id}') not found in HDF5 file {self.file_path}"
                )
            ds = self._h5_file[ds_path]

            # Index the dataset along the first axis.
            # h5py returns numpy types/arrays; copy to memory to avoid depending on file-backed objects.
            try:
                value = ds[idx]
            except Exception as e:
                # Provide a helpful error message if indexing fails
                raise IndexError(
                    f"Could not read index {idx} from dataset '{ds_path}': {e!s}"
                ) from e

            # Convert h5py scalar types to native numpy (or Python) objects by wrapping with np.asarray
            # and making a copy to decouple from the file-backed dataset.
            # Many consumers expect numpy arrays or scalars.
            value = (
                np.array(value) if not isinstance(value, np.ndarray) else value.copy()
            )
            sample[sensor_id] = value

        return sample

    def get_timestamp(self, idx: int) -> int:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset length {len(self)}")
        return int(self._timestamps[idx])

    def get_timestamp_idx(self, timestamp: int) -> int:
        """
        Return the index for the exact timestamp match. Raises ValueError if not found.
        """
        # Use list.index to preserve behavior identical to other dataset implementations.
        return self._timestamps.index(int(timestamp))

    # Properties ---------------------------------------------------------------

    @property
    def timestamps(self) -> list[int]:
        # Return the in-memory list of timestamps
        return self._timestamps

    @property
    def sensor_ids(self) -> list[str]:
        # The public sensor ids are the keys of the sensor_map
        return list(self.sensor_map.keys())

    @property
    def id(self) -> str:
        return self._id

    def get_machine_name(self) -> str:
        # Return the provided machine name (DatasetBase.machine_name will validate it)
        return self._machine_name

    # Representation -----------------------------------------------------------

    def __repr__(self) -> str:
        sensor_str = ", ".join(self.sensor_ids)
        return f"{self.machine_name} - {sensor_str}: {len(self)} samples"
