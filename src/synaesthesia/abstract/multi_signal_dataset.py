from loguru import logger
from tqdm import tqdm

from .dataset_base import DatasetBase


# Small, lightweight dummy context used when progress display is disabled.
# We intentionally keep this minimal: it supports `with self._tqdm_ctx(total=...) as pbar:`
# and `pbar.update(n)` calls made in the code.
class _DummyTqdmCtx:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n: int = 1):
        return


class MultiSignalDataset(DatasetBase):
    """
    Dataset class for handling multiple signal datasets.
    """

    def __init__(
        self,
        single_signal_datasets: list[DatasetBase],
        aggregation: str = "all",
        fill: str = "none",
        time_cut: int = 60,  # in minutes
        return_indices: bool = False,
        show_progress: bool = False,
    ):
        """
        Initializes the MultiSignalDataset.

        Args:
            single_signal_datasets (list): List of DatasetBase objects representing single signal datasets.
            aggregation (str): Aggregation method for timestamps ("all", "common", "I:<idx>").
            fill (str): Method for filling missing timestamps ("none", "last", "closest").
            time_cut (int): Time cut-off in minutes for "closest" fill method.
            show_progress (bool): When True, show progress bars via tqdm. Default False.
        """
        super().__init__()

        self.single_signal_datasets = single_signal_datasets
        self.aggregation = aggregation
        self.fill = fill
        self.time_cut = time_cut
        self.return_indices = return_indices
        self.show_progress = show_progress

        # When show_progress is False we avoid creating progress bar wrappers to keep library output clean.
        self._tqdm_iter = (
            tqdm if self.show_progress else (lambda iterable, **kwargs: iterable)
        )
        self._tqdm_ctx = tqdm if self.show_progress else _DummyTqdmCtx

        # Initialize timestamps and data dictionary
        logger.debug("Initializing timestamps...")
        self._timestamps = self._initialize_timestamps()

        logger.debug("Initializing data dictionary...")
        self.data_dict = self._initialize_data_dict()

    def _initialize_timestamps(self) -> list[int]:
        """
        Initializes the DataFrame to store timestamps and corresponding indices.
        """

        if self.aggregation == "all":
            merged_timestamps = self.single_signal_datasets[0].timestamps

            for ds in self._tqdm_iter(
                self.single_signal_datasets[1:], desc="Merging timestamps"
            ):
                i, j = 0, 0

                timestmaps_to_merge = ds.timestamps
                tmp_merged_timestamps = [
                    min(merged_timestamps[0], timestmaps_to_merge[0])
                ]

                # use context wrapper that is a no-op when progress is disabled
                with self._tqdm_ctx(
                    total=len(merged_timestamps) + len(timestmaps_to_merge)
                ) as pbar:
                    while i < len(merged_timestamps) and j < len(timestmaps_to_merge):
                        if tmp_merged_timestamps[-1] == merged_timestamps[i]:
                            i += 1
                            pbar.update(1)
                            continue

                        if tmp_merged_timestamps[-1] == timestmaps_to_merge[j]:
                            j += 1
                            pbar.update(1)
                            continue

                        if merged_timestamps[i] < timestmaps_to_merge[j]:
                            tmp_merged_timestamps.append(merged_timestamps[i])
                            i += 1
                        else:
                            tmp_merged_timestamps.append(timestmaps_to_merge[j])
                            j += 1
                        pbar.update(1)

                # Append any remaining elements from list1 or list2
                tmp_merged_timestamps.extend(merged_timestamps[i:])
                tmp_merged_timestamps.extend(timestmaps_to_merge[j:])
                merged_timestamps = tmp_merged_timestamps

            return merged_timestamps

        elif self.aggregation == "common":
            merged_timestamps = self.single_signal_datasets[0].timestamps

            for ds in self._tqdm_iter(
                self.single_signal_datasets[1:], desc="Merging timestamps"
            ):
                to_delete = []

                with self._tqdm_ctx(total=len(merged_timestamps)) as pbar:
                    i, j = 0, 0
                    while i < len(merged_timestamps) and j < len(ds):
                        if merged_timestamps[i] == ds.get_timestamp(j):
                            i += 1
                            j += 1
                        elif merged_timestamps[i] < ds.get_timestamp(j):
                            to_delete.append(i)
                            i += 1
                        else:
                            j += 1
                        pbar.update(1)

                merged_timestamps = [
                    timestamp
                    for i, timestamp in enumerate(merged_timestamps)
                    if i not in to_delete
                ]

            return merged_timestamps

        elif self.aggregation.startswith("I:"):
            # For 'I:<idx>' aggregation, mark timestamps from a specific dataset
            idx = int(self.aggregation[2:])
            return self.single_signal_datasets[idx].timestamps

        else:
            raise ValueError(f"Invalid aggregation method: {self.aggregation}")

    def _initialize_data_dict(self) -> dict[int, list[int | None]]:
        """
        Fills data vs timestamps based on the specified fill method.
        """

        # Build a mapping timestamp -> [index_for_each_dataset_or_None]
        data_dict = {
            t: [None] * len(self.single_signal_datasets) for t in self._timestamps
        }
        for i, ds in self._tqdm_iter(
            enumerate(self.single_signal_datasets), desc="Filling:"
        ):
            for j, timestamp in self._tqdm_iter(
                enumerate(ds.timestamps), desc=f"Dataset: {i}"
            ):
                if timestamp in data_dict:
                    data_dict[timestamp][i] = j

        if self.fill == "none":
            return data_dict

        elif self.fill == "last":
            # For 'last' fill we keep timestamps that start at the maximum first timestamp
            # across datasets (i.e., drop timestamps before all datasets have started).
            min_common_timestamp = max(
                [ds.get_timestamp(0) for ds in self.single_signal_datasets]
            )

            # Filter timestamps to those >= min_common_timestamp (do not mutate while iterating)
            new_timestamps = [t for t in self._timestamps if t >= min_common_timestamp]
            new_data_dict = {t: data_dict[t] for t in new_timestamps}

            # For each dataset, fill missing indices with the last seen index (previous valid)
            for i, _ in self._tqdm_iter(
                enumerate(self.single_signal_datasets), desc="Filling:"
            ):
                last_seen = None
                for t in new_timestamps:
                    if new_data_dict[t][i] is None:
                        # if we have a previously seen index, use it; otherwise keep None
                        new_data_dict[t][i] = last_seen
                    else:
                        last_seen = new_data_dict[t][i]

            # Update internal timestamps and return new mapping
            self._timestamps = new_timestamps
            return new_data_dict

        elif self.fill == "closest":
            for i, ds in self._tqdm_iter(
                enumerate(self.single_signal_datasets), desc="Filling:"
            ):
                global_timestamp_idx, ds_timestamp_idx = 0, 1

                while global_timestamp_idx < len(self._timestamps):
                    timestamps_to_fill = self._timestamps[global_timestamp_idx]

                    # advance ds_timestamp_idx until we bracket timestamps_to_fill, or reach end
                    while (
                        ds_timestamp_idx < len(ds) - 1
                        and ds.get_timestamp(ds_timestamp_idx) < timestamps_to_fill
                    ):
                        ds_timestamp_idx += 1

                    # choose closest of ds_timestamp_idx and ds_timestamp_idx-1
                    prev_idx = max(0, ds_timestamp_idx - 1)
                    if abs(ds.get_timestamp(prev_idx) - timestamps_to_fill) <= abs(
                        ds.get_timestamp(ds_timestamp_idx) - timestamps_to_fill
                    ):
                        data_dict[timestamps_to_fill][i] = prev_idx
                    else:
                        data_dict[timestamps_to_fill][i] = ds_timestamp_idx

                    global_timestamp_idx += 1

            return data_dict

    @property
    def timestamps(self) -> list[int]:
        """
        Returns the list of timestamps in numpy.datetime64 format.
        """
        return self._timestamps

    def __len__(self) -> int:
        """
        Returns the number of timestamps.
        """
        return len(self.timestamps)

    def get_data(self, idx: int) -> dict:
        """
        Retrieves the data at the specified index.

        Args:
            idx (int): Index of the timestamp.

        Returns:
            dict: Dictionary containing data from all datasets at the specified timestamp.
        """
        data_slice = self.data_dict[self.timestamps[idx]]
        data_dict = {}
        for i, ds in enumerate(self.single_signal_datasets):
            if data_slice[i] is None:
                for k in ds.sensor_ids:
                    data_dict[f"{ds.machine_name}_{ds.id}-{k}"] = None
            else:
                data = ds.get_data(data_slice[i])

                for k in data:
                    data_dict[f"{ds.machine_name}_{ds.id}-{k}"] = data[k]

                if self.return_indices:
                    data_dict[f"{ds.machine_name}_{ds.id}-index"] = data_slice[i]

        return data_dict

    def get_timestamp(self, idx: int) -> int:
        """
        Retrieves the timestamp at the specified index.

        Args:
            idx (int): Index of the timestamp.

        Returns:
            int: Timestamp at the specified index.
        """
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp: int) -> int:
        """
        Retrieves the index of the specified timestamp.

        Args:
            timestamp (pd.Timestamp): Timestamp to find the index for.

        Returns:
            int: Index of the specified timestamp.
        """
        return self.timestamps.index(timestamp)

    def __repr__(self) -> str:
        """
        Returns a string representation of the MultiSignalDataset object.
        """
        print_string = f"MultiSignalDataset - {len(self)} samples\nDatasets: {len(self.single_signal_datasets)}\n"
        for i, d in enumerate(self.single_signal_datasets):
            inner_repr = repr(d)
            lines = inner_repr.split("\n")
            inner_repr = "\n".join(["\t" + line for line in lines])

            print_string += f"{i} -------------\n"
            print_string += inner_repr
        print_string += "\n------------------\n"
        return print_string

    @property
    def id(self):
        """
        Returns the ID of the dataset.
        """
        return ""

    @property
    def sensor_ids(self):
        sids = []
        for ds in self.single_signal_datasets:
            sids += ds.sensor_ids
        return sids
