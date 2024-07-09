from datetime import timedelta

from .abstract_dataset import DatasetBase
from .utils import convert_to_datetime


class MultiSignalDataset(DatasetBase):
    def __init__(
        self,
        single_signal_datasets: list[DatasetBase],
        aggregation="all",
        fill="none",
        time_cut=60,  # in minutes
    ):
        super().__init__()

        self.single_signal_datasets = single_signal_datasets
        n_ssds = len(self.single_signal_datasets)
        assert n_ssds > 0, "No datasets provided."

        self.timestamp_dict = {}
        if aggregation == "all":
            for ssd in self.single_signal_datasets:
                for idx in range(len(ssd)):
                    self.timestamp_dict[ssd.get_timestamp(idx)] = [None] * n_ssds

        elif aggregation == "common":
            common_timestamps = set(self.single_signal_datasets[0].timestamps)
            for ssd in self.single_signal_datasets[1:]:
                common_timestamps.intersection_update(set(ssd.timestamps))
            self.timestamp_dict = {k: [None] * n_ssds for k in common_timestamps}

        elif aggregation[:2] == "I:":
            idx = int(aggregation[2:])
            i_timestamps = set(self.single_signal_datasets[idx].timestamps)
            self.timestamp_dict = {k: [None] * n_ssds for k in i_timestamps}

        else:
            raise ValueError(f"Aggregation {aggregation} not valid.")

        if fill == "none":
            print("Filling missing timestamps: none")
            for i, ssd in enumerate(self.single_signal_datasets):
                for j, timestamp in enumerate(ssd.timestamps):
                    if timestamp in self.timestamp_dict:
                        self.timestamp_dict[timestamp][i] = j

        elif fill == "last":
            timestamps = sorted(self.timestamp_dict.keys())

            found = False
            while not found:
                found = True
                for ssd in self.single_signal_datasets:
                    if ssd.get_timestamp(0) > timestamps[0]:
                        k = timestamps.pop(0)
                        del self.timestamp_dict[k]
                        found = False
                        break

            for i, ssd in enumerate(self.single_signal_datasets):
                for idx in range(len(ssd)):
                    if ssd.get_timestamp(idx) > timestamps[0]:
                        self.timestamp_dict[timestamps[0]][i] = idx - 1
                        break

            for t0, t1 in zip(timestamps[:-1], timestamps[1:]):
                for i, ssd in enumerate(self.single_signal_datasets):
                    idx_before = self.timestamp_dict[t0][i]
                    for idx in range(idx_before, len(ssd)):
                        if ssd.get_timestamp(idx) == t1:
                            self.timestamp_dict[t1][i] = idx
                            break

                        elif ssd.get_timestamp(idx) > t1:
                            self.timestamp_dict[t1][i] = idx - 1
                            break

                    if self.timestamp_dict[t1][i] is None:
                        self.timestamp_dict[t1][i] = len(ssd) - 1

        elif fill == "closest":
            timestamps = sorted(self.timestamp_dict.keys())
            timestamps_dt = [convert_to_datetime(ts) for ts in timestamps]

            found = False
            while not found:
                found = True
                for ssd in self.single_signal_datasets:
                    if convert_to_datetime(ssd.get_timestamp(0)) > timestamps_dt[0]:
                        k = timestamps.pop(0)
                        timestamps_dt.pop(0)
                        del self.timestamp_dict[k]
                        found = False
                        break

            for i, ssd in enumerate(self.single_signal_datasets):
                for idx in range(len(ssd)):
                    current_ts = convert_to_datetime(ssd.get_timestamp(idx))
                    if current_ts >= timestamps_dt[0]:
                        if idx == 0 or abs(current_ts - timestamps_dt[0]) < abs(
                            convert_to_datetime(ssd.get_timestamp(idx - 1))
                            - timestamps_dt[0]
                        ):
                            self.timestamp_dict[timestamps[0]][i] = idx
                        else:
                            self.timestamp_dict[timestamps[0]][i] = idx - 1
                        break

            for t0, t1 in zip(timestamps_dt[:-1], timestamps_dt[1:]):
                t0_key = timestamps[timestamps_dt.index(t0)]
                t1_key = timestamps[timestamps_dt.index(t1)]
                for i, ssd in enumerate(self.single_signal_datasets):
                    idx_before = self.timestamp_dict[t0_key][i]
                    closest_idx = None
                    closest_diff = timedelta(minutes=time_cut)
                    for idx in range(idx_before, len(ssd)):
                        current_ts = convert_to_datetime(ssd.get_timestamp(idx))
                        current_diff = abs(current_ts - t1)
                        if current_diff < closest_diff:
                            closest_diff = current_diff
                            closest_idx = idx

                        if current_ts >= t1:
                            break

                    if closest_idx is not None:
                        self.timestamp_dict[t1_key][i] = closest_idx
                    else:
                        raise ValueError(
                            f"No sufficiently close timestamp found for index {i} between {t0} and {t1}."
                        )

        else:
            raise ValueError(f"Fill {fill} not valid.")

        self._timestamps = sorted(self.timestamp_dict.keys())

    @property
    def timestamps(self):
        return self._timestamps

    def __len__(self):
        return len(self.timestamps)

    def get_data(self, idx):
        timestamp = self.get_timestamp(idx)
        return_dict = {}
        for dataset, i in zip(
            self.single_signal_datasets, self.timestamp_dict[timestamp]
        ):
            if i is None:
                tmp_data = {
                    f"{dataset.satellite_name}_{sensor_id}": None
                    for sensor_id in dataset.sensor_ids
                }
            else:
                data = dataset.get_data(i)
                tmp_data = {f"{dataset.satellite_name}_{k}": v for k, v in data.items()}

            for k in tmp_data:
                assert k not in return_dict, f"Key {k} already exists."

            return_dict |= tmp_data

        return return_dict

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp):
        return self.timestamps.index(timestamp)

    @property
    def sensor_id(self):
        return "multi"

    @property
    def satellite_id(self):
        return "multi"

    def __repr__(self) -> str:
        print_string = f"MultiSignalDataset - {len(self)} samples\n"
        print_string += f"Datasets: {len(self.single_signal_datasets)}\n"

        for i, d in enumerate(self.single_signal_datasets):
            inner_repr = repr(d)
            lines = inner_repr.split("\n")
            inner_repr = "\n".join(["\t" + line for line in lines])

            print_string += f"{i} -------------\n"
            print_string += inner_repr
            print_string += "------------------\n"
        return print_string
