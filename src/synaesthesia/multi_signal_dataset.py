from datetime import datetime, timedelta

from tqdm import tqdm

from .abstract_dataset import DatasetBase


def convert_to_datetime(timestamp):
    # Adjust this function based on the format of your timestamps
    return datetime.strptime(timestamp, "%Y%m%dT%H%M%S%f")


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

        self.timestamp_dict = {}
        for ssd in self.single_signal_datasets:
            for idx in range(len(ssd)):
                self.timestamp_dict[ssd.get_timestamp(idx)] = [None] * n_ssds

        if aggregation == "all":
            pass

        elif aggregation == "common":
            print("Aggregating common timestamps")
            for k in tqdm(list(self.timestamp_dict.keys())):
                for ssd in self.single_signal_datasets:
                    if k not in ssd:
                        del self.timestamp_dict[k]
                        break

        elif aggregation[:2] == "I:":
            idx = int(aggregation[2:])
            for k in list(self.timestamp_dict.keys()):
                if k not in self.single_signal_datasets[idx]:
                    del self.timestamp_dict[k]

        else:
            raise ValueError(f"Aggregation {aggregation} not valid.")

        if fill == "none":
            print("Filling missing timestamps: none")
            for k in tqdm(self.timestamp_dict):
                for i, ssd in enumerate(self.single_signal_datasets):
                    self.timestamp_dict[k][i] = ssd.get_timestamp_idx(k)

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
            timestamps = sorted(self.timestamp_dict.keys(), key=convert_to_datetime)
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

        self.timestamps = sorted(self.timestamp_dict.keys())

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "timestamp": self.get_timestamp(idx),
            "data": self.get_data(idx),
        }

    def __len__(self):
        return len(self.timestamps)

    def get_data(self, idx):
        timestamp = self.get_timestamp(idx)
        return_dict = {}
        for dataset, i in zip(
            self.single_signal_datasets, self.timestamp_dict[timestamp]
        ):
            return_dict[f"{dataset.satellite_name}_{dataset.sensor_id}"] = None

            if i is not None:
                return_dict[f"{dataset.satellite_name}_{dataset.sensor_id}"] = (
                    dataset.get_data(i)
                )

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
        print_string = f"MultiSignalDataset - len() samples\n"
        print_string += f"Datasets: {len(self.single_signal_datasets)}\n"

        for i, d in enumerate(self.single_signal_datasets):
            inner_repr = repr(d)
            lines = inner_repr.split("\n")
            inner_repr = "\n".join(["\t" + line for line in lines])

            print_string += f"{i} -------------\n"
            print_string += inner_repr
            print_string += "------------------\n"
        return print_string
