from pathlib import Path

from src.synaesthesia.abstract.concat_dataset import CustomConcatDataset

from .simple_csv_dataset import SimpleCsvDataset


def test_custom_concat_dataset_timestamps_property():
    """
    Verify that CustomConcatDataset.timestamps returns a flat list (not a generator),
    that it concatenates the timestamps of the underlying datasets in order, and
    that the length and endpoints match expectations.
    """
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH_10 = BASE_DIR / "test_data" / "test_data_10_s.csv"
    DATA_PATH_10_CONCAT = BASE_DIR / "test_data" / "test_data_10_s_concat_test.csv"

    ds1 = SimpleCsvDataset(DATA_PATH_10)
    ds2 = SimpleCsvDataset(DATA_PATH_10_CONCAT)

    concat = CustomConcatDataset([ds1, ds2])

    # timestamps should be a list (not a generator) and equal to ds1.timestamps + ds2.timestamps
    assert isinstance(concat.timestamps, list), "timestamps should be a list"
    expected = list(ds1.timestamps) + list(ds2.timestamps)
    assert concat.timestamps == expected, "Concat timestamps should equal concatenation of inner timestamps"

    # length should match
    assert len(concat.timestamps) == len(ds1) + len(ds2)

    # first and last timestamp sanity checks
    assert concat.timestamps[0] == ds1.timestamps[0]
    assert concat.timestamps[-1] == ds2.timestamps[-1]
