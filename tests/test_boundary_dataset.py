from pathlib import Path

import numpy as np
import pytest

from src.synaesthesia.abstract.boundary_filtered_dataset import BoundaryFilteredDataset

from .simple_csv_dataset import SimpleCsvDataset

# Determine the path to the directory of the current script
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH_10 = BASE_DIR / "test_data" / "test_data_10_s.csv"


@pytest.fixture
def base_dataset():
    """
    Fixture to load the base dataset for testing.
    """
    return SimpleCsvDataset(DATA_PATH_10)


@pytest.fixture
def single_boundary():
    """
    Fixture for a single time boundary.
    """
    # Assuming timestamps are in a reasonable range for the test data
    return [("20220101T000010000", "20220101T000110000")]


@pytest.fixture
def multiple_boundaries():
    """
    Fixture for multiple time boundaries.
    """
    return [
        ("20220101T000010000", "20220101T000110000"),
        ("20220101T000230000", "20220101T000350000"),
        ("20220101T000420000", "20220101T000440000"),
    ]


@pytest.fixture
def boundary_filtered_dataset_single(base_dataset, single_boundary):
    """
    Fixture to create a BoundaryFilteredDataset with a single boundary.
    """
    dataset = BoundaryFilteredDataset(base_dataset, single_boundary)
    assert len(dataset) == 6
    return dataset


@pytest.fixture
def boundary_filtered_dataset_multiple(base_dataset, multiple_boundaries):
    """
    Fixture to create a BoundaryFilteredDataset with multiple boundaries.
    """
    dataset = BoundaryFilteredDataset(base_dataset, multiple_boundaries)
    assert len(dataset) == 6 + 8 + 2
    return dataset


# Basic functionality tests
def test_boundary_filtered_dataset_initialization(base_dataset, single_boundary):
    """Test that the dataset initializes correctly."""
    filtered_dataset = BoundaryFilteredDataset(base_dataset, single_boundary)

    assert filtered_dataset.dataset is base_dataset
    assert filtered_dataset.boundaries == single_boundary
    assert hasattr(filtered_dataset, "fwd_indices")
    assert hasattr(filtered_dataset, "bwd_indices")
    assert len(filtered_dataset.fwd_indices) == len(filtered_dataset.bwd_indices)


def test_boundary_filtered_dataset_empty_boundaries(base_dataset):
    """Test that empty boundaries raise ValueError."""
    with pytest.raises(ValueError, match="At least one boundary must be provided"):
        BoundaryFilteredDataset(base_dataset, [])


def test_boundary_filtered_dataset_length(
    base_dataset, boundary_filtered_dataset_single
):
    """Test that the filtered dataset has correct length."""
    original_length = len(base_dataset)
    filtered_length = len(boundary_filtered_dataset_single)

    # Filtered dataset should be smaller or equal to original
    assert filtered_length <= original_length
    assert filtered_length >= 0


def test_boundary_filtered_dataset_properties(
    base_dataset, boundary_filtered_dataset_single
):
    """Test that properties are correctly inherited from base dataset."""
    assert boundary_filtered_dataset_single.id == base_dataset.id
    assert boundary_filtered_dataset_single.sensor_ids == base_dataset.sensor_ids
    assert (
        boundary_filtered_dataset_single.get_machine_name()
        == base_dataset.get_machine_name()
    )


def test_boundary_filtered_dataset_get_data(
    base_dataset, boundary_filtered_dataset_single
):
    """Test getting data from filtered dataset."""
    if len(boundary_filtered_dataset_single) > 0:
        # Test first item
        filtered_data = boundary_filtered_dataset_single.get_data(0)
        original_idx = boundary_filtered_dataset_single.fwd_indices[0]
        original_data = base_dataset.get_data(original_idx)

        # Data should be identical
        for key in original_data:
            if isinstance(original_data[key], np.ndarray):
                assert np.array_equal(filtered_data[key], original_data[key])
            else:
                assert filtered_data[key] == original_data[key]


def test_boundary_filtered_dataset_get_timestamp(
    base_dataset, boundary_filtered_dataset_single
):
    """Test getting timestamps from filtered dataset."""
    if len(boundary_filtered_dataset_single) > 0:
        # Test first timestamp
        filtered_timestamp = boundary_filtered_dataset_single.get_timestamp(0)
        original_idx = boundary_filtered_dataset_single.fwd_indices[0]
        original_timestamp = base_dataset.get_timestamp(original_idx)

        assert filtered_timestamp == original_timestamp


def test_boundary_filtered_dataset_timestamps_property(
    boundary_filtered_dataset_single,
):
    """Test the timestamps property."""
    timestamps = boundary_filtered_dataset_single.timestamps

    assert isinstance(timestamps, list)
    assert len(timestamps) == len(boundary_filtered_dataset_single)

    # Check that timestamps match individual get_timestamp calls
    for i, timestamp in enumerate(timestamps):
        assert timestamp == boundary_filtered_dataset_single.get_timestamp(i)


def test_boundary_filtered_dataset_index_bounds(boundary_filtered_dataset_single):
    """Test index bounds checking."""
    dataset_length = len(boundary_filtered_dataset_single)

    if dataset_length > 0:
        # Valid indices should work
        boundary_filtered_dataset_single.get_data(0)
        boundary_filtered_dataset_single.get_data(dataset_length - 1)
        boundary_filtered_dataset_single.get_timestamp(0)
        boundary_filtered_dataset_single.get_timestamp(dataset_length - 1)

    # Invalid indices should raise IndexError
    with pytest.raises(IndexError):
        boundary_filtered_dataset_single.get_data(dataset_length)

    with pytest.raises(IndexError):
        boundary_filtered_dataset_single.get_data(-1)

    with pytest.raises(IndexError):
        boundary_filtered_dataset_single.get_timestamp(dataset_length)

    with pytest.raises(IndexError):
        boundary_filtered_dataset_single.get_timestamp(-1)


def test_boundary_filtered_dataset_get_timestamp_idx(
    base_dataset, boundary_filtered_dataset_single
):
    """Test getting index from timestamp."""
    if len(boundary_filtered_dataset_single) > 0:
        # Get a valid timestamp
        test_timestamp = boundary_filtered_dataset_single.get_timestamp(0)

        # Should be able to find it
        found_idx = boundary_filtered_dataset_single.get_timestamp_idx(test_timestamp)
        assert found_idx == 0

        # Verify the timestamp matches
        assert (
            boundary_filtered_dataset_single.get_timestamp(found_idx) == test_timestamp
        )


def test_boundary_filtered_dataset_get_timestamp_idx_not_found(
    base_dataset, boundary_filtered_dataset_single
):
    """Test getting index from non-existent timestamp."""
    # Use a timestamp that's unlikely to exist
    fake_timestamp = "21000101T000000"

    with pytest.raises(ValueError):
        boundary_filtered_dataset_single.get_timestamp_idx(fake_timestamp)


def test_boundary_filtered_dataset_multiple_boundaries(
    boundary_filtered_dataset_multiple,
):
    """Test dataset with multiple boundaries."""
    # Should have valid structure
    assert len(boundary_filtered_dataset_multiple.fwd_indices) == len(
        boundary_filtered_dataset_multiple
    )
    assert len(boundary_filtered_dataset_multiple.bwd_indices) == len(
        boundary_filtered_dataset_multiple
    )

    # Forward and backward indices should be consistent
    for (
        filtered_idx,
        original_idx,
    ) in boundary_filtered_dataset_multiple.fwd_indices.items():
        assert (
            boundary_filtered_dataset_multiple.bwd_indices[original_idx] == filtered_idx
        )


def test_boundary_filtered_dataset_repr(boundary_filtered_dataset_single):
    """Test string representation."""
    repr_str = repr(boundary_filtered_dataset_single)

    assert "BoundaryFilteredDataset" in repr_str
    assert "samples" in repr_str
    assert "filtered from" in repr_str
    assert "Boundaries:" in repr_str
    assert "Wrapped Dataset:" in repr_str


def test_boundary_filtered_dataset_getitem(boundary_filtered_dataset_single):
    """Test __getitem__ method (inherited from DatasetBase)."""
    if len(boundary_filtered_dataset_single) > 0:
        item = boundary_filtered_dataset_single[0]

        # Should contain required keys
        assert "idx" in item
        assert "timestamp" in item
        assert item["idx"] == 0
        assert item["timestamp"] == boundary_filtered_dataset_single.get_timestamp(0)

        # Should contain data from get_data
        data = boundary_filtered_dataset_single.get_data(0)
        for key, value in data.items():
            expected_key = (
                f"{boundary_filtered_dataset_single.id}-{key}"
                if boundary_filtered_dataset_single.id
                else key
            )
            if isinstance(value, np.ndarray):
                assert np.array_equal(item[expected_key], value)
            else:
                assert item[expected_key] == value


def test_boundary_filtered_dataset_contains(
    base_dataset, boundary_filtered_dataset_single
):
    """Test __contains__ method (inherited from DatasetBase)."""
    if len(boundary_filtered_dataset_single) > 0:
        # Should contain valid timestamps
        valid_timestamp = boundary_filtered_dataset_single.get_timestamp(0)
        assert valid_timestamp in boundary_filtered_dataset_single

        # Should not contain invalid timestamps
        invalid_timestamp = 999999999
        assert invalid_timestamp not in boundary_filtered_dataset_single


def test_boundary_filtered_dataset_bidirectional_mapping(
    boundary_filtered_dataset_single,
):
    """Test that forward and backward index mappings are consistent."""
    for filtered_idx in range(len(boundary_filtered_dataset_single)):
        original_idx = boundary_filtered_dataset_single.fwd_indices[filtered_idx]
        mapped_back = boundary_filtered_dataset_single.bwd_indices[original_idx]
        assert mapped_back == filtered_idx


def test_boundary_filtered_dataset_preserves_order(
    base_dataset, boundary_filtered_dataset_single
):
    """Test that the filtered dataset preserves timestamp order."""
    if len(boundary_filtered_dataset_single) > 1:
        timestamps = boundary_filtered_dataset_single.timestamps

        # Check if timestamps are in ascending order (assuming original dataset is ordered)
        for i in range(1, len(timestamps)):
            # Allow for equal timestamps but not decreasing
            assert timestamps[i] >= timestamps[i - 1]


# Edge case tests
def test_boundary_filtered_dataset_no_matching_data(base_dataset):
    """Test dataset with boundaries that don't match any data."""
    # Use boundaries that are unlikely to contain any data
    future_boundaries = [("20990101T000000", "20990101T010000")]

    filtered_dataset = BoundaryFilteredDataset(base_dataset, future_boundaries)

    # Should have zero length
    assert len(filtered_dataset) == 0
    assert len(filtered_dataset.fwd_indices) == 0
    assert len(filtered_dataset.bwd_indices) == 0


def test_boundary_filtered_dataset_all_data_matches(base_dataset):
    """Test dataset with boundaries that include all data."""
    # Use very wide boundaries
    wide_boundaries = [("19000101T000000", "21000101T000000")]

    filtered_dataset = BoundaryFilteredDataset(base_dataset, wide_boundaries)

    # Should have same length as original
    assert len(filtered_dataset) == len(base_dataset)
