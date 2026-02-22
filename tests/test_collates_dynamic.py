from src.synaesthesia.collates import CollateBase


class RecordingCollate(CollateBase):
    """A small CollateBase subclass that returns the keys it was given in do_collate.

    This helps test that key matching is computed per-call (dynamic), not permanently cached.
    """

    def __init__(self, item_keys):
        # Keep default delete_original False so original keys remain for introspection if needed
        super().__init__(item_keys=item_keys, delete_original=False)

    def do_collate(self, items):
        # Return the matched keys so tests can assert which keys were seen this call
        return {"matched_keys": sorted(list(items.keys()))}


def test_match_keys_is_dynamic_across_calls():
    """
    Ensure CollateBase.match_keys matches keys for each call independently.

    This guards against the previous behavior where matched keys were cached once and
    subsequent batches with different keys could be incorrectly ignored.
    """
    collate = RecordingCollate(item_keys=["^a", "^b", "^c"])

    # First batch contains keys 'a' and 'b'
    batch1 = [{"a": 1, "b": 2}]
    out1 = collate(batch1)
    assert set(out1["matched_keys"]) == {"a", "b"}, (
        "First call should match 'a' and 'b'"
    )

    # Second batch contains keys 'a' and 'c' (different from the first batch)
    batch2 = [{"a": 3, "c": 4}]
    out2 = collate(batch2)
    assert set(out2["matched_keys"]) == {"a", "c"}, (
        "Second call should match 'a' and 'c' (dynamic keys)"
    )

    # Third call: only a single key matching none of the patterns -> matched_keys should be empty
    batch3 = [{"x": 10, "y": 20}]
    out3 = collate(batch3)
    assert out3["matched_keys"] == [], "No keys should match the provided patterns"

    # Fourth call: multiple items in the batch with differing key sets; the union of keys should be considered
    batch4 = [{"a": 1}, {"b": 2}, {"c": 3, "z": 0}]
    out4 = collate(batch4)
    assert set(out4["matched_keys"]) == {"a", "b", "c"}, (
        "Union of keys across items should be matched"
    )
