import os
import tempfile
import unittest

import numpy as np

from src.config import CONFIG
from src.dataset import load_memmap_dataset
from src.utils import write_memmap_metadata


class TestDataset(unittest.TestCase):
    """Dataset loading tests that validate metadata-driven shape handling."""

    def test_load_memmap_dataset_uses_metadata_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "toy.dat")
            arr = np.memmap(path, dtype="float32", mode="w+", shape=(5, 4, 1))
            arr[:] = 1.5
            arr.flush()
            write_memmap_metadata(
                path,
                {"num_sequences": 5, "sequence_length": 4, "dtype": "float32", "stride": 1},
            )

            old_path = CONFIG["memmap_file"]
            try:
                CONFIG["memmap_file"] = path
                loaded = load_memmap_dataset(flatten_for_tree=False, split="all")
                self.assertEqual(loaded.shape, (5, 4, 1))
                self.assertAlmostEqual(float(loaded[0, 0, 0]), 1.5)
            finally:
                CONFIG["memmap_file"] = old_path


if __name__ == "__main__":
    unittest.main()
