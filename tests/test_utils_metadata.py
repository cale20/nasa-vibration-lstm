import os
import tempfile
import unittest

from src.utils import read_memmap_metadata, write_memmap_metadata


class TestMemmapMetadata(unittest.TestCase):
    """Utility tests for memmap metadata persistence helpers."""

    def test_roundtrip_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            memmap_path = os.path.join(tmp, "sample.dat")
            with open(memmap_path, "wb") as fh:
                fh.write(b"")
            payload = {"num_sequences": 42, "sequence_length": 100, "dtype": "float32"}
            write_memmap_metadata(memmap_path, payload)
            loaded = read_memmap_metadata(memmap_path)
            self.assertEqual(loaded["num_sequences"], 42)
            self.assertEqual(loaded["sequence_length"], 100)
            self.assertEqual(loaded["dtype"], "float32")
            self.assertIn("created_at", loaded)


if __name__ == "__main__":
    unittest.main()
