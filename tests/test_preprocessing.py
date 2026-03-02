import unittest

import numpy as np

from src.preprocessing import create_sequences, count_sequences, _split_name_for_file_idx


class TestPreprocessing(unittest.TestCase):
    """Tests for sequence generation and split assignment helpers."""

    def test_create_sequences_includes_final_window(self):
        # Final window should be included when it lands exactly on the tail.
        signal = np.arange(10, dtype=np.float32).reshape(-1, 1)
        seqs = create_sequences(signal, seq_length=4, stride=3)
        self.assertEqual(seqs.shape, (3, 4, 1))
        np.testing.assert_array_equal(seqs[-1].reshape(-1), np.array([6, 7, 8, 9], dtype=np.float32))

    def test_split_name_assignment(self):
        # Split thresholds are configured as 80 train + 20 val healthy files.
        self.assertEqual(_split_name_for_file_idx(0), "healthy_train")
        self.assertEqual(_split_name_for_file_idx(79), "healthy_train")
        self.assertEqual(_split_name_for_file_idx(80), "healthy_val")
        self.assertEqual(_split_name_for_file_idx(99), "healthy_val")
        self.assertEqual(_split_name_for_file_idx(100), "test_mixed")

    def test_count_sequences_matches_expected_math(self):
        self.assertEqual(count_sequences(signal_length=10, seq_length=4, stride=3), 3)
        self.assertEqual(count_sequences(signal_length=4, seq_length=4, stride=2), 1)
        self.assertEqual(count_sequences(signal_length=3, seq_length=4, stride=1), 0)


if __name__ == "__main__":
    unittest.main()
