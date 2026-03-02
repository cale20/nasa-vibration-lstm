import unittest

import numpy as np

from src.evaluate_unsupervised import (
    _first_persistent_alert_idx,
    _spearman_rank_correlation,
)


class TestEvaluateUnsupervisedHelpers(unittest.TestCase):
    def test_spearman_detects_increasing_trend(self):
        x = np.arange(6, dtype=np.float64)
        y = np.array([0.1, 0.2, 0.2, 0.5, 0.7, 1.1], dtype=np.float64)
        rho = _spearman_rank_correlation(x, y)
        self.assertGreater(rho, 0.8)

    def test_first_persistent_alert_idx(self):
        alerts = np.array([0, 0, 1, 0, 1, 1, 0], dtype=np.uint8)
        idx = _first_persistent_alert_idx(alerts, k=2, m=3, start_idx=0)
        self.assertEqual(idx, 4)


if __name__ == "__main__":
    unittest.main()
