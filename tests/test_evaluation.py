from __future__ import annotations

import importlib.util
import unittest

from hdb_resale_mlops.evaluation import build_segment_metrics_frame, regression_metrics


class RegressionMetricsTest(unittest.TestCase):
    def test_regression_metrics_returns_rmse_and_mae(self) -> None:
        metrics = regression_metrics([10.0, 20.0, 30.0], [12.0, 18.0, 33.0])
        self.assertAlmostEqual(metrics["rmse"], 2.3804761428)
        self.assertAlmostEqual(metrics["mae"], 2.3333333333)


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is required for segment metrics tests")
class SegmentMetricsTest(unittest.TestCase):
    def test_segment_metrics_frame_groups_by_segment(self) -> None:
        import pandas as pd

        scored = pd.DataFrame(
            {
                "town": ["ANG MO KIO", "ANG MO KIO", "BEDOK"],
                "resale_price": [500000.0, 520000.0, 610000.0],
                "prediction": [490000.0, 530000.0, 620000.0],
            }
        )

        segment_frame = build_segment_metrics_frame(scored, "town")
        self.assertEqual(list(segment_frame["segment"]), ["ANG MO KIO", "BEDOK"])
        self.assertEqual(int(segment_frame.iloc[0]["count"]), 2)
        self.assertGreater(segment_frame.iloc[1]["rmse"], 0.0)

