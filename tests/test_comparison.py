"""Tests for the comparison module."""

import unittest

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta, compare_models
from hdb_resale_mlops.tabular_state import serialize_for_state


class TestCompareModelsNoChampion(unittest.TestCase):
    def test_no_champion_returns_no_deltas(self):
        result = compare_models(
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            champion_metrics=None,
        )
        self.assertFalse(result.has_champion)
        self.assertEqual(result.metric_deltas, {})
        self.assertEqual(result.segment_deltas, [])
        self.assertIsNone(result.champion_metrics)

    def test_no_champion_preserves_candidate_metrics(self):
        metrics = {"rmse": 150_000, "mae": 120_000}
        result = compare_models(candidate_metrics=metrics, champion_metrics=None)
        self.assertEqual(result.candidate_metrics, metrics)


class TestCompareModelsWithChampion(unittest.TestCase):
    def test_metric_deltas_computed(self):
        result = compare_models(
            candidate_metrics={"rmse": 160_000, "mae": 130_000},
            champion_metrics={"rmse": 150_000, "mae": 120_000},
        )
        self.assertTrue(result.has_champion)
        self.assertAlmostEqual(result.metric_deltas["rmse_delta"], 10_000)
        self.assertAlmostEqual(result.metric_deltas["mae_delta"], 10_000)
        self.assertAlmostEqual(result.metric_deltas["rmse_delta_pct"], 10_000 / 150_000)

    def test_candidate_better_than_champion(self):
        result = compare_models(
            candidate_metrics={"rmse": 140_000, "mae": 110_000},
            champion_metrics={"rmse": 150_000, "mae": 120_000},
        )
        self.assertLess(result.metric_deltas["rmse_delta"], 0)
        self.assertLess(result.metric_deltas["rmse_delta_pct"], 0)

    def test_segment_deltas_computed(self):
        import pandas as pd

        candidate_segments = {
            "town": pd.DataFrame(
                {"segment": ["ANG MO KIO", "BEDOK"], "rmse": [140_000, 160_000], "mae": [0, 0], "count": [100, 100], "mean_actual": [0, 0], "mean_prediction": [0, 0]}
            )
        }
        champion_segments = {
            "town": pd.DataFrame(
                {"segment": ["ANG MO KIO", "BEDOK"], "rmse": [130_000, 150_000], "mae": [0, 0], "count": [100, 100], "mean_actual": [0, 0], "mean_prediction": [0, 0]}
            )
        }
        result = compare_models(
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            champion_metrics={"rmse": 140_000, "mae": 115_000},
            candidate_segment_metrics=candidate_segments,
            champion_segment_metrics=champion_segments,
            segment_columns=("town",),
        )
        self.assertEqual(len(result.segment_deltas), 2)
        amk = [sd for sd in result.segment_deltas if sd.segment_value == "ANG MO KIO"][0]
        self.assertAlmostEqual(amk.rmse_delta, 10_000)
        self.assertAlmostEqual(amk.rmse_delta_pct, 10_000 / 130_000)

    def test_segment_deltas_computed_from_serialized_frames(self):
        import pandas as pd

        candidate_segments = serialize_for_state(
            {
                "town": pd.DataFrame(
                    {
                        "segment": ["ANG MO KIO"],
                        "rmse": [140_000],
                        "mae": [0],
                        "count": [100],
                    }
                )
            }
        )
        champion_segments = serialize_for_state(
            {
                "town": pd.DataFrame(
                    {
                        "segment": ["ANG MO KIO"],
                        "rmse": [130_000],
                        "mae": [0],
                        "count": [100],
                    }
                )
            }
        )

        result = compare_models(
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            champion_metrics={"rmse": 140_000, "mae": 115_000},
            candidate_segment_metrics=candidate_segments,
            champion_segment_metrics=champion_segments,
            segment_columns=("town",),
        )

        self.assertEqual(len(result.segment_deltas), 1)
        self.assertAlmostEqual(result.segment_deltas[0].rmse_delta, 10_000)


if __name__ == "__main__":
    unittest.main()
