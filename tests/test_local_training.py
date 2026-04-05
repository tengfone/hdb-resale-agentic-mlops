from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is required")
class LocalTrainingTest(unittest.TestCase):
    def _make_raw_frame(self, n: int = 40):
        import pandas as pd

        rows = []
        towns = ["ANG MO KIO", "BEDOK", "TAMPINES", "CLEMENTI"]
        flat_types = ["3 ROOM", "4 ROOM", "5 ROOM"]
        flat_models = ["Model A", "Improved", "New Generation"]
        storey_ranges = ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12"]
        for i in range(n):
            month_num = (i % 24) + 1
            year = 2022 + (month_num - 1) // 12
            month_str = f"{year}-{((month_num - 1) % 12) + 1:02d}"
            rows.append(
                {
                    "month": month_str,
                    "town": towns[i % len(towns)],
                    "flat_type": flat_types[i % len(flat_types)],
                    "flat_model": flat_models[i % len(flat_models)],
                    "storey_range": storey_ranges[i % len(storey_ranges)],
                    "floor_area_sqm": 80.0 + i,
                    "lease_commence_date": 1990 + (i % 20),
                    "remaining_lease": f"{60 + (i % 30)} years 00 months",
                    "resale_price": 400000.0 + i * 5000,
                }
            )
        return pd.DataFrame(rows)

    def test_train_locally_returns_pipeline_and_evaluation(self) -> None:
        from hdb_resale_mlops.local_training import train_locally

        raw = self._make_raw_frame(40)
        train_df = raw.iloc[:30].copy()
        val_df = raw.iloc[30:].copy()

        pipeline, evaluation = train_locally(train_df, val_df, random_seed=42)

        # Pipeline should be fitted and callable
        from hdb_resale_mlops.features import (
            prepare_training_frame,
            split_features_and_target,
        )

        prepared = prepare_training_frame(val_df)
        X, _ = split_features_and_target(prepared)
        predictions = pipeline.predict(X)
        self.assertEqual(len(predictions), len(X))

        # Evaluation should have overall metrics
        self.assertIn("rmse", evaluation.overall_metrics)
        self.assertIn("mae", evaluation.overall_metrics)
        self.assertGreater(evaluation.overall_metrics["rmse"], 0)

        # Segment metrics should be present
        self.assertIn("town", evaluation.segment_metrics)
        self.assertIn("flat_type", evaluation.segment_metrics)

    def test_save_model_creates_file(self) -> None:
        import tempfile
        from pathlib import Path

        from hdb_resale_mlops.local_training import train_locally, save_model

        raw = self._make_raw_frame(40)
        pipeline, _ = train_locally(raw.iloc[:30], raw.iloc[30:], random_seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = save_model(pipeline, Path(tmpdir) / "sub" / "model.joblib")
            self.assertTrue(model_path.exists())


if __name__ == "__main__":
    unittest.main()
