from __future__ import annotations

import importlib.util
import unittest

from hdb_resale_mlops.features import FEATURE_COLUMNS


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is required for smoke tests")
class NotebookSchemaSmokeTest(unittest.TestCase):
    def test_prepared_frame_matches_expected_feature_columns(self) -> None:
        import pandas as pd

        from hdb_resale_mlops.features import split_features_and_target, prepare_training_frame

        raw = pd.DataFrame(
            [
                {
                    "month": "2024-01",
                    "town": "ANG MO KIO",
                    "flat_type": "4 ROOM",
                    "flat_model": "Model A",
                    "storey_range": "04 TO 06",
                    "floor_area_sqm": 92.0,
                    "lease_commence_date": 2000,
                    "remaining_lease": "75 years 00 months",
                    "resale_price": 520000.0,
                },
                {
                    "month": "2024-02",
                    "town": "BEDOK",
                    "flat_type": "5 ROOM",
                    "flat_model": "Improved",
                    "storey_range": "10 TO 12",
                    "floor_area_sqm": 110.0,
                    "lease_commence_date": 1998,
                    "remaining_lease": "71 years 10 months",
                    "resale_price": 640000.0,
                },
            ]
        )

        prepared = prepare_training_frame(raw)
        features, target = split_features_and_target(prepared)

        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertEqual(len(features), 2)
        self.assertEqual(len(target), 2)
