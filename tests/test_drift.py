"""Tests for the drift detection module."""

import unittest

import numpy as np
import pandas as pd

from hdb_resale_mlops.drift import (
    _compute_psi,
    detect_categorical_drift,
    detect_numeric_drift,
    run_drift_checks,
)


class TestPSI(unittest.TestCase):
    def test_identical_distributions_psi_near_zero(self):
        values = np.array(["A", "B", "C"] * 100)
        psi = _compute_psi(values, values)
        self.assertAlmostEqual(psi, 0.0, places=5)

    def test_shifted_distribution_psi_positive(self):
        reference = np.array(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
        current = np.array(["A"] * 20 + ["B"] * 30 + ["C"] * 50)
        psi = _compute_psi(reference, current)
        self.assertGreater(psi, 0.0)

    def test_very_different_distributions_high_psi(self):
        reference = np.array(["A"] * 95 + ["B"] * 5)
        current = np.array(["A"] * 5 + ["B"] * 95)
        psi = _compute_psi(reference, current)
        self.assertGreater(psi, 0.2)


class TestCategoricalDrift(unittest.TestCase):
    def test_no_drift_detected_for_identical_data(self):
        df = pd.DataFrame({"town": ["ANG MO KIO", "BEDOK", "CLEMENTI"] * 100})
        results = detect_categorical_drift(df, df, ["town"])
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].is_drifted)

    def test_drift_detected_for_shifted_data(self):
        ref = pd.DataFrame({"town": ["ANG MO KIO"] * 95 + ["BEDOK"] * 5})
        cur = pd.DataFrame({"town": ["ANG MO KIO"] * 5 + ["BEDOK"] * 95})
        results = detect_categorical_drift(ref, cur, ["town"])
        self.assertTrue(results[0].is_drifted)


class TestNumericDrift(unittest.TestCase):
    def test_no_drift_for_same_distribution(self):
        rng = np.random.RandomState(42)
        values = rng.normal(100, 10, 1000)
        ref = pd.DataFrame({"floor_area_sqm": values})
        cur = pd.DataFrame({"floor_area_sqm": values})
        results = detect_numeric_drift(ref, cur, ["floor_area_sqm"])
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].is_drifted)

    def test_drift_for_shifted_distribution(self):
        rng = np.random.RandomState(42)
        ref = pd.DataFrame({"floor_area_sqm": rng.normal(100, 10, 1000)})
        cur = pd.DataFrame({"floor_area_sqm": rng.normal(200, 10, 1000)})
        results = detect_numeric_drift(ref, cur, ["floor_area_sqm"])
        self.assertTrue(results[0].is_drifted)


class TestRunDriftChecks(unittest.TestCase):
    def test_combined_report_no_drift(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "town": ["ANG MO KIO", "BEDOK", "CLEMENTI"] * 100,
            "floor_area_sqm": rng.normal(100, 10, 300),
        })
        report = run_drift_checks(
            df, df, categorical_columns=["town"], numeric_columns=["floor_area_sqm"]
        )
        self.assertFalse(report.overall_drift_detected)
        self.assertEqual(len(report.column_results), 2)


if __name__ == "__main__":
    unittest.main()
