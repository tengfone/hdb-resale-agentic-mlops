"""Tests for the policy engine."""

import unittest

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta
from hdb_resale_mlops.drift import ColumnDriftResult, DriftReport
from hdb_resale_mlops.policy import PolicyDecision, evaluate_policy


class TestAbsoluteThresholds(unittest.TestCase):
    def _no_champion(self):
        return ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )

    def test_good_metrics_promote(self):
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            comparison=self._no_champion(),
        )
        self.assertEqual(verdict.decision, PolicyDecision.PROMOTE)

    def test_high_rmse_reject(self):
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 250_000, "mae": 120_000},
            comparison=self._no_champion(),
        )
        self.assertEqual(verdict.decision, PolicyDecision.REJECT)
        self.assertIn("absolute_rmse", verdict.checks_failed)

    def test_high_mae_reject(self):
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 150_000, "mae": 180_000},
            comparison=self._no_champion(),
        )
        self.assertEqual(verdict.decision, PolicyDecision.REJECT)
        self.assertIn("absolute_mae", verdict.checks_failed)


class TestChampionComparison(unittest.TestCase):
    def test_regression_beyond_threshold_rejects(self):
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 170_000, "mae": 120_000},
            champion_metrics={"rmse": 150_000, "mae": 115_000},
            metric_deltas={
                "rmse_delta": 20_000,
                "rmse_delta_pct": 20_000 / 150_000,  # ~13.3%
                "mae_delta": 5_000,
                "mae_delta_pct": 5_000 / 115_000,
            },
        )
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 170_000, "mae": 120_000},
            comparison=comparison,
        )
        self.assertEqual(verdict.decision, PolicyDecision.REJECT)
        self.assertIn("champion_rmse_regression", verdict.checks_failed)

    def test_slight_regression_promotes(self):
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 153_000, "mae": 120_000},
            champion_metrics={"rmse": 150_000, "mae": 118_000},
            metric_deltas={
                "rmse_delta": 3_000,
                "rmse_delta_pct": 0.02,
                "mae_delta": 2_000,
                "mae_delta_pct": 2_000 / 118_000,
            },
        )
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 153_000, "mae": 120_000},
            comparison=comparison,
        )
        self.assertEqual(verdict.decision, PolicyDecision.PROMOTE)


class TestSegmentRegression(unittest.TestCase):
    def test_large_segment_regression_manual_review(self):
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 155_000, "mae": 120_000},
            champion_metrics={"rmse": 150_000, "mae": 118_000},
            metric_deltas={
                "rmse_delta": 5_000,
                "rmse_delta_pct": 0.033,
                "mae_delta": 2_000,
                "mae_delta_pct": 0.017,
            },
            segment_deltas=[
                SegmentDelta(
                    segment_column="town",
                    segment_value="PUNGGOL",
                    candidate_rmse=180_000,
                    champion_rmse=140_000,
                    rmse_delta=40_000,
                    rmse_delta_pct=40_000 / 140_000,  # ~28.6%
                ),
            ],
        )
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 155_000, "mae": 120_000},
            comparison=comparison,
        )
        self.assertEqual(verdict.decision, PolicyDecision.MANUAL_REVIEW)
        self.assertIn("segment_rmse_regression", verdict.checks_failed)


class TestDriftCheck(unittest.TestCase):
    def test_drift_triggers_manual_review(self):
        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        drift_report = DriftReport(
            column_results=[
                ColumnDriftResult(
                    column="town",
                    drift_type="psi",
                    statistic=0.35,
                    threshold=0.2,
                    is_drifted=True,
                ),
            ],
            overall_drift_detected=True,
        )
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            comparison=comparison,
            drift_report=drift_report,
        )
        self.assertEqual(verdict.decision, PolicyDecision.MANUAL_REVIEW)
        self.assertIn("drift_check", verdict.checks_failed)

    def test_no_drift_allows_promote(self):
        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        drift_report = DriftReport(
            column_results=[
                ColumnDriftResult(
                    column="town",
                    drift_type="psi",
                    statistic=0.05,
                    threshold=0.2,
                    is_drifted=False,
                ),
            ],
            overall_drift_detected=False,
        )
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            comparison=comparison,
            drift_report=drift_report,
        )
        self.assertEqual(verdict.decision, PolicyDecision.PROMOTE)


class TestRejectOverridesManualReview(unittest.TestCase):
    def test_reject_takes_priority_over_manual_review(self):
        """If both REJECT and MANUAL_REVIEW conditions are met, REJECT wins."""
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 250_000, "mae": 120_000},
            champion_metrics={"rmse": 150_000, "mae": 118_000},
            metric_deltas={
                "rmse_delta": 100_000,
                "rmse_delta_pct": 100_000 / 150_000,
                "mae_delta": 2_000,
                "mae_delta_pct": 0.017,
            },
            segment_deltas=[
                SegmentDelta(
                    segment_column="town",
                    segment_value="PUNGGOL",
                    candidate_rmse=200_000,
                    champion_rmse=140_000,
                    rmse_delta=60_000,
                    rmse_delta_pct=60_000 / 140_000,
                ),
            ],
        )
        drift_report = DriftReport(
            column_results=[
                ColumnDriftResult(
                    column="town", drift_type="psi", statistic=0.35,
                    threshold=0.2, is_drifted=True,
                ),
            ],
            overall_drift_detected=True,
        )
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 250_000, "mae": 120_000},
            comparison=comparison,
            drift_report=drift_report,
        )
        self.assertEqual(verdict.decision, PolicyDecision.REJECT)


class TestEvidenceFailures(unittest.TestCase):
    def test_missing_registry_evidence_rejects(self):
        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        verdict = evaluate_policy(
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            comparison=comparison,
            evidence_errors=[
                "Promotion blocked because champion evidence could not be loaded from MLflow.",
            ],
        )
        self.assertEqual(verdict.decision, PolicyDecision.REJECT)
        self.assertIn("promotion_evidence_unavailable", verdict.checks_failed)
        self.assertNotIn("champion_comparison_skipped", verdict.checks_passed)


if __name__ == "__main__":
    unittest.main()
