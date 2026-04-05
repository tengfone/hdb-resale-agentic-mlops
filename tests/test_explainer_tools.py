"""Tier 1 — Unit tests for each explainer agent tool in isolation.

These tests verify that every tool returned by ``_make_tools`` produces
correct, well-structured output without requiring an LLM or external services.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta
from hdb_resale_mlops.drift import ColumnDriftResult, DriftReport
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict
from hdb_resale_mlops.explainer import _make_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tools(**overrides):
    """Build tool list with sensible defaults; override any kwarg."""
    market_research_provider = overrides.pop("market_research_provider", "none")
    defaults = dict(
        candidate_metrics={"rmse": 150_000.0, "mae": 120_000.0},
        champion_info=None,
        comparison=ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000.0, "mae": 120_000.0},
        ),
        drift_report=None,
        policy_verdict=PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["All checks passed"],
            checks_passed=["absolute_rmse", "absolute_mae"],
            checks_failed=[],
        ),
        model_name="test-model",
    )
    defaults.update(overrides)
    with patch.dict(
        os.environ,
        {"MARKET_RESEARCH_PROVIDER": market_research_provider},
        clear=False,
    ):
        return {t.name: t for t in _make_tools(**defaults)}


# ---- Candidate metrics ----


class TestQueryCandidateMetrics(unittest.TestCase):
    def test_returns_candidate_metrics_json(self):
        tools = _build_tools(candidate_metrics={"rmse": 145_000.5, "mae": 115_000.3})
        result = json.loads(tools["query_candidate_metrics"].invoke({}))
        self.assertAlmostEqual(result["rmse"], 145_000.5)
        self.assertAlmostEqual(result["mae"], 115_000.3)

    def test_returns_valid_json(self):
        tools = _build_tools()
        raw = tools["query_candidate_metrics"].invoke({})
        self.assertIsInstance(json.loads(raw), dict)


# ---- Champion metrics ----


class TestQueryChampionMetrics(unittest.TestCase):
    def test_no_champion(self):
        tools = _build_tools(champion_info=None)
        result = tools["query_champion_metrics"].invoke({})
        self.assertIn("No champion", result)

    def test_with_champion(self):
        champ = {"version": "2", "metrics": {"rmse": 140_000.0, "mae": 110_000.0}}
        tools = _build_tools(champion_info=champ)
        result = json.loads(tools["query_champion_metrics"].invoke({}))
        self.assertEqual(result["rmse"], 140_000.0)
        self.assertEqual(result["mae"], 110_000.0)

    def test_champion_missing_metrics_key(self):
        champ = {"version": "2"}  # no "metrics" key
        tools = _build_tools(champion_info=champ)
        result = json.loads(tools["query_champion_metrics"].invoke({}))
        self.assertEqual(result, {})


# ---- Segment comparison ----


class TestCompareSegmentPerformance(unittest.TestCase):
    def test_no_champion_returns_message(self):
        tools = _build_tools()
        result = tools["compare_segment_performance"].invoke({"segment_type": "town"})
        self.assertIn("No champion", result)

    def test_town_segments_sorted_by_worst(self):
        deltas = [
            SegmentDelta("town", "BEDOK", 160_000.0, 150_000.0, 10_000.0, 0.0667),
            SegmentDelta("town", "YISHUN", 180_000.0, 140_000.0, 40_000.0, 0.2857),
            SegmentDelta("town", "CHOA CHU KANG", 165_000.0, 140_000.0, 25_000.0, 0.1786),
            SegmentDelta("flat_type", "4 ROOM", 155_000.0, 150_000.0, 5_000.0, 0.0333),
        ]
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 160_000.0},
            champion_metrics={"rmse": 150_000.0},
            segment_deltas=deltas,
        )
        tools = _build_tools(comparison=comparison)
        result = json.loads(tools["compare_segment_performance"].invoke({"segment_type": "town"}))
        self.assertEqual(len(result), 3)
        # Only town segments returned; flat_type excluded
        segments = [r["segment"] for r in result]
        self.assertEqual(segments, ["YISHUN", "CHOA CHU KANG", "BEDOK"])

    def test_flat_type_segments(self):
        deltas = [
            SegmentDelta("flat_type", "3 ROOM", 170_000.0, 160_000.0, 10_000.0, 0.0625),
        ]
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 160_000.0},
            champion_metrics={"rmse": 150_000.0},
            segment_deltas=deltas,
        )
        tools = _build_tools(comparison=comparison)
        result = json.loads(
            tools["compare_segment_performance"].invoke({"segment_type": "flat_type"})
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["segment"], "3 ROOM")

    def test_empty_segments_for_type(self):
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 160_000.0},
            champion_metrics={"rmse": 150_000.0},
            segment_deltas=[],
        )
        tools = _build_tools(comparison=comparison)
        result = tools["compare_segment_performance"].invoke({"segment_type": "town"})
        self.assertIn("No segment deltas", result)


# ---- Drift report ----


class TestCheckDriftReport(unittest.TestCase):
    def test_no_drift_report(self):
        tools = _build_tools(drift_report=None)
        result = tools["check_drift_report"].invoke({})
        self.assertIn("not run", result)

    def test_clean_drift(self):
        dr = DriftReport(
            column_results=[
                ColumnDriftResult("town", "psi", 0.05, 0.2, is_drifted=False),
                ColumnDriftResult("floor_area_sqm", "ks", 0.03, 0.05, p_value=0.45, is_drifted=False),
            ],
            overall_drift_detected=False,
        )
        tools = _build_tools(drift_report=dr)
        result = json.loads(tools["check_drift_report"].invoke({}))
        self.assertFalse(result["overall_drift_detected"])
        self.assertEqual(len(result["columns"]), 2)
        self.assertFalse(any(c["is_drifted"] for c in result["columns"]))

    def test_drifted_columns(self):
        dr = DriftReport(
            column_results=[
                ColumnDriftResult("town", "psi", 0.35, 0.2, is_drifted=True),
                ColumnDriftResult("floor_area_sqm", "ks", 0.12, 0.05, p_value=0.001, is_drifted=True),
            ],
            overall_drift_detected=True,
        )
        tools = _build_tools(drift_report=dr)
        result = json.loads(tools["check_drift_report"].invoke({}))
        self.assertTrue(result["overall_drift_detected"])
        self.assertTrue(all(c["is_drifted"] for c in result["columns"]))
        # Numeric column should include p_value
        ks_col = [c for c in result["columns"] if c["type"] == "ks"][0]
        self.assertIn("p_value", ks_col)
        self.assertAlmostEqual(ks_col["p_value"], 0.001, places=4)


# ---- Policy verdict ----


class TestGetPolicyVerdict(unittest.TestCase):
    def test_promote_verdict(self):
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["All checks passed"],
            checks_passed=["absolute_rmse", "absolute_mae"],
            checks_failed=[],
        )
        tools = _build_tools(policy_verdict=verdict)
        result = json.loads(tools["get_policy_verdict"].invoke({}))
        self.assertEqual(result["decision"], "PROMOTE")
        self.assertEqual(result["checks_passed"], ["absolute_rmse", "absolute_mae"])
        self.assertEqual(result["checks_failed"], [])

    def test_reject_verdict(self):
        verdict = PolicyVerdict(
            decision=PolicyDecision.REJECT,
            reasons=["Test RMSE 250,000 exceeds maximum threshold 200,000"],
            checks_passed=[],
            checks_failed=["absolute_rmse"],
        )
        tools = _build_tools(policy_verdict=verdict)
        result = json.loads(tools["get_policy_verdict"].invoke({}))
        self.assertEqual(result["decision"], "REJECT")
        self.assertIn("absolute_rmse", result["checks_failed"])

    def test_manual_review_verdict(self):
        verdict = PolicyVerdict(
            decision=PolicyDecision.MANUAL_REVIEW,
            reasons=["Drift detected in: town"],
            checks_passed=["absolute_rmse"],
            checks_failed=["drift_check"],
        )
        tools = _build_tools(policy_verdict=verdict)
        result = json.loads(tools["get_policy_verdict"].invoke({}))
        self.assertEqual(result["decision"], "MANUAL_REVIEW")


# ---- Training history ----


class TestGetTrainingHistory(unittest.TestCase):
    @patch("hdb_resale_mlops.mlflow_registry.get_training_history")
    def test_returns_formatted_history(self, mock_history):
        mock_history.return_value = [
            {
                "version": "3",
                "run_id": "r3",
                "metrics": {"test_rmse": 148_000, "test_mae": 118_000},
                "tags": {
                    "promotion_status": "champion",
                    "policy_verdict": "PROMOTE",
                    "decision_source": "human_review",
                },
            },
            {
                "version": "2",
                "run_id": "r2",
                "metrics": {"test_rmse": 155_000, "test_mae": 122_000},
                "tags": {
                    "promotion_status": "rejected",
                    "policy_verdict": "REJECT",
                    "decision_source": "policy_auto_reject",
                    "rejection_reasons": "RMSE too high",
                },
            },
        ]
        tools = _build_tools(model_name="test-model")
        result = json.loads(tools["get_training_history"].invoke({}))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["version"], "3")
        self.assertEqual(result[0]["test_rmse"], 148_000)
        self.assertEqual(result[0]["promotion_status"], "champion")
        self.assertEqual(result[1]["rejection_reasons"], "RMSE too high")

    @patch("hdb_resale_mlops.mlflow_registry.get_training_history")
    def test_empty_history(self, mock_history):
        mock_history.return_value = []
        tools = _build_tools(model_name="test-model")
        result = tools["get_training_history"].invoke({})
        self.assertIn("No training history", result)


# ---- Tool list completeness ----


class TestToolListCompleteness(unittest.TestCase):
    def test_six_core_tools_returned(self):
        tools = _build_tools()
        expected = {
            "query_candidate_metrics",
            "query_champion_metrics",
            "compare_segment_performance",
            "check_drift_report",
            "get_policy_verdict",
            "get_training_history",
        }
        self.assertEqual(set(tools.keys()), expected)

    def test_all_tools_have_descriptions(self):
        tools = _build_tools()
        for name, t in tools.items():
            self.assertTrue(t.description, f"Tool '{name}' missing description")

    def test_all_tools_return_strings(self):
        """Every tool must return a string (the agent consumes text)."""
        tools = _build_tools()
        # Tools with no required args
        for name in ("query_candidate_metrics", "query_champion_metrics",
                     "check_drift_report", "get_policy_verdict"):
            result = tools[name].invoke({})
            self.assertIsInstance(result, str, f"Tool '{name}' did not return str")

        result = tools["compare_segment_performance"].invoke({"segment_type": "town"})
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
