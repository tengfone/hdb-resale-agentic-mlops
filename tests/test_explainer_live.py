"""Opt-in live smoke test for the explainer agent."""

from __future__ import annotations

import json
import os
import pathlib
import unittest
from unittest.mock import patch

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta
from hdb_resale_mlops.drift import ColumnDriftResult, DriftReport
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "eval_scenarios"
_LIVE_EXPLAINER_ENV = "RUN_LLM_EXPLAINER_TESTS"
_LIVE_EXPLAINER_READY = (
    os.environ.get(_LIVE_EXPLAINER_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    and bool(os.environ.get("OPENAI_API_KEY"))
)
requires_live_explainer = unittest.skipUnless(
    _LIVE_EXPLAINER_READY,
    f"Set {_LIVE_EXPLAINER_ENV}=1 and OPENAI_API_KEY to run live explainer tests",
)


def _load_scenario(name: str) -> dict:
    raw = json.loads((FIXTURES_DIR / f"{name}.json").read_text())
    comp = raw["comparison"]
    comparison = ComparisonResult(
        has_champion=comp["has_champion"],
        candidate_metrics=comp["candidate_metrics"],
        champion_metrics=comp.get("champion_metrics"),
        metric_deltas=comp.get("metric_deltas", {}),
        segment_deltas=[SegmentDelta(**sd) for sd in comp.get("segment_deltas", [])],
    )

    drift_raw = raw.get("drift_report")
    drift_report = None
    if drift_raw:
        drift_report = DriftReport(
            column_results=[ColumnDriftResult(**cr) for cr in drift_raw["column_results"]],
            overall_drift_detected=drift_raw["overall_drift_detected"],
        )

    pv = raw["policy_verdict"]
    policy_verdict = PolicyVerdict(
        decision=PolicyDecision(pv["decision"]),
        reasons=pv["reasons"],
        checks_passed=pv["checks_passed"],
        checks_failed=pv["checks_failed"],
    )

    return {
        "candidate_metrics": raw["candidate_metrics"],
        "champion_info": raw.get("champion_info"),
        "comparison": comparison,
        "drift_report": drift_report,
        "policy_verdict": policy_verdict,
        "model_name": raw["model_name"],
        "model_version": raw["model_version"],
    }


@requires_live_explainer
class TestExplainerLive(unittest.TestCase):
    """Exercise the real explainer agent against one golden scenario."""

    @patch("hdb_resale_mlops.mlflow_registry.get_training_history", return_value=[])
    def test_live_explainer_agent_returns_non_fallback_report(self, _mock_history):
        from hdb_resale_mlops.explainer import run_explainer_agent_detailed

        scenario = _load_scenario("promote_with_champion")
        with patch.dict(os.environ, {"MARKET_RESEARCH_PROVIDER": "none"}, clear=False):
            result = run_explainer_agent_detailed(**scenario)

        self.assertFalse(result.used_fallback, result.fallback_note)
        self.assertTrue(result.structured_report.summary.strip())
        self.assertIn("Recommendation", result.report_text)
        self.assertGreaterEqual(result.run_metadata.get("tool_call_count", 0), 1)
