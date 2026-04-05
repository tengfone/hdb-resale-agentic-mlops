"""Tier 3 — Report quality evaluation tests using LLM-as-a-Judge.

These tests generate reports (template or LLM-backed) for golden scenarios
and score them with a stronger LLM judge model.

The LLM-backed tiers are explicitly opt-in. They require both:
- ``OPENAI_API_KEY``
- ``RUN_LLM_REPORT_QUALITY_TESTS=1``

Run the dedicated target to opt in:

    make test-report-llm

To run ONLY the non-LLM structure checks:

    make test-report-template
"""

import json
import os
import pathlib
import unittest

import pytest

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta
from hdb_resale_mlops.drift import ColumnDriftResult, DriftReport
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict
from hdb_resale_mlops.explainer import _generate_template_report

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "eval_scenarios"
_LLM_REPORT_QUALITY_ENV = "RUN_LLM_REPORT_QUALITY_TESTS"


def _llm_report_quality_enabled() -> bool:
    return os.environ.get(_LLM_REPORT_QUALITY_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_LLM_REPORT_QUALITY_READY = _llm_report_quality_enabled() and bool(os.environ.get("OPENAI_API_KEY"))
requires_llm = pytest.mark.requires_llm
requires_openai_key_unittest = unittest.skipUnless(
    _LLM_REPORT_QUALITY_READY,
    f"Set {_LLM_REPORT_QUALITY_ENV}=1 and OPENAI_API_KEY to run LLM-backed tests",
)


def _load_scenario(name: str) -> dict:
    """Load a raw scenario JSON fixture."""
    with open(FIXTURES_DIR / f"{name}.json") as f:
        return json.load(f)


def _reconstruct_dataclasses(raw: dict) -> dict:
    """Reconstruct dataclasses from raw scenario dict."""
    comp = raw["comparison"]
    segment_deltas = [SegmentDelta(**sd) for sd in comp.get("segment_deltas", [])]
    comparison = ComparisonResult(
        has_champion=comp["has_champion"],
        candidate_metrics=comp["candidate_metrics"],
        champion_metrics=comp.get("champion_metrics"),
        metric_deltas=comp.get("metric_deltas", {}),
        segment_deltas=segment_deltas,
    )

    drift_raw = raw.get("drift_report")
    drift_report = None
    if drift_raw:
        col_results = [ColumnDriftResult(**cr) for cr in drift_raw["column_results"]]
        drift_report = DriftReport(
            column_results=col_results,
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
        "model_version": raw["model_version"],
    }


# ---------------------------------------------------------------------------
# Template report structure tests (no LLM required)
# ---------------------------------------------------------------------------


class TestTemplateReportStructure(unittest.TestCase):
    """Verify template reports contain required sections for all scenarios."""

    def _get_scenarios(self):
        return [p.stem for p in FIXTURES_DIR.glob("*.json")]

    def test_all_scenarios_have_required_sections(self):
        for name in self._get_scenarios():
            raw = _load_scenario(name)
            dc = _reconstruct_dataclasses(raw)
            report = _generate_template_report(**dc)

            with self.subTest(scenario=name):
                self.assertIn("Summary", report)
                self.assertIn("Candidate Metrics", report)
                self.assertIn("Recommendation", report)

    def test_all_scenarios_mention_verdict(self):
        for name in self._get_scenarios():
            raw = _load_scenario(name)
            dc = _reconstruct_dataclasses(raw)
            report = _generate_template_report(**dc)
            verdict = raw["policy_verdict"]["decision"]

            with self.subTest(scenario=name):
                self.assertIn(verdict, report)

    def test_drift_scenarios_mention_drifted_columns(self):
        raw = _load_scenario("manual_review_drift")
        dc = _reconstruct_dataclasses(raw)
        report = _generate_template_report(**dc)
        self.assertIn("town", report)
        self.assertIn("floor_area_sqm", report)

    def test_segment_regression_mentions_worst_segment(self):
        raw = _load_scenario("manual_review_segment_regression")
        dc = _reconstruct_dataclasses(raw)
        report = _generate_template_report(**dc)
        self.assertIn("YISHUN", report)


# ---------------------------------------------------------------------------
# LLM-as-a-Judge tests (require OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@requires_openai_key_unittest
@requires_llm
class TestTemplateReportQuality(unittest.TestCase):
    """Score template-generated reports with LLM judge."""

    MIN_COMPLETENESS = 3
    MIN_ACCURACY = 4
    MIN_ACTIONABILITY = 3
    MIN_SAFETY = 5

    def _get_scenarios(self):
        return [p.stem for p in FIXTURES_DIR.glob("*.json")]

    def test_template_reports_meet_quality_bar(self):
        from hdb_resale_mlops.eval_judge import evaluate_report

        for name in self._get_scenarios():
            raw = _load_scenario(name)
            dc = _reconstruct_dataclasses(raw)
            report = _generate_template_report(**dc)

            score = evaluate_report(report=report, scenario=raw)

            with self.subTest(scenario=name):
                self.assertGreaterEqual(
                    score.completeness, self.MIN_COMPLETENESS,
                    f"Completeness too low for '{name}': {score.completeness}/5. {score.reasoning}",
                )
                self.assertGreaterEqual(
                    score.accuracy, self.MIN_ACCURACY,
                    f"Accuracy too low for '{name}': {score.accuracy}/5. {score.reasoning}",
                )
                self.assertGreaterEqual(
                    score.actionability, self.MIN_ACTIONABILITY,
                    f"Actionability too low for '{name}': {score.actionability}/5. {score.reasoning}",
                )
                self.assertGreaterEqual(
                    score.safety, self.MIN_SAFETY,
                    f"Safety too low for '{name}': {score.safety}/5. {score.reasoning}",
                )


@requires_openai_key_unittest
@requires_llm
class TestLLMReportQuality(unittest.TestCase):
    """Score LLM-generated reports with LLM judge.

    This is the most expensive test — it calls the explainer agent (gpt-5-nano)
    for each scenario, then scores the output with the judge (gpt-5-mini).
    """

    MIN_COMPLETENESS = 4
    MIN_ACCURACY = 4
    MIN_ACTIONABILITY = 4
    MIN_SAFETY = 5

    def _get_scenarios(self):
        return [p.stem for p in FIXTURES_DIR.glob("*.json")]

    def test_llm_reports_meet_quality_bar(self):
        from unittest.mock import patch
        from hdb_resale_mlops.explainer import run_explainer_agent
        from hdb_resale_mlops.eval_judge import evaluate_report

        with patch.dict(os.environ, {"MARKET_RESEARCH_PROVIDER": "none"}, clear=False):
            for name in self._get_scenarios():
                raw = _load_scenario(name)
                dc = _reconstruct_dataclasses(raw)

                with patch(
                    "hdb_resale_mlops.mlflow_registry.get_training_history",
                    return_value=[],
                ):
                    report = run_explainer_agent(
                        candidate_metrics=dc["candidate_metrics"],
                        champion_info=dc["champion_info"],
                        comparison=dc["comparison"],
                        drift_report=dc["drift_report"],
                        policy_verdict=dc["policy_verdict"],
                        model_name=raw["model_name"],
                        model_version=dc["model_version"],
                    )

                score = evaluate_report(report=report, scenario=raw)

                with self.subTest(scenario=name):
                    self.assertGreaterEqual(
                        score.completeness, self.MIN_COMPLETENESS,
                        f"Completeness too low for '{name}': {score.completeness}/5. {score.reasoning}",
                    )
                    self.assertGreaterEqual(
                        score.accuracy, self.MIN_ACCURACY,
                        f"Accuracy too low for '{name}': {score.accuracy}/5. {score.reasoning}",
                    )
                    self.assertGreaterEqual(
                        score.actionability, self.MIN_ACTIONABILITY,
                        f"Actionability too low for '{name}': {score.actionability}/5. {score.reasoning}",
                    )
                    self.assertGreaterEqual(
                        score.safety, self.MIN_SAFETY,
                        f"Safety too low for '{name}': {score.safety}/5. {score.reasoning}",
                    )


if __name__ == "__main__":
    unittest.main()
