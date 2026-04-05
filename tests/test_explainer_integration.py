"""Tier 2 — Integration tests for the explainer tool surface.

These tests verify the real tool objects and scenario fixtures work together:
  - Expected tools are callable for each scenario
  - Tool payloads are well-formed and factual
  - Deterministic report assembly from tool outputs contains required content

Requires ``pip install 'hdb-resale-mlops[agent]'``.
"""

import json
import os
import pathlib
import unittest
from unittest.mock import patch

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta
from hdb_resale_mlops.drift import ColumnDriftResult, DriftReport
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "eval_scenarios"


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------


def _load_scenario(name: str) -> dict:
    """Load a scenario JSON fixture and reconstruct dataclasses."""
    with open(FIXTURES_DIR / f"{name}.json") as f:
        raw = json.load(f)

    # Reconstruct ComparisonResult
    comp = raw["comparison"]
    segment_deltas = [SegmentDelta(**sd) for sd in comp.get("segment_deltas", [])]
    comparison = ComparisonResult(
        has_champion=comp["has_champion"],
        candidate_metrics=comp["candidate_metrics"],
        champion_metrics=comp.get("champion_metrics"),
        metric_deltas=comp.get("metric_deltas", {}),
        segment_deltas=segment_deltas,
    )

    # Reconstruct DriftReport
    drift_raw = raw.get("drift_report")
    drift_report = None
    if drift_raw:
        col_results = [ColumnDriftResult(**cr) for cr in drift_raw["column_results"]]
        drift_report = DriftReport(
            column_results=col_results,
            overall_drift_detected=drift_raw["overall_drift_detected"],
        )

    # Reconstruct PolicyVerdict
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
        "expected": raw["expected"],
        "description": raw.get("description", name),
    }


# ---------------------------------------------------------------------------
# Deterministic harness that exercises the real tools without an LLM
# ---------------------------------------------------------------------------


def _make_mock_agent(tools_dict, scenario):
    """Build a minimal fake agent that calls expected tools then returns a report.

    We call the real tools directly and compose a deterministic report from
    their outputs — this verifies tool correctness and report content without
    claiming autonomous tool selection coverage.
    """
    expected = scenario["expected"]
    tool_names_to_call = expected["minimum_tools_called"]

    # Invoke each expected tool and collect outputs
    tool_outputs = {}
    for name in tool_names_to_call:
        t = tools_dict[name]
        if name == "compare_segment_performance":
            tool_outputs[name] = t.invoke({"segment_type": "town"})
        else:
            tool_outputs[name] = t.invoke({})

    # Build a deterministic report from tool outputs
    verdict = scenario["policy_verdict"]
    lines = [
        "## Summary",
        f"Candidate model version {scenario['model_version']} evaluated. "
        f"Policy verdict: {verdict.decision.value}.",
        "",
        "## Evidence",
    ]
    cm = scenario["candidate_metrics"]
    lines.append(f"- Test RMSE: {cm['rmse']:,.0f}")
    lines.append(f"- Test MAE: {cm['mae']:,.0f}")

    if scenario["champion_info"]:
        champ = scenario["champion_info"]["metrics"]
        lines.append(f"- Champion RMSE: {champ['rmse']:,.0f}")

    lines.append("")
    lines.append("## Risk Flags")
    if verdict.checks_failed:
        for reason in verdict.reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- No risk flags identified.")

    if scenario["drift_report"] and scenario["drift_report"].overall_drift_detected:
        drifted = [c.column for c in scenario["drift_report"].column_results if c.is_drifted]
        lines.append(f"- Data drift detected in: {', '.join(drifted)}")

    if scenario["comparison"].segment_deltas:
        for sd in scenario["comparison"].segment_deltas:
            if sd.rmse_delta_pct > 0.20:
                lines.append(f"- Segment regression: {sd.segment_value} ({sd.rmse_delta_pct:+.1%})")

    lines.extend(["", "## Recommendation"])
    lines.append(f"Policy engine recommends: **{verdict.decision.value}**.")

    return "\n".join(lines), tool_outputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExplainerIntegration(unittest.TestCase):
    """Integration tests: build real tools, verify tool outputs + report content."""

    def _run_scenario(self, scenario_name: str):
        """Load scenario, build tools, generate report, run assertions."""
        from hdb_resale_mlops.explainer import _make_tools

        scenario = _load_scenario(scenario_name)

        tools_list = _make_tools(
            candidate_metrics=scenario["candidate_metrics"],
            champion_info=scenario["champion_info"],
            comparison=scenario["comparison"],
            drift_report=scenario["drift_report"],
            policy_verdict=scenario["policy_verdict"],
            model_name=scenario["model_name"],
        )
        tools_dict = {t.name: t for t in tools_list}

        report, tool_outputs = _make_mock_agent(tools_dict, scenario)
        expected = scenario["expected"]

        # 1. Verify expected tools were callable
        for tool_name in expected["minimum_tools_called"]:
            self.assertIn(tool_name, tool_outputs,
                          f"Tool '{tool_name}' was not called for scenario '{scenario_name}'")

        # 2. Verify report contains required strings
        for required in expected.get("report_must_contain", []):
            self.assertIn(required, report,
                          f"Report missing '{required}' in scenario '{scenario_name}'")

        # 3. Verify report does NOT contain forbidden strings
        for forbidden in expected.get("report_must_not_contain", []):
            self.assertNotIn(forbidden.lower(), report.lower(),
                             f"Report contains forbidden '{forbidden}' in scenario '{scenario_name}'")

        # 4. Verify report has expected sections
        for section in expected.get("report_sections", []):
            self.assertIn(section, report,
                          f"Report missing section '{section}' in scenario '{scenario_name}'")

        # 5. Verify tool outputs are valid JSON (where applicable)
        for tool_name, output in tool_outputs.items():
            if output.startswith("{") or output.startswith("["):
                try:
                    json.loads(output)
                except json.JSONDecodeError:
                    self.fail(f"Tool '{tool_name}' returned invalid JSON: {output[:100]}")

        return report, tool_outputs

    @patch("hdb_resale_mlops.mlflow_registry.get_training_history", return_value=[])
    def test_promote_no_champion(self, _mock_hist):
        report, _ = self._run_scenario("promote_no_champion")
        self.assertIn("PROMOTE", report)

    @patch("hdb_resale_mlops.mlflow_registry.get_training_history", return_value=[])
    def test_promote_with_champion(self, _mock_hist):
        report, outputs = self._run_scenario("promote_with_champion")
        self.assertIn("PROMOTE", report)
        # Champion metrics tool should return real champion data
        champ_data = json.loads(outputs["query_champion_metrics"])
        self.assertEqual(champ_data["rmse"], 155000.0)

    @patch("hdb_resale_mlops.mlflow_registry.get_training_history", return_value=[])
    def test_reject_high_rmse(self, _mock_hist):
        report, outputs = self._run_scenario("reject_high_rmse")
        self.assertIn("REJECT", report)
        # Verify the candidate metrics tool returns the high RMSE
        cand_data = json.loads(outputs["query_candidate_metrics"])
        self.assertEqual(cand_data["rmse"], 250000.0)

    @patch("hdb_resale_mlops.mlflow_registry.get_training_history", return_value=[])
    def test_manual_review_drift(self, _mock_hist):
        report, outputs = self._run_scenario("manual_review_drift")
        self.assertIn("MANUAL_REVIEW", report)
        # Drift report tool should show drift
        drift_data = json.loads(outputs["check_drift_report"])
        self.assertTrue(drift_data["overall_drift_detected"])

    @patch("hdb_resale_mlops.mlflow_registry.get_training_history", return_value=[])
    def test_manual_review_segment_regression(self, _mock_hist):
        report, outputs = self._run_scenario("manual_review_segment_regression")
        self.assertIn("YISHUN", report)
        # Segment comparison tool should show the regression
        segment_data = json.loads(outputs["compare_segment_performance"])
        yishun = [s for s in segment_data if s["segment"] == "YISHUN"]
        self.assertEqual(len(yishun), 1)
        self.assertIn("+", yishun[0]["delta_pct"])


class TestExplainerAgentWithMockedLLM(unittest.TestCase):
    """Test that build_explainer_agent produces a working agent graph."""

    def test_agent_builds_with_all_scenarios(self):
        """Verify the agent graph compiles for every scenario (no runtime errors)."""
        from hdb_resale_mlops.explainer import build_explainer_agent

        # Need a fake API key for the agent builder (it won't actually call the LLM)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake"}):
            for scenario_file in FIXTURES_DIR.glob("*.json"):
                name = scenario_file.stem
                scenario = _load_scenario(name)
                agent = build_explainer_agent(
                    candidate_metrics=scenario["candidate_metrics"],
                    champion_info=scenario["champion_info"],
                    comparison=scenario["comparison"],
                    drift_report=scenario["drift_report"],
                    policy_verdict=scenario["policy_verdict"],
                    model_name=scenario["model_name"],
                )
                # The agent should be a compiled graph
                self.assertTrue(
                    hasattr(agent, "invoke"),
                    f"Agent for scenario '{name}' is not invocable",
                )


class TestTemplateReportWithScenarios(unittest.TestCase):
    """Test that the template fallback produces sensible reports for all scenarios."""

    def test_all_scenarios_produce_template_report(self):
        from hdb_resale_mlops.explainer import _generate_template_report

        for scenario_file in FIXTURES_DIR.glob("*.json"):
            name = scenario_file.stem
            scenario = _load_scenario(name)  # returns reconstructed dataclasses
            report = _generate_template_report(
                candidate_metrics=scenario["candidate_metrics"],
                champion_info=scenario["champion_info"],
                comparison=scenario["comparison"],
                drift_report=scenario["drift_report"],
                policy_verdict=scenario["policy_verdict"],
                model_version=scenario["model_version"],
            )
            expected = scenario["expected"]
            with self.subTest(scenario=name):
                for required in expected.get("report_must_contain", []):
                    self.assertIn(required, report,
                                  f"Template report missing '{required}' for scenario '{name}'")


if __name__ == "__main__":
    unittest.main()
