"""Tests for the explainer module — template fallback and tool construction."""

import unittest
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta
from hdb_resale_mlops.drift import ColumnDriftResult, DriftReport
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict
from hdb_resale_mlops.explainer import (
    ExplainerRunResult,
    PromotionReport,
    _coerce_structured_report,
    _enrich_structured_report,
    _extract_final_report,
    _generate_template_report,
    _market_research_provider_from_env,
    _parse_report_text,
    _render_markdown_report,
    _run_openai_market_research,
    _selected_market_research_providers,
    build_explainer_agent,
    run_explainer_agent_detailed,
)


class _FakeSpan:
    def __init__(self, trace_id="trace-123"):
        self.trace_id = trace_id
        self.inputs = []
        self.outputs = []
        self.attributes = {}

    def set_inputs(self, inputs):
        self.inputs.append(inputs)

    def set_outputs(self, outputs):
        self.outputs.append(outputs)

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeSpanContext:
    def __init__(self, span):
        self._span = span

    def __enter__(self):
        return self._span

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMlflow:
    def __init__(self):
        self.started_spans = []
        self.update_calls = []

    def start_span(self, name, span_type=None, attributes=None):
        span = _FakeSpan()
        span.name = name
        span.span_type = span_type
        span.attributes.update(attributes or {})
        self.started_spans.append(span)
        return _FakeSpanContext(span)

    def update_current_trace(self, **kwargs):
        self.update_calls.append(kwargs)


class TestTemplateReport(unittest.TestCase):
    def test_template_report_no_champion(self):
        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["No existing champion — only absolute thresholds applied"],
            checks_passed=["absolute_rmse", "absolute_mae"],
            checks_failed=[],
        )
        report = _generate_template_report(
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            champion_info=None,
            comparison=comparison,
            drift_report=None,
            policy_verdict=verdict,
            model_version="1",
        )
        self.assertIn("Version 1", report)
        self.assertIn("150,000", report)
        self.assertIn("PROMOTE", report)
        self.assertIn("First model candidate", report)

    def test_template_report_with_champion(self):
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 155_000, "mae": 125_000},
            champion_metrics={"rmse": 150_000, "mae": 120_000},
            metric_deltas={
                "rmse_delta": 5_000,
                "rmse_delta_pct": 0.033,
                "mae_delta": 5_000,
                "mae_delta_pct": 0.042,
            },
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.MANUAL_REVIEW,
            reasons=["Test drift detected"],
            checks_passed=["absolute_rmse"],
            checks_failed=["drift_check"],
        )
        report = _generate_template_report(
            candidate_metrics={"rmse": 155_000, "mae": 125_000},
            champion_info={"version": "1", "metrics": {"rmse": 150_000, "mae": 120_000}},
            comparison=comparison,
            drift_report=None,
            policy_verdict=verdict,
            model_version="2",
        )
        self.assertIn("Champion RMSE", report)
        self.assertIn("MANUAL_REVIEW", report)

    def test_template_report_with_drift(self):
        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        drift_report = DriftReport(
            column_results=[
                ColumnDriftResult(
                    column="town", drift_type="psi", statistic=0.35,
                    threshold=0.2, is_drifted=True,
                ),
                ColumnDriftResult(
                    column="floor_area_sqm", drift_type="ks", statistic=0.12,
                    threshold=0.05, p_value=0.001, is_drifted=True,
                ),
            ],
            overall_drift_detected=True,
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.MANUAL_REVIEW,
            reasons=["Data drift detected"],
            checks_passed=[],
            checks_failed=["drift_check"],
        )
        report = _generate_template_report(
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            champion_info=None,
            comparison=comparison,
            drift_report=drift_report,
            policy_verdict=verdict,
            model_version="1",
        )
        self.assertIn("Drift detected in 2 feature(s)", report)
        self.assertIn("town", report)
        self.assertIn("floor_area_sqm", report)

    def test_template_report_reject_recommendation_has_next_steps(self):
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 250_000, "mae": 160_000},
            champion_metrics={"rmse": 155_000, "mae": 125_000},
            metric_deltas={
                "rmse_delta": 95_000,
                "rmse_delta_pct": 0.6129,
            },
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.REJECT,
            reasons=["Test RMSE 250,000 exceeds maximum threshold 200,000"],
            checks_passed=["absolute_mae"],
            checks_failed=["absolute_rmse"],
        )

        report = _generate_template_report(
            candidate_metrics={"rmse": 250_000, "mae": 160_000},
            champion_info={"version": "3", "metrics": {"rmse": 155_000, "mae": 125_000}},
            comparison=comparison,
            drift_report=None,
            policy_verdict=verdict,
            model_version="4",
        )

        self.assertIn("verify data quality and split integrity", report)
        self.assertIn("retraining or feature changes", report)

    def test_template_report_labels_segment_improvements_correctly(self):
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 145_000, "mae": 115_000},
            champion_metrics={"rmse": 155_000, "mae": 120_000},
            metric_deltas={
                "rmse_delta": -10_000,
                "rmse_delta_pct": -0.0645,
            },
            segment_deltas=[
                SegmentDelta(
                    segment_column="town",
                    segment_value="BEDOK",
                    candidate_rmse=138_000,
                    champion_rmse=150_000,
                    rmse_delta=-12_000,
                    rmse_delta_pct=-0.08,
                )
            ],
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["Candidate outperforms champion within thresholds"],
            checks_passed=["absolute_rmse", "absolute_mae", "champion_rmse_regression"],
            checks_failed=[],
        )

        report = _generate_template_report(
            candidate_metrics={"rmse": 145_000, "mae": 115_000},
            champion_info={"version": "2", "metrics": {"rmse": 155_000, "mae": 120_000}},
            comparison=comparison,
            drift_report=None,
            policy_verdict=verdict,
            model_version="5",
        )

        self.assertIn("Best Segment Improvements", report)
        self.assertNotIn("Worst Segment Regressions", report)


class TestExplainerFallback(unittest.TestCase):
    @patch("hdb_resale_mlops.explainer._configure_mlflow_tracing", return_value=None)
    def test_no_api_key_uses_template(self, _mock_tracing):
        """When OPENAI_API_KEY is unset, run_explainer_agent should use the template."""
        # Ensure the key is not set
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            from hdb_resale_mlops.explainer import run_explainer_agent

            comparison = ComparisonResult(
                has_champion=False,
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
            )
            verdict = PolicyVerdict(
                decision=PolicyDecision.PROMOTE,
                reasons=["No champion"],
                checks_passed=["absolute_rmse"],
                checks_failed=[],
            )
            report = run_explainer_agent(
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
                champion_info=None,
                comparison=comparison,
                drift_report=None,
                policy_verdict=verdict,
                model_name="test-model",
                model_version="1",
            )
            self.assertIn("template-based report", report)
            self.assertIn("PROMOTE", report)
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved

    @patch("hdb_resale_mlops.explainer._configure_mlflow_tracing", return_value=None)
    @patch("hdb_resale_mlops.explainer.build_explainer_agent", side_effect=ImportError("missing langgraph"))
    def test_missing_agent_dependencies_fall_back_to_template(self, _mock_build, _mock_tracing):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake"}):
            from hdb_resale_mlops.explainer import run_explainer_agent

            comparison = ComparisonResult(
                has_champion=False,
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
            )
            verdict = PolicyVerdict(
                decision=PolicyDecision.PROMOTE,
                reasons=["No champion"],
                checks_passed=["absolute_rmse"],
                checks_failed=[],
            )
            report = run_explainer_agent(
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
                champion_info=None,
                comparison=comparison,
                drift_report=None,
                policy_verdict=verdict,
                model_name="test-model",
                model_version="1",
            )

        self.assertIn("PROMOTE", report)
        self.assertIn("ImportError", report)


class TestExplainerMessageExtraction(unittest.TestCase):
    def test_extracts_last_non_tool_ai_message(self):
        messages = [
            SimpleNamespace(type="human", content="brief", tool_calls=[]),
            SimpleNamespace(type="ai", content="thinking", tool_calls=[{"name": "query_candidate_metrics"}]),
            SimpleNamespace(type="tool", content='{"rmse": 1}', tool_calls=[]),
            SimpleNamespace(type="ai", content="## Summary\nFinal report", tool_calls=[]),
        ]

        report = _extract_final_report(messages, brief="brief")
        self.assertEqual(report, "## Summary\nFinal report")

    def test_parse_report_text_extracts_sections(self):
        report = """## Summary
Candidate looks healthy.

## Evidence
- Test RMSE: 150,000
- Champion RMSE: 155,000

## Risk Flags
- Mild town regression in YISHUN

## Market Context
No major market shifts found.

## Recommendation
Use this as input to the final human decision.
"""
        structured = _parse_report_text(report)
        self.assertEqual(structured.summary, "Candidate looks healthy.")
        self.assertIn("Test RMSE: 150,000", structured.evidence)
        self.assertIn("Mild town regression in YISHUN", structured.risk_flags)
        self.assertEqual(structured.market_context, "No major market shifts found.")
        self.assertIn("final human decision", structured.recommendation)

    def test_coerces_json_report_to_structured_payload(self):
        report = """{
  "summary": "Candidate is slightly stronger than the baseline.",
  "evidence": ["Test RMSE: 150,000", "Champion RMSE: 155,000"],
  "risk_flags": ["No material regressions detected."],
  "market_context": "No major external shifts identified.",
  "recommendation": "Use this as evidence for the final human review.",
  "citations": ["https://example.com/report"]
}"""
        structured = _coerce_structured_report(report)
        self.assertEqual(
            structured.summary,
            "Candidate is slightly stronger than the baseline.",
        )
        self.assertEqual(structured.citations, ["https://example.com/report"])


class TestExplainerReportEnrichment(unittest.TestCase):
    def test_enrich_structured_report_backfills_core_metrics_and_checks(self):
        comparison = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 195_000, "mae": 150_000},
            champion_metrics={"rmse": 150_000, "mae": 120_000},
            metric_deltas={
                "rmse_delta": 45_000,
                "rmse_delta_pct": 0.30,
            },
            segment_deltas=[
                SegmentDelta(
                    segment_column="town",
                    segment_value="YISHUN",
                    candidate_rmse=195_000,
                    champion_rmse=150_000,
                    rmse_delta=45_000,
                    rmse_delta_pct=0.30,
                )
            ],
        )
        drift_report = DriftReport(
            column_results=[
                ColumnDriftResult(
                    column="town",
                    drift_type="psi",
                    statistic=0.31,
                    threshold=0.2,
                    is_drifted=True,
                )
            ],
            overall_drift_detected=True,
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.MANUAL_REVIEW,
            reasons=["Segment regression needs human review"],
            checks_passed=["absolute_rmse"],
            checks_failed=["segment_rmse_regression", "drift_check"],
        )
        report = PromotionReport(
            summary="Manual review recommended.",
            evidence=["Model needs closer inspection."],
            risk_flags=[],
            market_context="",
            recommendation="Inspect the regressions before any promotion decision.",
            citations=[],
        )

        enriched = _enrich_structured_report(
            report,
            candidate_metrics={"rmse": 195_000, "mae": 150_000},
            champion_info={"metrics": {"rmse": 150_000, "mae": 120_000}},
            comparison=comparison,
            drift_report=drift_report,
            policy_verdict=verdict,
        )
        rendered = _render_markdown_report(enriched)

        self.assertIn("Candidate test RMSE: 195,000", rendered)
        self.assertIn("Candidate test MAE: 150,000", rendered)
        self.assertIn("Champion test RMSE: 150,000", rendered)
        self.assertIn("RMSE delta vs champion: +45,000 (+30.0%)", rendered)
        self.assertIn("Worst segment regressions: town=YISHUN (+30.0%)", rendered)
        self.assertIn("Drift status: detected in town.", rendered)
        self.assertIn("Checks passed: absolute RMSE threshold", rendered)
        self.assertIn(
            "Checks failed: segment RMSE regression threshold, drift check",
            rendered,
        )


class TestExplainerDetailedResult(unittest.TestCase):
    @patch("hdb_resale_mlops.explainer._configure_mlflow_tracing", return_value=None)
    def test_detailed_fallback_returns_structured_payload(self, _mock_tracing):
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            comparison = ComparisonResult(
                has_champion=False,
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
            )
            verdict = PolicyVerdict(
                decision=PolicyDecision.PROMOTE,
                reasons=["No champion"],
                checks_passed=["absolute_rmse"],
                checks_failed=[],
            )
            result = run_explainer_agent_detailed(
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
                champion_info=None,
                comparison=comparison,
                drift_report=None,
                policy_verdict=verdict,
                model_name="test-model",
                model_version="1",
            )
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved

        self.assertIsInstance(result, ExplainerRunResult)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.agent_trace, [])
        self.assertIn("OPENAI_API_KEY not set", result.fallback_note)
        self.assertIn("PROMOTE", result.report_text)
        self.assertIn("Policy verdict", result.structured_report.summary)
        self.assertEqual(result.run_metadata["report_format"], "template_markdown")
        self.assertEqual(result.run_metadata["tool_call_count"], 0)


class TestExplainerOpenAIConfig(unittest.TestCase):
    @patch("hdb_resale_mlops.explainer._configure_mlflow_tracing", return_value=None)
    def test_build_explainer_agent_uses_custom_base_url_and_model(self, _mock_tracing):
        fake_chat = MagicMock(return_value="fake-llm")
        fake_create_react_agent = MagicMock(return_value="fake-agent")

        fake_langchain_openai = types.ModuleType("langchain_openai")
        fake_langchain_openai.ChatOpenAI = fake_chat

        fake_langgraph = types.ModuleType("langgraph")
        fake_langgraph_prebuilt = types.ModuleType("langgraph.prebuilt")
        fake_langgraph_prebuilt.create_react_agent = fake_create_react_agent
        fake_langgraph.prebuilt = fake_langgraph_prebuilt

        fake_langchain_core = types.ModuleType("langchain_core")
        fake_langchain_core_tools = types.ModuleType("langchain_core.tools")
        fake_langchain_core_tools.tool = lambda fn: fn
        fake_langchain_core.tools = fake_langchain_core_tools

        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["No champion"],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )

        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "custom-review-model",
                "OPENAI_BASE_URL": "https://openai-proxy.example.com/v1",
            },
            clear=False,
        ), patch.dict(
            sys.modules,
            {
                "langchain_openai": fake_langchain_openai,
                "langgraph": fake_langgraph,
                "langgraph.prebuilt": fake_langgraph_prebuilt,
                "langchain_core": fake_langchain_core,
                "langchain_core.tools": fake_langchain_core_tools,
            },
        ):
            agent = build_explainer_agent(
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
                champion_info=None,
                comparison=comparison,
                drift_report=None,
                policy_verdict=verdict,
                model_name="test-model",
            )

        self.assertEqual(agent, "fake-agent")
        fake_chat.assert_called_once_with(
            model="custom-review-model",
            temperature=0,
            base_url="https://openai-proxy.example.com/v1",
        )
        fake_create_react_agent.assert_called_once()


class TestExplainerMarketResearchConfig(unittest.TestCase):
    def test_market_research_provider_defaults_to_auto(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_market_research_provider_from_env(), "auto")
            self.assertEqual(_selected_market_research_providers(), [])

    def test_market_research_provider_auto_prefers_tavily(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "TAVILY_API_KEY": "tvly-test"},
            clear=True,
        ):
            self.assertEqual(_selected_market_research_providers(), ["tavily"])

    def test_market_research_provider_can_force_openai(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test",
                "TAVILY_API_KEY": "tvly-test",
                "MARKET_RESEARCH_PROVIDER": "openai",
            },
            clear=True,
        ):
            self.assertEqual(_selected_market_research_providers(), ["openai"])

    def test_openai_market_research_formats_sources_and_summary(self):
        class _FakeAnnotation:
            type = "url_citation"

            def __init__(self, url, title):
                self.url = url
                self.title = title

        class _FakeContent:
            type = "output_text"

            def __init__(self, annotations):
                self.annotations = annotations

        class _FakeMessage:
            type = "message"

            def __init__(self, content):
                self.content = content

        class _FakeSearchSource:
            def __init__(self, url):
                self.url = url

        class _FakeSearchAction:
            def __init__(self, sources):
                self.sources = sources

        class _FakeSearchCall:
            type = "web_search_call"

            def __init__(self, action):
                self.action = action

        class _FakeResponse:
            output_text = "Recent policy commentary and transaction signals remain stable."

            def __init__(self):
                self.output = [
                    _FakeSearchCall(_FakeSearchAction([_FakeSearchSource("https://example.com/news")])),
                    _FakeMessage([_FakeContent([_FakeAnnotation("https://example.com/report", "Market report")])]),
                ]

        fake_client = MagicMock()
        fake_client.responses.create.return_value = _FakeResponse()
        fake_openai_module = types.ModuleType("openai")
        fake_openai_module.OpenAI = MagicMock(return_value=fake_client)

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "gpt-5-nano"},
            clear=True,
        ), patch.dict(sys.modules, {"openai": fake_openai_module}):
            payload = _run_openai_market_research("latest HDB resale policy changes")

        self.assertEqual(payload["provider"], "openai")
        self.assertIn("Recent policy commentary", payload["summary"])
        self.assertEqual(
            payload["sources"],
            ["https://example.com/news", "https://example.com/report"],
        )
        self.assertEqual(payload["model"], "gpt-5-nano")


class TestExplainerTracing(unittest.TestCase):
    @patch("hdb_resale_mlops.explainer.build_explainer_agent")
    def test_detailed_result_includes_mlflow_trace_id(self, mock_build_agent):
        fake_mlflow = _FakeMlflow()
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                SimpleNamespace(type="human", content="brief", tool_calls=[]),
                SimpleNamespace(
                    type="ai",
                    content="thinking",
                    tool_calls=[{"id": "call-1", "name": "query_candidate_metrics", "args": {}}],
                ),
                SimpleNamespace(
                    type="tool",
                    content='{"rmse": 150000}',
                    tool_call_id="call-1",
                    name="query_candidate_metrics",
                ),
                SimpleNamespace(
                    type="ai",
                    content='{"summary":"Healthy candidate","evidence":["RMSE acceptable"],'
                    '"risk_flags":["None"],"market_context":"","recommendation":"Human review can approve.","citations":[]}',
                    tool_calls=[],
                ),
            ]
        }
        mock_build_agent.return_value = mock_agent

        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["No champion"],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake"}, clear=False), patch(
            "hdb_resale_mlops.explainer._configure_mlflow_tracing",
            return_value=fake_mlflow,
        ):
            result = run_explainer_agent_detailed(
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
                champion_info=None,
                comparison=comparison,
                drift_report=None,
                policy_verdict=verdict,
                model_name="test-model",
                model_version="1",
            )

        self.assertEqual(result.run_metadata["mlflow_trace_id"], "trace-123")
        self.assertTrue(result.run_metadata["mlflow_tracing_enabled"])
        self.assertFalse(result.used_fallback)
        self.assertGreaterEqual(len(fake_mlflow.update_calls), 2)
        self.assertEqual(fake_mlflow.started_spans[0].name, "promotion_explainer_review")
        self.assertIn(
            "tool_call:query_candidate_metrics",
            [span.name for span in fake_mlflow.started_spans],
        )
        self.assertIn(
            "tool_result:query_candidate_metrics",
            [span.name for span in fake_mlflow.started_spans],
        )

    def test_fallback_result_includes_mlflow_trace_id(self):
        fake_mlflow = _FakeMlflow()
        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["No champion"],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )

        with patch.dict(os.environ, {}, clear=True), patch(
            "hdb_resale_mlops.explainer._configure_mlflow_tracing",
            return_value=fake_mlflow,
        ):
            result = run_explainer_agent_detailed(
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
                champion_info=None,
                comparison=comparison,
                drift_report=None,
                policy_verdict=verdict,
                model_name="test-model",
                model_version="1",
            )

        self.assertEqual(result.run_metadata["mlflow_trace_id"], "trace-123")
        self.assertTrue(result.run_metadata["mlflow_tracing_enabled"])
        self.assertTrue(result.used_fallback)


class TestExplainerFailureClassification(unittest.TestCase):
    @patch("hdb_resale_mlops.explainer._configure_mlflow_tracing", return_value=None)
    @patch("hdb_resale_mlops.explainer.build_explainer_agent")
    def test_proxy_response_shape_error_has_actionable_fallback_note(
        self,
        mock_build_agent,
        _mock_tracing,
    ):
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = AttributeError("'str' object has no attribute 'model_dump'")
        mock_build_agent.return_value = mock_agent

        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["No champion"],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test-fake",
                "OPENAI_BASE_URL": "https://proxy.example.com/v1",
            },
            clear=False,
        ):
            result = run_explainer_agent_detailed(
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
                champion_info=None,
                comparison=comparison,
                drift_report=None,
                policy_verdict=verdict,
                model_name="test-model",
                model_version="1",
            )

        self.assertTrue(result.used_fallback)
        self.assertIn("OPENAI_BASE_URL", result.fallback_note)
        self.assertIn("not fully compatible", result.fallback_note)
        self.assertEqual(result.run_metadata["fallback_error_type"], "AttributeError")
        self.assertIn(
            "langchain_openai could not parse",
            result.run_metadata["fallback_error_detail"],
        )


if __name__ == "__main__":
    unittest.main()
