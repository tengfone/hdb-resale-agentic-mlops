"""Unit tests for the native LangChain-backed report-quality judge wrapper."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from hdb_resale_mlops.eval_judge import (
    JudgeScore,
    _coerce_feedback_scores,
    _resolve_judge_model,
    evaluate_report,
)


_SCENARIO = {
    "candidate_metrics": {"rmse": 123456.0, "mae": 98765.0},
    "champion_info": {"metrics": {"rmse": 120000.0, "mae": 95000.0}},
    "policy_verdict": {
        "decision": "MANUAL_REVIEW",
        "reasons": ["Drift detected"],
        "checks_passed": ["test_mae_within_limit"],
        "checks_failed": ["drift_detected"],
    },
    "comparison": {
        "segment_deltas": [
            {
                "segment_column": "town",
                "segment_value": "YISHUN",
                "rmse_delta_pct": 0.22,
            }
        ]
    },
    "drift_report": {
        "overall_drift_detected": True,
        "column_results": [
            {"column": "town", "is_drifted": True},
        ],
    },
}


class _StructuredJudgeResponse(BaseModel):
    completeness: int
    accuracy: int
    actionability: int
    safety: int
    reasoning: str = ""


class _FakeStructuredJudge:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _FakeJudgeClient:
    def __init__(self, structured_response, text_response):
        self.structured_response = structured_response
        self.text_response = text_response
        self.with_structured_output_calls = []
        self.invoke_messages = None
        self.structured_runner = None

    def with_structured_output(self, schema, method="json_schema", strict=True):
        self.with_structured_output_calls.append(
            {"schema": schema, "method": method, "strict": strict}
        )
        self.structured_runner = _FakeStructuredJudge(self.structured_response)
        return self.structured_runner

    def invoke(self, messages):
        self.invoke_messages = messages
        if isinstance(self.text_response, Exception):
            raise self.text_response
        return self.text_response


class TestEvalJudgeHelpers(unittest.TestCase):
    def test_judge_score_to_dict_includes_average(self):
        score = JudgeScore(
            completeness=4,
            accuracy=5,
            actionability=3,
            safety=5,
            reasoning="Useful advisory score.",
        )

        self.assertEqual(
            score.to_dict(),
            {
                "completeness": 4,
                "accuracy": 5,
                "actionability": 3,
                "safety": 5,
                "average": 4.25,
                "reasoning": "Useful advisory score.",
            },
        )

    def test_resolve_judge_model_normalizes_plain_openai_name(self):
        self.assertEqual(_resolve_judge_model("gpt-5-mini"), "gpt-5-mini")

    def test_resolve_judge_model_strips_legacy_openai_prefix(self):
        self.assertEqual(_resolve_judge_model("openai:/gpt-5-mini"), "gpt-5-mini")

    def test_resolve_judge_model_rejects_non_openai_provider_uri(self):
        with self.assertRaisesRegex(RuntimeError, "plain OpenAI-compatible model name"):
            _resolve_judge_model("anthropic:/claude-sonnet-4-5")

    def test_coerce_feedback_scores_accepts_json_string(self):
        scores = _coerce_feedback_scores(
            '{"completeness": 4, "accuracy": "5", "actionability": 3, "safety": 5}'
        )
        self.assertEqual(
            scores,
            {
                "completeness": 4,
                "accuracy": 5,
                "actionability": 3,
                "safety": 5,
            },
        )

    def test_coerce_feedback_scores_accepts_key_value_string(self):
        scores = _coerce_feedback_scores(
            "completeness=3, accuracy=5, actionability=5, safety=5"
        )
        self.assertEqual(
            scores,
            {
                "completeness": 3,
                "accuracy": 5,
                "actionability": 5,
                "safety": 5,
            },
        )

    def test_coerce_feedback_scores_accepts_colon_delimited_string(self):
        scores = _coerce_feedback_scores(
            "completeness: 3, accuracy: 5, actionability: 5, safety: 5"
        )
        self.assertEqual(
            scores,
            {
                "completeness": 3,
                "accuracy": 5,
                "actionability": 5,
                "safety": 5,
            },
        )

    def test_coerce_feedback_scores_requires_all_metrics(self):
        with self.assertRaises(KeyError):
            _coerce_feedback_scores({"completeness": 4})


class TestEvaluateReport(unittest.TestCase):
    @patch("hdb_resale_mlops.eval_judge._build_judge_client")
    def test_evaluate_report_uses_structured_langchain_contract(self, mock_build_client):
        fake_client = _FakeJudgeClient(
            structured_response=_StructuredJudgeResponse(
                completeness=4,
                accuracy=5,
                actionability=4,
                safety=5,
                reasoning="Covered the key metrics and kept the decision with the reviewer.",
            ),
            text_response=None,
        )
        mock_build_client.return_value = fake_client

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://judge.example/v1",
                "OPENAI_JUDGE_MODEL": "gpt-5-mini",
            },
            clear=True,
        ):
            score = evaluate_report(
                report="Summary with evidence and recommendation.",
                scenario=_SCENARIO,
            )

        self.assertEqual(score.completeness, 4)
        self.assertEqual(score.accuracy, 5)
        self.assertEqual(score.actionability, 4)
        self.assertEqual(score.safety, 5)
        self.assertIn("reviewer", score.reasoning)

        mock_build_client.assert_called_once_with(
            model_name="gpt-5-mini",
            api_key="test-key",
            base_url="https://judge.example/v1",
        )
        self.assertEqual(fake_client.with_structured_output_calls[0]["method"], "json_schema")
        self.assertEqual(fake_client.with_structured_output_calls[0]["strict"], True)
        self.assertEqual(fake_client.structured_runner.messages[0][0], "system")
        self.assertEqual(fake_client.structured_runner.messages[1][0], "human")
        self.assertIn("Scenario context", fake_client.structured_runner.messages[1][1])

    @patch("hdb_resale_mlops.eval_judge._build_judge_client")
    def test_evaluate_report_reuses_main_openai_credentials(self, mock_build_client):
        fake_client = _FakeJudgeClient(
            structured_response=_StructuredJudgeResponse(
                completeness=4,
                accuracy=5,
                actionability=4,
                safety=5,
                reasoning="Reused the main OpenAI credentials.",
            ),
            text_response=None,
        )
        mock_build_client.return_value = fake_client

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "shared-key",
                "OPENAI_BASE_URL": "https://proxy.example.com/v1",
            },
            clear=True,
        ):
            evaluate_report(
                report="Summary with evidence and recommendation.",
                scenario=_SCENARIO,
            )

        mock_build_client.assert_called_once_with(
            model_name="gpt-5-mini",
            api_key="shared-key",
            base_url="https://proxy.example.com/v1",
        )

    def test_evaluate_report_requires_openai_key_for_openai_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                evaluate_report(
                    report="report",
                    scenario=_SCENARIO,
                    judge_model="gpt-5-mini",
                )

    def test_evaluate_report_rejects_unsupported_provider_uri(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "plain OpenAI-compatible model name"):
                evaluate_report(
                    report="report",
                    scenario=_SCENARIO,
                    judge_model="anthropic:/claude-sonnet-4-5",
                )

    @patch("hdb_resale_mlops.eval_judge._build_judge_client")
    def test_evaluate_report_falls_back_to_plain_text_judge_output(self, mock_build_client):
        fake_client = _FakeJudgeClient(
            structured_response=RuntimeError("structured output unsupported"),
            text_response='{"completeness": 4, "accuracy": 4, "actionability": 4, "safety": 5, "reasoning": "Recovered from plain text fallback."}',
        )
        mock_build_client.return_value = fake_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            score = evaluate_report(
                report="report",
                scenario=_SCENARIO,
                judge_model="gpt-5-mini",
            )

        self.assertEqual(score.accuracy, 4)
        self.assertEqual(score.safety, 5)
        self.assertIn("plain text fallback", score.reasoning)
        self.assertIsNotNone(fake_client.invoke_messages)


if __name__ == "__main__":
    unittest.main()
