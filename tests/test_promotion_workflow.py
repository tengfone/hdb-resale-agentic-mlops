"""Tests for the promotion workflow — all three decision paths."""

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pandas as pd

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta
from hdb_resale_mlops.drift import DriftReport
from hdb_resale_mlops.explainer import ExplainerRunResult, PromotionReport
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict
from hdb_resale_mlops.promotion_workflow import (
    gather_evidence,
    check_drift,
    apply_policy,
    generate_report,
    execute_decision,
    route_after_report,
    resume_promotion_review,
    start_promotion_review,
    start_promotion_review_from_handoff,
)
from hdb_resale_mlops.tabular_state import serialize_for_state
from hdb_resale_mlops.mlflow_registry import MlflowRegistryError, PromotionReviewPersistenceError


def _make_state(**overrides):
    """Create a minimal PromotionState dict for testing."""
    base = {
        "model_name": "test-model",
        "model_version": "1",
        "review_id": "review-1",
        "candidate_metrics": {"rmse": 150_000, "mae": 120_000},
        "candidate_segment_metrics": {
            "town": [
                {"segment": "ANG MO KIO", "rmse": 140_000, "mae": 110_000, "count": 50}
            ],
            "flat_type": [
                {"segment": "4 ROOM", "rmse": 145_000, "mae": 115_000, "count": 100}
            ],
        },
        "train_df": None,
        "test_df": None,
        "champion_info": None,
        "comparison": None,
        "drift_report": None,
        "policy_verdict": None,
        "report": None,
        "human_decision": None,
        "outcome": None,
    }
    base.update(overrides)
    return base


class TestGatherEvidence(unittest.TestCase):
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version")
    @patch("hdb_resale_mlops.promotion_workflow.compare_models")
    def test_no_champion(self, mock_compare, mock_champion):
        mock_champion.return_value = None
        mock_compare.return_value = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        state = _make_state()
        updates = gather_evidence(state)
        self.assertIsNone(updates["champion_info"])
        self.assertFalse(updates["comparison"].has_champion)

    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version")
    @patch("hdb_resale_mlops.promotion_workflow.compare_models")
    def test_with_champion(self, mock_compare, mock_champion):
        champion_segments = {
            "town": pd.DataFrame(
                {
                    "segment": ["ANG MO KIO"],
                    "rmse": [135_000],
                    "mae": [105_000],
                    "count": [50],
                }
            )
        }
        champion_info = {
            "version": "0",
            "run_id": "abc",
            "metrics": {"rmse": 160_000},
            "segment_metrics": champion_segments,
        }
        mock_champion.return_value = champion_info
        mock_compare.return_value = ComparisonResult(
            has_champion=True,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
            champion_metrics={"rmse": 160_000},
            metric_deltas={"rmse_delta": -10_000, "rmse_delta_pct": -0.0625},
        )
        state = _make_state()
        updates = gather_evidence(state)
        self.assertEqual(updates["champion_info"], serialize_for_state(champion_info))
        self.assertTrue(updates["comparison"].has_champion)
        # Verify champion segment metrics were passed through
        _, kwargs = mock_compare.call_args
        self.assertIs(kwargs["champion_segment_metrics"], champion_segments)

    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version")
    def test_registry_error_becomes_evidence_error(self, mock_champion):
        mock_champion.side_effect = MlflowRegistryError("registry unavailable")
        state = _make_state()

        updates = gather_evidence(state)

        self.assertIsNone(updates["champion_info"])
        self.assertIn("champion evidence could not be loaded", updates["evidence_errors"][0])


class TestCheckDrift(unittest.TestCase):
    def test_drift_skipped_when_no_data(self):
        state = _make_state(train_df=None, test_df=None)
        updates = check_drift(state)
        self.assertIsNone(updates["drift_report"])


class TestApplyPolicy(unittest.TestCase):
    def test_promote_verdict(self):
        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        state = _make_state(comparison=comparison, drift_report=None)
        updates = apply_policy(state)
        self.assertEqual(updates["policy_verdict"].decision, PolicyDecision.PROMOTE)

    def test_reject_verdict_high_rmse(self):
        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 250_000, "mae": 120_000},
        )
        state = _make_state(
            candidate_metrics={"rmse": 250_000, "mae": 120_000},
            comparison=comparison,
            drift_report=None,
        )
        updates = apply_policy(state)
        self.assertEqual(updates["policy_verdict"].decision, PolicyDecision.REJECT)

    def test_manual_review_drift(self):
        comparison = ComparisonResult(
            has_champion=False,
            candidate_metrics={"rmse": 150_000, "mae": 120_000},
        )
        drift = DriftReport(column_results=[], overall_drift_detected=True)
        state = _make_state(comparison=comparison, drift_report=drift)
        updates = apply_policy(state)
        self.assertEqual(
            updates["policy_verdict"].decision, PolicyDecision.MANUAL_REVIEW
        )


class TestGenerateReport(unittest.TestCase):
    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=True)
    @patch("hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed")
    def test_returns_structured_report_and_trace(self, mock_explainer, mock_log_review):
        mock_explainer.return_value = ExplainerRunResult(
            report_text="## Summary\nStructured report",
            structured_report=PromotionReport(
                summary="Structured report",
                evidence=["RMSE reviewed"],
                risk_flags=["None"],
                market_context="Stable market",
                recommendation="Human should approve or reject.",
            ),
            agent_trace=[{"event": "tool_call", "tool_name": "query_candidate_metrics"}],
            run_metadata={"tool_call_count": 1, "report_format": "markdown"},
            used_fallback=False,
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["All checks passed"],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )
        state = _make_state(
            comparison=ComparisonResult(
                has_champion=False,
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
            ),
            policy_verdict=verdict,
        )

        updates = generate_report(state)

        self.assertEqual(updates["report"], "## Summary\nStructured report")
        self.assertEqual(updates["report_structured"]["summary"], "Structured report")
        self.assertEqual(len(updates["agent_trace"]), 1)
        self.assertEqual(updates["agent_run_metadata"]["tool_call_count"], 1)
        self.assertEqual(updates["agent_run_metadata"]["judge_status"], "disabled")
        self.assertEqual(updates["judge_evaluation"]["status"], "disabled")
        mock_log_review.assert_called_once()

    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=True)
    @patch("hdb_resale_mlops.eval_judge.evaluate_report")
    @patch("hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed")
    def test_enabled_judge_scores_are_attached_to_review(
        self,
        mock_explainer,
        mock_evaluate_report,
        mock_log_review,
    ):
        mock_explainer.return_value = ExplainerRunResult(
            report_text="## Summary\nStructured report",
            structured_report=PromotionReport(
                summary="Structured report",
                evidence=["RMSE reviewed"],
                risk_flags=["None"],
                market_context="Stable market",
                recommendation="Human should approve or reject.",
            ),
            agent_trace=[],
            run_metadata={"tool_call_count": 0, "report_format": "markdown"},
            used_fallback=False,
        )
        mock_evaluate_report.return_value = SimpleNamespace(
            to_dict=lambda: {
                "completeness": 4,
                "accuracy": 5,
                "actionability": 4,
                "safety": 5,
                "average": 4.5,
                "reasoning": "Clear report.",
            }
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["All checks passed"],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )
        state = _make_state(
            comparison=ComparisonResult(
                has_champion=False,
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
            ),
            policy_verdict=verdict,
        )

        with patch.dict(os.environ, {"ENABLE_JUDGE_EVAL": "true", "OPENAI_API_KEY": "sk-test"}, clear=False):
            updates = generate_report(state)

        self.assertEqual(updates["judge_evaluation"]["status"], "scored")
        self.assertEqual(updates["judge_evaluation"]["scores"]["average"], 4.5)
        self.assertEqual(updates["agent_run_metadata"]["judge_status"], "scored")
        self.assertEqual(updates["agent_run_metadata"]["judge_average_score"], 4.5)
        review_payload = mock_log_review.call_args.kwargs["review_payload"]
        self.assertEqual(review_payload["judge_evaluation"]["status"], "scored")

    @patch("hdb_resale_mlops.promotion_workflow._mlflow_tracking_is_configured", return_value=True)
    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=False)
    @patch("hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed")
    def test_raises_when_mlflow_review_logging_fails(
        self,
        mock_explainer,
        _mock_log_review,
        _mock_tracking_configured,
    ):
        mock_explainer.return_value = ExplainerRunResult(
            report_text="## Summary\nStructured report",
            structured_report=PromotionReport(
                summary="Structured report",
                evidence=["RMSE reviewed"],
                risk_flags=["None"],
                market_context="Stable market",
                recommendation="Human should approve or reject.",
            ),
            agent_trace=[],
            run_metadata={"tool_call_count": 0, "report_format": "markdown"},
            used_fallback=False,
        )
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["All checks passed"],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )
        state = _make_state(
            comparison=ComparisonResult(
                has_champion=False,
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
            ),
            policy_verdict=verdict,
        )

        with self.assertRaises(PromotionReviewPersistenceError):
            generate_report(state)


class TestExecuteDecision(unittest.TestCase):
    @patch("hdb_resale_mlops.mlflow_registry.promote_to_champion")
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="reviewer-1")
    def test_approve(self, _mock_reviewer, mock_promote):
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=["All checks passed"],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )
        state = _make_state(
            policy_verdict=verdict,
            human_decision="approve",
        )
        updates = execute_decision(state)
        self.assertEqual(updates["outcome"], "promoted")
        mock_promote.assert_called_once()
        self.assertEqual(mock_promote.call_args.args, ("test-model", "1"))
        self.assertEqual(
            mock_promote.call_args.kwargs["decision_metadata"]["decision_source"],
            "human_review",
        )
        self.assertEqual(
            mock_promote.call_args.kwargs["decision_metadata"]["decision_reviewer"],
            "reviewer-1",
        )
        self.assertEqual(
            mock_promote.call_args.kwargs["decision_metadata"]["policy_verdict"],
            "PROMOTE",
        )
        self.assertFalse(
            mock_promote.call_args.kwargs["decision_metadata"]["rejection_overridden"]
        )

    @patch("hdb_resale_mlops.mlflow_registry.reject_candidate")
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="reviewer-2")
    def test_reject(self, _mock_reviewer, mock_reject):
        verdict = PolicyVerdict(
            decision=PolicyDecision.REJECT,
            reasons=["RMSE too high"],
            checks_passed=[],
            checks_failed=["absolute_rmse"],
        )
        state = _make_state(
            policy_verdict=verdict,
            human_decision="reject",
        )
        updates = execute_decision(state)
        self.assertEqual(updates["outcome"], "rejected")
        mock_reject.assert_called_once()
        self.assertEqual(mock_reject.call_args.args[:3], ("test-model", "1", ["RMSE too high"]))
        self.assertEqual(
            mock_reject.call_args.kwargs["decision_metadata"]["decision_source"],
            "human_review",
        )
        self.assertEqual(
            mock_reject.call_args.kwargs["decision_metadata"]["decision_reviewer"],
            "reviewer-2",
        )

    @patch("hdb_resale_mlops.mlflow_registry.reject_candidate")
    def test_auto_reject(self, mock_reject):
        verdict = PolicyVerdict(
            decision=PolicyDecision.REJECT,
            reasons=["RMSE too high"],
            checks_passed=[],
            checks_failed=["absolute_rmse"],
        )
        state = _make_state(
            policy_verdict=verdict,
            human_decision="",
        )
        updates = execute_decision(state)
        self.assertEqual(updates["outcome"], "rejected")
        self.assertEqual(
            mock_reject.call_args.kwargs["decision_metadata"]["decision_source"],
            "policy_auto_reject",
        )
        self.assertNotIn("decision_reviewer", mock_reject.call_args.kwargs["decision_metadata"])


class TestRouteAfterReport(unittest.TestCase):
    def test_reject_routes_directly_to_execution(self):
        verdict = PolicyVerdict(
            decision=PolicyDecision.REJECT,
            reasons=["RMSE too high"],
            checks_passed=[],
            checks_failed=["absolute_rmse"],
        )
        state = _make_state(policy_verdict=verdict)
        self.assertEqual(route_after_report(state), "execute_decision")

    def test_promote_routes_to_human_review(self):
        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=[],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )
        state = _make_state(policy_verdict=verdict)
        self.assertEqual(route_after_report(state), "human_review")


class TestSequentialFallback(unittest.TestCase):
    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=True)
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=ExplainerRunResult(
            report_text="## Summary\nFallback workflow report",
            structured_report=PromotionReport(
                summary="Fallback workflow report",
                evidence=["RMSE reviewed"],
                risk_flags=["None"],
                market_context="Stable market",
                recommendation="Human should approve or reject.",
            ),
            agent_trace=[],
            run_metadata={"tool_call_count": 0, "report_format": "markdown"},
            used_fallback=True,
            fallback_note="langgraph not installed",
        ),
    )
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version", return_value=None)
    @patch(
        "hdb_resale_mlops.promotion_workflow.build_promotion_graph",
        side_effect=ModuleNotFoundError("No module named 'langgraph'"),
    )
    def test_start_review_degrades_gracefully_without_langgraph(
        self,
        _mock_graph,
        _mock_champion,
        _mock_explainer,
        _mock_log_review,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            review = start_promotion_review(
                model_name="test-model",
                model_version="1",
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
                review_dir=tmp_dir,
                use_langgraph=True,
            )

        self.assertEqual(review["status"], "pending_review")
        self.assertEqual(review["policy_verdict"]["decision"], "PROMOTE")
        self.assertEqual(
            review["agent_run_metadata"]["workflow_runtime"],
            "sequential_fallback",
        )


class TestPipelineReviewHandoff(unittest.TestCase):
    @patch("hdb_resale_mlops.promotion_workflow._mlflow_tracking_is_configured", return_value=False)
    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=True)
    @patch("hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed")
    def test_pending_review_handoff_stays_pending(
        self,
        mock_explainer,
        _mock_log_review,
        _mock_tracking_configured,
    ):
        mock_explainer.return_value = ExplainerRunResult(
            report_text="## Summary\nPipeline handoff report",
            structured_report=PromotionReport(
                summary="Pipeline handoff report",
                evidence=["Evidence"],
                risk_flags=["None"],
                market_context="Stable market",
                recommendation="Human should approve or reject.",
            ),
            agent_trace=[],
            run_metadata={"tool_call_count": 0, "report_format": "markdown"},
            used_fallback=False,
        )
        handoff = {
            "status": "pending_review",
            "registration": {
                "model_name": "test-model",
                "model_version": "5",
            },
            "candidate_metrics": {"rmse": 150_000, "mae": 120_000},
            "champion_info": None,
            "comparison": asdict(
                ComparisonResult(
                    has_champion=False,
                    candidate_metrics={"rmse": 150_000, "mae": 120_000},
                )
            ),
            "drift_report": {
                "overall_drift_detected": False,
                "column_results": [],
            },
            "policy_verdict": {
                "decision": "PROMOTE",
                "reasons": ["All checks passed"],
                "checks_passed": ["absolute_rmse"],
                "checks_failed": [],
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            review = start_promotion_review_from_handoff(
                handoff,
                review_dir=tmp_dir,
            )

        self.assertEqual(review["status"], "pending_review")
        self.assertEqual(review["policy_verdict"]["decision"], "PROMOTE")
        self.assertEqual(review["outcome"], "pending_review")

    @patch("hdb_resale_mlops.promotion_workflow._mlflow_tracking_is_configured", return_value=False)
    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=True)
    @patch("hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed")
    def test_auto_rejected_handoff_preserves_rejected_state(
        self,
        mock_explainer,
        _mock_log_review,
        _mock_tracking_configured,
    ):
        mock_explainer.return_value = ExplainerRunResult(
            report_text="## Summary\nRejected report",
            structured_report=PromotionReport(
                summary="Rejected report",
                evidence=["RMSE too high"],
                risk_flags=["High RMSE"],
                market_context="Stable market",
                recommendation="Human should approve or reject.",
            ),
            agent_trace=[],
            run_metadata={"tool_call_count": 0, "report_format": "markdown"},
            used_fallback=False,
        )
        handoff = {
            "status": "auto_rejected",
            "registration": {
                "model_name": "test-model",
                "model_version": "6",
            },
            "candidate_metrics": {"rmse": 250_000, "mae": 120_000},
            "champion_info": None,
            "comparison": asdict(
                ComparisonResult(
                    has_champion=False,
                    candidate_metrics={"rmse": 250_000, "mae": 120_000},
                )
            ),
            "drift_report": {
                "overall_drift_detected": False,
                "column_results": [],
            },
            "policy_verdict": {
                "decision": "REJECT",
                "reasons": ["Test RMSE 250,000 exceeds maximum threshold 200,000"],
                "checks_passed": ["absolute_mae"],
                "checks_failed": ["absolute_rmse"],
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            review = start_promotion_review_from_handoff(
                handoff,
                review_dir=tmp_dir,
            )

        self.assertEqual(review["status"], "auto_rejected")
        self.assertEqual(review["outcome"], "rejected")
        self.assertEqual(review["human_decision"], "auto_reject")

    @patch("hdb_resale_mlops.promotion_workflow._mlflow_tracking_is_configured", return_value=False)
    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=True)
    @patch("hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed")
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="alice")
    @patch("hdb_resale_mlops.mlflow_registry.promote_to_champion")
    def test_auto_rejected_handoff_can_be_overridden_to_promote(
        self,
        mock_promote,
        _mock_reviewer,
        mock_explainer,
        _mock_log_review,
        _mock_tracking_configured,
    ):
        mock_explainer.return_value = ExplainerRunResult(
            report_text="## Summary\nRejected report",
            structured_report=PromotionReport(
                summary="Rejected report",
                evidence=["RMSE too high"],
                risk_flags=["High RMSE"],
                market_context="Stable market",
                recommendation="Human should approve or reject.",
            ),
            agent_trace=[],
            run_metadata={"tool_call_count": 0, "report_format": "markdown"},
            used_fallback=False,
        )
        handoff = {
            "status": "auto_rejected",
            "registration": {
                "model_name": "test-model",
                "model_version": "7",
            },
            "candidate_metrics": {"rmse": 250_000, "mae": 120_000},
            "champion_info": None,
            "comparison": asdict(
                ComparisonResult(
                    has_champion=True,
                    candidate_metrics={"rmse": 250_000, "mae": 120_000},
                    champion_metrics={"rmse": 210_000, "mae": 110_000},
                    metric_deltas={"rmse_delta": 40_000, "rmse_delta_pct": 0.1905},
                    segment_deltas=[
                        SegmentDelta(
                            segment_column="town",
                            segment_value="BEDOK",
                            candidate_rmse=240_000.0,
                            champion_rmse=200_000.0,
                            rmse_delta=40_000.0,
                            rmse_delta_pct=0.20,
                        )
                    ],
                )
            ),
            "drift_report": {
                "overall_drift_detected": False,
                "column_results": [],
            },
            "policy_verdict": {
                "decision": "REJECT",
                "reasons": ["Candidate RMSE is 19.0% worse than champion (threshold: 10%)"],
                "checks_passed": ["absolute_rmse", "absolute_mae"],
                "checks_failed": ["champion_rmse_regression"],
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            review = start_promotion_review_from_handoff(
                handoff,
                review_dir=tmp_dir,
            )
            completed = resume_promotion_review(
                review["review_id"],
                "approve",
                review_dir=tmp_dir,
            )

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["outcome"], "promoted")
        mock_promote.assert_called_once()


class TestLoadSegmentArtifacts(unittest.TestCase):
    """Tests for loading champion segment metrics from MLflow artifacts."""

    @patch("mlflow.MlflowClient", autospec=True)
    def test_loads_segment_json_files(self, MockClient):
        """Verify DataFrames are reconstructed from downloaded JSON artifacts."""
        from hdb_resale_mlops.mlflow_registry import _load_segment_artifacts

        # Create a temp directory with segment JSON files
        tmp = tempfile.mkdtemp()
        eval_dir = os.path.join(tmp, "evaluation")
        os.makedirs(eval_dir)

        town_data = [
            {"segment": "ANG MO KIO", "rmse": 130_000, "mae": 100_000, "count": 50}
        ]
        flat_data = [
            {"segment": "4 ROOM", "rmse": 140_000, "mae": 110_000, "count": 80}
        ]

        with open(os.path.join(eval_dir, "test_segments_by_town.json"), "w") as f:
            json.dump(town_data, f)
        with open(os.path.join(eval_dir, "test_segments_by_flat_type.json"), "w") as f:
            json.dump(flat_data, f)

        MockClient.return_value.download_artifacts.return_value = eval_dir

        result = _load_segment_artifacts("run-123")

        self.assertIn("town", result)
        self.assertIn("flat_type", result)
        self.assertEqual(list(result["town"]["segment"]), ["ANG MO KIO"])
        self.assertEqual(list(result["town"]["rmse"]), [130_000])
        self.assertEqual(list(result["flat_type"]["segment"]), ["4 ROOM"])

    @patch("mlflow.MlflowClient", autospec=True)
    def test_returns_empty_dict_when_no_artifacts(self, MockClient):
        """Gracefully handle runs that have no segment artifacts."""
        from hdb_resale_mlops.mlflow_registry import _load_segment_artifacts

        MockClient.return_value.download_artifacts.side_effect = Exception("Not found")

        result = _load_segment_artifacts("run-old")
        self.assertEqual(result, {})

    @patch("mlflow.MlflowClient", autospec=True)
    def test_partial_segments(self, MockClient):
        """Only town segments exist — flat_type key should be absent."""
        from hdb_resale_mlops.mlflow_registry import _load_segment_artifacts

        tmp = tempfile.mkdtemp()
        eval_dir = os.path.join(tmp, "evaluation")
        os.makedirs(eval_dir)

        town_data = [{"segment": "BEDOK", "rmse": 155_000, "mae": 120_000, "count": 60}]
        with open(os.path.join(eval_dir, "test_segments_by_town.json"), "w") as f:
            json.dump(town_data, f)

        MockClient.return_value.download_artifacts.return_value = eval_dir

        result = _load_segment_artifacts("run-partial")
        self.assertIn("town", result)
        self.assertNotIn("flat_type", result)


class TestPromoteToChampion(unittest.TestCase):
    @patch("mlflow.MlflowClient", autospec=True)
    def test_clears_stale_rejection_reasons(self, MockClient):
        from hdb_resale_mlops.mlflow_registry import promote_to_champion

        promote_to_champion("test-model", "7")

        client = MockClient.return_value
        client.set_registered_model_alias.assert_called_once_with(
            "test-model", "champion", "7"
        )
        client.set_model_version_tag.assert_called_once_with(
            "test-model", "7", "promotion_status", "champion"
        )
        client.delete_model_version_tag.assert_called_once_with(
            "test-model", "7", "rejection_reasons"
        )

    @patch("mlflow.MlflowClient", autospec=True)
    def test_sets_decision_metadata_tags(self, MockClient):
        from hdb_resale_mlops.mlflow_registry import promote_to_champion

        promote_to_champion(
            "test-model",
            "8",
            decision_metadata={
                "decision_source": "human_override_after_policy_reject",
                "decision_reviewer": "alice",
                "rejection_overridden": True,
                "policy_verdict": "REJECT",
            },
        )

        client = MockClient.return_value
        tagged_pairs = [
            call.args[2:]
            for call in client.set_model_version_tag.call_args_list
        ]
        self.assertIn(("promotion_status", "champion"), tagged_pairs)
        self.assertIn(("decision_source", "human_override_after_policy_reject"), tagged_pairs)
        self.assertIn(("decision_reviewer", "alice"), tagged_pairs)
        self.assertIn(("rejection_overridden", "true"), tagged_pairs)
        self.assertIn(("policy_verdict", "REJECT"), tagged_pairs)


class TestRejectCandidate(unittest.TestCase):
    @patch("mlflow.MlflowClient", autospec=True)
    def test_sets_rejection_and_decision_tags(self, MockClient):
        from hdb_resale_mlops.mlflow_registry import reject_candidate

        reject_candidate(
            "test-model",
            "9",
            ["RMSE too high"],
            decision_metadata={
                "decision_source": "human_confirmed_reject",
                "decision_reviewer": "bob",
                "policy_verdict": "REJECT",
            },
        )

        client = MockClient.return_value
        tagged_pairs = [
            call.args[2:]
            for call in client.set_model_version_tag.call_args_list
        ]
        self.assertIn(("promotion_status", "rejected"), tagged_pairs)
        self.assertIn(("rejection_reasons", "RMSE too high"), tagged_pairs)
        self.assertIn(("decision_source", "human_confirmed_reject"), tagged_pairs)
        self.assertIn(("decision_reviewer", "bob"), tagged_pairs)
        self.assertIn(("policy_verdict", "REJECT"), tagged_pairs)


if __name__ == "__main__":
    unittest.main()
