"""Tier 2 — Integration tests for the LangGraph promotion workflow.

These tests compile the full promotion graph and run it with mocked
dependencies (MLflow, LLM), verifying:
  - State transitions follow the correct order
  - Policy routing (PROMOTE, REJECT, MANUAL_REVIEW) reaches correct nodes
  - REJECT auto-executes after report generation
  - PROMOTE / MANUAL_REVIEW still pause for human approval
  - Final outcomes (promoted/rejected) are correctly applied
"""

import json
import pathlib
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta
from hdb_resale_mlops.drift import ColumnDriftResult, DriftReport
from hdb_resale_mlops.explainer import ExplainerRunResult, PromotionReport
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "eval_scenarios"
_TRACKING_PATCHER = patch(
    "hdb_resale_mlops.promotion_workflow._mlflow_tracking_is_configured",
    return_value=False,
)


def setUpModule():
    _TRACKING_PATCHER.start()


def tearDownModule():
    _TRACKING_PATCHER.stop()


def _load_scenario(name: str) -> dict:
    """Load a scenario fixture and return raw dict for state construction."""
    with open(FIXTURES_DIR / f"{name}.json") as f:
        return json.load(f)


def _build_initial_state(scenario_raw: dict) -> dict:
    """Build a PromotionState-compatible dict from raw scenario data."""
    return {
        "model_name": scenario_raw["model_name"],
        "model_version": scenario_raw["model_version"],
        "review_id": f"review-{scenario_raw['model_version']}",
        "candidate_metrics": scenario_raw["candidate_metrics"],
        "candidate_segment_metrics": {},
        "train_df": None,
        "test_df": None,
    }


def _mock_explainer_result(report_text: str) -> ExplainerRunResult:
    return ExplainerRunResult(
        report_text=report_text,
        structured_report=PromotionReport(
            summary=report_text,
            evidence=["candidate metrics reviewed"],
            risk_flags=[],
            market_context="",
            recommendation="Human review required.",
        ),
        agent_trace=[{"event": "tool_call", "tool_name": "query_candidate_metrics"}],
    )


class TestWorkflowGraphConstruction(unittest.TestCase):
    """Verify the LangGraph compiles and has expected structure."""

    def test_graph_compiles(self):
        from hdb_resale_mlops.promotion_workflow import build_promotion_graph
        graph = build_promotion_graph()
        self.assertTrue(hasattr(graph, "invoke"))

    def test_graph_has_all_nodes(self):
        from hdb_resale_mlops.promotion_workflow import build_promotion_graph
        graph = build_promotion_graph()
        # LangGraph compiled graphs expose node names via get_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected_nodes = {
            "gather_evidence", "check_drift", "apply_policy",
            "generate_report", "human_review", "execute_decision",
        }
        # __start__ and __end__ are implicit
        self.assertTrue(expected_nodes.issubset(node_names),
                        f"Missing nodes: {expected_nodes - node_names}")


class TestWorkflowNodeSequence(unittest.TestCase):
    """Test individual nodes in sequence with mocked dependencies."""

    def _make_comparison(self, has_champion=False, **kwargs):
        return ComparisonResult(
            has_champion=has_champion,
            candidate_metrics=kwargs.get("candidate_metrics", {"rmse": 150_000, "mae": 120_000}),
            champion_metrics=kwargs.get("champion_metrics"),
            metric_deltas=kwargs.get("metric_deltas", {}),
            segment_deltas=kwargs.get("segment_deltas", []),
        )

    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version", return_value=None)
    def test_full_deterministic_layer_no_champion(self, _mock_champ):
        """gather_evidence → check_drift → apply_policy runs correctly without champion."""
        from hdb_resale_mlops.promotion_workflow import (
            gather_evidence, check_drift, apply_policy,
        )

        state = _build_initial_state(_load_scenario("promote_no_champion"))

        # Layer 1a
        updates = gather_evidence(state)
        state.update(updates)
        self.assertIsNone(state["champion_info"])
        self.assertFalse(state["comparison"].has_champion)

        # Layer 1b
        updates = check_drift(state)
        state.update(updates)
        self.assertIsNone(state["drift_report"])

        # Layer 1c
        updates = apply_policy(state)
        state.update(updates)
        self.assertEqual(state["policy_verdict"].decision, PolicyDecision.PROMOTE)

    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version")
    def test_full_deterministic_layer_reject(self, mock_champ):
        """REJECT path: high RMSE triggers rejection."""
        from hdb_resale_mlops.promotion_workflow import (
            gather_evidence, check_drift, apply_policy,
        )

        mock_champ.return_value = {
            "version": "2",
            "run_id": "def456",
            "metrics": {"rmse": 155_000, "mae": 125_000},
            "segment_metrics": {},
        }
        state = _build_initial_state(_load_scenario("reject_high_rmse"))

        updates = gather_evidence(state)
        state.update(updates)
        updates = check_drift(state)
        state.update(updates)
        updates = apply_policy(state)
        state.update(updates)

        self.assertEqual(state["policy_verdict"].decision, PolicyDecision.REJECT)
        self.assertIn("absolute_rmse", state["policy_verdict"].checks_failed)


class TestWorkflowExecuteDecision(unittest.TestCase):
    """Test the final execute_decision node."""

    @patch("hdb_resale_mlops.mlflow_registry.promote_to_champion")
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="reviewer-1")
    def test_approve_promotes(self, _mock_reviewer, mock_promote):
        from hdb_resale_mlops.promotion_workflow import execute_decision

        verdict = PolicyVerdict(
            decision=PolicyDecision.PROMOTE,
            reasons=[],
            checks_passed=["absolute_rmse"],
            checks_failed=[],
        )
        state = {
            "model_name": "test-model",
            "model_version": "1",
            "policy_verdict": verdict,
            "human_decision": "approve",
        }
        updates = execute_decision(state)
        self.assertEqual(updates["outcome"], "promoted")
        mock_promote.assert_called_once()
        self.assertEqual(mock_promote.call_args.args, ("test-model", "1"))
        self.assertEqual(
            mock_promote.call_args.kwargs["decision_metadata"]["decision_source"],
            "human_review",
        )

    @patch("hdb_resale_mlops.mlflow_registry.reject_candidate")
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="reviewer-2")
    def test_reject_records_reasons(self, _mock_reviewer, mock_reject):
        from hdb_resale_mlops.promotion_workflow import execute_decision

        verdict = PolicyVerdict(
            decision=PolicyDecision.REJECT,
            reasons=["RMSE too high", "Champion regression"],
            checks_passed=[],
            checks_failed=["absolute_rmse"],
        )
        state = {
            "model_name": "test-model",
            "model_version": "1",
            "policy_verdict": verdict,
            "human_decision": "reject",
        }
        updates = execute_decision(state)
        self.assertEqual(updates["outcome"], "rejected")
        mock_reject.assert_called_once()
        self.assertEqual(
            mock_reject.call_args.args[2], ["RMSE too high", "Champion regression"]
        )
        self.assertEqual(
            mock_reject.call_args.kwargs["decision_metadata"]["decision_source"],
            "human_review",
        )


class TestWorkflowInterruptResume(unittest.TestCase):
    """Test the graph-level interrupt/resume flow for human approval."""

    @patch("hdb_resale_mlops.mlflow_registry.promote_to_champion")
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version", return_value=None)
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=_mock_explainer_result("Test report"),
    )
    def test_graph_pauses_at_human_review_and_resumes(
        self, _mock_explainer, _mock_champ, mock_promote
    ):
        """Full graph run: should pause at human_review, then resume with approval."""
        from hdb_resale_mlops.promotion_workflow import build_promotion_graph
        from langgraph.types import Command

        graph = build_promotion_graph()
        config = {"configurable": {"thread_id": "test-promote-1"}}

        state = _build_initial_state(_load_scenario("promote_no_champion"))
        # First run — should pause at human_review (interrupt)
        result = graph.invoke(state, config=config)

        # The graph should have paused — check that outcome is not yet set
        snapshot = graph.get_state(config)
        self.assertIn("human_review", snapshot.next,
                       "Graph should be paused at human_review node")

        # Resume with approval
        result = graph.invoke(Command(resume="approve"), config=config)
        self.assertEqual(result.get("outcome"), "promoted")
        mock_promote.assert_called_once()

    @patch("hdb_resale_mlops.mlflow_registry.reject_candidate")
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version", return_value=None)
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=_mock_explainer_result("Test report"),
    )
    def test_graph_pauses_and_rejects(
        self, _mock_explainer, _mock_champ, mock_reject
    ):
        """Full graph run: pause at human_review, resume with rejection."""
        from hdb_resale_mlops.promotion_workflow import build_promotion_graph
        from langgraph.types import Command

        graph = build_promotion_graph()
        config = {"configurable": {"thread_id": "test-reject-1"}}

        state = _build_initial_state(_load_scenario("promote_no_champion"))
        graph.invoke(state, config=config)

        result = graph.invoke(Command(resume="reject"), config=config)
        self.assertEqual(result.get("outcome"), "rejected")
        mock_reject.assert_called_once()

    @patch("hdb_resale_mlops.mlflow_registry.reject_candidate")
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version")
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=_mock_explainer_result("Reject report"),
    )
    def test_reject_path_auto_executes_without_human_review(
        self, _mock_explainer, mock_champ, mock_reject
    ):
        """REJECT verdicts should skip human_review and execute immediately."""
        from hdb_resale_mlops.promotion_workflow import build_promotion_graph

        graph = build_promotion_graph()
        config = {"configurable": {"thread_id": "test-auto-reject-1"}}
        mock_champ.return_value = {
            "version": "2",
            "run_id": "def456",
            "metrics": {"rmse": 155_000, "mae": 125_000},
            "segment_metrics": {},
        }

        state = _build_initial_state(_load_scenario("reject_high_rmse"))
        result = graph.invoke(state, config=config)

        self.assertEqual(result.get("outcome"), "rejected")
        self.assertEqual(result["policy_verdict"].decision, PolicyDecision.REJECT)
        mock_reject.assert_called_once()


class TestWorkflowRunnerAutoReject(unittest.TestCase):
    """Test the notebook convenience runner for the auto-reject path."""

    @patch("builtins.input", return_value="")
    @patch("hdb_resale_mlops.mlflow_registry.reject_candidate")
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version")
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=_mock_explainer_result("Reject report"),
    )
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="reviewer-3")
    def test_runner_keeps_default_auto_reject(
        self, _mock_reviewer, _mock_explainer, mock_champ, mock_reject, _mock_input
    ):
        from hdb_resale_mlops.promotion_workflow import run_promotion_workflow

        mock_champ.return_value = {
            "version": "2",
            "run_id": "def456",
            "metrics": {"rmse": 155_000, "mae": 125_000},
            "segment_metrics": {},
        }
        scenario = _load_scenario("reject_high_rmse")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_promotion_workflow(
                model_name=scenario["model_name"],
                model_version=scenario["model_version"],
                candidate_metrics=scenario["candidate_metrics"],
                thread_id="test-auto-reject-runner",
                review_dir=tmp_dir,
            )

        self.assertEqual(result["outcome"], "rejected")
        self.assertEqual(result["human_decision"], "auto_reject")
        mock_reject.assert_called_once()
        self.assertEqual(
            mock_reject.call_args.kwargs["decision_metadata"]["decision_source"],
            "policy_auto_reject",
        )
        self.assertNotIn("decision_reviewer", mock_reject.call_args.kwargs["decision_metadata"])

    @patch("builtins.input", return_value="approve")
    @patch("hdb_resale_mlops.mlflow_registry.promote_to_champion")
    @patch("hdb_resale_mlops.mlflow_registry.reject_candidate")
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version")
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=_mock_explainer_result("Reject report"),
    )
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="reviewer-4")
    def test_runner_can_override_auto_reject(
        self, _mock_reviewer, _mock_explainer, mock_champ, mock_reject, mock_promote, _mock_input
    ):
        from hdb_resale_mlops.promotion_workflow import run_promotion_workflow

        mock_champ.return_value = {
            "version": "2",
            "run_id": "def456",
            "metrics": {"rmse": 155_000, "mae": 125_000},
            "segment_metrics": {},
        }
        scenario = _load_scenario("reject_high_rmse")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_promotion_workflow(
                model_name=scenario["model_name"],
                model_version=scenario["model_version"],
                candidate_metrics=scenario["candidate_metrics"],
                thread_id="test-auto-reject-override",
                review_dir=tmp_dir,
            )

        self.assertEqual(result["outcome"], "promoted")
        self.assertEqual(result["human_decision"], "approve")
        mock_reject.assert_called_once()
        mock_promote.assert_called_once()
        self.assertEqual(
            mock_promote.call_args.args,
            (scenario["model_name"], scenario["model_version"]),
        )
        self.assertEqual(
            mock_promote.call_args.kwargs["decision_metadata"]["decision_source"],
            "human_override_after_policy_reject",
        )
        self.assertEqual(
            mock_promote.call_args.kwargs["decision_metadata"]["decision_reviewer"],
            "reviewer-4",
        )
        self.assertTrue(
            mock_promote.call_args.kwargs["decision_metadata"]["rejection_overridden"]
        )

    @patch("builtins.input", return_value="approve")
    @patch("hdb_resale_mlops.mlflow_registry.promote_to_champion")
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version", return_value=None)
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=_mock_explainer_result("Promote report"),
    )
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="reviewer-5")
    def test_runner_resumes_promote_path_after_interrupt(
        self, _mock_reviewer, _mock_explainer, _mock_champ, mock_promote, _mock_input
    ):
        from hdb_resale_mlops.promotion_workflow import run_promotion_workflow

        scenario = _load_scenario("promote_no_champion")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_promotion_workflow(
                model_name=scenario["model_name"],
                model_version=scenario["model_version"],
                candidate_metrics=scenario["candidate_metrics"],
                thread_id="test-promote-runner",
                review_dir=tmp_dir,
            )

        self.assertEqual(result["outcome"], "promoted")
        self.assertEqual(result["human_decision"], "approve")
        mock_promote.assert_called_once()
        self.assertEqual(
            mock_promote.call_args.args,
            (scenario["model_name"], scenario["model_version"]),
        )
        self.assertEqual(
            mock_promote.call_args.kwargs["decision_metadata"]["decision_reviewer"],
            "reviewer-5",
        )


class TestPersistedReviewFlow(unittest.TestCase):
    @patch("hdb_resale_mlops.mlflow_registry.promote_to_champion")
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version", return_value=None)
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=_mock_explainer_result("Persisted review report"),
    )
    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=True)
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="reviewer-persisted")
    def test_pending_review_can_be_resumed_from_disk(
        self,
        _mock_reviewer,
        _mock_log_review,
        _mock_explainer,
        _mock_champ,
        mock_promote,
    ):
        from hdb_resale_mlops.promotion_workflow import (
            load_promotion_review,
            resume_promotion_review,
            start_promotion_review,
        )

        scenario = _load_scenario("promote_no_champion")
        with tempfile.TemporaryDirectory() as tmp_dir:
            review = start_promotion_review(
                model_name=scenario["model_name"],
                model_version=scenario["model_version"],
                candidate_metrics=scenario["candidate_metrics"],
                thread_id="persisted-review",
                review_dir=tmp_dir,
            )

            self.assertEqual(review["status"], "pending_review")
            self.assertTrue(pathlib.Path(review["review_path"]).exists())
            self.assertIn("report_structured", review)
            self.assertIn("agent_trace", review)

            loaded = load_promotion_review(review["review_id"], review_dir=tmp_dir)
            self.assertEqual(loaded["status"], "pending_review")

            completed = resume_promotion_review(
                review["review_id"],
                "approve",
                review_dir=tmp_dir,
            )

            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["outcome"], "promoted")
            self.assertEqual(completed["human_decision"], "approve")
            mock_promote.assert_called_once()

    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version")
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=_mock_explainer_result("Serialized tabular report"),
    )
    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=True)
    def test_start_review_accepts_dataframe_inputs(
        self,
        _mock_log_review,
        _mock_explainer,
        mock_champ,
    ):
        from hdb_resale_mlops.promotion_workflow import load_promotion_review, start_promotion_review

        mock_champ.return_value = {
            "version": "2",
            "run_id": "champ-002",
            "metrics": {"rmse": 155_000, "mae": 125_000},
            "segment_metrics": {
                "town": pd.DataFrame(
                    {
                        "segment": ["ANG MO KIO"],
                        "rmse": [148_000],
                        "mae": [118_000],
                        "count": [50],
                    }
                ),
                "flat_type": pd.DataFrame(
                    {
                        "segment": ["4 ROOM"],
                        "rmse": [149_000],
                        "mae": [119_000],
                        "count": [100],
                    }
                ),
            },
        }

        candidate_segments = {
            "town": pd.DataFrame(
                {
                    "segment": ["ANG MO KIO"],
                    "rmse": [145_000],
                    "mae": [115_000],
                    "count": [50],
                }
            ),
            "flat_type": pd.DataFrame(
                {
                    "segment": ["4 ROOM"],
                    "rmse": [146_000],
                    "mae": [116_000],
                    "count": [100],
                }
            ),
        }
        train_df = pd.DataFrame(
            {
                "town": ["ANG MO KIO", "BEDOK", "QUEENSTOWN"],
                "flat_type": ["4 ROOM", "5 ROOM", "3 ROOM"],
                "flat_model": ["Model A", "Model B", "Model A"],
                "storey_range": ["01 TO 03", "04 TO 06", "07 TO 09"],
                "floor_area_sqm": [90.0, 110.0, 75.0],
                "flat_age_years": [25.0, 18.0, 30.0],
                "remaining_lease_years": [74.0, 81.0, 69.0],
                "storey_midpoint": [2.0, 5.0, 8.0],
            }
        )
        test_df = pd.DataFrame(
            {
                "town": ["ANG MO KIO", "BEDOK", "QUEENSTOWN"],
                "flat_type": ["4 ROOM", "5 ROOM", "3 ROOM"],
                "flat_model": ["Model A", "Model B", "Model A"],
                "storey_range": ["01 TO 03", "04 TO 06", "07 TO 09"],
                "floor_area_sqm": [91.0, 109.0, 76.0],
                "flat_age_years": [24.0, 19.0, 29.0],
                "remaining_lease_years": [75.0, 80.0, 70.0],
                "storey_midpoint": [2.0, 5.0, 8.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            review = start_promotion_review(
                model_name="test-model",
                model_version="3",
                candidate_metrics={"rmse": 150_000, "mae": 120_000},
                candidate_segment_metrics=candidate_segments,
                train_df=train_df,
                test_df=test_df,
                thread_id="dataframe-review",
                review_dir=tmp_dir,
            )

            self.assertEqual(review["status"], "pending_review")
            self.assertTrue(pathlib.Path(review["review_path"]).exists())
            self.assertEqual(review["policy_verdict"]["decision"], "PROMOTE")

            loaded = load_promotion_review(review["review_id"], review_dir=tmp_dir)
            self.assertEqual(loaded["status"], "pending_review")
            self.assertEqual(loaded["model_name"], "test-model")

    @patch("hdb_resale_mlops.mlflow_registry.promote_to_champion")
    @patch("hdb_resale_mlops.mlflow_registry.reject_candidate")
    @patch("hdb_resale_mlops.mlflow_registry.get_champion_version")
    @patch(
        "hdb_resale_mlops.promotion_workflow.run_explainer_agent_detailed",
        return_value=_mock_explainer_result("Persisted reject report"),
    )
    @patch("hdb_resale_mlops.mlflow_registry.log_promotion_review_artifacts", return_value=True)
    @patch("hdb_resale_mlops.promotion_workflow._resolve_reviewer_identity", return_value="reviewer-override")
    def test_auto_reject_record_can_be_overridden_later(
        self,
        _mock_reviewer,
        _mock_log_review,
        _mock_explainer,
        mock_champ,
        mock_reject,
        mock_promote,
    ):
        from hdb_resale_mlops.promotion_workflow import (
            resume_promotion_review,
            start_promotion_review,
        )

        mock_champ.return_value = {
            "version": "2",
            "run_id": "def456",
            "metrics": {"rmse": 155_000, "mae": 125_000},
            "segment_metrics": {},
        }
        scenario = _load_scenario("reject_high_rmse")

        with tempfile.TemporaryDirectory() as tmp_dir:
            review = start_promotion_review(
                model_name=scenario["model_name"],
                model_version=scenario["model_version"],
                candidate_metrics=scenario["candidate_metrics"],
                thread_id="persisted-auto-reject",
                review_dir=tmp_dir,
            )

            self.assertEqual(review["status"], "auto_rejected")
            mock_reject.assert_called_once()

            completed = resume_promotion_review(
                review["review_id"],
                "approve",
                review_dir=tmp_dir,
            )

            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["outcome"], "promoted")
            self.assertEqual(completed["human_decision"], "approve")
            mock_promote.assert_called_once()


if __name__ == "__main__":
    unittest.main()
