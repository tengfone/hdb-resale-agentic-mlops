"""
LangGraph state machine for the model-promotion review flow.

This file is the bridge between the notebook and the rest of the package.
After a notebook has trained and registered a candidate model in MLflow, it
hands the candidate metrics and optional train/test frames into this module.

The workflow then runs in three layers:

1. Deterministic layer
   - load the current champion from MLflow
   - compare candidate vs champion
   - run drift checks on train vs test
   - apply the rule-based promotion policy
2. Agentic layer
   - ask the explainer LLM to investigate the evidence and write a review report
3. Human layer
   - pause for approval when promotion is still possible
   - write the final decision back to MLflow

Important boundary:
- The LLM explains the evidence.
- The policy decides the routing.
- The human makes the final promotion call unless policy already rejected it.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
from typing import Any, Mapping, TypedDict

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta, compare_models
from hdb_resale_mlops.config import RuntimeConfig
from hdb_resale_mlops.drift import ColumnDriftResult, DriftReport, run_drift_checks
from hdb_resale_mlops.explainer import run_explainer_agent_detailed
from hdb_resale_mlops.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict, evaluate_policy
from hdb_resale_mlops.tabular_state import coerce_dataframe, serialize_for_state


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class PromotionState(TypedDict, total=False):
    # Inputs
    model_name: str
    model_version: str
    review_id: str
    candidate_metrics: dict[str, float]
    candidate_segment_metrics: dict[str, Any]
    train_df: Any
    test_df: Any

    # Intermediate results (populated by nodes)
    champion_info: dict[str, Any] | None
    comparison: ComparisonResult
    drift_report: dict[str, Any] | None
    policy_verdict: PolicyVerdict
    evidence_errors: list[str]

    # Agent output
    report: str
    report_structured: dict[str, Any]
    agent_trace: list[dict[str, Any]]
    agent_run_metadata: dict[str, Any]
    judge_evaluation: dict[str, Any] | None

    # Human decision
    human_decision: str  # "approve", "reject", or "auto_reject"

    # Final outcome
    outcome: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def gather_evidence(state: PromotionState) -> dict:
    """Layer 1a: Load champion info and compare with candidate."""
    from hdb_resale_mlops.mlflow_registry import (
        MlflowRegistryError,
        get_champion_version,
    )

    model_name = state["model_name"]
    evidence_errors: list[str] = []
    try:
        champion_info = get_champion_version(model_name)
    except MlflowRegistryError as exc:
        champion_info = None
        evidence_errors.append(
            f"Promotion blocked because champion evidence could not be loaded from MLflow: {exc}"
        )

    # The workflow compares against the current MLflow champion if one exists.
    champion_metrics = champion_info["metrics"] if champion_info else None
    champion_segment_metrics = (
        champion_info.get("segment_metrics") if champion_info else None
    )
    if champion_info and not state.get("candidate_segment_metrics"):
        evidence_errors.append(
            "Promotion blocked because candidate segment metrics are unavailable for champion comparison."
        )

    comparison = compare_models(
        candidate_metrics=state["candidate_metrics"],
        champion_metrics=champion_metrics,
        candidate_segment_metrics=state.get("candidate_segment_metrics"),
        champion_segment_metrics=champion_segment_metrics,
    )

    return {
        "champion_info": serialize_for_state(champion_info),
        "comparison": comparison,
        "evidence_errors": evidence_errors,
    }


def check_drift(state: PromotionState) -> dict:
    """Layer 1b: Run drift detection on train vs test distributions."""
    train_df = coerce_dataframe(state.get("train_df"))
    test_df = coerce_dataframe(state.get("test_df"))

    if train_df is None or test_df is None:
        return {"drift_report": None}

    drift_report = run_drift_checks(
        train_df=train_df,
        test_df=test_df,
        categorical_columns=CATEGORICAL_FEATURES,
        numeric_columns=NUMERIC_FEATURES,
    )
    return {"drift_report": _drift_report_to_dict(drift_report)}


def apply_policy(state: PromotionState) -> dict:
    """Layer 1c: Apply deterministic policy rules to produce a verdict."""
    # This is the actual decision engine for routing. The explainer only
    # receives the verdict after these checks have already been computed.
    verdict = evaluate_policy(
        candidate_metrics=state["candidate_metrics"],
        comparison=state["comparison"],
        drift_report=_drift_report_from_dict(state.get("drift_report")),
        evidence_errors=list(state.get("evidence_errors", [])),
    )
    return {"policy_verdict": verdict}


def generate_report(state: PromotionState) -> dict:
    """Layer 2: Run the ReAct explainer agent to produce a promotion report."""
    from hdb_resale_mlops.mlflow_registry import (
        PromotionReviewPersistenceError,
        log_promotion_review_artifacts,
    )

    # The agent gets structured evidence, not raw notebook state.
    explainer_result = run_explainer_agent_detailed(
        candidate_metrics=state["candidate_metrics"],
        champion_info=state.get("champion_info"),
        comparison=state["comparison"],
        drift_report=_drift_report_from_dict(state.get("drift_report")),
        policy_verdict=state["policy_verdict"],
        model_name=state["model_name"],
        model_version=state["model_version"],
    )
    # The optional judge scores report quality only. It never changes
    # promotion routing or final model-registry actions.
    judge_evaluation = _run_optional_judge_evaluation(
        state,
        explainer_result.report_text,
    )
    agent_run_metadata = dict(explainer_result.run_metadata)
    agent_run_metadata["judge_status"] = judge_evaluation["status"]
    if judge_evaluation["status"] == "scored":
        agent_run_metadata["judge_average_score"] = judge_evaluation["scores"][
            "average"
        ]
    if judge_evaluation["status"] == "failed":
        agent_run_metadata["judge_error_type"] = judge_evaluation.get("error_type")

    review_logged = log_promotion_review_artifacts(
        model_name=state["model_name"],
        model_version=state["model_version"],
        review_payload={
            "review_id": state["review_id"],
            "model_name": state["model_name"],
            "model_version": state["model_version"],
            "policy_verdict": _policy_verdict_to_dict(state["policy_verdict"]),
            "report_text": explainer_result.report_text,
            "report_structured": explainer_result.structured_report.to_dict(),
            "agent_trace": explainer_result.agent_trace,
            "run_metadata": agent_run_metadata,
            "judge_evaluation": judge_evaluation,
            "used_fallback": explainer_result.used_fallback,
            "fallback_note": explainer_result.fallback_note,
        },
    )
    if _mlflow_tracking_is_configured() and not review_logged:
        raise PromotionReviewPersistenceError(
            f"Promotion review artifacts for {state['model_name']} v{state['model_version']} "
            "could not be persisted to MLflow."
        )

    return {
        "report": explainer_result.report_text,
        "report_structured": explainer_result.structured_report.to_dict(),
        "agent_trace": explainer_result.agent_trace,
        "agent_run_metadata": agent_run_metadata,
        "judge_evaluation": judge_evaluation,
    }


def route_after_report(state: PromotionState) -> str:
    """Route REJECT decisions directly to execution; others require review."""
    if state["policy_verdict"].decision == PolicyDecision.REJECT:
        return "execute_decision"
    return "human_review"


def human_review(state: PromotionState) -> dict:
    """Layer 3: Pause for human approval using LangGraph interrupt."""
    from langgraph.types import interrupt

    verdict = state["policy_verdict"]

    decision = interrupt(
        {
            "report": state["report"],
            "policy_verdict": verdict.decision.value,
            "policy_reasons": verdict.reasons,
            "action_required": (
                "Review the report above and respond with 'approve' or 'reject'."
            ),
        }
    )

    return {"human_decision": str(decision).strip().lower()}


def execute_decision(state: PromotionState) -> dict:
    """Execute the final promotion or rejection in MLflow registry."""
    from hdb_resale_mlops.mlflow_registry import promote_to_champion, reject_candidate

    model_name = state["model_name"]
    model_version = state["model_version"]
    human_decision = state.get("human_decision", "")
    verdict = state["policy_verdict"]

    if human_decision == "approve":
        promote_to_champion(
            model_name,
            model_version,
            decision_metadata=_build_decision_metadata(
                verdict,
                decision_source="human_review",
                reviewer=_resolve_reviewer_identity(),
                rejection_overridden=verdict.decision == PolicyDecision.REJECT,
            ),
        )
        return {"outcome": "promoted"}
    else:
        reasons = verdict.reasons if verdict.reasons else ["Rejected by human reviewer"]
        decision_source = (
            "policy_auto_reject"
            if verdict.decision == PolicyDecision.REJECT and not human_decision
            else "human_review"
        )
        reviewer = (
            None
            if decision_source == "policy_auto_reject"
            else _resolve_reviewer_identity()
        )
        reject_candidate(
            model_name,
            model_version,
            reasons,
            decision_metadata=_build_decision_metadata(
                verdict,
                decision_source=decision_source,
                reviewer=reviewer,
            ),
        )
        return {"outcome": "rejected"}


def _resolve_reviewer_identity() -> str:
    """Best-effort reviewer identity for notebook-driven human decisions."""
    for value in (
        os.environ.get("MODEL_REVIEWER"),
        os.environ.get("USER"),
        os.environ.get("USERNAME"),
    ):
        if value and value.strip():
            return value.strip()
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _build_decision_metadata(
    policy_verdict: PolicyVerdict,
    decision_source: str,
    reviewer: str | None = None,
    rejection_overridden: bool = False,
) -> dict[str, Any]:
    """Build MLflow model-version tags that explain the final decision."""
    metadata: dict[str, Any] = {
        "decision_source": decision_source,
        "decision_timestamp": datetime.now(timezone.utc).isoformat(),
        "policy_verdict": policy_verdict.decision.value,
        "rejection_overridden": rejection_overridden,
    }
    if policy_verdict.reasons:
        metadata["policy_reasons"] = "; ".join(policy_verdict.reasons)
    if reviewer:
        metadata["decision_reviewer"] = reviewer
    return metadata


def _policy_verdict_to_dict(policy_verdict: PolicyVerdict) -> dict[str, Any]:
    if isinstance(policy_verdict, dict):
        return {
            "decision": policy_verdict["decision"],
            "reasons": list(policy_verdict.get("reasons", [])),
            "checks_passed": list(policy_verdict.get("checks_passed", [])),
            "checks_failed": list(policy_verdict.get("checks_failed", [])),
        }
    return {
        "decision": policy_verdict.decision.value,
        "reasons": list(policy_verdict.reasons),
        "checks_passed": list(policy_verdict.checks_passed),
        "checks_failed": list(policy_verdict.checks_failed),
    }


def _policy_verdict_from_dict(payload: dict[str, Any]) -> PolicyVerdict:
    return PolicyVerdict(
        decision=PolicyDecision(payload["decision"]),
        reasons=list(payload.get("reasons", [])),
        checks_passed=list(payload.get("checks_passed", [])),
        checks_failed=list(payload.get("checks_failed", [])),
    )


def _drift_report_to_dict(
    drift_report: DriftReport | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if drift_report is None:
        return None
    if isinstance(drift_report, dict):
        return {
            "overall_drift_detected": bool(
                drift_report.get("overall_drift_detected", False)
            ),
            "column_results": [
                serialize_for_state(dict(item))
                for item in drift_report.get("column_results", [])
            ],
        }
    return {
        "overall_drift_detected": drift_report.overall_drift_detected,
        "column_results": [
            {
                "column": result.column,
                "drift_type": result.drift_type,
                "statistic": float(result.statistic),
                "threshold": float(result.threshold),
                "p_value": None if result.p_value is None else float(result.p_value),
                "is_drifted": bool(result.is_drifted),
            }
            for result in drift_report.column_results
        ],
    }


def _drift_report_from_dict(
    payload: DriftReport | dict[str, Any] | None,
) -> DriftReport | None:
    if payload is None or isinstance(payload, DriftReport):
        return payload
    return DriftReport(
        overall_drift_detected=bool(payload.get("overall_drift_detected", False)),
        column_results=[
            ColumnDriftResult(
                column=str(item["column"]),
                drift_type=str(item["drift_type"]),
                statistic=float(item["statistic"]),
                threshold=float(item["threshold"]),
                p_value=None if item.get("p_value") is None else float(item["p_value"]),
                is_drifted=bool(item.get("is_drifted", False)),
            )
            for item in payload.get("column_results", [])
        ],
    )


def _default_review_dir() -> Path:
    from hdb_resale_mlops.config import ProjectPaths

    paths = ProjectPaths.discover()
    review_dir = paths.artifacts_dir / "promotion_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    return review_dir


def _mlflow_tracking_is_configured() -> bool:
    return bool(RuntimeConfig.from_env().mlflow_tracking_uri)


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _compact_error_detail(exc: Exception, limit: int = 240) -> str:
    text = " ".join(str(exc).split())
    if not text:
        return type(exc).__name__
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _comparison_to_dict(
    comparison: ComparisonResult | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(comparison, dict):
        return comparison
    return serialize_for_state(asdict(comparison))


def _comparison_from_dict(payload: ComparisonResult | dict[str, Any]) -> ComparisonResult:
    if isinstance(payload, ComparisonResult):
        return payload
    return ComparisonResult(
        has_champion=bool(payload.get("has_champion", False)),
        candidate_metrics=dict(payload.get("candidate_metrics", {})),
        champion_metrics=(
            None
            if payload.get("champion_metrics") is None
            else dict(payload.get("champion_metrics", {}))
        ),
        metric_deltas=dict(payload.get("metric_deltas", {})),
        segment_deltas=[
            SegmentDelta(
                segment_column=str(item["segment_column"]),
                segment_value=str(item["segment_value"]),
                candidate_rmse=float(item["candidate_rmse"]),
                champion_rmse=float(item["champion_rmse"]),
                rmse_delta=float(item["rmse_delta"]),
                rmse_delta_pct=float(item["rmse_delta_pct"]),
            )
            for item in payload.get("segment_deltas", [])
        ],
    )


def _build_judge_scenario(state: PromotionState) -> dict[str, Any]:
    return {
        "candidate_metrics": serialize_for_state(state["candidate_metrics"]),
        "champion_info": serialize_for_state(state.get("champion_info")),
        "policy_verdict": _policy_verdict_to_dict(state["policy_verdict"]),
        "comparison": _comparison_to_dict(state["comparison"]),
        "drift_report": serialize_for_state(state.get("drift_report")),
    }


def _run_optional_judge_evaluation(
    state: PromotionState,
    report_text: str,
) -> dict[str, Any]:
    """Score the report only when explicitly enabled.

    The judge is advisory. It adds visibility in MLflow but never changes the
    deterministic promotion decision.
    """
    if not _env_flag_enabled("ENABLE_JUDGE_EVAL"):
        return {
            "enabled": False,
            "status": "disabled",
            "note": "Set ENABLE_JUDGE_EVAL=true to log advisory report-quality scores to MLflow.",
        }

    from hdb_resale_mlops.eval_judge import _resolve_judge_model, evaluate_report

    resolved_model = _resolve_judge_model(os.environ.get("OPENAI_JUDGE_MODEL"))
    try:
        score = evaluate_report(
            report=report_text,
            scenario=_build_judge_scenario(state),
        )
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "model": resolved_model,
            "error_type": type(exc).__name__,
            "error_detail": _compact_error_detail(exc),
        }

    return {
        "enabled": True,
        "status": "scored",
        "model": resolved_model,
        "scores": score.to_dict(),
    }


def _resolve_review_dir(review_dir: str | os.PathLike[str] | None) -> Path:
    if review_dir is None:
        return _default_review_dir()
    resolved = Path(review_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _sanitize_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.lower())
    cleaned = cleaned.strip("-")
    return cleaned or "review"


def _build_review_id(model_name: str, model_version: str, thread_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "-".join(
        [
            timestamp,
            _sanitize_token(model_name),
            f"v{_sanitize_token(model_version)}",
            _sanitize_token(thread_id),
        ]
    )


def _review_path(
    review_id: str, review_dir: str | os.PathLike[str] | None = None
) -> Path:
    return _resolve_review_dir(review_dir) / f"{review_id}.json"


def _write_review_record(
    record: dict[str, Any], review_dir: str | os.PathLike[str] | None = None
) -> Path:
    path = _review_path(record["review_id"], review_dir)
    payload = dict(record)
    payload["review_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_promotion_review(
    review_id: str,
    review_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load a persisted review packet from disk."""
    path = _review_path(review_id, review_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def _print_report(
    report: str,
    policy_verdict: PolicyVerdict | str | None,
    reasons: list[str] | None = None,
) -> None:
    """Render a consistent notebook-friendly review payload."""
    verdict_label = "N/A"
    verdict_reasons = reasons or []
    if isinstance(policy_verdict, PolicyVerdict):
        verdict_label = policy_verdict.decision.value
        verdict_reasons = policy_verdict.reasons
    elif isinstance(policy_verdict, str):
        verdict_label = policy_verdict

    print("\n" + "=" * 70)
    print("MODEL PROMOTION REPORT")
    print("=" * 70)
    print(report)
    print("\n" + "-" * 70)
    print(f"Policy verdict: {verdict_label}")
    if verdict_reasons:
        print("Reasons:")
        for reason in verdict_reasons:
            print(f"  • {reason}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_promotion_graph():
    """Build the LangGraph state machine for the promotion workflow.

    Returns a compiled graph with an in-memory checkpointer for interrupt support.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    builder = StateGraph(PromotionState)

    # Add nodes
    builder.add_node("gather_evidence", gather_evidence)
    builder.add_node("check_drift", check_drift)
    builder.add_node("apply_policy", apply_policy)
    builder.add_node("generate_report", generate_report)
    builder.add_node("human_review", human_review)
    builder.add_node("execute_decision", execute_decision)

    # Layer 1: deterministic evidence gathering → policy
    builder.set_entry_point("gather_evidence")
    builder.add_edge("gather_evidence", "check_drift")
    builder.add_edge("check_drift", "apply_policy")

    # Layer 1 → Layer 2: always generate a report
    builder.add_edge("apply_policy", "generate_report")

    # Layer 2 → Layer 3: only non-REJECT verdicts require approval before execution
    builder.add_conditional_edges(
        "generate_report",
        route_after_report,
        {
            "human_review": "human_review",
            "execute_decision": "execute_decision",
        },
    )

    # Layer 3 → execute
    builder.add_edge("human_review", "execute_decision")
    builder.add_edge("execute_decision", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def _is_missing_langgraph_error(exc: Exception) -> bool:
    if isinstance(exc, ModuleNotFoundError) and exc.name == "langgraph":
        return True
    return "langgraph" in str(exc)


def _run_promotion_until_review_sequential(
    initial_state: PromotionState,
    *,
    workflow_runtime: str,
    workflow_note: str,
) -> PromotionState:
    """Sequential notebook runner that avoids graph-state serialization."""
    state: PromotionState = dict(initial_state)
    for step in (gather_evidence, check_drift, apply_policy, generate_report):
        state.update(step(state))

    agent_run_metadata = dict(state.get("agent_run_metadata") or {})
    agent_run_metadata.setdefault("workflow_runtime", workflow_runtime)
    agent_run_metadata.setdefault("workflow_note", workflow_note)
    state["agent_run_metadata"] = agent_run_metadata

    if route_after_report(state) == "execute_decision":
        state["human_decision"] = "auto_reject"
        state.update(execute_decision(state))

    return state


# ---------------------------------------------------------------------------
# Review persistence and convenience runners
# ---------------------------------------------------------------------------


def start_promotion_review(
    model_name: str,
    model_version: str,
    candidate_metrics: dict[str, float],
    candidate_segment_metrics: dict[str, Any] | None = None,
    train_df: Any = None,
    test_df: Any = None,
    thread_id: str = "promotion-1",
    review_dir: str | os.PathLike[str] | None = None,
    use_langgraph: bool = False,
) -> dict[str, Any]:
    """Run the workflow until human input is needed and persist a review packet.

    This is the entry point the notebooks usually call. It runs the graph far
    enough to either:
    - auto-reject immediately, or
    - stop with a saved review packet that a human can inspect and resume.
    """
    from hdb_resale_mlops.mlflow_registry import (
        PromotionReviewPersistenceError,
        persist_promotion_review_record,
    )

    review_id = _build_review_id(model_name, model_version, thread_id)
    initial_state_raw: PromotionState = {
        "model_name": model_name,
        "model_version": model_version,
        "review_id": review_id,
        "candidate_metrics": candidate_metrics,
        "candidate_segment_metrics": candidate_segment_metrics or {},
        "train_df": train_df,
        "test_df": test_df,
    }
    config = {"configurable": {"thread_id": thread_id}}

    if not use_langgraph:
        graph = None
        result = _run_promotion_until_review_sequential(
            initial_state_raw,
            workflow_runtime="sequential_helper",
            workflow_note="Used notebook-side sequential runner to avoid large graph checkpoints.",
        )
    else:
        initial_state: PromotionState = {
            "model_name": model_name,
            "model_version": model_version,
            "review_id": review_id,
            "candidate_metrics": serialize_for_state(candidate_metrics),
            "candidate_segment_metrics": serialize_for_state(
                candidate_segment_metrics or {}
            ),
            "train_df": serialize_for_state(train_df),
            "test_df": serialize_for_state(test_df),
        }
        try:
            graph = build_promotion_graph()
        except Exception as exc:
            if not _is_missing_langgraph_error(exc):
                raise
            graph = None
            result = _run_promotion_until_review_sequential(
                initial_state_raw,
                workflow_runtime="sequential_fallback",
                workflow_note="langgraph not installed; used notebook-side sequential fallback.",
            )
        else:
            result = graph.invoke(initial_state, config=config)
    now = datetime.now(timezone.utc).isoformat()

    if "outcome" in result:
        # Policy REJECT routes straight through to execution, but we still save
        # a review packet so a later human can inspect or override it.
        record = {
            "review_id": review_id,
            "status": "auto_rejected",
            "created_at": now,
            "completed_at": now,
            "thread_id": thread_id,
            "model_name": model_name,
            "model_version": model_version,
            "candidate_metrics": candidate_metrics,
            "policy_verdict": _policy_verdict_to_dict(result["policy_verdict"]),
            "report": result.get("report", ""),
            "report_structured": result.get("report_structured", {}),
            "agent_trace": result.get("agent_trace", []),
            "agent_run_metadata": result.get("agent_run_metadata", {}),
            "judge_evaluation": result.get("judge_evaluation"),
            "human_decision": result.get("human_decision", "auto_reject"),
            "outcome": result["outcome"],
            "action_required": (
                "Policy already rejected this candidate. Optionally confirm or override with "
                "resume_promotion_review(review_id, 'approve'|'reject')."
            ),
        }
        path = _write_review_record(record, review_dir=review_dir)
        record["review_path"] = str(path)
        if _mlflow_tracking_is_configured():
            persisted = persist_promotion_review_record(
                model_name=model_name,
                model_version=model_version,
                review_id=review_id,
                review_record=record,
            )
            if not persisted:
                raise PromotionReviewPersistenceError(
                    f"Promotion review record {review_id!r} could not be mirrored to MLflow."
                )
        return record

    # PROMOTE and MANUAL_REVIEW stop at the interrupt point and become
    # resumable review packets on disk.
    if graph is None:
        values = result
    else:
        snapshot = graph.get_state(config)
        values = snapshot.values
    record = {
        "review_id": review_id,
        "status": "pending_review",
        "created_at": now,
        "thread_id": thread_id,
        "model_name": model_name,
        "model_version": model_version,
        "candidate_metrics": candidate_metrics,
        "policy_verdict": _policy_verdict_to_dict(values["policy_verdict"]),
        "report": values.get("report", ""),
        "report_structured": values.get("report_structured", {}),
        "agent_trace": values.get("agent_trace", []),
        "agent_run_metadata": values.get("agent_run_metadata", {}),
        "judge_evaluation": values.get("judge_evaluation"),
        "human_decision": "",
        "outcome": "pending_review",
        "action_required": "Resume this review with approve or reject.",
    }
    path = _write_review_record(record, review_dir=review_dir)
    record["review_path"] = str(path)
    if _mlflow_tracking_is_configured():
        persisted = persist_promotion_review_record(
            model_name=model_name,
            model_version=model_version,
            review_id=review_id,
            review_record=record,
        )
        if not persisted:
            raise PromotionReviewPersistenceError(
                f"Promotion review record {review_id!r} could not be mirrored to MLflow."
            )
    return record


def start_promotion_review_from_handoff(
    review_handoff: Mapping[str, Any],
    *,
    thread_id: str = "promotion-1",
    review_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Start notebook-side explainer + human review from frozen pipeline evidence."""
    from hdb_resale_mlops.mlflow_registry import (
        PromotionReviewPersistenceError,
        persist_promotion_review_record,
    )

    registration = dict(review_handoff.get("registration") or {})
    model_name = str(registration["model_name"])
    model_version = str(registration["model_version"])
    review_id = _build_review_id(model_name, model_version, thread_id)
    now = datetime.now(timezone.utc).isoformat()

    state: PromotionState = {
        "model_name": model_name,
        "model_version": model_version,
        "review_id": review_id,
        "candidate_metrics": dict(review_handoff.get("candidate_metrics", {})),
        "champion_info": review_handoff.get("champion_info"),
        "comparison": _comparison_from_dict(review_handoff.get("comparison", {})),
        "drift_report": review_handoff.get("drift_report"),
        "policy_verdict": _policy_verdict_from_dict(review_handoff["policy_verdict"]),
    }
    state.update(generate_report(state))

    if review_handoff.get("status") == "auto_rejected":
        record = {
            "review_id": review_id,
            "status": "auto_rejected",
            "created_at": now,
            "completed_at": now,
            "thread_id": thread_id,
            "model_name": model_name,
            "model_version": model_version,
            "candidate_metrics": state["candidate_metrics"],
            "policy_verdict": _policy_verdict_to_dict(state["policy_verdict"]),
            "report": state.get("report", ""),
            "report_structured": state.get("report_structured", {}),
            "agent_trace": state.get("agent_trace", []),
            "agent_run_metadata": state.get("agent_run_metadata", {}),
            "judge_evaluation": state.get("judge_evaluation"),
            "human_decision": "auto_reject",
            "outcome": "rejected",
            "action_required": (
                "Policy already rejected this candidate in the pipeline. Optionally confirm or "
                "override with resume_promotion_review(review_id, 'approve'|'reject')."
            ),
        }
    else:
        record = {
            "review_id": review_id,
            "status": "pending_review",
            "created_at": now,
            "thread_id": thread_id,
            "model_name": model_name,
            "model_version": model_version,
            "candidate_metrics": state["candidate_metrics"],
            "policy_verdict": _policy_verdict_to_dict(state["policy_verdict"]),
            "report": state.get("report", ""),
            "report_structured": state.get("report_structured", {}),
            "agent_trace": state.get("agent_trace", []),
            "agent_run_metadata": state.get("agent_run_metadata", {}),
            "judge_evaluation": state.get("judge_evaluation"),
            "human_decision": "",
            "outcome": "pending_review",
            "action_required": "Resume this review with approve or reject.",
        }

    path = _write_review_record(record, review_dir=review_dir)
    record["review_path"] = str(path)
    if _mlflow_tracking_is_configured():
        persisted = persist_promotion_review_record(
            model_name=model_name,
            model_version=model_version,
            review_id=review_id,
            review_record=record,
        )
        if not persisted:
            raise PromotionReviewPersistenceError(
                f"Promotion review record {review_id!r} could not be mirrored to MLflow."
            )
    return record


def resume_promotion_review(
    review_id: str,
    decision: str,
    review_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Complete a persisted review packet with a human approve/reject decision."""
    from hdb_resale_mlops.mlflow_registry import (
        PromotionReviewPersistenceError,
        persist_promotion_review_record,
        promote_to_champion,
        reject_candidate,
    )

    normalized = decision.strip().lower()
    if normalized not in ("approve", "reject"):
        raise ValueError("Decision must be either 'approve' or 'reject'.")

    record = load_promotion_review(review_id, review_dir=review_dir)
    verdict = _policy_verdict_from_dict(record["policy_verdict"])
    now = datetime.now(timezone.utc).isoformat()

    if record["status"] == "pending_review":
        updates = execute_decision(
            {
                "model_name": record["model_name"],
                "model_version": record["model_version"],
                "policy_verdict": verdict,
                "human_decision": normalized,
            }
        )
    elif record["status"] == "auto_rejected":
        reasons = verdict.reasons if verdict.reasons else ["Rejected by human reviewer"]
        if normalized == "approve":
            promote_to_champion(
                record["model_name"],
                record["model_version"],
                decision_metadata=_build_decision_metadata(
                    verdict,
                    decision_source="human_override_after_policy_reject",
                    reviewer=_resolve_reviewer_identity(),
                    rejection_overridden=True,
                ),
            )
            updates = {"outcome": "promoted"}
        else:
            reject_candidate(
                record["model_name"],
                record["model_version"],
                reasons,
                decision_metadata=_build_decision_metadata(
                    verdict,
                    decision_source="human_confirmed_reject",
                    reviewer=_resolve_reviewer_identity(),
                ),
            )
            updates = {"outcome": "rejected"}
    else:
        raise ValueError(f"Review {review_id!r} is already completed.")

    record.update(
        {
            "status": "completed",
            "human_decision": normalized,
            "outcome": updates["outcome"],
            "completed_at": now,
        }
    )
    path = _write_review_record(record, review_dir=review_dir)
    record["review_path"] = str(path)
    if _mlflow_tracking_is_configured():
        persisted = persist_promotion_review_record(
            model_name=record["model_name"],
            model_version=record["model_version"],
            review_id=record["review_id"],
            review_record=record,
        )
        if not persisted:
            raise PromotionReviewPersistenceError(
                f"Promotion review record {record['review_id']!r} could not be mirrored to MLflow."
            )
    return record


def run_promotion_workflow(
    model_name: str,
    model_version: str,
    candidate_metrics: dict[str, float],
    candidate_segment_metrics: dict[str, Any] | None = None,
    train_df: Any = None,
    test_df: Any = None,
    thread_id: str = "promotion-1",
    review_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Notebook-friendly wrapper around start/resume review helpers."""
    review = start_promotion_review(
        model_name=model_name,
        model_version=model_version,
        candidate_metrics=candidate_metrics,
        candidate_segment_metrics=candidate_segment_metrics,
        train_df=train_df,
        test_df=test_df,
        thread_id=thread_id,
        review_dir=review_dir,
    )

    policy_verdict = _policy_verdict_from_dict(review["policy_verdict"])
    review_response = dict(review)
    review_response["policy_verdict"] = policy_verdict
    _print_report(review["report"], policy_verdict, policy_verdict.reasons)
    print(f"Review packet saved to: {review['review_path']}")

    if review["status"] == "auto_rejected":
        decision = (
            input(
                "\nModel was auto-rejected by policy. "
                "Type 'approve' to override and promote anyway, "
                "or press Enter to keep it rejected [approve/keep rejected]: "
            )
            .strip()
            .lower()
        )
        if decision not in ("", "approve", "reject"):
            print(f"Invalid input '{decision}', keeping the automatic rejection.")
            decision = ""
        if decision == "":
            return review_response
    else:
        decision = (
            input("\nDo you approve this model for promotion? [approve/reject]: ")
            .strip()
            .lower()
        )
        if decision not in ("approve", "reject"):
            print(f"Invalid input '{decision}', defaulting to 'reject'.")
            decision = "reject"

    completed = resume_promotion_review(
        review["review_id"],
        decision,
        review_dir=review_dir,
    )
    completed_response = dict(completed)
    completed_response["policy_verdict"] = policy_verdict
    return completed_response
