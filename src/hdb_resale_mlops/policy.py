"""Rule-based policy engine for model promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hdb_resale_mlops.comparison import ComparisonResult
from hdb_resale_mlops.drift import DriftReport


class PolicyDecision(str, Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(frozen=True)
class PolicyVerdict:
    decision: PolicyDecision
    reasons: list[str] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


DEFAULT_THRESHOLDS: dict[str, Any] = {
    "max_test_rmse": 200_000,
    "max_test_mae": 170_000,
    "max_rmse_regression_pct": 0.10,
    "max_segment_rmse_regression_pct": 0.20,
    "drift_blocks_promotion": True,
}


def evaluate_policy(
    candidate_metrics: dict[str, float],
    comparison: ComparisonResult,
    drift_report: DriftReport | None = None,
    thresholds: dict[str, Any] | None = None,
    evidence_errors: list[str] | None = None,
) -> PolicyVerdict:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons: list[str] = []
    passed: list[str] = []
    failed: list[str] = []
    has_reject = False
    has_manual_review = False

    # --- Absolute threshold checks ---
    test_rmse = candidate_metrics.get("rmse", 0.0)
    max_rmse = t["max_test_rmse"]
    if test_rmse > max_rmse:
        failed.append("absolute_rmse")
        reasons.append(
            f"Test RMSE {test_rmse:,.0f} exceeds maximum threshold {max_rmse:,.0f}"
        )
        has_reject = True
    else:
        passed.append("absolute_rmse")

    test_mae = candidate_metrics.get("mae", 0.0)
    max_mae = t["max_test_mae"]
    if test_mae > max_mae:
        failed.append("absolute_mae")
        reasons.append(
            f"Test MAE {test_mae:,.0f} exceeds maximum threshold {max_mae:,.0f}"
        )
        has_reject = True
    else:
        passed.append("absolute_mae")

    evidence_issues = [issue for issue in (evidence_errors or []) if issue]
    if evidence_issues:
        failed.append("promotion_evidence_unavailable")
        reasons.extend(evidence_issues)
        has_reject = True

    # --- Champion comparison checks ---
    if comparison.has_champion:
        rmse_delta_pct = comparison.metric_deltas.get("rmse_delta_pct", 0.0)
        max_regression = t["max_rmse_regression_pct"]
        if rmse_delta_pct > max_regression:
            failed.append("champion_rmse_regression")
            reasons.append(
                f"Candidate RMSE is {rmse_delta_pct:.1%} worse than champion "
                f"(threshold: {max_regression:.0%})"
            )
            has_reject = True
        else:
            passed.append("champion_rmse_regression")

        # Segment regression checks
        max_segment_regression = t["max_segment_rmse_regression_pct"]
        worst_segments: list[str] = []
        for sd in comparison.segment_deltas:
            if sd.rmse_delta_pct > max_segment_regression:
                worst_segments.append(
                    f"{sd.segment_column}={sd.segment_value} ({sd.rmse_delta_pct:+.1%})"
                )
        if worst_segments:
            failed.append("segment_rmse_regression")
            reasons.append(
                f"Segment RMSE regressions exceed {max_segment_regression:.0%}: "
                + ", ".join(worst_segments[:5])
            )
            has_manual_review = True
        else:
            passed.append("segment_rmse_regression")
    elif not evidence_issues:
        passed.append("champion_comparison_skipped")
        reasons.append("No existing champion — only absolute thresholds applied")

    # --- Drift checks ---
    if drift_report is not None:
        if drift_report.overall_drift_detected and t["drift_blocks_promotion"]:
            drifted_cols = [
                r.column for r in drift_report.column_results if r.is_drifted
            ]
            failed.append("drift_check")
            reasons.append(f"Data drift detected in: {', '.join(drifted_cols)}")
            has_manual_review = True
        else:
            passed.append("drift_check")
    else:
        passed.append("drift_check_skipped")

    # --- Final decision ---
    if has_reject:
        decision = PolicyDecision.REJECT
    elif has_manual_review:
        decision = PolicyDecision.MANUAL_REVIEW
    else:
        decision = PolicyDecision.PROMOTE

    return PolicyVerdict(
        decision=decision,
        reasons=reasons,
        checks_passed=passed,
        checks_failed=failed,
    )
