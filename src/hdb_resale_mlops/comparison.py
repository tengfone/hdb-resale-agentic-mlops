"""
Champion vs candidate model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hdb_resale_mlops.tabular_state import iter_tabular_rows


@dataclass(frozen=True)
class SegmentDelta:
    segment_column: str
    segment_value: str
    candidate_rmse: float
    champion_rmse: float
    rmse_delta: float
    rmse_delta_pct: float


@dataclass(frozen=True)
class ComparisonResult:
    has_champion: bool
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None = None
    metric_deltas: dict[str, float] = field(default_factory=dict)
    segment_deltas: list[SegmentDelta] = field(default_factory=list)


def _build_segment_lookup(
    segment_metrics: Any, segment_column: str
) -> dict[str, float]:
    """Build a {segment_value: rmse} lookup from DataFrame or serialized rows."""
    lookup: dict[str, float] = {}
    if segment_metrics is None:
        return lookup
    frame = segment_metrics.get(segment_column)
    if frame is None:
        return lookup
    for row in iter_tabular_rows(frame):
        if "segment" not in row or "rmse" not in row:
            continue
        lookup[str(row["segment"])] = float(row["rmse"])
    return lookup


def compare_models(
    candidate_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    candidate_segment_metrics: dict[str, Any] | None = None,
    champion_segment_metrics: dict[str, Any] | None = None,
    segment_columns: tuple[str, ...] = ("town", "flat_type"),
) -> ComparisonResult:
    if champion_metrics is None:
        return ComparisonResult(
            has_champion=False,
            candidate_metrics=candidate_metrics,
        )

    metric_deltas: dict[str, float] = {}
    for key in ("rmse", "mae"):
        candidate_val = candidate_metrics.get(key, 0.0)
        champion_val = champion_metrics.get(key, 0.0)
        metric_deltas[f"{key}_delta"] = candidate_val - champion_val
        if champion_val > 0:
            metric_deltas[f"{key}_delta_pct"] = (
                candidate_val - champion_val
            ) / champion_val
        else:
            metric_deltas[f"{key}_delta_pct"] = 0.0

    segment_deltas: list[SegmentDelta] = []
    if candidate_segment_metrics and champion_segment_metrics:
        for col in segment_columns:
            candidate_lookup = _build_segment_lookup(candidate_segment_metrics, col)
            champion_lookup = _build_segment_lookup(champion_segment_metrics, col)
            for segment_value, candidate_rmse in candidate_lookup.items():
                champion_rmse = champion_lookup.get(segment_value)
                if champion_rmse is not None and champion_rmse > 0:
                    delta = candidate_rmse - champion_rmse
                    delta_pct = delta / champion_rmse
                    segment_deltas.append(
                        SegmentDelta(
                            segment_column=col,
                            segment_value=segment_value,
                            candidate_rmse=candidate_rmse,
                            champion_rmse=champion_rmse,
                            rmse_delta=delta,
                            rmse_delta_pct=delta_pct,
                        )
                    )

    return ComparisonResult(
        has_champion=True,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        metric_deltas=metric_deltas,
        segment_deltas=segment_deltas,
    )
