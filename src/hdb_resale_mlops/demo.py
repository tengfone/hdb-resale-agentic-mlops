from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hdb_resale_mlops.comparison import ComparisonResult, SegmentDelta
from hdb_resale_mlops.config import ProjectPaths
from hdb_resale_mlops.drift import ColumnDriftResult, DriftReport
from hdb_resale_mlops.explainer import (
    _generate_template_report,
    _make_tools,
    _parse_report_text,
    run_explainer_agent_detailed,
)
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict


@dataclass(frozen=True)
class DemoScenario:
    name: str
    description: str
    candidate_metrics: dict[str, float]
    champion_info: dict[str, Any] | None
    comparison: ComparisonResult
    drift_report: DriftReport | None
    policy_verdict: PolicyVerdict
    model_name: str
    model_version: str
    expected: dict[str, Any]


def _fixture_dir() -> Path:
    paths = ProjectPaths.discover()
    return paths.repo_root / "tests" / "fixtures" / "eval_scenarios"


def list_demo_scenarios() -> list[str]:
    return sorted(path.stem for path in _fixture_dir().glob("*.json"))


def _load_raw_scenario(name: str) -> dict[str, Any]:
    path = _fixture_dir() / f"{name}.json"
    if not path.exists():
        available = ", ".join(list_demo_scenarios())
        raise FileNotFoundError(f"Unknown scenario {name!r}. Available: {available}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_demo_scenario(name: str) -> DemoScenario:
    raw = _load_raw_scenario(name)

    comparison_raw = raw["comparison"]
    comparison = ComparisonResult(
        has_champion=comparison_raw["has_champion"],
        candidate_metrics=comparison_raw["candidate_metrics"],
        champion_metrics=comparison_raw.get("champion_metrics"),
        metric_deltas=comparison_raw.get("metric_deltas", {}),
        segment_deltas=[
            SegmentDelta(**row) for row in comparison_raw.get("segment_deltas", [])
        ],
    )

    drift_report = None
    drift_raw = raw.get("drift_report")
    if drift_raw:
        drift_report = DriftReport(
            column_results=[
                ColumnDriftResult(**row) for row in drift_raw["column_results"]
            ],
            overall_drift_detected=drift_raw["overall_drift_detected"],
        )

    verdict_raw = raw["policy_verdict"]
    policy_verdict = PolicyVerdict(
        decision=PolicyDecision(verdict_raw["decision"]),
        reasons=list(verdict_raw.get("reasons", [])),
        checks_passed=list(verdict_raw.get("checks_passed", [])),
        checks_failed=list(verdict_raw.get("checks_failed", [])),
    )

    return DemoScenario(
        name=name,
        description=raw.get("description", name),
        candidate_metrics=raw["candidate_metrics"],
        champion_info=raw.get("champion_info"),
        comparison=comparison,
        drift_report=drift_report,
        policy_verdict=policy_verdict,
        model_name=raw["model_name"],
        model_version=raw["model_version"],
        expected=raw.get("expected", {}),
    )


def _maybe_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _collect_tool_outputs(scenario: DemoScenario) -> dict[str, Any]:
    tools = {
        tool.name: tool
        for tool in _make_tools(
            candidate_metrics=scenario.candidate_metrics,
            champion_info=scenario.champion_info,
            comparison=scenario.comparison,
            drift_report=scenario.drift_report,
            policy_verdict=scenario.policy_verdict,
            model_name=scenario.model_name,
        )
    }

    outputs: dict[str, Any] = {}
    for tool_name in scenario.expected.get("minimum_tools_called", []):
        tool = tools[tool_name]
        if tool_name == "compare_segment_performance":
            outputs[tool_name] = {
                "town": _maybe_json(tool.invoke({"segment_type": "town"})),
                "flat_type": _maybe_json(tool.invoke({"segment_type": "flat_type"})),
            }
        else:
            outputs[tool_name] = _maybe_json(tool.invoke({}))
    return outputs


def build_demo_review(
    scenario_name: str,
    *,
    mode: str = "template",
) -> dict[str, Any]:
    scenario = load_demo_scenario(scenario_name)

    if mode == "template":
        fallback_note = "Fixture demo mode. Set --mode auto or --mode agent to exercise the explainer path."
        report_text = _generate_template_report(
            candidate_metrics=scenario.candidate_metrics,
            champion_info=scenario.champion_info,
            comparison=scenario.comparison,
            drift_report=scenario.drift_report,
            policy_verdict=scenario.policy_verdict,
            model_version=scenario.model_version,
            fallback_note=fallback_note,
        )
        structured_report = _parse_report_text(report_text)
        run_metadata = {
            "demo_mode": mode,
            "used_fallback": True,
            "fallback_note": fallback_note,
            "report_format": "template_markdown",
        }
        report_payload = {
            "report_text": report_text,
            "structured_report": structured_report.to_dict(),
            "agent_trace": [],
            "run_metadata": run_metadata,
            "used_fallback": True,
            "fallback_note": fallback_note,
        }
    elif mode in ("auto", "agent"):
        result = run_explainer_agent_detailed(
            candidate_metrics=scenario.candidate_metrics,
            champion_info=scenario.champion_info,
            comparison=scenario.comparison,
            drift_report=scenario.drift_report,
            policy_verdict=scenario.policy_verdict,
            model_name=scenario.model_name,
            model_version=scenario.model_version,
        )
        report_payload = result.to_dict()
        report_payload["run_metadata"] = {
            **report_payload.get("run_metadata", {}),
            "demo_mode": mode,
        }
    else:
        raise ValueError("mode must be one of: template, auto, agent")

    return {
        "scenario_name": scenario.name,
        "description": scenario.description,
        "model_name": scenario.model_name,
        "model_version": scenario.model_version,
        "candidate_metrics": scenario.candidate_metrics,
        "champion_info": scenario.champion_info,
        "policy_verdict": {
            "decision": scenario.policy_verdict.decision.value,
            "reasons": list(scenario.policy_verdict.reasons),
            "checks_passed": list(scenario.policy_verdict.checks_passed),
            "checks_failed": list(scenario.policy_verdict.checks_failed),
        },
        "expected": scenario.expected,
        "tool_outputs": _collect_tool_outputs(scenario),
        **report_payload,
    }


def write_demo_review(review: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    report_path = destination / "report.md"
    review_path = destination / "review.json"
    tools_path = destination / "tool_outputs.json"

    report_path.write_text(str(review["report_text"]), encoding="utf-8")
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    tools_path.write_text(
        json.dumps(review["tool_outputs"], indent=2), encoding="utf-8"
    )

    return {
        "report_path": str(report_path),
        "review_path": str(review_path),
        "tool_outputs_path": str(tools_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a promotion review from fixture scenarios without retraining."
    )
    parser.add_argument(
        "--list", action="store_true", help="List available demo scenarios."
    )
    parser.add_argument(
        "--scenario", default="promote_no_champion", help="Scenario fixture to replay."
    )
    parser.add_argument(
        "--mode",
        choices=("template", "auto", "agent"),
        default="template",
        help="template: deterministic report; auto: explainer with fallback; agent: same as auto but intended for LLM-backed runs.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Console output format.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory where report.md, review.json, and tool_outputs.json will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list:
        for name in list_demo_scenarios():
            print(name)
        return

    review = build_demo_review(args.scenario, mode=args.mode)
    written_paths: dict[str, str] | None = None
    if args.output_dir:
        written_paths = write_demo_review(review, args.output_dir)

    if args.format == "json":
        payload = dict(review)
        if written_paths:
            payload["written_paths"] = written_paths
        print(json.dumps(payload, indent=2))
        return

    print(f"Scenario: {review['scenario_name']}")
    print(review["description"])
    print("")
    print(review["report_text"])
    print("")
    print("Minimum tool outputs:")
    print(json.dumps(review["tool_outputs"], indent=2))
    if written_paths:
        print("")
        print("Wrote demo artifacts:")
        print(json.dumps(written_paths, indent=2))


if __name__ == "__main__":
    main()
