"""
LLM-backed explainer for promotion reviews.

This file does not train or score the resale-price model. Its job is narrower:
given already-computed evidence from the promotion workflow, ask an LLM to
inspect that evidence and write a structured review report for a human.

The main ideas:
- The workflow computes the facts first: metrics, champion comparison, drift,
  policy verdict, and training history.
- This module wraps those facts as tools for a ReAct agent.
- The agent can optionally do web research for HDB market context.
- If OpenAI access is unavailable, the module falls back to a deterministic
  template report so the workflow still completes.

So when reading this file, think "report writer" rather than "model trainer".
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

from hdb_resale_mlops.comparison import ComparisonResult
from hdb_resale_mlops.drift import DriftReport
from hdb_resale_mlops.env import maestro_proxy_env
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict


@dataclass(frozen=True)
class PromotionReport:
    summary: str
    evidence: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    market_context: str = ""
    recommendation: str = ""
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplainerRunResult:
    report_text: str
    structured_report: PromotionReport
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    run_metadata: dict[str, Any] = field(default_factory=dict)
    used_fallback: bool = False
    fallback_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_text": self.report_text,
            "structured_report": self.structured_report.to_dict(),
            "agent_trace": self.agent_trace,
            "run_metadata": self.run_metadata,
            "used_fallback": self.used_fallback,
            "fallback_note": self.fallback_note,
        }


_SECTION_ALIASES = {
    "summary": "summary",
    "evidence": "evidence",
    "risk flags": "risk_flags",
    "market context": "market_context",
    "recommendation": "recommendation",
    "sources": "citations",
    "citations": "citations",
}

_PROMPT_VERSION = "v1"
_REPORT_SCHEMA_VERSION = "v1"
_NO_OP_TRACE_ID = "MLFLOW_NO_OP_SPAN_TRACE_ID"
_VALID_MARKET_RESEARCH_PROVIDERS = {"auto", "tavily", "openai", "both", "none"}
_CHAMPION_WORD_PATTERN = re.compile(r"champion", re.IGNORECASE)


def _market_research_provider_from_env() -> str:
    """Return the configured market research backend selection."""
    provider = (os.environ.get("MARKET_RESEARCH_PROVIDER") or "auto").strip().lower()
    if provider not in _VALID_MARKET_RESEARCH_PROVIDERS:
        return "auto"
    return provider


def _selected_market_research_providers() -> list[str]:
    """Return the enabled market research backends in execution order."""
    provider = _market_research_provider_from_env()
    has_tavily = bool(os.environ.get("TAVILY_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if provider == "none":
        return []
    if provider == "tavily":
        return ["tavily"] if has_tavily else []
    if provider == "openai":
        return ["openai"] if has_openai else []
    if provider == "both":
        selected: list[str] = []
        if has_tavily:
            selected.append("tavily")
        if has_openai:
            selected.append("openai")
        return selected

    if has_tavily:
        return ["tavily"]
    if has_openai:
        return ["openai"]
    return []


def _openai_web_search_model_from_env() -> str:
    """Return the model to use for OpenAI native web search calls."""
    return os.environ.get("OPENAI_WEB_SEARCH_MODEL") or os.environ.get(
        "OPENAI_MODEL",
        "gpt-5-nano",
    )


def _dedupe_result_urls(results: list[dict[str, Any]]) -> list[str]:
    """Collect unique URLs from result dictionaries while preserving order."""
    urls: list[str] = []
    seen: set[str] = set()
    for result in results:
        url = str(result.get("url", "") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _describe_openai_web_search_failure(
    exc: Exception,
    *,
    base_url_configured: bool,
) -> str:
    """Return an actionable OpenAI web-search error description."""
    exc_type = type(exc).__name__
    exc_text = _sanitize_exception_text(exc)

    if exc_type == "AttributeError" and "model_dump" in exc_text:
        if base_url_configured:
            detail = (
                "Configured OPENAI_BASE_URL returned a non-standard Responses payload. "
                "This endpoint is likely not compatible with OpenAI native web search."
            )
        else:
            detail = "The Responses API returned a payload the SDK could not parse."
    elif exc_type == "AuthenticationError" or "invalid_api_key" in exc_text:
        detail = (
            "The configured endpoint rejected the OpenAI credentials."
            if base_url_configured
            else "The official OpenAI endpoint rejected the API key."
        )
    else:
        detail = exc_text or exc_type

    return f"OpenAI native web search failed ({exc_type}). {detail}"


def _run_tavily_market_research(query: str) -> dict[str, Any]:
    """Execute Tavily-backed market research and normalize the response shape."""
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return {
            "provider": "tavily",
            "query": query,
            "summary": "",
            "results": [],
            "sources": [],
            "error": "TAVILY_API_KEY is not set.",
        }

    try:
        from tavily import TavilyClient
    except ImportError:
        return {
            "provider": "tavily",
            "query": query,
            "summary": "",
            "results": [],
            "sources": [],
            "error": "tavily-python is not installed.",
        }

    client = TavilyClient(api_key=tavily_api_key)
    with maestro_proxy_env():
        response = client.search(query=query, max_results=5)
    results = [
        {
            "provider": "tavily",
            "title": r.get("title", ""),
            "content": (r.get("content", "") or "")[:500],
            "url": r.get("url", ""),
        }
        for r in response.get("results", [])
    ]
    return {
        "provider": "tavily",
        "query": query,
        "summary": "Recent market-search results from Tavily.",
        "results": results,
        "sources": _dedupe_result_urls(results),
    }


def _run_openai_market_research(query: str) -> dict[str, Any]:
    """Execute OpenAI native web search and normalize the response shape."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "provider": "openai",
            "query": query,
            "summary": "",
            "results": [],
            "sources": [],
            "error": "OPENAI_API_KEY is not set.",
        }

    try:
        from openai import OpenAI
    except ImportError:
        return {
            "provider": "openai",
            "query": query,
            "summary": "",
            "results": [],
            "sources": [],
            "error": "openai is not installed.",
        }

    base_url = _openai_base_url_from_env()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        client = OpenAI(**client_kwargs)
        response = client.responses.create(
            model=_openai_web_search_model_from_env(),
            input=query,
            include=["web_search_call.action.sources"],
            tool_choice={"type": "web_search"},
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "medium",
                    "user_location": {
                        "type": "approximate",
                        "city": "Singapore",
                        "region": "Singapore",
                        "country": "SG",
                        "timezone": "Asia/Singapore",
                    },
                }
            ],
        )
    except Exception as exc:
        return {
            "provider": "openai",
            "query": query,
            "summary": "",
            "results": [],
            "sources": [],
            "error": _describe_openai_web_search_failure(
                exc,
                base_url_configured=bool(base_url),
            ),
        }

    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "web_search_call":
            action = getattr(item, "action", None)
            for source in getattr(action, "sources", None) or []:
                url = str(getattr(source, "url", "") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(
                    {
                        "provider": "openai",
                        "title": "",
                        "content": "",
                        "url": url,
                    }
                )
            continue

        if item_type != "message":
            continue

        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                url = str(getattr(annotation, "url", "") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(
                    {
                        "provider": "openai",
                        "title": str(getattr(annotation, "title", "") or ""),
                        "content": "",
                        "url": url,
                    }
                )

    summary = str(getattr(response, "output_text", "") or "").strip()
    return {
        "provider": "openai",
        "query": query,
        "summary": summary,
        "results": results,
        "sources": _dedupe_result_urls(results),
        "model": _openai_web_search_model_from_env(),
    }


def _run_market_research(query: str, providers: list[str]) -> dict[str, Any]:
    """Run the configured market research backends and merge the payload."""
    payloads: list[dict[str, Any]] = []
    for provider in providers:
        if provider == "tavily":
            payloads.append(_run_tavily_market_research(query))
        elif provider == "openai":
            payloads.append(_run_openai_market_research(query))

    if len(payloads) == 1:
        return payloads[0]

    merged_results: list[dict[str, Any]] = []
    merged_sources: list[str] = []
    merged_errors: list[str] = []
    summaries: list[str] = []
    models: list[str] = []

    for payload in payloads:
        merged_results.extend(payload.get("results", []))
        for url in payload.get("sources", []):
            if url not in merged_sources:
                merged_sources.append(url)
        error = payload.get("error")
        if error:
            merged_errors.append(f"{payload['provider']}: {error}")
        summary = str(payload.get("summary", "") or "").strip()
        if summary:
            summaries.append(f"{payload['provider']}: {summary}")
        model = payload.get("model")
        if model:
            models.append(str(model))

    merged_payload: dict[str, Any] = {
        "provider": "both",
        "providers_used": [payload["provider"] for payload in payloads],
        "query": query,
        "summary": "\n\n".join(summaries),
        "results": merged_results,
        "sources": merged_sources,
    }
    if merged_errors:
        merged_payload["errors"] = merged_errors
    if models:
        merged_payload["models"] = models
    return merged_payload


# ---------------------------------------------------------------------------
# Tool factories — each tool closes over the pre-computed workflow state
# so the agent can query evidence on demand.
# ---------------------------------------------------------------------------


def _make_tools(
    candidate_metrics: dict[str, float],
    champion_info: dict[str, Any] | None,
    comparison: ComparisonResult,
    drift_report: DriftReport | None,
    policy_verdict: PolicyVerdict,
    model_name: str,
) -> list:
    """Build LangChain tool objects that the explainer agent can call.

    Each tool closes over already-computed workflow state. The agent is not
    reaching back into the notebook. It is querying this frozen evidence bundle.
    """
    from langchain_core.tools import tool

    @tool
    def query_candidate_metrics() -> str:
        """Return the candidate model's overall test metrics (RMSE, MAE)."""
        return json.dumps(candidate_metrics, indent=2)

    @tool
    def query_champion_metrics() -> str:
        """Return the current champion model's test metrics, or state that no champion exists."""
        if champion_info is None:
            return "No champion model exists yet. This is the first candidate."
        return json.dumps(champion_info.get("metrics", {}), indent=2)

    @tool
    def compare_segment_performance(segment_type: str = "town") -> str:
        """Return per-segment RMSE deltas between candidate and champion.

        Args:
            segment_type: Either 'town' or 'flat_type'.
        """
        if not comparison.has_champion:
            return (
                "No champion to compare against — segment comparison is not available."
            )
        relevant = sorted(
            (
                sd
                for sd in comparison.segment_deltas
                if sd.segment_column == segment_type
            ),
            key=lambda sd: sd.rmse_delta_pct,
            reverse=True,
        )
        if not relevant:
            return f"No segment deltas available for '{segment_type}'."
        return json.dumps(
            [
                {
                    "segment": sd.segment_value,
                    "candidate_rmse": round(sd.candidate_rmse, 2),
                    "champion_rmse": round(sd.champion_rmse, 2),
                    "delta_pct": f"{sd.rmse_delta_pct:+.1%}",
                }
                for sd in relevant
            ],
            indent=2,
        )

    @tool
    def check_drift_report() -> str:
        """Return the data drift analysis results (PSI for categorical, KS test for numeric features)."""
        if drift_report is None:
            return "Drift detection was not run."
        results = []
        for cr in drift_report.column_results:
            entry = {
                "column": cr.column,
                "type": cr.drift_type,
                "statistic": round(cr.statistic, 4),
                "threshold": cr.threshold,
                "is_drifted": cr.is_drifted,
            }
            if cr.p_value is not None:
                entry["p_value"] = round(cr.p_value, 4)
            results.append(entry)
        return json.dumps(
            {
                "overall_drift_detected": drift_report.overall_drift_detected,
                "columns": results,
            },
            indent=2,
        )

    @tool
    def get_policy_verdict() -> str:
        """Return the deterministic policy engine's decision and reasoning."""
        return json.dumps(
            {
                "decision": policy_verdict.decision.value,
                "reasons": policy_verdict.reasons,
                "checks_passed": policy_verdict.checks_passed,
                "checks_failed": policy_verdict.checks_failed,
            },
            indent=2,
        )

    @tool
    def get_training_history() -> str:
        """Return metrics and past review outcomes from recent model versions."""
        from hdb_resale_mlops.mlflow_registry import (
            get_training_history as _get_history,
        )

        history = _get_history(model_name)
        if not history:
            return (
                "No training history available — this may be the first model version."
            )
        summary = []
        for entry in history:
            tags = entry.get("tags", {})
            summary.append(
                {
                    "version": entry["version"],
                    "test_rmse": entry["metrics"].get("test_rmse"),
                    "test_mae": entry["metrics"].get("test_mae"),
                    "promotion_status": tags.get("promotion_status"),
                    "policy_verdict": tags.get("policy_verdict"),
                    "decision_source": tags.get("decision_source"),
                    "rejection_reasons": tags.get("rejection_reasons"),
                }
            )
        return json.dumps(summary, indent=2)

    tools = [
        query_candidate_metrics,
        query_champion_metrics,
        compare_segment_performance,
        check_drift_report,
        get_policy_verdict,
        get_training_history,
    ]

    market_research_providers = _selected_market_research_providers()
    if market_research_providers:

        @tool
        def research_market_trends(query: str) -> str:
            """Search the web for recent HDB market news, policy changes, or property trends.

            Args:
                query: Search query about Singapore HDB resale market trends.
            """
            payload = _run_market_research(query, market_research_providers)
            return json.dumps(payload, indent=2)

        tools.append(research_market_trends)

    return tools


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are an MLOps model review analyst. Your job is to investigate a candidate
ML model for HDB resale price prediction and produce a thorough promotion report.

You have access to tools that let you query metrics, compare with the champion model,
check data drift, review training history, and research market trends.

IMPORTANT RULES:
1. You MUST call tools to gather evidence — do NOT guess or hallucinate metrics.
2. Autonomously decide which tools to call and in what order based on the situation.
3. If the policy verdict flagged specific issues, investigate those deeply.
4. If segments are underperforming, check for drift and market context.
5. Be concise but thorough. Cite specific numbers from tool outputs.
6. Explicitly mention the candidate's overall RMSE and MAE.
7. If a champion exists, explicitly mention the champion's overall RMSE and MAE.
8. Explicitly mention policy checks passed and checks failed when available.
9. Your report does NOT make the final promote/reject decision — that is for the human reviewer.
10. When you use external market research, include source URLs in citations.
11. If no champion exists, avoid the word "champion" in the final report; describe it as the first candidate or no incumbent baseline instead.

Respond with ONLY a JSON object using this schema:
{
  "summary": "<one paragraph>",
  "evidence": ["<fact 1>", "<fact 2>"],
  "risk_flags": ["<risk 1>", "<risk 2>"],
  "market_context": "<short paragraph or empty string>",
  "recommendation": "<assessment for the human reviewer>",
  "citations": ["<url>", "<url>"]
}

If you did not use external research, return "citations": [].
"""


def _openai_base_url_from_env() -> str | None:
    """Return an optional OpenAI-compatible base URL override from the environment."""
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")


def build_explainer_agent(
    candidate_metrics: dict[str, float],
    champion_info: dict[str, Any] | None,
    comparison: ComparisonResult,
    drift_report: DriftReport | None,
    policy_verdict: PolicyVerdict,
    model_name: str,
    model: str | None = None,
):
    """Build a ReAct agent with tools for model investigation.

    Returns a LangGraph CompiledGraph that can be invoked with a brief message.
    """
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    llm_kwargs = {
        "model": model or os.environ.get("OPENAI_MODEL", "gpt-5-nano"),
        "temperature": 0,
    }
    base_url = _openai_base_url_from_env()
    if base_url:
        llm_kwargs["base_url"] = base_url

    # OPENAI_MODEL controls the report-writing model, not the tabular
    # prediction model. This is the main "second model" in the repo.
    llm = ChatOpenAI(**llm_kwargs)

    tools = _make_tools(
        candidate_metrics=candidate_metrics,
        champion_info=champion_info,
        comparison=comparison,
        drift_report=drift_report,
        policy_verdict=policy_verdict,
        model_name=model_name,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=_SYSTEM_PROMPT,
    )

    return agent


def _message_text(content: Any) -> str:
    """Best-effort conversion of LangChain message content into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return ""


def _json_safe(value: Any) -> Any:
    """Best-effort conversion into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _section_key_for_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None

    candidates = [stripped]
    if stripped.startswith("#"):
        candidates.append(stripped.lstrip("#").strip())
    if stripped.endswith(":"):
        candidates.append(stripped[:-1].strip())

    for candidate in candidates:
        key = _SECTION_ALIASES.get(candidate.lower())
        if key:
            return key
    return None


def _parse_list_section(lines: list[str]) -> list[str]:
    items: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        items.append(stripped)
    return items


def _normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    return [str(value).strip()]


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    if "```json" in stripped:
        start = stripped.find("```json")
        end = stripped.rfind("```")
        if start != -1 and end > start:
            candidates.append(stripped[start + len("```json") : end].strip())
    elif stripped.startswith("```") and stripped.endswith("```"):
        candidates.append(stripped[3:-3].strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _report_from_payload(payload: dict[str, Any]) -> PromotionReport:
    return PromotionReport(
        summary=str(payload.get("summary", "")).strip(),
        evidence=_normalize_text_list(payload.get("evidence")),
        risk_flags=_normalize_text_list(payload.get("risk_flags")),
        market_context=str(payload.get("market_context", "")).strip(),
        recommendation=str(payload.get("recommendation", "")).strip(),
        citations=_normalize_text_list(payload.get("citations")),
    )


def _parse_report_text(report_text: str) -> PromotionReport:
    """Parse a markdown-style report into a structured PromotionReport."""
    sections = {
        "summary": [],
        "evidence": [],
        "risk_flags": [],
        "market_context": [],
        "recommendation": [],
        "citations": [],
    }
    intro_lines: list[str] = []
    current_section: str | None = None

    for line in report_text.splitlines():
        section_key = _section_key_for_line(line)
        if section_key is not None:
            current_section = section_key
            continue
        if current_section is None:
            intro_lines.append(line)
            continue
        sections[current_section].append(line)

    summary_lines = [line.strip() for line in sections["summary"] if line.strip()]
    if not summary_lines:
        summary_lines = [line.strip() for line in intro_lines if line.strip()]

    recommendation_lines = [
        line.strip() for line in sections["recommendation"] if line.strip()
    ]
    market_context = "\n".join(
        line.strip() for line in sections["market_context"] if line.strip()
    )

    return PromotionReport(
        summary=" ".join(summary_lines).strip() or report_text.strip(),
        evidence=_parse_list_section(sections["evidence"]),
        risk_flags=_parse_list_section(sections["risk_flags"]),
        market_context=market_context,
        recommendation=" ".join(recommendation_lines).strip(),
        citations=_parse_list_section(sections["citations"]),
    )


def _coerce_structured_report(report_text: str) -> PromotionReport:
    payload = _extract_json_payload(report_text)
    if payload is not None:
        report = _report_from_payload(payload)
        if (
            report.summary
            or report.evidence
            or report.risk_flags
            or report.recommendation
        ):
            return report
    return _parse_report_text(report_text)


def _render_markdown_report(report: PromotionReport) -> str:
    evidence = report.evidence or ["No evidence items captured."]
    risk_flags = report.risk_flags or ["No major risk flags identified."]
    lines = [
        "## Summary",
        report.summary or "No summary provided.",
        "",
        "## Evidence",
        *[f"- {item}" for item in evidence],
        "",
        "## Risk Flags",
        *[f"- {item}" for item in risk_flags],
        "",
        "## Market Context",
        report.market_context or "No external market context provided.",
        "",
        "## Recommendation",
        report.recommendation
        or "Use this report as input to the final human decision.",
    ]
    if report.citations:
        lines.extend(["", "## Sources", *[f"- {item}" for item in report.citations]])
    return "\n".join(lines)


def _append_unique_line(lines: list[str], line: str) -> None:
    normalized = line.strip().lower()
    if not normalized:
        return
    if any(existing.strip().lower() == normalized for existing in lines):
        return
    lines.append(line)


def _humanize_check_name(check_name: str) -> str:
    if check_name == "absolute_rmse":
        return "absolute RMSE threshold"
    if check_name == "absolute_mae":
        return "absolute MAE threshold"
    if check_name == "champion_rmse_regression":
        return "champion RMSE comparison"
    if check_name == "champion_comparison_skipped":
        return "incumbent baseline comparison skipped for first candidate"
    if check_name == "segment_rmse_regression":
        return "segment RMSE regression threshold"
    if check_name == "drift_check":
        return "drift check"
    if check_name == "drift_check_skipped":
        return "drift check skipped"
    if check_name == "promotion_evidence_unavailable":
        return "promotion evidence availability"
    return check_name.replace("_", " ")


def _format_check_names(check_names: list[str]) -> str:
    return ", ".join(_humanize_check_name(check_name) for check_name in check_names)


def _replace_champion_with_baseline(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        return "Baseline" if match.group(0)[0].isupper() else "baseline"

    return _CHAMPION_WORD_PATTERN.sub(replace, text)


def _normalize_first_candidate_language(report: PromotionReport) -> PromotionReport:
    """Remove `champion` wording for first-candidate scenarios."""
    return PromotionReport(
        summary=_replace_champion_with_baseline(report.summary),
        evidence=[
            _replace_champion_with_baseline(line) for line in report.evidence
        ],
        risk_flags=[
            _replace_champion_with_baseline(line) for line in report.risk_flags
        ],
        market_context=_replace_champion_with_baseline(report.market_context),
        recommendation=_replace_champion_with_baseline(report.recommendation),
        citations=report.citations,
    )


def _segment_performance_summary(
    segment_deltas: list[Any],
    *,
    limit: int = 5,
) -> tuple[str, str, list[Any]] | None:
    if not segment_deltas:
        return None

    regressions = [segment for segment in segment_deltas if segment.rmse_delta_pct > 0]
    if regressions:
        return (
            "Worst Segment Regressions",
            "Worst segment regressions",
            sorted(
                regressions,
                key=lambda segment: segment.rmse_delta_pct,
                reverse=True,
            )[:limit],
        )

    return (
        "Best Segment Improvements",
        "Best segment improvements",
        sorted(segment_deltas, key=lambda segment: segment.rmse_delta_pct)[:limit],
    )


def _enrich_structured_report(
    report: PromotionReport,
    *,
    candidate_metrics: dict[str, float],
    champion_info: dict[str, Any] | None,
    comparison: ComparisonResult,
    drift_report: DriftReport | None,
    policy_verdict: PolicyVerdict,
) -> PromotionReport:
    """Backfill core deterministic evidence so final reports stay review-ready."""
    evidence = list(report.evidence)
    risk_flags = list(report.risk_flags)

    _append_unique_line(
        evidence,
        f"Candidate test RMSE: {candidate_metrics.get('rmse', 0):,.0f}",
    )
    _append_unique_line(
        evidence,
        f"Candidate test MAE: {candidate_metrics.get('mae', 0):,.0f}",
    )

    if comparison.has_champion and champion_info:
        champ_metrics = champion_info.get("metrics", {})
        _append_unique_line(
            evidence,
            f"Champion test RMSE: {champ_metrics.get('rmse', 0):,.0f}",
        )
        _append_unique_line(
            evidence,
            f"Champion test MAE: {champ_metrics.get('mae', 0):,.0f}",
        )
        _append_unique_line(
            evidence,
            f"RMSE delta vs champion: {comparison.metric_deltas.get('rmse_delta', 0):+,.0f} "
            f"({comparison.metric_deltas.get('rmse_delta_pct', 0):+.1%})",
        )
        segment_summary = _segment_performance_summary(
            comparison.segment_deltas,
            limit=3,
        )
        if segment_summary is not None:
            _, evidence_prefix, segments = segment_summary
            _append_unique_line(
                evidence,
                f"{evidence_prefix}: "
                + "; ".join(
                    f"{segment.segment_column}={segment.segment_value} "
                    f"({segment.rmse_delta_pct:+.1%})"
                    for segment in segments
                ),
            )
    else:
        _append_unique_line(
            evidence,
            "Baseline comparison: first candidate; no incumbent baseline available.",
        )

    if drift_report is None:
        _append_unique_line(evidence, "Drift status: no drift report provided.")
    else:
        drifted = [
            column_result.column
            for column_result in drift_report.column_results
            if column_result.is_drifted
        ]
        if drifted:
            _append_unique_line(
                evidence,
                "Drift status: detected in " + ", ".join(drifted) + ".",
            )
        else:
            _append_unique_line(evidence, "Drift status: no significant drift detected.")

    if policy_verdict.checks_passed:
        _append_unique_line(
            evidence,
            "Checks passed: " + _format_check_names(policy_verdict.checks_passed),
        )
    if policy_verdict.checks_failed:
        _append_unique_line(
            risk_flags,
            "Checks failed: " + _format_check_names(policy_verdict.checks_failed),
        )
    for reason in policy_verdict.reasons:
        _append_unique_line(risk_flags, reason)
    if not risk_flags:
        risk_flags.append("No major risk flags identified.")

    enriched = PromotionReport(
        summary=report.summary,
        evidence=evidence,
        risk_flags=risk_flags,
        market_context=report.market_context,
        recommendation=report.recommendation,
        citations=report.citations,
    )
    if not comparison.has_champion:
        return _normalize_first_candidate_language(enriched)
    return enriched


def _extract_agent_trace(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract tool-call and tool-result events from agent messages."""
    trace: list[dict[str, Any]] = []
    tool_names_by_id: dict[str, str] = {}

    for index, message in enumerate(messages):
        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            call_id = None
            name = None
            args = None
            if isinstance(tool_call, dict):
                call_id = tool_call.get("id")
                name = tool_call.get("name")
                args = tool_call.get("args")
            else:
                call_id = getattr(tool_call, "id", None)
                name = getattr(tool_call, "name", None)
                args = getattr(tool_call, "args", None)
            if call_id and name:
                tool_names_by_id[str(call_id)] = str(name)
            trace.append(
                {
                    "event": "tool_call",
                    "message_index": index,
                    "tool_call_id": str(call_id) if call_id else None,
                    "tool_name": str(name) if name else None,
                    "tool_args": _json_safe(args),
                }
            )

        if getattr(message, "type", None) != "tool":
            continue

        tool_call_id = getattr(message, "tool_call_id", None)
        tool_name = getattr(message, "name", None)
        if tool_name is None and tool_call_id is not None:
            tool_name = tool_names_by_id.get(str(tool_call_id))
        trace.append(
            {
                "event": "tool_result",
                "message_index": index,
                "tool_call_id": str(tool_call_id) if tool_call_id else None,
                "tool_name": tool_name,
                "content": _message_text(getattr(message, "content", "")),
            }
        )

    return trace


def _extract_final_report(messages: list[Any], brief: str) -> str:
    """Return the final non-tool agent message from a LangGraph result."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "tool":
            continue
        if getattr(msg, "tool_calls", None):
            continue
        text = _message_text(getattr(msg, "content", ""))
        if text and text != brief:
            return text
    return ""


def _log_agent_trace_spans(mlflow: Any, agent_trace: list[dict[str, Any]]) -> None:
    """Mirror the serialized tool trace into compact MLflow spans."""
    if mlflow is None or not agent_trace:
        return

    for event in agent_trace:
        event_type = str(event.get("event") or "event")
        tool_name = str(event.get("tool_name") or "unknown_tool")
        attributes: dict[str, Any] = {
            "event": event_type,
            "tool_name": tool_name,
        }
        tool_call_id = event.get("tool_call_id")
        if tool_call_id:
            attributes["tool_call_id"] = str(tool_call_id)
        if event.get("message_index") is not None:
            attributes["message_index"] = int(event["message_index"])

        try:
            span_cm = mlflow.start_span(
                name=f"{event_type}:{tool_name}",
                span_type="TOOL",
                attributes=attributes,
            )
        except Exception:
            continue

        try:
            with span_cm as span:
                if span is None:
                    continue
                if event_type == "tool_call":
                    span.set_inputs({"tool_args": _json_safe(event.get("tool_args"))})
                elif event_type == "tool_result":
                    span.set_outputs(
                        {
                            "content_preview": _trace_preview(
                                _message_text(event.get("content", ""))
                            )
                        }
                    )
        except Exception:
            continue


def _build_run_metadata(
    *,
    duration_ms: int,
    llm_model: str,
    agent_trace: list[dict[str, Any]],
    used_fallback: bool,
    fallback_note: str | None,
    report_format: str,
    fallback_error_type: str | None = None,
    fallback_error_detail: str | None = None,
) -> dict[str, Any]:
    tool_calls = [event for event in agent_trace if event.get("event") == "tool_call"]
    tool_results = [
        event for event in agent_trace if event.get("event") == "tool_result"
    ]
    unique_tools = sorted(
        {str(event.get("tool_name")) for event in agent_trace if event.get("tool_name")}
    )
    return {
        "prompt_version": _PROMPT_VERSION,
        "report_schema_version": _REPORT_SCHEMA_VERSION,
        "llm_model": llm_model,
        "duration_ms": duration_ms,
        "tool_call_count": len(tool_calls),
        "tool_result_count": len(tool_results),
        "unique_tools_used": unique_tools,
        "used_fallback": used_fallback,
        "fallback_note": fallback_note,
        "fallback_error_type": fallback_error_type,
        "fallback_error_detail": fallback_error_detail,
        "report_format": report_format,
    }


def _sanitize_exception_text(exc: Exception) -> str:
    """Return a compact exception string without leaking long raw payloads."""
    text = " ".join(str(exc).split())
    if not text:
        return ""
    if len(text) > 240:
        return f"{text[:237]}..."
    return text


def _describe_agent_failure(
    exc: Exception,
    *,
    base_url_configured: bool,
) -> tuple[str, str]:
    """Classify agent failures into an actionable fallback note and short detail."""
    exc_type = type(exc).__name__
    exc_text = _sanitize_exception_text(exc)

    if exc_type == "AttributeError" and "model_dump" in exc_text:
        if base_url_configured:
            detail = (
                "Configured OPENAI_BASE_URL returned a response format that "
                "langchain_openai could not parse. This usually means the proxy "
                "is not fully compatible with OpenAI Chat Completions/tool calling."
            )
        else:
            detail = (
                "The OpenAI client returned a response format that "
                "langchain_openai could not parse."
            )
    elif exc_type == "AuthenticationError" or "invalid_api_key" in exc_text:
        detail = (
            "The OpenAI credentials were rejected by the configured endpoint."
            if base_url_configured
            else "The OpenAI credentials were rejected by the official OpenAI endpoint."
        )
    else:
        detail = exc_text or exc_type

    fallback_note = (
        f"AI-generated analysis was unavailable ({exc_type}). {detail} "
        "Used the template-based fallback instead."
    )
    return fallback_note, detail


def _configure_mlflow_tracing():
    """Best-effort MLflow tracing setup for the explainer workflow.

    Returns the imported ``mlflow`` module when available, otherwise ``None``.
    This helper intentionally swallows setup errors so model review still works
    even when tracing cannot be enabled in the current environment.
    """
    try:
        import mlflow
    except Exception:
        return None

    try:
        from hdb_resale_mlops.config import RuntimeConfig
        from hdb_resale_mlops.mlflow_registry import configure_mlflow

        runtime_config = RuntimeConfig.from_env()
        if runtime_config.mlflow_tracking_uri:
            configure_mlflow(runtime_config)
        else:
            mlflow.set_experiment(runtime_config.mlflow_experiment_name)
    except Exception:
        # If MLflow is already configured in-process, leave the existing config alone.
        pass

    try:
        mlflow.langchain.autolog(
            silent=True,
            log_traces=True,
            run_tracer_inline=True,
        )
    except Exception:
        pass

    try:
        mlflow.openai.autolog(
            silent=True,
            log_traces=True,
        )
    except Exception:
        pass

    return mlflow


def _update_current_trace_compat(mlflow_module, **kwargs) -> None:
    """Best-effort wrapper around MLflow trace updates across API variants."""
    if mlflow_module is None or not hasattr(mlflow_module, "update_current_trace"):
        return

    update_fn = mlflow_module.update_current_trace
    try:
        parameters = inspect.signature(update_fn).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    filtered_kwargs = (
        dict(kwargs)
        if accepts_var_kwargs or not parameters
        else {key: value for key, value in kwargs.items() if key in parameters}
    )
    if not filtered_kwargs:
        return

    try:
        update_fn(**filtered_kwargs)
    except TypeError:
        # Older MLflow builds expose narrower trace-update signatures. Ignore
        # trace enrichment rather than failing the promotion review.
        return


def _trace_preview(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _mlflow_trace_id(span: Any) -> str | None:
    trace_id = getattr(span, "trace_id", None)
    if not trace_id or trace_id == _NO_OP_TRACE_ID:
        return None
    return str(trace_id)


def _attach_trace_metadata(
    run_metadata: dict[str, Any], trace_id: str | None
) -> dict[str, Any]:
    enriched = dict(run_metadata)
    enriched["mlflow_trace_id"] = trace_id
    enriched["mlflow_tracing_enabled"] = trace_id is not None
    return enriched


def run_explainer_agent_detailed(
    candidate_metrics: dict[str, float],
    champion_info: dict[str, Any] | None,
    comparison: ComparisonResult,
    drift_report: DriftReport | None,
    policy_verdict: PolicyVerdict,
    model_name: str,
    model_version: str,
    model: str | None = None,
) -> ExplainerRunResult:
    """Run the explainer agent and return report text plus structured metadata.

    Falls back to a template-based report if OPENAI_API_KEY is not set.
    """
    resolved_model = model or os.environ.get("OPENAI_MODEL", "gpt-5-nano")
    started_at = perf_counter()
    brief = (
        f"Model version {model_version} has been registered as a candidate. "
        f"The policy engine verdict is: {policy_verdict.decision.value}. "
        f"Reasons: {'; '.join(policy_verdict.reasons) if policy_verdict.reasons else 'None'}. "
        f"Please investigate this candidate and produce your promotion report."
    )
    mlflow = _configure_mlflow_tracing()
    root_span_cm = (
        mlflow.start_span(
            name="promotion_explainer_review",
            span_type="WORKFLOW",
            attributes={
                "workflow": "promotion_review",
                "model_name": model_name,
                "model_version": model_version,
                "policy_verdict": policy_verdict.decision.value,
                "llm_model": resolved_model,
            },
        )
        if mlflow is not None
        else nullcontext(None)
    )

    try:
        with root_span_cm as root_span:
            if root_span is not None:
                root_span.set_inputs(
                    {
                        "brief": brief,
                        "candidate_metrics": candidate_metrics,
                        "champion_present": champion_info is not None,
                        "policy_verdict": policy_verdict.decision.value,
                    }
                )

            if mlflow is not None:
                _update_current_trace_compat(
                    mlflow,
                    tags={
                        "workflow": "promotion_review",
                        "model_name": model_name,
                        "model_version": model_version,
                        "policy_verdict": policy_verdict.decision.value,
                    },
                    metadata={
                        "prompt_version": _PROMPT_VERSION,
                        "report_schema_version": _REPORT_SCHEMA_VERSION,
                        "llm_model": resolved_model,
                    },
                    request_preview=_trace_preview(brief),
                )

            # Missing OpenAI credentials is not fatal. The workflow still
            # produces a review packet via a template-only fallback.
            if not os.environ.get("OPENAI_API_KEY"):
                fallback_note = (
                    "OPENAI_API_KEY not set. Used the template-based report instead."
                )
                fallback_span_cm = (
                    mlflow.start_span(
                        name="template_report_fallback",
                        span_type="CHAIN",
                        attributes={"fallback_reason": "missing_openai_api_key"},
                    )
                    if mlflow is not None
                    else nullcontext(None)
                )
                with fallback_span_cm as fallback_span:
                    report_text = _generate_template_report(
                        candidate_metrics=candidate_metrics,
                        champion_info=champion_info,
                        comparison=comparison,
                        drift_report=drift_report,
                        policy_verdict=policy_verdict,
                        model_version=model_version,
                        fallback_note=fallback_note,
                    )
                    structured_report = _parse_report_text(report_text)
                    if fallback_span is not None:
                        fallback_span.set_outputs(
                            {
                                "used_fallback": True,
                                "report_format": "template_markdown",
                                "summary": structured_report.summary,
                            }
                        )

                duration_ms = int((perf_counter() - started_at) * 1000)
                run_metadata = _attach_trace_metadata(
                    _build_run_metadata(
                        duration_ms=duration_ms,
                        llm_model=resolved_model,
                        agent_trace=[],
                        used_fallback=True,
                        fallback_note=fallback_note,
                        report_format="template_markdown",
                        fallback_error_type="MissingAPIKey",
                        fallback_error_detail="OPENAI_API_KEY not set.",
                    ),
                    _mlflow_trace_id(root_span),
                )
                if root_span is not None:
                    root_span.set_outputs(
                        {
                            "used_fallback": True,
                            "report_format": "template_markdown",
                            "summary": structured_report.summary,
                        }
                    )
                if mlflow is not None:
                    _update_current_trace_compat(
                        mlflow,
                        response_preview=_trace_preview(
                            structured_report.summary or report_text
                        ),
                    )
                return ExplainerRunResult(
                    report_text=report_text,
                    structured_report=structured_report,
                    agent_trace=[],
                    run_metadata=run_metadata,
                    used_fallback=True,
                    fallback_note=fallback_note,
                )

            try:
                invoke_span_cm = (
                    mlflow.start_span(
                        name="explainer_agent_invoke",
                        span_type="AGENT",
                        attributes={"llm_model": resolved_model},
                    )
                    if mlflow is not None
                    else nullcontext(None)
                )
                with invoke_span_cm as invoke_span:
                    agent = build_explainer_agent(
                        candidate_metrics=candidate_metrics,
                        champion_info=champion_info,
                        comparison=comparison,
                        drift_report=drift_report,
                        policy_verdict=policy_verdict,
                        model_name=model_name,
                        model=resolved_model,
                    )
                    # The agent decides which tools to call, but all of those
                    # tools are read-only views over workflow state.
                    result = agent.invoke({"messages": [("user", brief)]})
                    messages = result.get("messages", [])
                    if invoke_span is not None:
                        invoke_span.set_outputs({"message_count": len(messages)})

                final_text = _extract_final_report(messages, brief=brief)
                if not final_text:
                    raise RuntimeError("Agent did not produce a final report.")

                parse_span_cm = (
                    mlflow.start_span(
                        name="parse_explainer_report",
                        span_type="PARSER",
                    )
                    if mlflow is not None
                    else nullcontext(None)
                )
                with parse_span_cm as parse_span:
                    structured_report = _enrich_structured_report(
                        _coerce_structured_report(final_text),
                        candidate_metrics=candidate_metrics,
                        champion_info=champion_info,
                        comparison=comparison,
                        drift_report=drift_report,
                        policy_verdict=policy_verdict,
                    )
                    agent_trace = _extract_agent_trace(messages)
                    report_format = (
                        "json_structured"
                        if _extract_json_payload(final_text) is not None
                        else "markdown"
                    )
                    rendered_report = _render_markdown_report(structured_report)
                    if parse_span is not None:
                        parse_span.set_outputs(
                            {
                                "report_format": report_format,
                                "summary": structured_report.summary,
                                "tool_call_count": len(
                                    [
                                        event
                                        for event in agent_trace
                                        if event.get("event") == "tool_call"
                                    ]
                                ),
                            }
                        )

                # Keep the rich JSON trace artifact, but also mirror each tool
                # step into child spans so the MLflow trace UI is easier to scan.
                _log_agent_trace_spans(mlflow, agent_trace)

                duration_ms = int((perf_counter() - started_at) * 1000)
                run_metadata = _attach_trace_metadata(
                    _build_run_metadata(
                        duration_ms=duration_ms,
                        llm_model=resolved_model,
                        agent_trace=agent_trace,
                        used_fallback=False,
                        fallback_note=None,
                        report_format=report_format,
                    ),
                    _mlflow_trace_id(root_span),
                )
                if root_span is not None:
                    root_span.set_outputs(
                        {
                            "used_fallback": False,
                            "report_format": report_format,
                            "summary": structured_report.summary,
                            "tool_call_count": run_metadata["tool_call_count"],
                        }
                    )
                if mlflow is not None:
                    _update_current_trace_compat(
                        mlflow,
                        response_preview=_trace_preview(
                            structured_report.summary or rendered_report
                        ),
                    )
                return ExplainerRunResult(
                    report_text=rendered_report,
                    structured_report=structured_report,
                    agent_trace=agent_trace,
                    run_metadata=run_metadata,
                )
            except Exception as exc:
                fallback_note, fallback_detail = _describe_agent_failure(
                    exc,
                    base_url_configured=bool(_openai_base_url_from_env()),
                )
                fallback_reason = type(exc).__name__
                fallback_span_cm = (
                    mlflow.start_span(
                        name="template_report_fallback",
                        span_type="CHAIN",
                        attributes={"fallback_reason": fallback_reason},
                    )
                    if mlflow is not None
                    else nullcontext(None)
                )
                with fallback_span_cm as fallback_span:
                    report_text = _generate_template_report(
                        candidate_metrics=candidate_metrics,
                        champion_info=champion_info,
                        comparison=comparison,
                        drift_report=drift_report,
                        policy_verdict=policy_verdict,
                        model_version=model_version,
                        fallback_note=fallback_note,
                    )
                    structured_report = _parse_report_text(report_text)
                    if fallback_span is not None:
                        fallback_span.set_outputs(
                            {
                                "used_fallback": True,
                                "report_format": "template_markdown",
                                "summary": structured_report.summary,
                            }
                        )

                duration_ms = int((perf_counter() - started_at) * 1000)
                run_metadata = _attach_trace_metadata(
                    _build_run_metadata(
                        duration_ms=duration_ms,
                        llm_model=resolved_model,
                        agent_trace=[],
                        used_fallback=True,
                        fallback_note=fallback_note,
                        report_format="template_markdown",
                        fallback_error_type=fallback_reason,
                        fallback_error_detail=fallback_detail,
                    ),
                    _mlflow_trace_id(root_span),
                )
                if root_span is not None:
                    root_span.set_outputs(
                        {
                            "used_fallback": True,
                            "report_format": "template_markdown",
                            "summary": structured_report.summary,
                            "fallback_reason": fallback_reason,
                        }
                    )
                if mlflow is not None:
                    _update_current_trace_compat(
                        mlflow,
                        tags={"used_fallback": "true"},
                        metadata={
                            "fallback_reason": fallback_reason,
                            "fallback_error_detail": fallback_detail,
                        },
                        response_preview=_trace_preview(
                            structured_report.summary or report_text
                        ),
                    )
                return ExplainerRunResult(
                    report_text=report_text,
                    structured_report=structured_report,
                    agent_trace=[],
                    run_metadata=run_metadata,
                    used_fallback=True,
                    fallback_note=fallback_note,
                )
    finally:
        pass


def run_explainer_agent(
    candidate_metrics: dict[str, float],
    champion_info: dict[str, Any] | None,
    comparison: ComparisonResult,
    drift_report: DriftReport | None,
    policy_verdict: PolicyVerdict,
    model_name: str,
    model_version: str,
    model: str | None = None,
) -> str:
    """Backward-compatible wrapper that returns only the final report text."""
    return run_explainer_agent_detailed(
        candidate_metrics=candidate_metrics,
        champion_info=champion_info,
        comparison=comparison,
        drift_report=drift_report,
        policy_verdict=policy_verdict,
        model_name=model_name,
        model_version=model_version,
        model=model,
    ).report_text


# ---------------------------------------------------------------------------
# Template-based fallback (no LLM required)
# ---------------------------------------------------------------------------


def _generate_template_report(
    candidate_metrics: dict[str, float],
    champion_info: dict[str, Any] | None,
    comparison: ComparisonResult,
    drift_report: DriftReport | None,
    policy_verdict: PolicyVerdict,
    model_version: str,
    fallback_note: str | None = None,
) -> str:
    """Generate a structured report without an LLM."""
    evidence_lines = [
        f"- Candidate test RMSE: {candidate_metrics.get('rmse', 0):,.0f}",
        f"- Candidate test MAE: {candidate_metrics.get('mae', 0):,.0f}",
    ]
    risk_lines: list[str] = []

    lines = [
        f"# Model Promotion Report — Version {model_version}",
        "",
        "## Summary",
        (
            f"Policy verdict: **{policy_verdict.decision.value}**. "
            "This report is an assessment for the human reviewer; "
            "the final promotion decision remains human-controlled."
        ),
        "",
        "## Candidate Metrics",
        f"- Test RMSE: {candidate_metrics.get('rmse', 0):,.0f}",
        f"- Test MAE: {candidate_metrics.get('mae', 0):,.0f}",
        "",
    ]

    if comparison.has_champion and champion_info:
        champ_metrics = champion_info.get("metrics", {})
        lines.extend(
            [
                "## Champion Comparison",
                f"- Champion RMSE: {champ_metrics.get('rmse', 0):,.0f}",
                f"- Champion MAE: {champ_metrics.get('mae', 0):,.0f}",
                f"- RMSE delta: {comparison.metric_deltas.get('rmse_delta', 0):+,.0f} "
                f"({comparison.metric_deltas.get('rmse_delta_pct', 0):+.1%})",
                "",
            ]
        )
        evidence_lines.extend(
            [
                f"- Champion test RMSE: {champ_metrics.get('rmse', 0):,.0f}",
                f"- Champion test MAE: {champ_metrics.get('mae', 0):,.0f}",
                f"- RMSE delta vs champion: {comparison.metric_deltas.get('rmse_delta', 0):+,.0f} "
                f"({comparison.metric_deltas.get('rmse_delta_pct', 0):+.1%})",
            ]
        )

        segment_summary = _segment_performance_summary(
            comparison.segment_deltas,
            limit=5,
        )
        if segment_summary is not None:
            heading, evidence_prefix, segments = segment_summary
            lines.append(f"## {heading}")
            for sd in segments:
                lines.append(
                    f"- {sd.segment_column}={sd.segment_value}: {sd.rmse_delta_pct:+.1%}"
                )
            evidence_lines.append(
                f"- {evidence_prefix}: "
                + "; ".join(
                    f"{sd.segment_column}={sd.segment_value} ({sd.rmse_delta_pct:+.1%})"
                    for sd in segments
                )
            )
            lines.append("")
        else:
            evidence_lines.append("- Segment comparison: no segment regressions reported.")
    else:
        lines.extend(
            [
                "## Champion Comparison",
                "First model candidate — no incumbent baseline is available.",
                "",
            ]
        )
        evidence_lines.append(
            "- Baseline comparison: first candidate; no incumbent baseline available."
        )

    if drift_report:
        drifted = [r for r in drift_report.column_results if r.is_drifted]
        lines.append("## Drift Detection")
        if drifted:
            lines.append(f"Drift detected in {len(drifted)} feature(s):")
            for r in drifted:
                stat_str = f"{r.drift_type.upper()}={r.statistic:.4f}"
                if r.p_value is not None:
                    stat_str += f" (p={r.p_value:.4f})"
                lines.append(f"- {r.column}: {stat_str}")
            evidence_lines.append(
                "- Drift status: detected in "
                + ", ".join(r.column for r in drifted)
                + "."
            )
        else:
            lines.append("No significant drift detected.")
            evidence_lines.append("- Drift status: no significant drift detected.")
        lines.append("")
    else:
        evidence_lines.append("- Drift status: no drift report provided.")

    if policy_verdict.checks_passed:
        evidence_lines.append(
            "- Checks passed: " + _format_check_names(policy_verdict.checks_passed)
        )
    if policy_verdict.checks_failed:
        risk_lines.append(
            "- Checks failed: " + _format_check_names(policy_verdict.checks_failed)
        )

    for reason in policy_verdict.reasons:
        risk_lines.append(f"- {reason}")
    if not risk_lines:
        risk_lines.append("- No policy risk flags were identified.")

    lines.extend(["## Evidence", *evidence_lines, "", "## Risk Flags", *risk_lines, ""])
    if policy_verdict.decision == PolicyDecision.REJECT:
        recommendation = (
            f"Policy engine recommends: **{policy_verdict.decision.value}**. "
            "A human reviewer should confirm the rejection, inspect the failed "
            "checks, verify data quality and split integrity, and require "
            "retraining or feature changes before reconsidering promotion."
        )
    elif policy_verdict.decision == PolicyDecision.MANUAL_REVIEW:
        recommendation = (
            f"Policy engine recommends: **{policy_verdict.decision.value}**. "
            "A human reviewer should inspect the flagged drift or segment "
            "regressions, validate whether the degradation is operationally "
            "acceptable, and document the final decision."
        )
    else:
        recommendation = (
            f"Policy engine recommends: **{policy_verdict.decision.value}**. "
            "A human reviewer should confirm the passed checks, spot-check "
            "segment behaviour, and then approve or hold promotion."
        )

    lines.extend(
        [
            "## Policy Reasons",
            *[f"- {reason}" for reason in policy_verdict.reasons],
            "",
            "## Recommendation",
            recommendation,
        ]
    )

    if fallback_note:
        lines.extend(["", f"*{fallback_note}*"])
    else:
        lines.extend(
            [
                "",
                "*(This is a template-based report. Set OPENAI_API_KEY for AI-generated analysis.)*",
            ]
        )

    return "\n".join(lines)
