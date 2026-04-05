"""
Environment loading utilities for local and notebook workflows.

This module exists because the package runs in two distinct environments —
local/Colab notebooks and SageMaker training containers — each with different
ways of supplying secrets and configuration. Rather than requiring every
developer to manually export variables before running a notebook, this module
auto-discovers and loads a repo-local ``.env`` file into the process at import
time, without overriding variables that are already set (e.g. from CI or a
SageMaker container). It also centralises the list of project env vars that
must be forwarded into SageMaker training jobs, keeping that contract in one
place instead of scattered across notebook cells.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import MutableMapping
from pathlib import Path
import ast
import os
import shlex

_LOADED_ENV_FILES: set[Path] = set()
FORWARDED_SAGEMAKER_ENV_VARS: tuple[str, ...] = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_MODEL_NAME",
    "MLFLOW_EXPERIMENT_NAME",
    "HDB_DATA_COLLECTION_ID",
    "HDB_DATASET_ID",
    "HDB_DATASET_NAME",
    "DATA_GOV_API_KEY",
    "MAESTRO_HTTP_PROXY",
    "MAESTRO_HTTPS_PROXY",
    "RANDOM_SEED",
    "VALIDATION_MONTHS",
    "TEST_MONTHS",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ENABLE_JUDGE_EVAL",
    "MARKET_RESEARCH_PROVIDER",
    "OPENAI_WEB_SEARCH_MODEL",
    "OPENAI_JUDGE_MODEL",
    "TAVILY_API_KEY",
    "MODEL_REVIEWER",
    "TRAINING_PIP_INDEX_URL",
    "PIP_INDEX_URL",
)


def _maestro_proxy_env_vars(
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    target_env = os.environ if env is None else env
    proxy_env: dict[str, str] = {}
    http_proxy = target_env.get("MAESTRO_HTTP_PROXY")
    https_proxy = target_env.get("MAESTRO_HTTPS_PROXY")
    if http_proxy:
        proxy_env["HTTP_PROXY"] = http_proxy
        proxy_env["http_proxy"] = http_proxy
    if https_proxy:
        proxy_env["HTTPS_PROXY"] = https_proxy
        proxy_env["https_proxy"] = https_proxy
    return proxy_env


@contextmanager
def maestro_proxy_env(env: MutableMapping[str, str] | None = None):
    target_env = os.environ if env is None else env
    proxy_env = _maestro_proxy_env_vars(target_env)
    previous_values = {key: target_env.get(key) for key in proxy_env}

    for key, value in proxy_env.items():
        target_env[key] = value

    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                target_env.pop(key, None)
            else:
                target_env[key] = previous_value


def _discover_repo_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _candidate_env_files() -> list[Path]:
    candidates: list[Path] = []
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        candidates.append(cwd_env.resolve())

    repo_root = _discover_repo_root(Path(__file__).resolve())
    if repo_root is not None:
        repo_env = (repo_root / ".env").resolve()
        if repo_env.exists() and repo_env not in candidates:
            candidates.append(repo_env)

    return candidates


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            return str(ast.literal_eval(value))
        except Exception:
            return value[1:-1]

    lexer = shlex.shlex(value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens = list(lexer)
    if not tokens:
        return ""
    return " ".join(tokens)


def _load_env_file(path: Path, *, override: bool = False) -> bool:
    changed = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = _parse_env_value(raw_value)
        changed = True
    return changed


def load_repo_env(*, override: bool = False) -> Path | None:
    """Best-effort load of a local `.env` file into the current process.

    This is intended for notebook and local development workflows. Existing
    environment variables win by default.
    """
    for env_file in _candidate_env_files():
        if env_file in _LOADED_ENV_FILES and not override:
            return env_file
        _load_env_file(env_file, override=override)
        _LOADED_ENV_FILES.add(env_file)
        return env_file
    return None


def collect_sagemaker_forwarded_env() -> dict[str, str]:
    """Return project env vars that should be forwarded into SageMaker jobs."""
    forwarded = {
        key: value
        for key in FORWARDED_SAGEMAKER_ENV_VARS
        if (value := os.environ.get(key))
    }

    base_url = forwarded.get("OPENAI_BASE_URL")
    api_base = forwarded.get("OPENAI_API_BASE")
    if base_url and not api_base:
        forwarded["OPENAI_API_BASE"] = base_url
    elif api_base and not base_url:
        forwarded["OPENAI_BASE_URL"] = api_base

    return forwarded


__all__ = [
    "FORWARDED_SAGEMAKER_ENV_VARS",
    "collect_sagemaker_forwarded_env",
    "load_repo_env",
    "maestro_proxy_env",
]
