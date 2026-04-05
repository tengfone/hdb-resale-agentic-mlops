"""
MLflow integration for training runs, registry aliases, and review artifacts.

This module is the persistence layer for the project.

The training notebook uses it to:
- configure MLflow
- log metrics, artifacts, and the sklearn model
- register a new model version
- point the `candidate` alias at that version

The promotion workflow uses it to:
- load the current `champion`
- read the champion's test metrics and segment artifacts
- write review packets and explainer outputs back to the backing run
- tag the model version as `champion` or `rejected`

If the notebook is the orchestrator, this file is the durable handoff point.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import os
from typing import Any, Mapping

from hdb_resale_mlops.config import RuntimeConfig
from hdb_resale_mlops.evaluation import EvaluationResult
from hdb_resale_mlops.features import FEATURE_COLUMNS, NUMERIC_FEATURES, feature_schema


@dataclass(frozen=True)
class RegistrationResult:
    run_id: str
    model_name: str
    model_version: str
    model_uri: str


class MlflowRegistryError(RuntimeError):
    """Raised when the MLflow registry or tracking store is unavailable."""


class ChampionDataUnavailableError(MlflowRegistryError):
    """Raised when champion metadata exists but cannot be loaded safely."""


class PromotionReviewPersistenceError(MlflowRegistryError):
    """Raised when promotion review artifacts cannot be persisted safely."""


def _flatten(prefix: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        composite = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten(composite, value))
        else:
            flattened[composite] = value
    return flattened


def _set_model_version_tags(
    client,
    model_name: str,
    model_version: str,
    tags: Mapping[str, Any],
) -> None:
    """Set model version tags, skipping empty values and normalizing scalars."""
    for key, value in tags.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            normalized = str(value).lower()
        else:
            normalized = str(value)
        client.set_model_version_tag(model_name, model_version, key, normalized)


def _error_code(exc: Exception) -> str | None:
    return getattr(exc, "error_code", None)


def _is_missing_registered_model_error(exc: Exception) -> bool:
    return _error_code(exc) in {
        "RESOURCE_DOES_NOT_EXIST",
        "INVALID_PARAMETER_VALUE",
        "NOT_FOUND",
    }


def _sanitize_artifact_token(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.strip()
    )
    cleaned = cleaned.strip("-_")
    return cleaned or "review"


def _review_artifact_path(review_id: str | None) -> str:
    token = _sanitize_artifact_token(str(review_id or "latest-review"))
    return f"promotion_review/{token}"


def _judge_metrics_from_payload(
    judge_evaluation: Mapping[str, Any] | None,
) -> dict[str, float]:
    if not isinstance(judge_evaluation, Mapping):
        return {}
    if judge_evaluation.get("status") != "scored":
        return {}

    scores = judge_evaluation.get("scores")
    if not isinstance(scores, Mapping):
        return {}

    metrics: dict[str, float] = {}
    for key in ("completeness", "accuracy", "actionability", "safety", "average"):
        value = scores.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[f"judge_{key}"] = float(value)
    return metrics


def configure_mlflow(config: RuntimeConfig) -> None:
    import mlflow

    tracking_uri = config.resolved_mlflow_tracking_uri()
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    if config.mlflow_tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = config.mlflow_tracking_username
    if config.mlflow_tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = config.mlflow_tracking_password
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)


def _log_sklearn_model_with_registry(
    *,
    mlflow_sklearn,
    model,
    input_example,
    signature,
    registered_model_name: str,
):
    """Call MLflow's sklearn log_model across API variants.

    Newer MLflow releases use `name=...`, while older releases still expect
    `artifact_path=...`.
    """

    common_kwargs = {
        "sk_model": model,
        "input_example": input_example,
        "signature": signature,
        "registered_model_name": registered_model_name,
    }
    try:
        parameters = inspect.signature(mlflow_sklearn.log_model).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "name" in parameters:
        return mlflow_sklearn.log_model(name="model", **common_kwargs)
    if "artifact_path" in parameters:
        return mlflow_sklearn.log_model(artifact_path="model", **common_kwargs)

    try:
        return mlflow_sklearn.log_model(name="model", **common_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'name'" not in str(exc):
            raise
        return mlflow_sklearn.log_model(artifact_path="model", **common_kwargs)


def _write_evaluation_artifacts(
    artifact_dir: Path,
    prefix: str,
    evaluation: EvaluationResult,
) -> list[Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    predictions_path = artifact_dir / f"{prefix}_predictions.csv"
    evaluation.scored_frame.to_csv(predictions_path, index=False)
    written_paths.append(predictions_path)

    metrics_path = artifact_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(
        json.dumps(evaluation.overall_metrics, indent=2), encoding="utf-8"
    )
    written_paths.append(metrics_path)

    for segment_name, frame in evaluation.segment_metrics.items():
        segment_csv_path = artifact_dir / f"{prefix}_segments_by_{segment_name}.csv"
        segment_json_path = artifact_dir / f"{prefix}_segments_by_{segment_name}.json"
        frame.to_csv(segment_csv_path, index=False)
        segment_json_path.write_text(
            frame.to_json(orient="records", indent=2), encoding="utf-8"
        )
        written_paths.extend([segment_csv_path, segment_json_path])

    return written_paths


def log_and_register_candidate_model(
    model,
    validation_evaluation: EvaluationResult,
    test_evaluation: EvaluationResult,
    runtime_config: RuntimeConfig,
    artifact_dir: Path,
    dataset_snapshot: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
    training_job_metadata: Mapping[str, Any],
) -> RegistrationResult:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    configure_mlflow(runtime_config)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="hdb-resale-candidate-training") as run:
        mlflow.set_tags(
            {
                "phase": "phase-2",
                "candidate_status": "candidate",
                "dataset_id": runtime_config.dataset_id,
                "dataset_collection_id": runtime_config.dataset_collection_id,
            }
        )
        mlflow.log_params(
            _flatten(
                "",
                {
                    "hyperparameters": dict(hyperparameters),
                    "training_job": dict(training_job_metadata),
                },
            )
        )
        mlflow.log_dict(dict(dataset_snapshot), "dataset_snapshot.json")
        mlflow.log_dict(dict(split_summary), "split_summary.json")
        mlflow.log_dict(feature_schema(), "feature_schema.json")

        for prefix, evaluation in (
            ("validation", validation_evaluation),
            ("test", test_evaluation),
        ):
            for metric_name, metric_value in evaluation.overall_metrics.items():
                mlflow.log_metric(f"{prefix}_{metric_name}", float(metric_value))

        for path in _write_evaluation_artifacts(
            artifact_dir, "validation", validation_evaluation
        ):
            mlflow.log_artifact(str(path), artifact_path="evaluation")
        for path in _write_evaluation_artifacts(artifact_dir, "test", test_evaluation):
            mlflow.log_artifact(str(path), artifact_path="evaluation")

        input_example = (
            test_evaluation.scored_frame.loc[:, FEATURE_COLUMNS].head(5).copy()
        )
        for column in NUMERIC_FEATURES:
            if column in input_example.columns:
                input_example[column] = input_example[column].astype(float)
        signature = infer_signature(
            model_input=input_example,
            model_output=model.predict(input_example),
        )
        # Registering during log_model gives us both a run artifact and a
        # versioned entry in the MLflow Model Registry.
        logged_model = _log_sklearn_model_with_registry(
            mlflow_sklearn=mlflow.sklearn,
            model=model,
            input_example=input_example,
            signature=signature,
            registered_model_name=runtime_config.mlflow_model_name,
        )

        client = MlflowClient()
        registered_version = getattr(logged_model, "registered_model_version", None)
        if registered_version is None:
            raise MlflowRegistryError(
                f"MLflow did not return a registered model version for {runtime_config.mlflow_model_name!r}."
            )
        model_uri = (
            str(getattr(logged_model, "model_uri", ""))
            or f"runs:/{run.info.run_id}/model"
        )
        # The latest successfully trained model becomes the active candidate.
        client.set_registered_model_alias(
            runtime_config.mlflow_model_name,
            "candidate",
            str(registered_version),
        )
        client.set_model_version_tag(
            runtime_config.mlflow_model_name,
            str(registered_version),
            "promotion_status",
            "candidate",
        )

    return RegistrationResult(
        run_id=run.info.run_id,
        model_name=runtime_config.mlflow_model_name,
        model_version=str(registered_version),
        model_uri=model_uri,
    )


def _load_segment_artifacts(
    run_id: str,
    segment_columns: tuple[str, ...] = ("town", "flat_type"),
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Load champion segment metrics from MLflow run artifacts.

    Returns a dict keyed by segment column name (e.g. ``{"town": DataFrame,
    "flat_type": DataFrame}``).  Returns an empty dict when the artifacts are
    missing (backward-compatible with runs logged before segments were stored).
    """
    import tempfile

    import pandas as pd
    from mlflow import MlflowClient

    client = MlflowClient()
    segments: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            local_dir = client.download_artifacts(run_id, "evaluation", tmp_dir)
        except Exception as exc:
            if required:
                raise ChampionDataUnavailableError(
                    f"Champion run {run_id!r} evaluation artifacts could not be downloaded."
                ) from exc
            return segments

        local_path = Path(local_dir)
        missing_columns: list[str] = []
        for col in segment_columns:
            json_file = local_path / f"test_segments_by_{col}.json"
            if json_file.exists():
                segments[col] = pd.read_json(json_file, orient="records")
            else:
                missing_columns.append(col)

    if required and missing_columns:
        missing = ", ".join(missing_columns)
        raise ChampionDataUnavailableError(
            f"Champion run {run_id!r} is missing required segment metric artifacts for: {missing}."
        )
    return segments


def get_champion_version(model_name: str) -> dict[str, Any] | None:
    """Return champion model version info, or None if no champion exists."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException as exc:
        if _is_missing_registered_model_error(exc):
            return None
        raise MlflowRegistryError(
            f"MLflow could not resolve the champion alias for model {model_name!r}."
        ) from exc
    except Exception as exc:
        raise MlflowRegistryError(
            f"MLflow could not resolve the champion alias for model {model_name!r}."
        ) from exc

    try:
        run = client.get_run(version.run_id)
    except Exception as exc:
        raise MlflowRegistryError(
            f"MLflow could not load the backing run for champion model {model_name!r} version {version.version}."
        ) from exc

    # Champion comparison uses test metrics from the backing MLflow run plus
    # per-segment artifacts saved during registration.
    metrics = {
        k.replace("test_", ""): v
        for k, v in run.data.metrics.items()
        if k.startswith("test_")
    }
    segment_metrics = _load_segment_artifacts(version.run_id, required=True)
    return {
        "version": version.version,
        "run_id": version.run_id,
        "metrics": metrics,
        "segment_metrics": segment_metrics,
    }


def get_training_history(
    model_name: str, max_versions: int = 10
) -> list[dict[str, Any]]:
    """Return metrics from the last N registered model versions."""
    from mlflow import MlflowClient

    client = MlflowClient()
    try:
        versions = client.search_model_versions(
            f"name='{model_name}'",
            order_by=["version_number DESC"],
            max_results=max_versions,
        )
    except Exception:
        return []

    history: list[dict[str, Any]] = []
    for mv in versions:
        try:
            run = client.get_run(mv.run_id)
            history.append(
                {
                    "version": mv.version,
                    "run_id": mv.run_id,
                    "metrics": dict(run.data.metrics),
                    "tags": dict(run.data.tags),
                }
            )
        except Exception:
            continue
    return history


def log_promotion_review_artifacts(
    model_name: str,
    model_version: str,
    review_payload: Mapping[str, Any],
) -> bool:
    """Attach explainer outputs to the model version's backing MLflow run.

    Returns ``True`` when the payload was logged successfully and ``False`` when
    the model version or tracking run could not be resolved.
    """
    import tempfile

    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    try:
        version = client.get_model_version(model_name, model_version)
    except Exception:
        return False

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            artifact_path = _review_artifact_path(review_payload.get("review_id"))
            report_text = str(review_payload.get("report_text", ""))
            run_metadata = review_payload.get("run_metadata")
            judge_evaluation = review_payload.get("judge_evaluation")
            trace_id = None
            if isinstance(run_metadata, Mapping):
                candidate_trace_id = run_metadata.get("mlflow_trace_id")
                if candidate_trace_id:
                    trace_id = str(candidate_trace_id)
            (tmp_path / "report.md").write_text(report_text, encoding="utf-8")
            (tmp_path / "review.json").write_text(
                json.dumps(dict(review_payload), indent=2),
                encoding="utf-8",
            )
            structured = review_payload.get("report_structured")
            if structured is not None:
                (tmp_path / "report_structured.json").write_text(
                    json.dumps(structured, indent=2),
                    encoding="utf-8",
                )
            trace = review_payload.get("agent_trace")
            if trace is not None:
                (tmp_path / "agent_trace.json").write_text(
                    json.dumps(trace, indent=2),
                    encoding="utf-8",
                )
            if judge_evaluation is not None:
                (tmp_path / "judge_evaluation.json").write_text(
                    json.dumps(judge_evaluation, indent=2),
                    encoding="utf-8",
                )
            if trace_id:
                (tmp_path / "mlflow_trace_id.txt").write_text(
                    trace_id, encoding="utf-8"
                )

            with mlflow.start_run(run_id=version.run_id):
                for artifact in tmp_path.iterdir():
                    mlflow.log_artifact(
                        str(artifact),
                        artifact_path=artifact_path,
                    )
                if hasattr(mlflow, "log_metric"):
                    for metric_name, metric_value in _judge_metrics_from_payload(
                        judge_evaluation
                    ).items():
                        mlflow.log_metric(metric_name, metric_value)
            if trace_id:
                try:
                    client.link_traces_to_run([trace_id], run_id=version.run_id)
                except Exception:
                    pass
    except Exception:
        return False

    return True


def persist_promotion_review_record(
    model_name: str,
    model_version: str,
    review_id: str,
    review_record: Mapping[str, Any],
) -> bool:
    """Persist the review packet itself to the candidate run for durable audit."""
    import tempfile

    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    try:
        version = client.get_model_version(model_name, model_version)
    except Exception:
        return False

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            review_path = tmp_path / "review_record.json"
            review_path.write_text(
                json.dumps(dict(review_record), indent=2), encoding="utf-8"
            )
            with mlflow.start_run(run_id=version.run_id):
                mlflow.log_artifact(
                    str(review_path),
                    artifact_path=_review_artifact_path(review_id),
                )
    except Exception:
        return False

    return True


def promote_to_champion(
    model_name: str,
    model_version: str,
    decision_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Set the champion alias and attach decision-audit tags."""
    from mlflow import MlflowClient

    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", model_version)
    _set_model_version_tags(
        client,
        model_name,
        model_version,
        {
            "promotion_status": "champion",
            **dict(decision_metadata or {}),
        },
    )
    try:
        client.delete_model_version_tag(model_name, model_version, "rejection_reasons")
    except Exception:
        pass


def reject_candidate(
    model_name: str,
    model_version: str,
    reasons: list[str],
    decision_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Tag a model version as rejected with reasons and decision-audit tags."""
    from mlflow import MlflowClient

    client = MlflowClient()
    _set_model_version_tags(
        client,
        model_name,
        model_version,
        {
            "promotion_status": "rejected",
            "rejection_reasons": "; ".join(reasons),
            **dict(decision_metadata or {}),
        },
    )
