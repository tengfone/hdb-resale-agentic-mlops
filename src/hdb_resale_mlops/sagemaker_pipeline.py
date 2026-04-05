"""
SageMaker Pipelines builder and reusable step helpers.

The pipeline mode intentionally stays close to the direct MAESTRO notebook:
- prepare data with the existing data.gov.sg + split helpers
- train with the existing SageMaker script-mode entrypoint
- evaluate/register with the existing MLflow helpers
- compute the deterministic policy gate before handing off to notebook-driven
  explainer + human review

MLflow remains the only registry of record. SageMaker Model Registry is not
used here.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping

from hdb_resale_mlops.comparison import compare_models
from hdb_resale_mlops.config import ProjectPaths, RuntimeConfig
from hdb_resale_mlops.data import (
    chronological_split,
    load_or_download_snapshot,
    load_raw_resale_frame,
)
from hdb_resale_mlops.drift import DriftReport, run_drift_checks
from hdb_resale_mlops.env import collect_sagemaker_forwarded_env, load_repo_env
from hdb_resale_mlops.evaluation import evaluate_model
from hdb_resale_mlops.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_training_frame,
)
from hdb_resale_mlops.mlflow_registry import (
    MlflowRegistryError,
    log_and_register_candidate_model,
    reject_candidate,
)
from hdb_resale_mlops.policy import PolicyDecision, PolicyVerdict, evaluate_policy
from hdb_resale_mlops.sagemaker_job import _resolve_sklearn_image_uri
from hdb_resale_mlops.tabular_state import serialize_for_state


PREPARE_DATA_STEP_NAME = "PrepareData"
TRAIN_CANDIDATE_STEP_NAME = "TrainCandidate"
EVALUATE_REGISTER_STEP_NAME = "EvaluateRegisterCandidate"
POLICY_GATE_STEP_NAME = "PolicyGate"

PREPARE_DATA_OUTPUTS = {
    "train": "train",
    "validation": "validation",
    "test": "test",
    "metadata": "metadata",
}
EVALUATE_REGISTER_OUTPUTS = {
    "registration": "registration",
    "evaluation": "evaluation",
}
POLICY_GATE_OUTPUTS = {
    "policy": "policy",
    "handoff": "handoff",
}

PIPELINE_STATUS_PENDING_REVIEW = "pending_review"
PIPELINE_STATUS_AUTO_REJECTED = "auto_rejected"


def require_pipeline_mlflow_tracking_uri(config: RuntimeConfig) -> str:
    """Pipeline mode requires a shared MLflow backend across isolated jobs."""
    tracking_uri = config.mlflow_tracking_uri
    if tracking_uri and not tracking_uri.startswith("sqlite:///"):
        return tracking_uri
    raise ValueError(
        "SageMaker Pipeline mode requires a non-SQLite MLFLOW_TRACKING_URI because "
        "the prepare/evaluate/policy steps run in separate isolated jobs."
    )


def build_pipeline_output_s3_uri(
    *,
    bucket: str,
    training_job_prefix: str,
    pipeline_name: str,
    execution_id: str,
    step_name: str,
    output_name: str,
    filename: str,
) -> str:
    return (
        f"s3://{bucket}/{training_job_prefix}/pipelines/{pipeline_name}/"
        f"{execution_id}/{step_name}/{output_name}/{filename}"
    )


def _pipeline_output_join(bucket_param, training_job_prefix_param, pipeline_name: str, step_name: str, output_name: str):
    from sagemaker.workflow.execution_variables import ExecutionVariables
    from sagemaker.workflow.functions import Join

    return Join(
        on="/",
        values=[
            "s3:/",
            bucket_param,
            training_job_prefix_param,
            "pipelines",
            pipeline_name,
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            step_name,
            output_name,
        ],
    )


def _processing_source_dir(project_paths: ProjectPaths) -> Path:
    source_dir = project_paths.source_dir / "pipeline_steps"
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Expected processing source directory at {source_dir}."
        )
    return source_dir


def _processing_dependencies(project_paths: ProjectPaths) -> list[str]:
    return [str(project_paths.source_dir / "hdb_resale_mlops")]


def _pipeline_environment(config: RuntimeConfig) -> dict[str, str]:
    env = collect_sagemaker_forwarded_env()
    env["AWS_DEFAULT_REGION"] = config.aws_region
    env["AWS_REGION"] = config.aws_region
    if config.training_pip_index_url:
        env["PIP_INDEX_URL"] = config.training_pip_index_url
    require_pipeline_mlflow_tracking_uri(config)
    env["MLFLOW_TRACKING_URI"] = config.mlflow_tracking_uri or ""
    return env


def build_sagemaker_pipeline(
    *,
    pipeline_name: str,
    runtime_config: RuntimeConfig,
    project_paths: ProjectPaths,
    role_arn: str,
    pipeline_session,
):
    """
    Build the SageMaker Pipeline for the MAESTRO DAG workflow.

    The resulting DAG is linear by design:
    PrepareData -> TrainCandidate -> EvaluateRegisterCandidate -> PolicyGate
    """
    from sagemaker.inputs import TrainingInput
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.sklearn.processing import SKLearnProcessor
    from sagemaker.workflow.execution_variables import ExecutionVariables
    from sagemaker.workflow.parameters import ParameterInteger, ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

    require_pipeline_mlflow_tracking_uri(runtime_config)

    processing_source_dir = _processing_source_dir(project_paths)
    shared_dependencies = _processing_dependencies(project_paths)
    step_environment = _pipeline_environment(runtime_config)

    bucket_param = ParameterString(
        name="Bucket",
        default_value=runtime_config.require_s3_bucket(),
    )
    dataset_collection_id_param = ParameterString(
        name="DatasetCollectionId",
        default_value=runtime_config.dataset_collection_id,
    )
    dataset_id_param = ParameterString(
        name="DatasetId",
        default_value=runtime_config.dataset_id,
    )
    dataset_name_param = ParameterString(
        name="DatasetName",
        default_value=runtime_config.dataset_name,
    )
    random_seed_param = ParameterInteger(
        name="RandomSeed",
        default_value=runtime_config.random_seed,
    )
    validation_months_param = ParameterInteger(
        name="ValidationMonths",
        default_value=runtime_config.validation_months,
    )
    test_months_param = ParameterInteger(
        name="TestMonths",
        default_value=runtime_config.test_months,
    )
    training_job_prefix_param = ParameterString(
        name="TrainingJobPrefix",
        default_value=runtime_config.training_job_prefix,
    )
    instance_type_param = ParameterString(
        name="SageMakerInstanceType",
        default_value=runtime_config.sagemaker_instance_type,
    )
    processor = SKLearnProcessor(
        framework_version=runtime_config.sagemaker_framework_version,
        role=role_arn,
        instance_type=instance_type_param,
        instance_count=1,
        sagemaker_session=pipeline_session,
        env=step_environment,
    )

    prepare_step_args = processor.run(
        code="prepare_data.py",
        source_dir=str(processing_source_dir),
        dependencies=shared_dependencies,
        arguments=[
            "--dataset-collection-id",
            dataset_collection_id_param,
            "--dataset-id",
            dataset_id_param,
            "--dataset-name",
            dataset_name_param,
            "--random-seed",
            random_seed_param,
            "--validation-months",
            validation_months_param,
            "--test-months",
            test_months_param,
            "--train-dir",
            "/opt/ml/processing/output/train",
            "--validation-dir",
            "/opt/ml/processing/output/validation",
            "--test-dir",
            "/opt/ml/processing/output/test",
            "--metadata-dir",
            "/opt/ml/processing/output/metadata",
        ],
        outputs=[
            ProcessingOutput(
                output_name=PREPARE_DATA_OUTPUTS["train"],
                source="/opt/ml/processing/output/train",
                destination=_pipeline_output_join(
                    bucket_param,
                    training_job_prefix_param,
                    pipeline_name,
                    PREPARE_DATA_STEP_NAME,
                    PREPARE_DATA_OUTPUTS["train"],
                ),
            ),
            ProcessingOutput(
                output_name=PREPARE_DATA_OUTPUTS["validation"],
                source="/opt/ml/processing/output/validation",
                destination=_pipeline_output_join(
                    bucket_param,
                    training_job_prefix_param,
                    pipeline_name,
                    PREPARE_DATA_STEP_NAME,
                    PREPARE_DATA_OUTPUTS["validation"],
                ),
            ),
            ProcessingOutput(
                output_name=PREPARE_DATA_OUTPUTS["test"],
                source="/opt/ml/processing/output/test",
                destination=_pipeline_output_join(
                    bucket_param,
                    training_job_prefix_param,
                    pipeline_name,
                    PREPARE_DATA_STEP_NAME,
                    PREPARE_DATA_OUTPUTS["test"],
                ),
            ),
            ProcessingOutput(
                output_name=PREPARE_DATA_OUTPUTS["metadata"],
                source="/opt/ml/processing/output/metadata",
                destination=_pipeline_output_join(
                    bucket_param,
                    training_job_prefix_param,
                    pipeline_name,
                    PREPARE_DATA_STEP_NAME,
                    PREPARE_DATA_OUTPUTS["metadata"],
                ),
            ),
        ],
    )
    step_prepare = ProcessingStep(
        name=PREPARE_DATA_STEP_NAME,
        step_args=prepare_step_args,
    )

    image_uri = _resolve_sklearn_image_uri(
        runtime_config.aws_region,
        runtime_config.sagemaker_framework_version,
        runtime_config.sagemaker_instance_type,
    )
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(project_paths.source_dir),
        role=role_arn,
        instance_type=instance_type_param,
        instance_count=runtime_config.sagemaker_instance_count,
        image_uri=image_uri,
        output_path=_pipeline_output_join(
            bucket_param,
            training_job_prefix_param,
            pipeline_name,
            TRAIN_CANDIDATE_STEP_NAME,
            "model",
        ),
        max_run=runtime_config.sagemaker_max_run_seconds,
        base_job_name=runtime_config.training_job_prefix,
        hyperparameters={"random_seed": random_seed_param},
        sagemaker_session=pipeline_session,
        environment=step_environment,
    )
    train_step_args = estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_prepare.properties.ProcessingOutputConfig.Outputs[
                    PREPARE_DATA_OUTPUTS["train"]
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_prepare.properties.ProcessingOutputConfig.Outputs[
                    PREPARE_DATA_OUTPUTS["validation"]
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )
    step_train = TrainingStep(
        name=TRAIN_CANDIDATE_STEP_NAME,
        step_args=train_step_args,
    )

    evaluate_step_args = processor.run(
        code="evaluate_register.py",
        source_dir=str(processing_source_dir),
        dependencies=shared_dependencies,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
                input_name="model",
            ),
            ProcessingInput(
                source=step_prepare.properties.ProcessingOutputConfig.Outputs[
                    PREPARE_DATA_OUTPUTS["validation"]
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/validation",
                input_name="validation",
            ),
            ProcessingInput(
                source=step_prepare.properties.ProcessingOutputConfig.Outputs[
                    PREPARE_DATA_OUTPUTS["test"]
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/test",
                input_name="test",
            ),
            ProcessingInput(
                source=step_prepare.properties.ProcessingOutputConfig.Outputs[
                    PREPARE_DATA_OUTPUTS["metadata"]
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/metadata",
                input_name="metadata",
            ),
        ],
        arguments=[
            "--model-artifact-input",
            "/opt/ml/processing/input/model",
            "--validation-input",
            "/opt/ml/processing/input/validation/validation.csv",
            "--test-input",
            "/opt/ml/processing/input/test/test.csv",
            "--metadata-input",
            "/opt/ml/processing/input/metadata",
            "--registration-dir",
            "/opt/ml/processing/output/registration",
            "--evaluation-dir",
            "/opt/ml/processing/output/evaluation",
            "--training-job-name",
            step_train.properties.TrainingJobName,
            "--model-artifact-s3-uri",
            step_train.properties.ModelArtifacts.S3ModelArtifacts,
            "--random-seed",
            random_seed_param,
        ],
        outputs=[
            ProcessingOutput(
                output_name=EVALUATE_REGISTER_OUTPUTS["registration"],
                source="/opt/ml/processing/output/registration",
                destination=_pipeline_output_join(
                    bucket_param,
                    training_job_prefix_param,
                    pipeline_name,
                    EVALUATE_REGISTER_STEP_NAME,
                    EVALUATE_REGISTER_OUTPUTS["registration"],
                ),
            ),
            ProcessingOutput(
                output_name=EVALUATE_REGISTER_OUTPUTS["evaluation"],
                source="/opt/ml/processing/output/evaluation",
                destination=_pipeline_output_join(
                    bucket_param,
                    training_job_prefix_param,
                    pipeline_name,
                    EVALUATE_REGISTER_STEP_NAME,
                    EVALUATE_REGISTER_OUTPUTS["evaluation"],
                ),
            ),
        ],
    )
    step_evaluate = ProcessingStep(
        name=EVALUATE_REGISTER_STEP_NAME,
        step_args=evaluate_step_args,
    )

    policy_gate_args = processor.run(
        code="policy_gate.py",
        source_dir=str(processing_source_dir),
        dependencies=shared_dependencies,
        inputs=[
            ProcessingInput(
                source=step_prepare.properties.ProcessingOutputConfig.Outputs[
                    PREPARE_DATA_OUTPUTS["train"]
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/train",
                input_name="train",
            ),
            ProcessingInput(
                source=step_prepare.properties.ProcessingOutputConfig.Outputs[
                    PREPARE_DATA_OUTPUTS["test"]
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/test",
                input_name="test",
            ),
            ProcessingInput(
                source=step_evaluate.properties.ProcessingOutputConfig.Outputs[
                    EVALUATE_REGISTER_OUTPUTS["registration"]
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/registration",
                input_name="registration",
            ),
            ProcessingInput(
                source=step_evaluate.properties.ProcessingOutputConfig.Outputs[
                    EVALUATE_REGISTER_OUTPUTS["evaluation"]
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/evaluation",
                input_name="evaluation",
            ),
        ],
        arguments=[
            "--train-input",
            "/opt/ml/processing/input/train/train.csv",
            "--test-input",
            "/opt/ml/processing/input/test/test.csv",
            "--registration-input",
            "/opt/ml/processing/input/registration/registration.json",
            "--evaluation-input",
            "/opt/ml/processing/input/evaluation",
            "--policy-dir",
            "/opt/ml/processing/output/policy",
            "--handoff-dir",
            "/opt/ml/processing/output/handoff",
            "--pipeline-execution-id",
            ExecutionVariables.PIPELINE_EXECUTION_ID,
        ],
        outputs=[
            ProcessingOutput(
                output_name=POLICY_GATE_OUTPUTS["policy"],
                source="/opt/ml/processing/output/policy",
                destination=_pipeline_output_join(
                    bucket_param,
                    training_job_prefix_param,
                    pipeline_name,
                    POLICY_GATE_STEP_NAME,
                    POLICY_GATE_OUTPUTS["policy"],
                ),
            ),
            ProcessingOutput(
                output_name=POLICY_GATE_OUTPUTS["handoff"],
                source="/opt/ml/processing/output/handoff",
                destination=_pipeline_output_join(
                    bucket_param,
                    training_job_prefix_param,
                    pipeline_name,
                    POLICY_GATE_STEP_NAME,
                    POLICY_GATE_OUTPUTS["handoff"],
                ),
            ),
        ],
    )
    step_policy = ProcessingStep(
        name=POLICY_GATE_STEP_NAME,
        step_args=policy_gate_args,
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            bucket_param,
            dataset_collection_id_param,
            dataset_id_param,
            dataset_name_param,
            random_seed_param,
            validation_months_param,
            test_months_param,
            training_job_prefix_param,
            instance_type_param,
        ],
        steps=[step_prepare, step_train, step_evaluate, step_policy],
        sagemaker_session=pipeline_session,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_artifact(model_artifact_input: Path):
    import joblib

    artifact_root = model_artifact_input
    if artifact_root.is_file():
        archive_path = artifact_root
        extract_dir = artifact_root.parent / "_extracted_model"
    else:
        archive_candidates = sorted(artifact_root.rglob("*.tar.gz"))
        if not archive_candidates:
            raise FileNotFoundError(
                f"Could not find a SageMaker model artifact under {artifact_root}."
            )
        archive_path = archive_candidates[0]
        extract_dir = artifact_root / "_extracted_model"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        try:
            archive.extractall(extract_dir, filter="data")
        except TypeError:
            archive.extractall(extract_dir)

    model_path = extract_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected model.joblib in extracted artifact at {model_path}."
        )

    model = joblib.load(model_path)
    hyperparameters_path = extract_dir / "hyperparameters.json"
    training_metrics_path = extract_dir / "training_metrics.json"
    return (
        model,
        hyperparameters_path if hyperparameters_path.exists() else None,
        training_metrics_path if training_metrics_path.exists() else None,
    )


def _resolve_runtime_config(
    *,
    dataset_collection_id: str | None = None,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    random_seed: int | None = None,
    validation_months: int | None = None,
    test_months: int | None = None,
) -> RuntimeConfig:
    from dataclasses import replace

    load_repo_env()
    base = RuntimeConfig.from_env()
    return replace(
        base,
        dataset_collection_id=dataset_collection_id or base.dataset_collection_id,
        dataset_id=dataset_id or base.dataset_id,
        dataset_name=dataset_name or base.dataset_name,
        random_seed=base.random_seed if random_seed is None else int(random_seed),
        validation_months=(
            base.validation_months
            if validation_months is None
            else int(validation_months)
        ),
        test_months=base.test_months if test_months is None else int(test_months),
    )


def _ephemeral_project_paths(base_dir: Path) -> ProjectPaths:
    root = base_dir.resolve()
    return ProjectPaths(
        repo_root=root,
        source_dir=root / "src",
        data_dir=root / "data",
        cache_dir=root / "data" / "cache",
        processed_dir=root / "data" / "processed",
        artifacts_dir=root / "artifacts",
        notebooks_dir=root,
    )


def run_prepare_data_step(
    *,
    train_dir: Path,
    validation_dir: Path,
    test_dir: Path,
    metadata_dir: Path,
    dataset_collection_id: str | None = None,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    random_seed: int | None = None,
    validation_months: int | None = None,
    test_months: int | None = None,
    force_snapshot_refresh: bool = False,
) -> dict[str, Any]:
    runtime_config = _resolve_runtime_config(
        dataset_collection_id=dataset_collection_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        random_seed=random_seed,
        validation_months=validation_months,
        test_months=test_months,
    )
    project_paths = _ephemeral_project_paths(metadata_dir / "_runtime")
    project_paths.ensure_local_dirs()

    snapshot = load_or_download_snapshot(
        project_paths,
        runtime_config,
        force=force_snapshot_refresh,
    )
    raw_frame = load_raw_resale_frame(snapshot)
    split = chronological_split(
        raw_frame,
        validation_months=runtime_config.validation_months,
        test_months=runtime_config.test_months,
    )

    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(train_dir / "train.csv", index=False)
    split.validation.to_csv(validation_dir / "validation.csv", index=False)
    split.test.to_csv(test_dir / "test.csv", index=False)
    _write_json(metadata_dir / "dataset_snapshot.json", snapshot.to_metadata())
    _write_json(metadata_dir / "split_summary.json", split.summary)
    metadata = {
        "dataset_snapshot": snapshot.to_metadata(),
        "split_summary": split.summary,
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
    }
    _write_json(metadata_dir / "prepare_data_metadata.json", metadata)
    return metadata


def _load_step_metadata(metadata_input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _read_json(metadata_input_dir / "dataset_snapshot.json"),
        _read_json(metadata_input_dir / "split_summary.json"),
    )


def run_evaluate_register_step(
    *,
    model_artifact_input: Path,
    validation_input: Path,
    test_input: Path,
    metadata_input_dir: Path,
    registration_dir: Path,
    evaluation_dir: Path,
    training_job_name: str | None = None,
    model_artifact_s3_uri: str | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    import pandas as pd

    runtime_config = _resolve_runtime_config(random_seed=random_seed)
    require_pipeline_mlflow_tracking_uri(runtime_config)

    model, hyperparameters_path, training_metrics_path = _load_model_artifact(
        model_artifact_input
    )
    dataset_snapshot, split_summary = _load_step_metadata(metadata_input_dir)

    validation_raw = pd.read_csv(validation_input)
    test_raw = pd.read_csv(test_input)
    prepared_validation = prepare_training_frame(validation_raw)
    prepared_test = prepare_training_frame(test_raw)

    validation_evaluation = evaluate_model(model, prepared_validation)
    test_evaluation = evaluate_model(model, prepared_test)

    if hyperparameters_path is not None:
        hyperparameters = _read_json(hyperparameters_path)
    else:
        hyperparameters = {"random_seed": runtime_config.random_seed}
    training_metrics = (
        _read_json(training_metrics_path) if training_metrics_path is not None else None
    )
    training_job_metadata = {
        "training_job_name": training_job_name,
        "model_artifact_s3_uri": model_artifact_s3_uri,
        "training_job_validation_metrics": training_metrics,
    }

    registration_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    registration = log_and_register_candidate_model(
        model=model,
        validation_evaluation=validation_evaluation,
        test_evaluation=test_evaluation,
        runtime_config=runtime_config,
        artifact_dir=evaluation_dir,
        dataset_snapshot=dataset_snapshot,
        split_summary=split_summary,
        hyperparameters=hyperparameters,
        training_job_metadata=training_job_metadata,
    )

    registration_payload = {
        "run_id": registration.run_id,
        "model_name": registration.model_name,
        "model_version": registration.model_version,
        "model_uri": registration.model_uri,
        "training_job_name": training_job_name,
        "model_artifact_s3_uri": model_artifact_s3_uri,
        "hyperparameters": hyperparameters,
        "validation_metrics": validation_evaluation.overall_metrics,
        "test_metrics": test_evaluation.overall_metrics,
        "training_job_validation_metrics": training_metrics,
    }
    _write_json(registration_dir / "registration.json", registration_payload)
    return registration_payload


def _load_candidate_segment_metrics(evaluation_dir: Path) -> dict[str, Any]:
    return {
        "town": _read_json(evaluation_dir / "test_segments_by_town.json"),
        "flat_type": _read_json(evaluation_dir / "test_segments_by_flat_type.json"),
    }


def _policy_verdict_payload(policy_verdict: PolicyVerdict) -> dict[str, Any]:
    return {
        "decision": policy_verdict.decision.value,
        "reasons": list(policy_verdict.reasons),
        "checks_passed": list(policy_verdict.checks_passed),
        "checks_failed": list(policy_verdict.checks_failed),
    }


def _drift_report_payload(drift_report: DriftReport | None) -> dict[str, Any] | None:
    if drift_report is None:
        return None
    return {
        "overall_drift_detected": drift_report.overall_drift_detected,
        "column_results": [
            {
                "column": result.column,
                "drift_type": result.drift_type,
                "statistic": float(result.statistic),
                "threshold": float(result.threshold),
                "p_value": (
                    None if result.p_value is None else float(result.p_value)
                ),
                "is_drifted": bool(result.is_drifted),
            }
            for result in drift_report.column_results
        ],
    }


def _decision_metadata_for_pipeline_reject(policy_verdict: PolicyVerdict) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "decision_source": "pipeline_policy_auto_reject",
        "decision_timestamp": datetime.now(timezone.utc).isoformat(),
        "policy_verdict": policy_verdict.decision.value,
        "rejection_overridden": False,
    }
    if policy_verdict.reasons:
        metadata["policy_reasons"] = "; ".join(policy_verdict.reasons)
    return metadata


def run_policy_gate_step(
    *,
    train_input: Path,
    test_input: Path,
    registration_input: Path,
    evaluation_input: Path,
    policy_dir: Path,
    handoff_dir: Path,
    pipeline_execution_id: str | None = None,
) -> dict[str, Any]:
    import pandas as pd

    runtime_config = _resolve_runtime_config()
    require_pipeline_mlflow_tracking_uri(runtime_config)

    registration = _read_json(registration_input)
    candidate_metrics = _read_json(evaluation_input / "test_metrics.json")
    candidate_segment_metrics = _load_candidate_segment_metrics(evaluation_input)

    evidence_errors: list[str] = []
    try:
        from hdb_resale_mlops.mlflow_registry import get_champion_version

        champion_info = get_champion_version(registration["model_name"])
    except MlflowRegistryError as exc:
        champion_info = None
        evidence_errors.append(
            f"Promotion blocked because champion evidence could not be loaded from MLflow: {exc}"
        )

    comparison = compare_models(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_info["metrics"] if champion_info else None,
        candidate_segment_metrics=candidate_segment_metrics,
        champion_segment_metrics=(
            champion_info.get("segment_metrics") if champion_info else None
        ),
    )

    prepared_train = prepare_training_frame(pd.read_csv(train_input))
    prepared_test = prepare_training_frame(pd.read_csv(test_input))
    drift_report = run_drift_checks(
        train_df=prepared_train,
        test_df=prepared_test,
        categorical_columns=CATEGORICAL_FEATURES,
        numeric_columns=NUMERIC_FEATURES,
    )
    policy_verdict = evaluate_policy(
        candidate_metrics=candidate_metrics,
        comparison=comparison,
        drift_report=drift_report,
        evidence_errors=evidence_errors,
    )

    if policy_verdict.decision == PolicyDecision.REJECT:
        reject_candidate(
            registration["model_name"],
            registration["model_version"],
            policy_verdict.reasons or ["Rejected by pipeline policy gate"],
            decision_metadata=_decision_metadata_for_pipeline_reject(policy_verdict),
        )

    now = datetime.now(timezone.utc).isoformat()
    policy_payload = {
        "created_at": now,
        "pipeline_execution_id": pipeline_execution_id,
        "model_name": registration["model_name"],
        "model_version": registration["model_version"],
        "candidate_metrics": candidate_metrics,
        "policy_verdict": _policy_verdict_payload(policy_verdict),
    }
    handoff_payload = {
        "created_at": now,
        "source": "sagemaker_pipeline_policy_gate",
        "pipeline_execution_id": pipeline_execution_id,
        "status": (
            PIPELINE_STATUS_AUTO_REJECTED
            if policy_verdict.decision == PolicyDecision.REJECT
            else PIPELINE_STATUS_PENDING_REVIEW
        ),
        "registration": registration,
        "candidate_metrics": candidate_metrics,
        "candidate_segment_metrics": candidate_segment_metrics,
        "champion_info": serialize_for_state(champion_info),
        "comparison": serialize_for_state(asdict(comparison)),
        "drift_report": _drift_report_payload(drift_report),
        "policy_verdict": _policy_verdict_payload(policy_verdict),
    }

    policy_dir.mkdir(parents=True, exist_ok=True)
    handoff_dir.mkdir(parents=True, exist_ok=True)
    _write_json(policy_dir / "policy_verdict.json", policy_payload)
    _write_json(handoff_dir / "review_handoff.json", handoff_payload)
    return handoff_payload
