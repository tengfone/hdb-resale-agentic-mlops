"""
SageMaker-specific training launcher and artifact downloader.

The enterprise notebook does not train the model inline. Instead it:
- prepares train/validation CSVs locally
- uploads them to S3
- starts a SageMaker scikit-learn script-mode job
- waits for that job to produce `model.joblib`
- downloads the artifact back into the notebook environment

After that download, the rest of the workflow becomes local again:
evaluation, MLflow registration, and promotion review all continue from the
notebook process rather than inside SageMaker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tarfile
import time
from typing import Any, Mapping

from hdb_resale_mlops.config import ProjectPaths, RuntimeConfig
from hdb_resale_mlops.data import s3_uri_parts
from hdb_resale_mlops.env import collect_sagemaker_forwarded_env, load_repo_env


def _resolve_sklearn_image_uri(
    region: str, framework_version: str, instance_type: str
) -> str:
    """
    Resolve the ECR image URI for a SageMaker sklearn container.
    """
    import sagemaker

    try:
        return sagemaker.image_uris.retrieve(
            "sklearn",
            region,
            version=framework_version,
            instance_type=instance_type,
        )
    except Exception:
        known_uri = sagemaker.image_uris.retrieve(
            "sklearn",
            region,
            version="1.2-1",
            instance_type=instance_type,
        )
        return known_uri.replace("1.2-1", framework_version)


@dataclass(frozen=True)
class TrainingJobResult:
    training_job_name: str
    model_artifact_s3_uri: str
    output_s3_uri: str
    train_input_s3_uri: str
    validation_input_s3_uri: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "training_job_name": self.training_job_name,
            "model_artifact_s3_uri": self.model_artifact_s3_uri,
            "output_s3_uri": self.output_s3_uri,
            "train_input_s3_uri": self.train_input_s3_uri,
            "validation_input_s3_uri": self.validation_input_s3_uri,
        }


@dataclass(frozen=True)
class DownloadedModel:
    archive_path: Path
    extract_dir: Path
    model_path: Path
    training_metrics_path: Path | None


def _wait_for_training_job(
    *,
    sagemaker_client,
    training_job_name: str,
    poll_seconds: int = 15,
) -> str:
    """Poll SageMaker directly when notebook log streaming is disabled.

    This avoids the SDK's blocking log/wait path, which can leave some
    notebook sessions attached even after the job has already completed.
    Returns the model artifact S3 URI for the completed job.
    """

    terminal_states = {"Completed", "Failed", "Stopped", "Stopping"}
    last_status: str | None = None
    last_secondary_status: str | None = None

    print(f"Submitted SageMaker training job: {training_job_name}")
    while True:
        description = sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        status = str(description.get("TrainingJobStatus", "Unknown"))
        secondary_status = str(description.get("SecondaryStatus", ""))
        if status != last_status or secondary_status != last_secondary_status:
            suffix = f" ({secondary_status})" if secondary_status else ""
            print(f"SageMaker training job status: {status}{suffix}")
            last_status = status
            last_secondary_status = secondary_status

        if status == "Completed":
            model_artifacts = description.get("ModelArtifacts") or {}
            model_artifact_s3_uri = model_artifacts.get("S3ModelArtifacts")
            if not model_artifact_s3_uri:
                raise RuntimeError(
                    f"SageMaker training job {training_job_name!r} completed without a model artifact URI."
                )
            return str(model_artifact_s3_uri)
        if status in terminal_states:
            failure_reason = description.get("FailureReason") or "Unknown failure"
            raise RuntimeError(
                f"SageMaker training job {training_job_name!r} ended with status {status!r}: {failure_reason}"
            )

        time.sleep(poll_seconds)


def launch_training_job(
    split_paths: Mapping[str, Path],
    runtime_config: RuntimeConfig,
    project_paths: ProjectPaths,
    hyperparameters: Mapping[str, Any] | None = None,
    wait: bool = True,
    stream_logs: bool = False,
) -> TrainingJobResult:
    import boto3
    import sagemaker
    from sagemaker.inputs import TrainingInput
    from sagemaker.sklearn.estimator import SKLearn

    load_repo_env()
    boto_session = boto3.Session(region_name=runtime_config.aws_region)
    sagemaker_client = boto3.client("sagemaker", region_name=runtime_config.aws_region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    bucket = runtime_config.require_s3_bucket()
    key_prefix = f"{runtime_config.training_job_prefix}/{timestamp}"

    # The notebook persists split CSVs first, then stages them to S3 as
    # separate SageMaker input channels.
    train_input_s3_uri = sagemaker_session.upload_data(
        path=str(split_paths["train"]),
        bucket=bucket,
        key_prefix=f"{key_prefix}/input/train",
    )
    validation_input_s3_uri = sagemaker_session.upload_data(
        path=str(split_paths["validation"]),
        bucket=bucket,
        key_prefix=f"{key_prefix}/input/validation",
    )
    output_s3_uri = f"s3://{bucket}/{key_prefix}/output"
    job_name = f"{runtime_config.training_job_prefix}-{timestamp}"
    estimator_environment = collect_sagemaker_forwarded_env()
    estimator_environment["AWS_DEFAULT_REGION"] = runtime_config.aws_region
    estimator_environment["AWS_REGION"] = runtime_config.aws_region
    if runtime_config.training_pip_index_url:
        estimator_environment["PIP_INDEX_URL"] = runtime_config.training_pip_index_url

    image_uri = _resolve_sklearn_image_uri(
        runtime_config.aws_region,
        runtime_config.sagemaker_framework_version,
        runtime_config.sagemaker_instance_type,
    )

    # SageMaker runs src/train.py inside the container. That file simply calls
    # the shared package entrypoint, so local and remote training stay aligned.
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(project_paths.source_dir),
        role=runtime_config.require_sagemaker_role(),
        instance_type=runtime_config.sagemaker_instance_type,
        instance_count=runtime_config.sagemaker_instance_count,
        image_uri=image_uri,
        output_path=output_s3_uri,
        max_run=runtime_config.sagemaker_max_run_seconds,
        base_job_name=runtime_config.training_job_prefix,
        hyperparameters=dict(hyperparameters or {}),
        sagemaker_session=sagemaker_session,
        environment=estimator_environment,
    )
    show_logs = bool(wait and stream_logs)
    fit_wait = bool(wait and stream_logs)
    estimator.fit(
        inputs={
            "train": TrainingInput(train_input_s3_uri, content_type="text/csv"),
            "validation": TrainingInput(
                validation_input_s3_uri, content_type="text/csv"
            ),
        },
        wait=fit_wait,
        logs=show_logs,
        job_name=job_name,
    )
    if wait and not stream_logs:
        model_artifact_s3_uri = _wait_for_training_job(
            sagemaker_client=sagemaker_client,
            training_job_name=job_name,
        )
    else:
        model_artifact_s3_uri = estimator.model_data

    return TrainingJobResult(
        training_job_name=job_name,
        model_artifact_s3_uri=model_artifact_s3_uri,
        output_s3_uri=output_s3_uri,
        train_input_s3_uri=train_input_s3_uri,
        validation_input_s3_uri=validation_input_s3_uri,
    )


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise RuntimeError(
                f"Refusing to extract unsafe archive member: {member.name}"
            )
    archive.extractall(destination)


def download_model_artifact(
    model_artifact_s3_uri: str,
    runtime_config: RuntimeConfig,
    destination_dir: Path,
) -> DownloadedModel:
    import boto3

    destination_dir.mkdir(parents=True, exist_ok=True)
    bucket, key = s3_uri_parts(model_artifact_s3_uri)
    archive_path = destination_dir / Path(key).name
    extract_dir = destination_dir / archive_path.name.removesuffix(".tar.gz")

    s3_client = boto3.client("s3", region_name=runtime_config.aws_region)
    s3_client.download_file(bucket, key, str(archive_path))

    with tarfile.open(archive_path) as archive:
        extract_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract(archive, extract_dir)

    # The notebook later loads this joblib pipeline and evaluates it on the
    # local validation/test splits before talking to MLflow.
    model_path = extract_dir / "model.joblib"
    training_metrics_path = extract_dir / "training_metrics.json"
    return DownloadedModel(
        archive_path=archive_path,
        extract_dir=extract_dir,
        model_path=model_path,
        training_metrics_path=(
            training_metrics_path if training_metrics_path.exists() else None
        ),
    )
