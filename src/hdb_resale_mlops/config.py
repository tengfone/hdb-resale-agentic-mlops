from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Mapping

from hdb_resale_mlops.env import load_repo_env

DEFAULT_COLLECTION_ID = "189"
DEFAULT_DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
DEFAULT_DATASET_NAME = "hdb-resale-prices-2017-onward"
DEFAULT_MODEL_NAME = "hdb-resale-price-regressor"
DEFAULT_EXPERIMENT_NAME = "hdb-resale-candidate"
DEFAULT_AWS_REGION = "ap-southeast-1"
DEFAULT_LOCAL_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def _maestro_proxies_from_env() -> dict[str, str] | None:
    http_proxy = os.getenv("MAESTRO_HTTP_PROXY")
    https_proxy = os.getenv("MAESTRO_HTTPS_PROXY")
    proxies = {
        key: value
        for key, value in {
            "http": http_proxy,
            "https": https_proxy,
        }.items()
        if value
    }
    return proxies or None


def _training_pip_index_url_from_env() -> str | None:
    return os.getenv("TRAINING_PIP_INDEX_URL") or os.getenv("PIP_INDEX_URL")


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root from the current path."
    )


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    source_dir: Path
    data_dir: Path
    cache_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    notebooks_dir: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        repo_root = discover_repo_root()
        return cls(
            repo_root=repo_root,
            source_dir=repo_root / "src",
            data_dir=repo_root / "data",
            cache_dir=repo_root / "data" / "cache",
            processed_dir=repo_root / "data" / "processed",
            artifacts_dir=repo_root / "artifacts",
            notebooks_dir=repo_root,
        )

    def ensure_local_dirs(self) -> None:
        for path in (
            self.cache_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.notebooks_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RuntimeConfig:
    aws_region: str = DEFAULT_AWS_REGION
    sagemaker_role_arn: str | None = None
    s3_bucket: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: str | None = None
    mlflow_model_name: str = DEFAULT_MODEL_NAME
    mlflow_experiment_name: str = DEFAULT_EXPERIMENT_NAME
    dataset_collection_id: str = DEFAULT_COLLECTION_ID
    dataset_id: str = DEFAULT_DATASET_ID
    dataset_name: str = DEFAULT_DATASET_NAME
    data_gov_api_key: str | None = None
    maestro_proxies: Mapping[str, str] | None = None
    training_pip_index_url: str | None = None
    random_seed: int = 7
    validation_months: int = 12
    test_months: int = 12
    sagemaker_framework_version: str = "1.4-2"
    sagemaker_instance_type: str = "ml.m5.xlarge"
    sagemaker_instance_count: int = 1
    sagemaker_max_run_seconds: int = 3600
    training_job_prefix: str = "hdb-resale"

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        load_repo_env()
        return cls(
            aws_region=os.getenv("AWS_REGION", DEFAULT_AWS_REGION),
            sagemaker_role_arn=os.getenv("SAGEMAKER_ROLE_ARN"),
            s3_bucket=os.getenv("S3_BUCKET"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            mlflow_tracking_username=os.getenv("MLFLOW_TRACKING_USERNAME"),
            mlflow_tracking_password=os.getenv("MLFLOW_TRACKING_PASSWORD"),
            mlflow_model_name=os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME),
            mlflow_experiment_name=os.getenv(
                "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME
            ),
            dataset_collection_id=os.getenv(
                "HDB_DATA_COLLECTION_ID", DEFAULT_COLLECTION_ID
            ),
            dataset_id=os.getenv("HDB_DATASET_ID", DEFAULT_DATASET_ID),
            dataset_name=os.getenv("HDB_DATASET_NAME", DEFAULT_DATASET_NAME),
            data_gov_api_key=os.getenv("DATA_GOV_API_KEY"),
            maestro_proxies=_maestro_proxies_from_env(),
            training_pip_index_url=_training_pip_index_url_from_env(),
            random_seed=int(os.getenv("RANDOM_SEED", "7")),
            validation_months=int(os.getenv("VALIDATION_MONTHS", "12")),
            test_months=int(os.getenv("TEST_MONTHS", "12")),
            sagemaker_framework_version=os.getenv(
                "SAGEMAKER_FRAMEWORK_VERSION", "1.4-2"
            ),
            sagemaker_instance_type=os.getenv(
                "SAGEMAKER_INSTANCE_TYPE", "ml.m5.xlarge"
            ),
            sagemaker_instance_count=int(os.getenv("SAGEMAKER_INSTANCE_COUNT", "1")),
            sagemaker_max_run_seconds=int(
                os.getenv("SAGEMAKER_MAX_RUN_SECONDS", "3600")
            ),
            training_job_prefix=os.getenv("TRAINING_JOB_PREFIX", "hdb-resale"),
        )

    def require_sagemaker_role(self) -> str:
        if not self.sagemaker_role_arn:
            raise ValueError(
                "SAGEMAKER_ROLE_ARN is required to launch the training job."
            )
        return self.sagemaker_role_arn

    def require_s3_bucket(self) -> str:
        if not self.s3_bucket:
            raise ValueError(
                "S3_BUCKET is required to stage training data and model artifacts."
            )
        return self.s3_bucket

    def require_mlflow_tracking_uri(self) -> str:
        if not self.mlflow_tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI is required to log the run and register the model."
            )
        return self.mlflow_tracking_uri

    def resolved_mlflow_tracking_uri(self) -> str:
        return self.mlflow_tracking_uri or DEFAULT_LOCAL_MLFLOW_TRACKING_URI
