import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from hdb_resale_mlops.config import ProjectPaths, RuntimeConfig
from hdb_resale_mlops.sagemaker_job import launch_training_job


class TestSageMakerJobEnvForwarding(unittest.TestCase):
    @patch("hdb_resale_mlops.sagemaker_job.load_repo_env", return_value=None)
    @patch("hdb_resale_mlops.sagemaker_job._resolve_sklearn_image_uri", return_value="123456.dkr.ecr/image:latest")
    def test_launch_training_job_forwards_project_env_to_estimator(
        self,
        _mock_image_uri,
        _mock_load_env,
    ):
        recorded: dict[str, object] = {}

        class _FakeTrainingInput:
            def __init__(self, s3_uri, content_type=None):
                self.s3_uri = s3_uri
                self.content_type = content_type

        class _FakeSKLearn:
            def __init__(self, **kwargs):
                recorded["estimator_kwargs"] = kwargs
                self.model_data = "s3://bucket/output/model.tar.gz"

            def fit(self, inputs, wait, logs, job_name):
                recorded["fit_inputs"] = inputs
                recorded["fit_wait"] = wait
                recorded["fit_logs"] = logs
                recorded["job_name"] = job_name

        class _FakeSageMakerSession:
            def __init__(self, boto_session=None):
                recorded["boto_session"] = boto_session

            def upload_data(self, path, bucket, key_prefix):
                return f"s3://{bucket}/{key_prefix}/{Path(path).name}"

        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.Session = lambda region_name=None: {"region_name": region_name}
        fake_boto3.client = lambda service_name, region_name=None: None

        fake_sagemaker = types.ModuleType("sagemaker")
        fake_sagemaker.Session = _FakeSageMakerSession

        fake_sagemaker_inputs = types.ModuleType("sagemaker.inputs")
        fake_sagemaker_inputs.TrainingInput = _FakeTrainingInput

        fake_sagemaker_sklearn = types.ModuleType("sagemaker.sklearn")
        fake_sagemaker_sklearn_estimator = types.ModuleType("sagemaker.sklearn.estimator")
        fake_sagemaker_sklearn_estimator.SKLearn = _FakeSKLearn

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            sys.modules,
            {
                "boto3": fake_boto3,
                "sagemaker": fake_sagemaker,
                "sagemaker.inputs": fake_sagemaker_inputs,
                "sagemaker.sklearn": fake_sagemaker_sklearn,
                "sagemaker.sklearn.estimator": fake_sagemaker_sklearn_estimator,
            },
        ), patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test",
                "OPENAI_BASE_URL": "https://proxy.example.com/v1",
                "OPENAI_MODEL": "gpt-5-nano",
                "ENABLE_JUDGE_EVAL": "true",
                "MARKET_RESEARCH_PROVIDER": "both",
                "TAVILY_API_KEY": "tvly-test",
                "MAESTRO_HTTP_PROXY": "http://maestro-proxy:8080",
                "MAESTRO_HTTPS_PROXY": "http://maestro-proxy:8443",
                "MLFLOW_TRACKING_URI": "https://mlflow.example.com",
                "MODEL_REVIEWER": "alice",
            },
            clear=True,
        ):
            tmp_path = Path(tmpdir)
            train_csv = tmp_path / "train.csv"
            validation_csv = tmp_path / "validation.csv"
            train_csv.write_text("x\n1\n", encoding="utf-8")
            validation_csv.write_text("x\n2\n", encoding="utf-8")

            runtime_config = RuntimeConfig(
                aws_region="ap-southeast-1",
                sagemaker_role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
                s3_bucket="demo-bucket",
                training_pip_index_url="https://pypi.example/simple/",
            )
            project_paths = ProjectPaths(
                repo_root=tmp_path,
                source_dir=tmp_path / "src",
                data_dir=tmp_path / "data",
                cache_dir=tmp_path / "data" / "cache",
                processed_dir=tmp_path / "data" / "processed",
                artifacts_dir=tmp_path / "artifacts",
                notebooks_dir=tmp_path,
            )

            result = launch_training_job(
                split_paths={"train": train_csv, "validation": validation_csv},
                runtime_config=runtime_config,
                project_paths=project_paths,
                wait=False,
            )

        environment = recorded["estimator_kwargs"]["environment"]
        self.assertEqual(environment["AWS_DEFAULT_REGION"], "ap-southeast-1")
        self.assertEqual(environment["AWS_REGION"], "ap-southeast-1")
        self.assertEqual(environment["OPENAI_API_KEY"], "sk-test")
        self.assertEqual(environment["OPENAI_BASE_URL"], "https://proxy.example.com/v1")
        self.assertEqual(environment["OPENAI_API_BASE"], "https://proxy.example.com/v1")
        self.assertEqual(environment["OPENAI_MODEL"], "gpt-5-nano")
        self.assertEqual(environment["ENABLE_JUDGE_EVAL"], "true")
        self.assertEqual(environment["MARKET_RESEARCH_PROVIDER"], "both")
        self.assertEqual(environment["TAVILY_API_KEY"], "tvly-test")
        self.assertEqual(environment["MAESTRO_HTTP_PROXY"], "http://maestro-proxy:8080")
        self.assertEqual(environment["MAESTRO_HTTPS_PROXY"], "http://maestro-proxy:8443")
        self.assertNotIn("HTTP_PROXY", environment)
        self.assertNotIn("HTTPS_PROXY", environment)
        self.assertEqual(environment["MLFLOW_TRACKING_URI"], "https://mlflow.example.com")
        self.assertEqual(environment["MODEL_REVIEWER"], "alice")
        self.assertEqual(environment["PIP_INDEX_URL"], "https://pypi.example/simple/")
        self.assertNotIn("PIP_TRUSTED_HOST", environment)
        self.assertTrue(result.output_s3_uri.startswith("s3://demo-bucket/hdb-resale/"))
        self.assertFalse(recorded["fit_wait"])
        self.assertFalse(recorded["fit_logs"])

    @patch("hdb_resale_mlops.sagemaker_job.load_repo_env", return_value=None)
    @patch("hdb_resale_mlops.sagemaker_job._resolve_sklearn_image_uri", return_value="123456.dkr.ecr/image:latest")
    def test_launch_training_job_only_streams_logs_when_requested(
        self,
        _mock_image_uri,
        _mock_load_env,
    ):
        recorded: dict[str, object] = {}

        class _FakeTrainingInput:
            def __init__(self, s3_uri, content_type=None):
                self.s3_uri = s3_uri
                self.content_type = content_type

        class _FakeSKLearn:
            def __init__(self, **kwargs):
                recorded["estimator_kwargs"] = kwargs
                self.model_data = "s3://bucket/output/model.tar.gz"

            def fit(self, inputs, wait, logs, job_name):
                recorded["fit_wait"] = wait
                recorded["fit_logs"] = logs

        class _FakeSageMakerSession:
            def __init__(self, boto_session=None):
                pass

            def upload_data(self, path, bucket, key_prefix):
                return f"s3://{bucket}/{key_prefix}/{Path(path).name}"

        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.Session = lambda region_name=None: {"region_name": region_name}
        fake_boto3.client = lambda service_name, region_name=None: None

        fake_sagemaker = types.ModuleType("sagemaker")
        fake_sagemaker.Session = _FakeSageMakerSession

        fake_sagemaker_inputs = types.ModuleType("sagemaker.inputs")
        fake_sagemaker_inputs.TrainingInput = _FakeTrainingInput

        fake_sagemaker_sklearn = types.ModuleType("sagemaker.sklearn")
        fake_sagemaker_sklearn_estimator = types.ModuleType("sagemaker.sklearn.estimator")
        fake_sagemaker_sklearn_estimator.SKLearn = _FakeSKLearn

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            sys.modules,
            {
                "boto3": fake_boto3,
                "sagemaker": fake_sagemaker,
                "sagemaker.inputs": fake_sagemaker_inputs,
                "sagemaker.sklearn": fake_sagemaker_sklearn,
                "sagemaker.sklearn.estimator": fake_sagemaker_sklearn_estimator,
            },
        ):
            tmp_path = Path(tmpdir)
            train_csv = tmp_path / "train.csv"
            validation_csv = tmp_path / "validation.csv"
            train_csv.write_text("x\n1\n", encoding="utf-8")
            validation_csv.write_text("x\n2\n", encoding="utf-8")

            runtime_config = RuntimeConfig(
                aws_region="ap-southeast-1",
                sagemaker_role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
                s3_bucket="demo-bucket",
            )
            project_paths = ProjectPaths(
                repo_root=tmp_path,
                source_dir=tmp_path / "src",
                data_dir=tmp_path / "data",
                cache_dir=tmp_path / "data" / "cache",
                processed_dir=tmp_path / "data" / "processed",
                artifacts_dir=tmp_path / "artifacts",
                notebooks_dir=tmp_path,
            )

            launch_training_job(
                split_paths={"train": train_csv, "validation": validation_csv},
                runtime_config=runtime_config,
                project_paths=project_paths,
                wait=True,
                stream_logs=True,
            )

        self.assertTrue(recorded["fit_wait"])
        self.assertTrue(recorded["fit_logs"])

    @patch("hdb_resale_mlops.sagemaker_job.load_repo_env", return_value=None)
    @patch("hdb_resale_mlops.sagemaker_job._resolve_sklearn_image_uri", return_value="123456.dkr.ecr/image:latest")
    @patch("hdb_resale_mlops.sagemaker_job.time.sleep", return_value=None)
    def test_launch_training_job_polls_directly_when_waiting_without_logs(
        self,
        _mock_sleep,
        _mock_image_uri,
        _mock_load_env,
    ):
        recorded: dict[str, object] = {}

        class _FakeTrainingInput:
            def __init__(self, s3_uri, content_type=None):
                self.s3_uri = s3_uri
                self.content_type = content_type

        class _FakeSKLearn:
            def __init__(self, **kwargs):
                self.model_data = "s3://bucket/output/model.tar.gz"

            def fit(self, inputs, wait, logs, job_name):
                recorded["fit_wait"] = wait
                recorded["fit_logs"] = logs
                recorded["job_name"] = job_name

        class _FakeSageMakerSession:
            def __init__(self, boto_session=None):
                pass

            def upload_data(self, path, bucket, key_prefix):
                return f"s3://{bucket}/{key_prefix}/{Path(path).name}"

        class _FakeSageMakerClient:
            def __init__(self):
                self.calls = 0

            def describe_training_job(self, TrainingJobName):
                self.calls += 1
                recorded.setdefault("describe_job_names", []).append(TrainingJobName)
                if self.calls == 1:
                    return {
                        "TrainingJobStatus": "InProgress",
                        "SecondaryStatus": "Training",
                    }
                return {
                    "TrainingJobStatus": "Completed",
                    "SecondaryStatus": "Completed",
                    "ModelArtifacts": {
                        "S3ModelArtifacts": "s3://demo-bucket/output/model.tar.gz"
                    },
                }

        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.Session = lambda region_name=None: {"region_name": region_name}
        fake_boto3.client = lambda service_name, region_name=None: _FakeSageMakerClient()

        fake_sagemaker = types.ModuleType("sagemaker")
        fake_sagemaker.Session = _FakeSageMakerSession

        fake_sagemaker_inputs = types.ModuleType("sagemaker.inputs")
        fake_sagemaker_inputs.TrainingInput = _FakeTrainingInput

        fake_sagemaker_sklearn = types.ModuleType("sagemaker.sklearn")
        fake_sagemaker_sklearn_estimator = types.ModuleType("sagemaker.sklearn.estimator")
        fake_sagemaker_sklearn_estimator.SKLearn = _FakeSKLearn

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            sys.modules,
            {
                "boto3": fake_boto3,
                "sagemaker": fake_sagemaker,
                "sagemaker.inputs": fake_sagemaker_inputs,
                "sagemaker.sklearn": fake_sagemaker_sklearn,
                "sagemaker.sklearn.estimator": fake_sagemaker_sklearn_estimator,
            },
        ):
            tmp_path = Path(tmpdir)
            train_csv = tmp_path / "train.csv"
            validation_csv = tmp_path / "validation.csv"
            train_csv.write_text("x\n1\n", encoding="utf-8")
            validation_csv.write_text("x\n2\n", encoding="utf-8")

            runtime_config = RuntimeConfig(
                aws_region="ap-southeast-1",
                sagemaker_role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
                s3_bucket="demo-bucket",
            )
            project_paths = ProjectPaths(
                repo_root=tmp_path,
                source_dir=tmp_path / "src",
                data_dir=tmp_path / "data",
                cache_dir=tmp_path / "data" / "cache",
                processed_dir=tmp_path / "data" / "processed",
                artifacts_dir=tmp_path / "artifacts",
                notebooks_dir=tmp_path,
            )

            result = launch_training_job(
                split_paths={"train": train_csv, "validation": validation_csv},
                runtime_config=runtime_config,
                project_paths=project_paths,
                wait=True,
                stream_logs=False,
            )

        self.assertFalse(recorded["fit_wait"])
        self.assertFalse(recorded["fit_logs"])
        self.assertEqual(result.model_artifact_s3_uri, "s3://demo-bucket/output/model.tar.gz")


if __name__ == "__main__":
    unittest.main()
