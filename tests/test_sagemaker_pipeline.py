from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd

from hdb_resale_mlops.config import ProjectPaths, RuntimeConfig
from hdb_resale_mlops.data import DatasetSnapshot
from hdb_resale_mlops.sagemaker_pipeline import (
    EVALUATE_REGISTER_STEP_NAME,
    POLICY_GATE_STEP_NAME,
    PREPARE_DATA_OUTPUTS,
    PREPARE_DATA_STEP_NAME,
    TRAIN_CANDIDATE_STEP_NAME,
    build_sagemaker_pipeline,
    run_evaluate_register_step,
    run_policy_gate_step,
    run_prepare_data_step,
)


class DummyModel:
    def predict(self, frame):
        return pd.Series([500_000.0] * len(frame))


def _raw_resale_frame(months: int = 30) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(months):
        rows.append(
            {
                "month": f"{2019 + idx // 12:04d}-{(idx % 12) + 1:02d}",
                "town": "BEDOK" if idx % 2 == 0 else "ANG MO KIO",
                "flat_type": "4 ROOM" if idx % 3 else "5 ROOM",
                "flat_model": "Improved",
                "storey_range": "04 TO 06",
                "floor_area_sqm": 90.0 + idx,
                "lease_commence_date": 2000,
                "remaining_lease": "79 years 6 months",
                "resale_price": 450_000.0 + (idx * 1_000),
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_model_artifact(root: Path) -> Path:
    artifact_input = root / "model"
    artifact_input.mkdir(parents=True, exist_ok=True)
    build_dir = root / "model_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(DummyModel(), build_dir / "model.joblib")
    _write_json(build_dir / "hyperparameters.json", {"random_seed": 7})
    _write_json(build_dir / "training_metrics.json", {"rmse": 10_000.0, "mae": 8_000.0})

    archive_path = artifact_input / "model.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(build_dir / "model.joblib", arcname="model.joblib")
        archive.add(build_dir / "hyperparameters.json", arcname="hyperparameters.json")
        archive.add(build_dir / "training_metrics.json", arcname="training_metrics.json")
    return artifact_input


class _FakeJoin:
    def __init__(self, on=None, values=None):
        self.on = on
        self.values = values or []


class _FakeParameter:
    def __init__(self, name, default_value=None):
        self.name = name
        self.default_value = default_value


class _FakeProcessingInput:
    def __init__(self, source, destination=None, input_name=None):
        self.source = source
        self.destination = destination
        self.input_name = input_name


class _FakeProcessingOutput:
    def __init__(self, output_name, source=None, destination=None):
        self.output_name = output_name
        self.source = source
        self.destination = destination


class _FakeTrainingInput:
    def __init__(self, s3_data, content_type=None):
        self.s3_data = s3_data
        self.content_type = content_type


class _FakeOutputs(dict):
    pass


class _FakeProcessingStep:
    def __init__(self, name, step_args):
        self.name = name
        self.step_args = step_args
        outputs = _FakeOutputs(
            {
                output.output_name: SimpleNamespace(
                    S3Output=SimpleNamespace(S3Uri=output.destination)
                )
                for output in step_args.get("outputs", [])
            }
        )
        self.properties = SimpleNamespace(
            ProcessingOutputConfig=SimpleNamespace(Outputs=outputs)
        )


class _FakeTrainingStep:
    def __init__(self, name, step_args):
        self.name = name
        self.step_args = step_args
        self.properties = SimpleNamespace(
            ModelArtifacts=SimpleNamespace(S3ModelArtifacts="s3://dummy/model.tar.gz"),
            TrainingJobName="train-job-123",
        )


class _FakeSKLearnProcessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, **kwargs):
        return kwargs


class _FakeSKLearn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, inputs):
        return {"inputs": inputs, "estimator_kwargs": self.kwargs}


class _FakePipeline:
    def __init__(self, name, parameters, steps, sagemaker_session):
        self.name = name
        self.parameters = parameters
        self.steps = steps
        self.sagemaker_session = sagemaker_session


class TestSageMakerPipelineBuilder(unittest.TestCase):
    @patch("hdb_resale_mlops.sagemaker_pipeline._resolve_sklearn_image_uri", return_value="123456.dkr.ecr/sklearn:latest")
    def test_builds_expected_four_step_pipeline(self, _mock_image_uri):
        fake_modules = {
            "sagemaker": types.ModuleType("sagemaker"),
            "sagemaker.inputs": types.ModuleType("sagemaker.inputs"),
            "sagemaker.processing": types.ModuleType("sagemaker.processing"),
            "sagemaker.sklearn": types.ModuleType("sagemaker.sklearn"),
            "sagemaker.sklearn.estimator": types.ModuleType("sagemaker.sklearn.estimator"),
            "sagemaker.sklearn.processing": types.ModuleType("sagemaker.sklearn.processing"),
            "sagemaker.workflow": types.ModuleType("sagemaker.workflow"),
            "sagemaker.workflow.execution_variables": types.ModuleType("sagemaker.workflow.execution_variables"),
            "sagemaker.workflow.functions": types.ModuleType("sagemaker.workflow.functions"),
            "sagemaker.workflow.parameters": types.ModuleType("sagemaker.workflow.parameters"),
            "sagemaker.workflow.pipeline": types.ModuleType("sagemaker.workflow.pipeline"),
            "sagemaker.workflow.steps": types.ModuleType("sagemaker.workflow.steps"),
        }
        fake_modules["sagemaker.inputs"].TrainingInput = _FakeTrainingInput
        fake_modules["sagemaker.processing"].ProcessingInput = _FakeProcessingInput
        fake_modules["sagemaker.processing"].ProcessingOutput = _FakeProcessingOutput
        fake_modules["sagemaker.sklearn.estimator"].SKLearn = _FakeSKLearn
        fake_modules["sagemaker.sklearn.processing"].SKLearnProcessor = _FakeSKLearnProcessor
        fake_modules["sagemaker.workflow.execution_variables"].ExecutionVariables = SimpleNamespace(
            PIPELINE_EXECUTION_ID="PIPELINE_EXECUTION_ID"
        )
        fake_modules["sagemaker.workflow.functions"].Join = _FakeJoin
        fake_modules["sagemaker.workflow.parameters"].ParameterInteger = _FakeParameter
        fake_modules["sagemaker.workflow.parameters"].ParameterString = _FakeParameter
        fake_modules["sagemaker.workflow.pipeline"].Pipeline = _FakePipeline
        fake_modules["sagemaker.workflow.steps"].ProcessingStep = _FakeProcessingStep
        fake_modules["sagemaker.workflow.steps"].TrainingStep = _FakeTrainingStep

        config = RuntimeConfig(
            aws_region="ap-southeast-1",
            s3_bucket="demo-bucket",
            mlflow_tracking_uri="https://mlflow.example.com",
            sagemaker_role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        )
        paths = ProjectPaths.discover()

        with patch.dict(sys.modules, fake_modules):
            pipeline = build_sagemaker_pipeline(
                pipeline_name="hdb-resale-maestro-pipeline",
                runtime_config=config,
                project_paths=paths,
                role_arn=config.require_sagemaker_role(),
                pipeline_session=object(),
            )

        self.assertEqual(
            [step.name for step in pipeline.steps],
            [
                PREPARE_DATA_STEP_NAME,
                TRAIN_CANDIDATE_STEP_NAME,
                EVALUATE_REGISTER_STEP_NAME,
                POLICY_GATE_STEP_NAME,
            ],
        )
        train_step = pipeline.steps[1]
        prepare_step = pipeline.steps[0]
        self.assertEqual(
            train_step.step_args["inputs"]["train"].s3_data,
            prepare_step.properties.ProcessingOutputConfig.Outputs[
                PREPARE_DATA_OUTPUTS["train"]
            ].S3Output.S3Uri,
        )
        self.assertEqual(
            train_step.step_args["estimator_kwargs"]["environment"]["MLFLOW_TRACKING_URI"],
            "https://mlflow.example.com",
        )
        policy_args = pipeline.steps[3].step_args["arguments"]
        self.assertIn("PIPELINE_EXECUTION_ID", policy_args)


class TestPipelineStepRunners(unittest.TestCase):
    def test_prepare_step_writes_split_outputs_and_metadata(self):
        raw_frame = _raw_resale_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            snapshot = DatasetSnapshot(
                dataset_id="d_demo",
                dataset_name="demo-dataset",
                csv_path=tmp / "cache.csv",
                metadata_path=tmp / "cache.metadata.json",
                source_url="https://data.gov.sg/demo",
                pulled_at="2026-04-04T00:00:00+00:00",
                api_url="https://api-open.data.gov.sg/demo",
                record_count=len(raw_frame),
            )

            with patch(
                "hdb_resale_mlops.sagemaker_pipeline.load_or_download_snapshot",
                return_value=snapshot,
            ), patch(
                "hdb_resale_mlops.sagemaker_pipeline.load_raw_resale_frame",
                return_value=raw_frame,
            ), patch.dict(
                os.environ,
                {"MLFLOW_TRACKING_URI": "https://mlflow.example.com"},
                clear=False,
            ):
                metadata = run_prepare_data_step(
                    train_dir=tmp / "train",
                    validation_dir=tmp / "validation",
                    test_dir=tmp / "test",
                    metadata_dir=tmp / "metadata",
                    validation_months=12,
                    test_months=12,
                )

            self.assertEqual(metadata["dataset_snapshot"]["dataset_id"], "d_demo")
            self.assertTrue((tmp / "train" / "train.csv").exists())
            self.assertTrue((tmp / "validation" / "validation.csv").exists())
            self.assertTrue((tmp / "test" / "test.csv").exists())
            self.assertTrue((tmp / "metadata" / "split_summary.json").exists())

    def test_evaluate_register_step_writes_registration_payload(self):
        raw_frame = _raw_resale_frame(26)
        validation_frame = raw_frame.iloc[-4:-2].copy()
        test_frame = raw_frame.iloc[-2:].copy()

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"MLFLOW_TRACKING_URI": "https://mlflow.example.com"},
            clear=False,
        ):
            tmp = Path(tmpdir)
            artifact_input = _make_model_artifact(tmp)
            validation_input = tmp / "validation.csv"
            test_input = tmp / "test.csv"
            validation_frame.to_csv(validation_input, index=False)
            test_frame.to_csv(test_input, index=False)
            metadata_dir = tmp / "metadata"
            _write_json(metadata_dir / "dataset_snapshot.json", {"dataset_id": "demo"})
            _write_json(metadata_dir / "split_summary.json", {"train_rows": 22})

            with patch(
                "hdb_resale_mlops.sagemaker_pipeline.log_and_register_candidate_model",
                return_value=SimpleNamespace(
                    run_id="run-123",
                    model_name="hdb-resale-price-regressor",
                    model_version="7",
                    model_uri="models:/demo/7",
                ),
            ) as mock_register:
                payload = run_evaluate_register_step(
                    model_artifact_input=artifact_input,
                    validation_input=validation_input,
                    test_input=test_input,
                    metadata_input_dir=metadata_dir,
                    registration_dir=tmp / "registration",
                    evaluation_dir=tmp / "evaluation",
                    training_job_name="job-123",
                    model_artifact_s3_uri="s3://bucket/model.tar.gz",
                    random_seed=7,
                )

            self.assertEqual(payload["model_version"], "7")
            self.assertTrue((tmp / "registration" / "registration.json").exists())
            register_kwargs = mock_register.call_args.kwargs
            self.assertEqual(register_kwargs["training_job_metadata"]["training_job_name"], "job-123")
            self.assertEqual(register_kwargs["hyperparameters"]["random_seed"], 7)

    def test_policy_gate_step_writes_handoff_and_auto_rejects(self):
        raw_frame = _raw_resale_frame(26)
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"MLFLOW_TRACKING_URI": "https://mlflow.example.com"},
            clear=False,
        ):
            tmp = Path(tmpdir)
            train_input = tmp / "train.csv"
            test_input = tmp / "test.csv"
            raw_frame.iloc[:-2].to_csv(train_input, index=False)
            raw_frame.iloc[-2:].to_csv(test_input, index=False)

            registration_input = tmp / "registration.json"
            _write_json(
                registration_input,
                {
                    "run_id": "run-123",
                    "model_name": "hdb-resale-price-regressor",
                    "model_version": "9",
                    "model_uri": "models:/demo/9",
                },
            )
            evaluation_dir = tmp / "evaluation"
            _write_json(evaluation_dir / "test_metrics.json", {"rmse": 250_000.0, "mae": 120_000.0})
            _write_json(
                evaluation_dir / "test_segments_by_town.json",
                [{"segment": "BEDOK", "count": 2, "rmse": 240_000.0, "mae": 110_000.0}],
            )
            _write_json(
                evaluation_dir / "test_segments_by_flat_type.json",
                [{"segment": "4 ROOM", "count": 2, "rmse": 240_000.0, "mae": 110_000.0}],
            )

            with patch(
                "hdb_resale_mlops.mlflow_registry.get_champion_version",
                return_value=None,
            ), patch(
                "hdb_resale_mlops.sagemaker_pipeline.reject_candidate"
            ) as mock_reject:
                handoff = run_policy_gate_step(
                    train_input=train_input,
                    test_input=test_input,
                    registration_input=registration_input,
                    evaluation_input=evaluation_dir,
                    policy_dir=tmp / "policy",
                    handoff_dir=tmp / "handoff",
                    pipeline_execution_id="exec-123",
                )

            self.assertEqual(handoff["status"], "auto_rejected")
            mock_reject.assert_called_once()
            self.assertTrue((tmp / "policy" / "policy_verdict.json").exists())
            self.assertTrue((tmp / "handoff" / "review_handoff.json").exists())
