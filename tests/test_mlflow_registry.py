"""Tests for MLflow registry helpers."""

import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from mlflow.exceptions import MlflowException

from hdb_resale_mlops.config import RuntimeConfig
from hdb_resale_mlops.evaluation import EvaluationResult
from hdb_resale_mlops.mlflow_registry import (
    ChampionDataUnavailableError,
    MlflowRegistryError,
    _load_segment_artifacts,
    configure_mlflow,
    get_champion_version,
    log_and_register_candidate_model,
    log_promotion_review_artifacts,
)


class _FakeRunContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestPromotionReviewArtifacts(unittest.TestCase):
    def test_links_trace_to_backing_run(self):
        fake_client = MagicMock()
        fake_client.get_model_version.return_value = SimpleNamespace(run_id="run-123")

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.MlflowClient = MagicMock(return_value=fake_client)
        fake_mlflow.start_run = MagicMock(return_value=_FakeRunContext())
        fake_mlflow.log_artifact = MagicMock()
        fake_mlflow.log_metric = MagicMock()

        payload = {
            "review_id": "review-123",
            "report_text": "## Summary\nReport",
            "report_structured": {"summary": "Report"},
            "agent_trace": [{"event": "tool_call", "tool_name": "query_candidate_metrics"}],
            "run_metadata": {"mlflow_trace_id": "trace-123"},
            "judge_evaluation": {
                "enabled": True,
                "status": "scored",
                "model": "gpt-5-mini",
                "scores": {
                    "completeness": 4,
                    "accuracy": 5,
                    "actionability": 4,
                    "safety": 5,
                    "average": 4.5,
                    "reasoning": "Good report.",
                },
            },
        }

        with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logged = log_promotion_review_artifacts(
                model_name="test-model",
                model_version="7",
                review_payload=payload,
            )

        self.assertTrue(logged)
        fake_client.link_traces_to_run.assert_called_once_with(
            ["trace-123"],
            run_id="run-123",
        )
        self.assertGreaterEqual(fake_mlflow.log_artifact.call_count, 3)
        logged_artifact_paths = {
            call.kwargs["artifact_path"] for call in fake_mlflow.log_artifact.call_args_list
        }
        self.assertEqual(logged_artifact_paths, {"promotion_review/review-123"})
        fake_mlflow.log_metric.assert_any_call("judge_completeness", 4.0)
        fake_mlflow.log_metric.assert_any_call("judge_average", 4.5)


class TestChampionLookup(unittest.TestCase):
    @patch("mlflow.MlflowClient", autospec=True)
    def test_missing_alias_returns_none(self, MockClient):
        from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

        MockClient.return_value.get_model_version_by_alias.side_effect = MlflowException(
            "missing alias",
            error_code=INVALID_PARAMETER_VALUE,
        )

        self.assertIsNone(get_champion_version("test-model"))

    @patch("mlflow.MlflowClient", autospec=True)
    def test_backend_failure_raises_registry_error(self, MockClient):
        from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

        MockClient.return_value.get_model_version_by_alias.side_effect = MlflowException(
            "backend unavailable",
            error_code=INTERNAL_ERROR,
        )

        with self.assertRaises(MlflowRegistryError):
            get_champion_version("test-model")


class TestSegmentArtifacts(unittest.TestCase):
    @patch("mlflow.MlflowClient", autospec=True)
    def test_missing_required_segments_raise(self, MockClient):
        import json
        import os
        import tempfile

        tmp = tempfile.mkdtemp()
        eval_dir = os.path.join(tmp, "evaluation")
        os.makedirs(eval_dir)
        with open(os.path.join(eval_dir, "test_segments_by_town.json"), "w") as f:
            json.dump([{"segment": "BEDOK", "rmse": 155_000, "mae": 120_000, "count": 60}], f)

        MockClient.return_value.download_artifacts.return_value = eval_dir

        with self.assertRaises(ChampionDataUnavailableError):
            _load_segment_artifacts("run-partial", required=True)


class TestCandidateRegistration(unittest.TestCase):
    def test_configure_mlflow_defaults_to_local_sqlite_when_uri_missing(self):
        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.set_tracking_uri = MagicMock()
        fake_mlflow.set_experiment = MagicMock()
        config = RuntimeConfig()

        with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            with patch.dict(os.environ, {}, clear=True):
                configure_mlflow(config)
                self.assertEqual(os.environ["MLFLOW_TRACKING_URI"], "sqlite:///mlflow.db")
        fake_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///mlflow.db")
        fake_mlflow.set_experiment.assert_called_once_with(
            config.mlflow_experiment_name
        )

    def test_uses_registered_model_version_from_log_model(self):
        fake_client = MagicMock()

        class _FakeRunContext:
            def __enter__(self):
                return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.start_run = MagicMock(return_value=_FakeRunContext())
        fake_mlflow.set_tracking_uri = MagicMock()
        fake_mlflow.set_experiment = MagicMock()
        fake_mlflow.set_tags = MagicMock()
        fake_mlflow.log_params = MagicMock()
        fake_mlflow.log_dict = MagicMock()
        fake_mlflow.log_metric = MagicMock()
        fake_mlflow.log_artifact = MagicMock()
        fake_mlflow.MlflowClient = MagicMock(return_value=fake_client)

        fake_sklearn = types.ModuleType("mlflow.sklearn")
        fake_sklearn.log_model = MagicMock(
            return_value=SimpleNamespace(
                registered_model_version="7",
                model_uri="models:/m-123",
            )
        )
        fake_mlflow.sklearn = fake_sklearn

        fake_models = types.ModuleType("mlflow.models")
        fake_models.infer_signature = MagicMock(return_value="sig")

        class _Model:
            def predict(self, frame):
                return [100000.0] * len(frame)

        scored = pd.DataFrame(
            {
                "town": ["BEDOK"],
                "flat_type": ["4 ROOM"],
                "flat_model": ["Improved"],
                "storey_range": ["04 TO 06"],
                "floor_area_sqm": [90.0],
                "flat_age_years": [20.0],
                "remaining_lease_years": [79.0],
                "storey_midpoint": [5.0],
                "resale_price": [500000.0],
                "prediction": [500000.0],
            }
        )
        segment = pd.DataFrame(
            {
                "segment": ["BEDOK"],
                "count": [1],
                "rmse": [0.0],
                "mae": [0.0],
                "mean_actual": [500000.0],
                "mean_prediction": [500000.0],
            }
        )
        evaluation = EvaluationResult(
            overall_metrics={"rmse": 0.0, "mae": 0.0},
            scored_frame=scored,
            segment_metrics={"town": segment, "flat_type": segment},
        )
        config = RuntimeConfig(mlflow_tracking_uri="sqlite:///mlflow.db")

        with patch.dict(
            sys.modules,
            {
                "mlflow": fake_mlflow,
                "mlflow.sklearn": fake_sklearn,
                "mlflow.models": fake_models,
            },
        ):
            result = log_and_register_candidate_model(
                model=_Model(),
                validation_evaluation=evaluation,
                test_evaluation=evaluation,
                runtime_config=config,
                artifact_dir=Path("/tmp/mlflow-registry-test"),
                dataset_snapshot={"rows": 1},
                split_summary={"train": 1, "validation": 1, "test": 1},
                hyperparameters={"n_estimators": 1},
                training_job_metadata={"mode": "local"},
            )

        self.assertEqual(result.model_version, "7")
        self.assertEqual(result.model_uri, "models:/m-123")
        fake_sklearn.log_model.assert_called_once()
        self.assertEqual(
            fake_sklearn.log_model.call_args.kwargs["registered_model_name"],
            config.mlflow_model_name,
        )
        fake_client.set_registered_model_alias.assert_called_once_with(
            config.mlflow_model_name,
            "candidate",
            "7",
        )

    def test_falls_back_to_artifact_path_for_older_mlflow_log_model_api(self):
        fake_client = MagicMock()

        class _FakeRunContext:
            def __enter__(self):
                return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.start_run = MagicMock(return_value=_FakeRunContext())
        fake_mlflow.set_tracking_uri = MagicMock()
        fake_mlflow.set_experiment = MagicMock()
        fake_mlflow.set_tags = MagicMock()
        fake_mlflow.log_params = MagicMock()
        fake_mlflow.log_dict = MagicMock()
        fake_mlflow.log_metric = MagicMock()
        fake_mlflow.log_artifact = MagicMock()
        fake_mlflow.MlflowClient = MagicMock(return_value=fake_client)

        call_kwargs: list[dict[str, object]] = []

        def _log_model(**kwargs):
            call_kwargs.append(dict(kwargs))
            if "name" in kwargs:
                raise TypeError("log_model() got an unexpected keyword argument 'name'")
            return SimpleNamespace(
                registered_model_version="7",
                model_uri="models:/m-123",
            )

        fake_sklearn = types.ModuleType("mlflow.sklearn")
        fake_sklearn.log_model = _log_model
        fake_mlflow.sklearn = fake_sklearn

        fake_models = types.ModuleType("mlflow.models")
        fake_models.infer_signature = MagicMock(return_value="sig")

        class _Model:
            def predict(self, frame):
                return [100000.0] * len(frame)

        scored = pd.DataFrame(
            {
                "town": ["BEDOK"],
                "flat_type": ["4 ROOM"],
                "flat_model": ["Improved"],
                "storey_range": ["04 TO 06"],
                "floor_area_sqm": [90.0],
                "flat_age_years": [20.0],
                "remaining_lease_years": [79.0],
                "storey_midpoint": [5.0],
                "resale_price": [500000.0],
                "prediction": [500000.0],
            }
        )
        segment = pd.DataFrame(
            {
                "segment": ["BEDOK"],
                "count": [1],
                "rmse": [0.0],
                "mae": [0.0],
                "mean_actual": [500000.0],
                "mean_prediction": [500000.0],
            }
        )
        evaluation = EvaluationResult(
            overall_metrics={"rmse": 0.0, "mae": 0.0},
            scored_frame=scored,
            segment_metrics={"town": segment, "flat_type": segment},
        )
        config = RuntimeConfig(mlflow_tracking_uri="sqlite:///mlflow.db")

        with patch.dict(
            sys.modules,
            {
                "mlflow": fake_mlflow,
                "mlflow.sklearn": fake_sklearn,
                "mlflow.models": fake_models,
            },
        ):
            result = log_and_register_candidate_model(
                model=_Model(),
                validation_evaluation=evaluation,
                test_evaluation=evaluation,
                runtime_config=config,
                artifact_dir=Path("/tmp/mlflow-registry-test"),
                dataset_snapshot={"rows": 1},
                split_summary={"train": 1, "validation": 1, "test": 1},
                hyperparameters={"n_estimators": 1},
                training_job_metadata={"mode": "local"},
            )

        self.assertEqual(result.model_version, "7")
        self.assertEqual(len(call_kwargs), 2)
        self.assertEqual(call_kwargs[0]["name"], "model")
        self.assertEqual(call_kwargs[1]["artifact_path"], "model")
        self.assertNotIn("name", call_kwargs[1])


if __name__ == "__main__":
    unittest.main()
