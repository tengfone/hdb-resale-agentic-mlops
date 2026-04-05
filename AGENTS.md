# AGENTS.md

Living operating context for this repo. Keep only current, operational guidance here.

## Purpose

Build an MLOps system for HDB resale price prediction where:

- the prediction model is a tabular `XGBRegressor`
- MLflow is the registry of record
- a deterministic policy gate plus an explainer agent govern candidate promotion
- a human makes the final promotion decision for non-auto-reject cases

## Public Repo Policy

- The public repo tracks only `hdb-resale-candidate-training-local-colab.ipynb`.
- The enterprise notebook variants are maintained internally and should stay out of the open-source tree.
- Public docs should describe enterprise flows generically as internal enterprise notebook variants and point internal users to MAESTRO internal docs for the actual entrypoints.

## Core Design

- Three layers:
  1. deterministic evidence gathering, champion comparison, drift checks, policy routing
  2. agentic report generation
  3. human approval or rejection
- The explainer agent investigates and explains. It never decides promotion.
- Promotion routing:
  - `REJECT`: auto-reject unless a human later overrides
  - `MANUAL_REVIEW`: human decides
  - `PROMOTE`: human approval still required

## Data And Model Decisions

- Data source: data.gov.sg HDB resale collection `189`
- Dataset: `d_8b84c4ee58e3cfc0ece0d773c8ca6abc`
- Split policy: chronological, with the most recent 12 months as test and the previous 12 months as validation
- Features:
  - categorical: `town`, `flat_type`, `flat_model`, `storey_range`
  - numeric: `floor_area_sqm`, `flat_age_years`, `remaining_lease_years`, `storey_midpoint`
  - excluded: `block`, `street_name`
- SageMaker training target: scikit-learn `1.4-2`, Python 3.10
- `src/requirements.txt` should not pin numpy, pandas, scipy, or scikit-learn because the container image provides them

## Promotion Semantics

- Log one MLflow run per notebook execution.
- Register the active trained model version as the `candidate` alias.
- On promotion, set the `champion` alias and `promotion_status=champion`.
- On rejection, set `promotion_status=rejected` and record rejection reasons.
- Mirror each promotion review packet both locally and into the candidate run under `promotion_review/<review_id>/`.
- If champion evidence or required segment artifacts cannot be loaded from MLflow, fail closed and block promotion unless a human later overrides.

Policy defaults:

- `max_test_rmse=200_000`
- `max_test_mae=170_000`
- `max_rmse_regression_pct=0.10`
- `max_segment_rmse_regression_pct=0.20`
- `drift_blocks_promotion=True`

Drift defaults:

- categorical drift: PSI `> 0.2`
- numeric drift: KS test `p < 0.05`

## Repo Shape

Key public entrypoints and modules:

- `hdb-resale-candidate-training-local-colab.ipynb`: public local/Colab workflow
- `src/hdb_resale_mlops/data.py`: download, cache, chronological split
- `src/hdb_resale_mlops/features.py`: feature engineering and preprocessing
- `src/hdb_resale_mlops/local_training.py`: shared local fit/evaluate core
- `src/hdb_resale_mlops/evaluation.py`: overall and segment metrics
- `src/hdb_resale_mlops/mlflow_registry.py`: MLflow tracking, registration, aliases
- `src/hdb_resale_mlops/sagemaker_job.py`: direct SageMaker training flow
- `src/hdb_resale_mlops/sagemaker_pipeline.py`: SageMaker Pipeline DAG flow
- `src/hdb_resale_mlops/promotion_workflow.py`: LangGraph workflow
- `src/hdb_resale_mlops/explainer.py`: ReAct explainer agent
- `src/hdb_resale_mlops/policy.py`: deterministic policy rules
- `src/hdb_resale_mlops/drift.py`: PSI and KS drift checks
- `src/hdb_resale_mlops/comparison.py`: candidate vs champion comparison
- `src/pipeline_steps/`: thin SageMaker processing-step wrappers
- `tests/`: unit, integration, notebook-alignment, and report-quality tests
- `.env.example`: combined local and enterprise environment template

## Runtime And Environment

- Open-source runtime: local Python 3.10+ or Google Colab
- Enterprise runtime: SageMaker notebook in `ap-southeast-1`
- Local toolchain may optionally use `.mise.toml` to provision Python `3.12.13`
- Local notebook/package imports auto-load a repo-local `.env` file without overriding already-set env vars
- Known project env vars are forwarded into SageMaker jobs and pipeline steps

Important env rules:

- `MLFLOW_TRACKING_URI` is optional for the public local notebook and the internal direct enterprise notebook variant
- `MLFLOW_TRACKING_URI` is required for the internal enterprise pipeline variant because isolated jobs cannot share the local SQLite fallback
- Defaults when unset: `MLFLOW_MODEL_NAME=hdb-resale-price-regressor`, `OPENAI_MODEL=gpt-5-nano`
- Enterprise-only proxy vars: `MAESTRO_HTTP_PROXY`, `MAESTRO_HTTPS_PROXY`
- Common optional vars: `AWS_REGION`, `SAGEMAKER_ROLE_ARN`, `S3_BUCKET`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_EXPERIMENT_NAME`, `DATA_GOV_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ENABLE_JUDGE_EVAL`, `MARKET_RESEARCH_PROVIDER`, `OPENAI_WEB_SEARCH_MODEL`, `OPENAI_JUDGE_MODEL`, `TAVILY_API_KEY`, `MODEL_REVIEWER`

## Execution Workflows

- Open-source workflow: run `hdb-resale-candidate-training-local-colab.ipynb`; in Colab, let the bootstrap cell install dependencies and restart the runtime once before continuing; then train locally, log to MLflow, register `candidate`, and run the promotion workflow
- Internal enterprise direct-job workflow: launch SageMaker script-mode training, download the artifact, evaluate notebook-side, register `candidate`, then run the promotion workflow
- Internal enterprise pipeline workflow: run `PrepareData -> TrainCandidate -> EvaluateRegisterCandidate -> PolicyGate`, download the frozen handoff payload, then continue explainer + human review notebook-side

Acceptance path:

- For notebook flows, top-to-bottom notebook execution in the intended environment is the real acceptance path

## Change Rules

- Preserve the notebook-first workflow.
- Keep the public local notebook and the internal enterprise notebook variants aligned on shared semantics.
- Move non-trivial shared logic into `src/hdb_resale_mlops/`.
- Keep SageMaker Pipeline logic in `src/hdb_resale_mlops/sagemaker_pipeline.py` with only thin wrappers under `src/pipeline_steps/`.
- `local_training.py` is the shared training core; do not duplicate fit/evaluate logic elsewhere.
- The public local notebook must not import SageMaker-specific modules.
- Agent dependencies are optional; guard imports and degrade gracefully when unavailable.
- Keep repo-tracked artifacts small; generated outputs belong under ignored directories.
- Use the `jupyter-notebook` skill when a task materially changes notebooks.

## Validation

- Static verification is acceptable when local dependencies are unavailable.
- Prefer local tests for parsing, splitting, metrics math, comparison, drift, policy behavior, registry behavior, and workflow routing.
- Test `PROMOTE`, `REJECT`, and `MANUAL_REVIEW` workflow paths with mocked dependencies.
- Test the explainer fallback path when `OPENAI_API_KEY` is absent.
- `make test` force-skips the opt-in LLM suites even if related env vars are already set.
- Only run `make test-agent-llm` or `make test-report-llm` intentionally.
- Do not claim SageMaker or MLflow end-to-end validation unless those systems were actually exercised.

## Maintenance

Update this file when any of these change:

- project scope
- public vs internal notebook boundary
- runtime environment
- dataset choice or fetch path
- model family
- split policy
- environment variables
- registry semantics
- workflow architecture
- validation expectations
- policy thresholds
