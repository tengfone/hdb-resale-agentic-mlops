"""
Dataset download, caching, and chronological train/val/test splits.
Optionally supports data.gov.sg API key for higher rate limits.

Steps:
- download the latest HDB resale CSV snapshot from data.gov.sg
- cache the CSV and its metadata locally so notebooks can be rerun cheaply
- load that snapshot into a pandas DataFrame
- split the data by time rather than at random
- optionally persist the split CSVs for SageMaker or debugging

Use chronological split (time-based) to ensure models are trained on
past data and tested on future data, preventing data leakage and accurately
simulating real-world forecasting scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.parse import urlparse

from hdb_resale_mlops.config import RuntimeConfig
from hdb_resale_mlops.features import DATE_COLUMN

DATASET_API_BASE = "https://api-open.data.gov.sg/v1/public/api/datasets"
DOWNLOAD_POLL_INTERVAL_SECONDS = 3
DOWNLOAD_MAX_POLL_ATTEMPTS = 60


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_id: str
    dataset_name: str
    csv_path: Path
    metadata_path: Path
    source_url: str
    pulled_at: str
    api_url: str
    record_count: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "csv_path": str(self.csv_path),
            "metadata_path": str(self.metadata_path),
            "source_url": self.source_url,
            "pulled_at": self.pulled_at,
            "api_url": self.api_url,
            "record_count": self.record_count,
        }


@dataclass(frozen=True)
class DataSplit:
    train: Any
    validation: Any
    test: Any
    summary: dict[str, Any]


def _download_dataset_csv(
    config: RuntimeConfig,
    csv_path: Path,
) -> str:
    """
    Download the full dataset CSV via the data.gov.sg v1 download API.

    Flow:
    1. GET  .../initiate-download  → triggers server-side CSV generation
    2. GET  .../poll-download      → poll until status is DOWNLOAD_SUCCESS
    3. Stream the pre-signed S3 URL to *csv_path*.

    Returns the API base URL used for metadata.
    """
    import requests

    base_url = f"{DATASET_API_BASE}/{config.dataset_id}"
    proxies = dict(config.maestro_proxies) if config.maestro_proxies else None
    headers = (
        {"x-api-key": config.data_gov_api_key} if config.data_gov_api_key else None
    )

    # Step 1 – initiate download
    initiate_url = f"{base_url}/initiate-download"
    resp = requests.get(initiate_url, timeout=60, proxies=proxies, headers=headers)
    resp.raise_for_status()
    initiate_body = resp.json()

    # The initiate response may already contain a direct download URL.
    download_url: str | None = (initiate_body.get("data") or {}).get("url")

    # Step 2 – poll until the download is ready (if not already provided)
    if not download_url:
        poll_url = f"{base_url}/poll-download"
        for _ in range(DOWNLOAD_MAX_POLL_ATTEMPTS):
            time.sleep(DOWNLOAD_POLL_INTERVAL_SECONDS)
            poll_resp = requests.get(
                poll_url, timeout=60, proxies=proxies, headers=headers
            )
            poll_resp.raise_for_status()
            poll_body = poll_resp.json()
            status = (poll_body.get("data") or {}).get("status", "")
            if status == "DOWNLOAD_SUCCESS":
                download_url = (poll_body.get("data") or {}).get("url")
                break
            if "FAIL" in status.upper():
                raise RuntimeError(
                    f"data.gov.sg download failed for dataset {config.dataset_id!r}: {poll_body}"
                )
        if not download_url:
            raise RuntimeError(
                f"Timed out waiting for data.gov.sg download for dataset {config.dataset_id!r}."
            )

    # Step 3 – stream the CSV to disk.
    # The pre-signed URL points to S3 which is typically not whitelisted on
    # the data.gov.sg proxy.  Use a throwaway session with trust_env=False
    # so that HTTPS_PROXY / HTTP_PROXY env vars are fully ignored.
    dl_session = requests.Session()
    dl_session.trust_env = False
    with dl_session.get(download_url, stream=True, timeout=300) as dl:
        dl.raise_for_status()
        with open(csv_path, "wb") as fh:
            for chunk in dl.iter_content(chunk_size=1 << 20):  # 1 MiB chunks
                fh.write(chunk)

    return initiate_url


def load_or_download_snapshot(
    paths, config: RuntimeConfig, force: bool = False
) -> DatasetSnapshot:
    # Reuse the cached snapshot by default so notebook reruns do not keep
    # hitting the remote API unless the caller explicitly asks for a refresh.
    paths.ensure_local_dirs()
    csv_path = paths.cache_dir / f"{config.dataset_name}.csv"
    metadata_path = paths.cache_dir / f"{config.dataset_name}.metadata.json"

    if csv_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return DatasetSnapshot(
            dataset_id=metadata["dataset_id"],
            dataset_name=metadata["dataset_name"],
            csv_path=Path(metadata["csv_path"]),
            metadata_path=Path(metadata["metadata_path"]),
            source_url=metadata["source_url"],
            pulled_at=metadata["pulled_at"],
            api_url=metadata.get("api_url", ""),
            record_count=int(metadata.get("record_count", 0)),
        )

    api_url = _download_dataset_csv(config, csv_path)
    import pandas as pd

    record_count = len(pd.read_csv(csv_path))
    pulled_at = datetime.now(timezone.utc).isoformat()
    source_url = f"https://data.gov.sg/datasets/{config.dataset_id}/view"
    snapshot = DatasetSnapshot(
        dataset_id=config.dataset_id,
        dataset_name=config.dataset_name,
        csv_path=csv_path,
        metadata_path=metadata_path,
        source_url=source_url,
        pulled_at=pulled_at,
        api_url=api_url,
        record_count=record_count,
    )
    metadata_path.write_text(
        json.dumps(snapshot.to_metadata(), indent=2), encoding="utf-8"
    )
    return snapshot


def load_raw_resale_frame(snapshot: DatasetSnapshot):
    import pandas as pd

    return pd.read_csv(snapshot.csv_path)


def chronological_split(
    raw_frame, validation_months: int = 12, test_months: int = 12
) -> DataSplit:
    import pandas as pd

    frame = raw_frame.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame["month"], format="%Y-%m")
    month_index = sorted(frame[DATE_COLUMN].dt.to_period("M").unique())
    holdout_months = validation_months + test_months
    if len(month_index) <= holdout_months:
        raise ValueError(
            f"Need more than {holdout_months} unique months to create train, validation, and test splits."
        )

    # Newest months become test, the block before that becomes validation,
    # and everything older becomes training.
    test_periods = month_index[-test_months:]
    validation_periods = month_index[-holdout_months:-test_months]
    train_periods = month_index[:-holdout_months]

    period_series = frame[DATE_COLUMN].dt.to_period("M")
    train = frame.loc[period_series.isin(train_periods)].copy()
    validation = frame.loc[period_series.isin(validation_periods)].copy()
    test = frame.loc[period_series.isin(test_periods)].copy()

    summary = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_month_start": str(train_periods[0]),
        "train_month_end": str(train_periods[-1]),
        "validation_month_start": str(validation_periods[0]),
        "validation_month_end": str(validation_periods[-1]),
        "test_month_start": str(test_periods[0]),
        "test_month_end": str(test_periods[-1]),
    }
    return DataSplit(train=train, validation=validation, test=test, summary=summary)


def persist_split_frames(split: DataSplit, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test": output_dir / "test.csv",
        "summary": output_dir / "split_summary.json",
    }
    split.train.to_csv(split_paths["train"], index=False)
    split.validation.to_csv(split_paths["validation"], index=False)
    split.test.to_csv(split_paths["test"], index=False)
    split_paths["summary"].write_text(
        json.dumps(split.summary, indent=2), encoding="utf-8"
    )
    return split_paths


def s3_uri_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")
