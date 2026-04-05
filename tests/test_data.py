from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from hdb_resale_mlops.config import RuntimeConfig
from hdb_resale_mlops.data import DATASET_API_BASE, _download_dataset_csv, chronological_split


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is required for split tests")
class ChronologicalSplitTest(unittest.TestCase):
    def test_split_uses_last_12_months_for_test_and_previous_12_for_validation(self) -> None:
        import pandas as pd

        records = []
        for year in range(2021, 2025):
            for month in range(1, 13):
                records.append(
                    {
                        "month": f"{year}-{month:02d}",
                        "town": "ANG MO KIO",
                        "flat_type": "4 ROOM",
                        "flat_model": "Model A",
                        "storey_range": "04 TO 06",
                        "floor_area_sqm": 92.0,
                        "lease_commence_date": 2000,
                        "remaining_lease": "75 years 00 months",
                        "resale_price": 450000 + year + month,
                    }
                )

        frame = pd.DataFrame.from_records(records)
        split = chronological_split(frame, validation_months=12, test_months=12)

        self.assertEqual(split.summary["train_month_start"], "2021-01")
        self.assertEqual(split.summary["train_month_end"], "2022-12")
        self.assertEqual(split.summary["validation_month_start"], "2023-01")
        self.assertEqual(split.summary["validation_month_end"], "2023-12")
        self.assertEqual(split.summary["test_month_start"], "2024-01")
        self.assertEqual(split.summary["test_month_end"], "2024-12")
        self.assertEqual(len(split.train), 24)
        self.assertEqual(len(split.validation), 12)
        self.assertEqual(len(split.test), 12)


class DatasetDownloadTest(unittest.TestCase):
    @patch("hdb_resale_mlops.data.time.sleep", return_value=None)
    @patch("requests.Session")
    @patch("requests.get")
    def test_download_uses_api_key_header_for_api_calls(
        self,
        mock_get,
        mock_session_cls,
        _mock_sleep,
    ) -> None:
        initiate_response = MagicMock()
        initiate_response.raise_for_status.return_value = None
        initiate_response.json.return_value = {"data": {}}

        poll_response = MagicMock()
        poll_response.raise_for_status.return_value = None
        poll_response.json.return_value = {
            "data": {
                "status": "DOWNLOAD_SUCCESS",
                "url": "https://example.com/hdb.csv",
            }
        }
        mock_get.side_effect = [initiate_response, poll_response]

        download_response = MagicMock()
        download_response.__enter__.return_value = download_response
        download_response.__exit__.return_value = False
        download_response.raise_for_status.return_value = None
        download_response.iter_content.return_value = [b"month,resale_price\n2024-01,500000\n"]
        mock_session = MagicMock()
        mock_session.get.return_value = download_response
        mock_session_cls.return_value = mock_session

        config = RuntimeConfig(data_gov_api_key="test-api-key")
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "snapshot.csv"
            _download_dataset_csv(config, csv_path)
            self.assertTrue(csv_path.exists())

        initiate_url = f"{DATASET_API_BASE}/{config.dataset_id}/initiate-download"
        poll_url = f"{DATASET_API_BASE}/{config.dataset_id}/poll-download"
        expected_headers = {"x-api-key": "test-api-key"}
        self.assertEqual(
            mock_get.call_args_list[0].kwargs,
            {"timeout": 60, "proxies": None, "headers": expected_headers},
        )
        self.assertEqual(mock_get.call_args_list[0].args, (initiate_url,))
        self.assertEqual(
            mock_get.call_args_list[1].kwargs,
            {"timeout": 60, "proxies": None, "headers": expected_headers},
        )
        self.assertEqual(mock_get.call_args_list[1].args, (poll_url,))
