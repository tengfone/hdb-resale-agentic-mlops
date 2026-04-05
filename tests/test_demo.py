from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hdb_resale_mlops.demo import (
    build_demo_review,
    list_demo_scenarios,
    load_demo_scenario,
    write_demo_review,
)


class DemoModuleTest(unittest.TestCase):
    def test_lists_known_scenarios(self) -> None:
        scenarios = list_demo_scenarios()
        self.assertIn("promote_no_champion", scenarios)
        self.assertIn("manual_review_drift", scenarios)

    def test_load_demo_scenario(self) -> None:
        scenario = load_demo_scenario("promote_no_champion")
        self.assertEqual(scenario.policy_verdict.decision.value, "PROMOTE")
        self.assertEqual(scenario.model_version, "1")

    def test_build_demo_review_in_template_mode(self) -> None:
        review = build_demo_review("manual_review_drift", mode="template")
        self.assertEqual(review["scenario_name"], "manual_review_drift")
        self.assertIn("MANUAL_REVIEW", review["report_text"])
        self.assertIn("check_drift_report", review["tool_outputs"])
        self.assertTrue(review["used_fallback"])

    def test_write_demo_review(self) -> None:
        review = build_demo_review("promote_no_champion", mode="template")
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = write_demo_review(review, tmp_dir)
            review_path = Path(paths["review_path"])
            self.assertTrue(review_path.exists())
            loaded = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["scenario_name"], "promote_no_champion")
