from __future__ import annotations

import json
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCAL_NOTEBOOK = REPO_ROOT / "hdb-resale-candidate-training-local-colab.ipynb"


def _notebook_text(path: pathlib.Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in notebook.get("cells", []):
        parts.extend(cell.get("source", []))
    return "".join(parts)


class NotebookAlignmentTest(unittest.TestCase):
    def test_local_notebook_uses_persisted_review_helpers(self) -> None:
        text = _notebook_text(LOCAL_NOTEBOOK)
        self.assertIn("start_promotion_review", text)
        self.assertIn("resume_promotion_review", text)

    def test_local_notebook_mentions_local_mlflow_fallback(self) -> None:
        text = _notebook_text(LOCAL_NOTEBOOK)
        self.assertIn("sqlite:///mlflow.db", text)
