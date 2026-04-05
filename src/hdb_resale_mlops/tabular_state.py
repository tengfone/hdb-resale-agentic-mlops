"""Helpers for moving tabular pandas data through LangGraph state safely."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date, datetime
import math
from pathlib import Path
from typing import Any


_DATAFRAME_STATE_KIND = "dataframe_records"
_DATAFRAME_STATE_MARKER = "__kind__"


def _is_dataframe(value: Any) -> bool:
    return hasattr(value, "to_dict") and hasattr(value, "columns")


def serialize_for_state(value: Any) -> Any:
    """Convert DataFrames and pandas/numpy scalars into plain Python objects."""
    if _is_dataframe(value):
        return {
            _DATAFRAME_STATE_MARKER: _DATAFRAME_STATE_KIND,
            "columns": [str(column) for column in value.columns],
            "records": [serialize_for_state(row) for row in value.to_dict(orient="records")],
        }

    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return None if math.isnan(value) else value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(key): serialize_for_state(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [serialize_for_state(item) for item in value]

    if hasattr(value, "item"):
        try:
            return serialize_for_state(value.item())
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return str(value)


def is_serialized_dataframe(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get(_DATAFRAME_STATE_MARKER) == _DATAFRAME_STATE_KIND
        and "records" in value
    )


def coerce_dataframe(value: Any):
    """Materialize a pandas DataFrame from serialized state when needed."""
    if value is None or _is_dataframe(value):
        return value

    import pandas as pd

    if is_serialized_dataframe(value):
        return pd.DataFrame(value.get("records", []), columns=value.get("columns"))

    if isinstance(value, list):
        return pd.DataFrame(value)

    raise TypeError(f"Expected a DataFrame-compatible value, got {type(value)!r}")


def iter_tabular_rows(value: Any) -> Iterator[dict[str, Any]]:
    """Yield row dictionaries from DataFrames or serialized tabular payloads."""
    if value is None:
        return

    if _is_dataframe(value):
        for row in value.to_dict(orient="records"):
            yield serialize_for_state(row)
        return

    if is_serialized_dataframe(value):
        for row in value.get("records", []):
            if isinstance(row, Mapping):
                yield {str(key): item for key, item in row.items()}
        return

    if isinstance(value, list):
        for row in value:
            if isinstance(row, Mapping):
                yield {str(key): serialize_for_state(item) for key, item in row.items()}
