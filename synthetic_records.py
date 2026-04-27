from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RECORDS_PATH = Path(__file__).resolve().parent / "synthetic_user_records.json"
HIDDEN_FIELDS = {"correct_answer", "correct_category"}


def load_records_payload(path: Path | str = DEFAULT_RECORDS_PATH) -> dict[str, Any]:
    records_path = Path(path)
    with records_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{records_path} must contain a top-level 'items' list")
    return data


def remove_hidden_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in HIDDEN_FIELDS}


def get_llm_entries(path: Path | str = DEFAULT_RECORDS_PATH) -> list[dict[str, Any]]:
    payload = load_records_payload(path)
    return [remove_hidden_fields(item) for item in payload["items"]]


def get_llm_payload(path: Path | str = DEFAULT_RECORDS_PATH) -> dict[str, Any]:
    payload = load_records_payload(path)
    return {**payload, "items": [remove_hidden_fields(item) for item in payload["items"]]}


def get_llm_json(
    path: Path | str = DEFAULT_RECORDS_PATH,
    *,
    indent: int | None = 2,
) -> str:
    return json.dumps(get_llm_payload(path), ensure_ascii=False, indent=indent)


def write_llm_payload(
    output: Path | str,
    path: Path | str = DEFAULT_RECORDS_PATH,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(get_llm_json(path) + "\n", encoding="utf-8")
