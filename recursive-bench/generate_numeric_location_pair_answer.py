from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


TARGET_CATEGORIES = {"numeric value", "location"}
DEFAULT_RECORDS_PATH = Path(__file__).resolve().parents[1] / "synthetic_user_records.json"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a top-level 'items' list")
    return items


def qualifying_users(records: list[dict[str, Any]]) -> list[int]:
    categories_by_user: dict[int, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        try:
            user = int(record["user"])
            category = record["correct_category"]
        except KeyError as exc:
            raise ValueError(f"Record {index} is missing {exc.args[0]!r}") from exc

        if not isinstance(category, str):
            raise ValueError(f"Record {index} has a non-string correct_category")
        categories_by_user[user].add(category)

    return sorted(
        user
        for user, categories in categories_by_user.items()
        if categories & TARGET_CATEGORIES
    )


def format_pairs(users: list[int]) -> str:
    lines = [f"({first}, {second})" for first, second in combinations(users, 2)]
    return "\n".join(lines) if lines else "[]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the expected pair answer for users with at least one "
            "numeric value or location record."
        )
    )
    parser.add_argument(
        "records",
        nargs="?",
        type=Path,
        default=DEFAULT_RECORDS_PATH,
        help="Path to the synthetic user records JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write the newline-separated pair answer to.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    users = qualifying_users(load_records(args.records))
    answer = format_pairs(users)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(answer + "\n", encoding="utf-8")
    else:
        print(answer)


if __name__ == "__main__":
    main()
