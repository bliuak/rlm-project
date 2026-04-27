from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_RECORDS_PATH = Path(__file__).resolve().parents[1] / "synthetic_user_records.json"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a top-level 'items' list")
    return items


def category_counts_by_user(records: list[dict[str, Any]]) -> dict[int, Counter[str]]:
    counts_by_user: dict[int, Counter[str]] = defaultdict(Counter)
    for index, record in enumerate(records):
        try:
            user = int(record["user"])
            category = record["correct_category"]
        except KeyError as exc:
            raise ValueError(f"Record {index} is missing {exc.args[0]!r}") from exc

        if not isinstance(category, str):
            raise ValueError(f"Record {index} has a non-string correct_category")
        counts_by_user[user][category] += 1

    return counts_by_user


def satisfies_location_entity(counts: Counter[str]) -> bool:
    return counts["location"] >= 2 and counts["entity"] >= 1


def satisfies_description_abbreviation(counts: Counter[str]) -> bool:
    return (
        counts["description and abstract concept"] == 1
        and counts["abbreviation"] == 1
    )


def expected_pairs(records: list[dict[str, Any]]) -> list[tuple[int, int]]:
    counts_by_user = category_counts_by_user(records)
    pairs: list[tuple[int, int]] = []

    for first, second in combinations(sorted(counts_by_user), 2):
        first_counts = counts_by_user[first]
        second_counts = counts_by_user[second]
        if (
            satisfies_location_entity(first_counts)
            and satisfies_description_abbreviation(second_counts)
        ) or (
            satisfies_description_abbreviation(first_counts)
            and satisfies_location_entity(second_counts)
        ):
            pairs.append((first, second))

    return pairs


def format_pairs(pairs: list[tuple[int, int]]) -> str:
    lines = [f"({first}, {second})" for first, second in pairs]
    return "\n".join(lines) if lines else "[]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the expected pair answer for users where one side has at "
            "least two location records and one entity record, and the other "
            "side has exactly one description/abstract record and exactly one "
            "abbreviation record."
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
    answer = format_pairs(expected_pairs(load_records(args.records)))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(answer + "\n", encoding="utf-8")
    else:
        print(answer)


if __name__ == "__main__":
    main()
