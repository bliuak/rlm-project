from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from synthetic_records import DEFAULT_RECORDS_PATH, load_records_payload


def _sorted_counts(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    }


def summarize_records(path: Path | str = DEFAULT_RECORDS_PATH) -> dict[str, object]:
    payload = load_records_payload(path)
    items = payload["items"]

    users = Counter(item.get("user", "<missing>") for item in items)
    categories = Counter(item.get("correct_category", "<missing>") for item in items)

    return {
        "total_entries": len(items),
        "entries_per_user": _sorted_counts(users),
        "entries_per_correct_category": _sorted_counts(categories),
    }


def format_summary(summary: dict[str, object]) -> str:
    lines = [f"Total entries: {summary['total_entries']}", ""]

    lines.append("Entries per user:")
    for user, count in summary["entries_per_user"].items():  # type: ignore[union-attr]
        lines.append(f"  {user}: {count}")

    lines.append("")
    lines.append("Entries per correct category:")
    for category, count in summary["entries_per_correct_category"].items():  # type: ignore[union-attr]
        lines.append(f"  {category}: {count}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize synthetic records by total count, user distribution, "
            "and correct category distribution."
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
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of plain text.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_records(args.records)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
