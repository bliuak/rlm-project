from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDS_PATH = PROJECT_ROOT / "synthetic_user_records.json"
DEFAULT_EXISTING_ANSWER = (
    PROJECT_ROOT / "results" / "oolong_pairs_verified_answers" / "task_05.txt"
)
CUTOFF = date(2023, 3, 15)
TARGET_CATEGORIES = {"entity", "numeric value"}


def load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a top-level 'items' list")
    return items


def parse_date(raw: Any, index: int) -> date:
    try:
        return datetime.strptime(str(raw), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"item[{index}] has invalid date: {raw!r}") from exc


def group_records_by_user(items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    records_by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(items):
        for key in ("date", "user", "correct_answer", "correct_category"):
            if key not in item:
                raise ValueError(f"item[{index}] missing {key!r}")
        records_by_user[int(item["user"])].append(
            {
                "record_index": index,
                "date": parse_date(item["date"], index),
                "correct_answer": item["correct_answer"],
                "correct_category": item["correct_category"],
            }
        )
    return {user: records_by_user[user] for user in sorted(records_by_user)}


def user_satisfies_task_05(records: list[dict[str, Any]]) -> bool:
    """Encode task_05 directly from the question text.

    A user qualifies if:
    1. They have at least one entry whose answer label is entity OR numeric value.
    2. Every entity entry they have is strictly before Mar 15, 2023.

    Clause 2 is universal: a user with numeric-value entries and no entity
    entries satisfies it, because they have no entity entries violating the date
    condition.
    """
    has_entity_or_numeric = any(
        record["correct_category"] in TARGET_CATEGORIES for record in records
    )
    all_entities_before_cutoff = all(
        record["date"] < CUTOFF
        for record in records
        if record["correct_category"] == "entity"
    )
    return has_entity_or_numeric and all_entities_before_cutoff


def user_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["correct_category"] for record in records)
    entity_entries = [
        {
            "record_index": record["record_index"],
            "date": record["date"].isoformat(),
            "correct_answer": record["correct_answer"],
            "is_before_2023_03_15": record["date"] < CUTOFF,
        }
        for record in records
        if record["correct_category"] == "entity"
    ]
    target_entries = [
        {
            "record_index": record["record_index"],
            "date": record["date"].isoformat(),
            "correct_answer": record["correct_answer"],
            "correct_category": record["correct_category"],
        }
        for record in records
        if record["correct_category"] in TARGET_CATEGORIES
    ]
    return {
        "category_counts": dict(sorted(counts.items())),
        "has_entity_or_numeric_value": bool(target_entries),
        "entity_entries": entity_entries,
        "all_entity_entries_before_2023_03_15": all(
            entry["is_before_2023_03_15"] for entry in entity_entries
        ),
        "qualifies": user_satisfies_task_05(records),
        "target_entries": target_entries,
    }


def expected_pairs(records_by_user: dict[int, list[dict[str, Any]]]) -> list[tuple[int, int]]:
    qualifying_users = [
        user for user, records in records_by_user.items() if user_satisfies_task_05(records)
    ]
    return list(combinations(qualifying_users, 2))


def format_pairs(pairs: list[tuple[int, int]]) -> str:
    return "\n".join(f"({first}, {second})" for first, second in pairs) if pairs else "[]"


def build_audit(items: list[dict[str, Any]], compare_to: Path | None) -> dict[str, Any]:
    records_by_user = group_records_by_user(items)
    pairs = expected_pairs(records_by_user)
    answer = format_pairs(pairs)
    existing = compare_to.read_text(encoding="utf-8").strip() if compare_to and compare_to.exists() else None
    users = {str(user): user_audit(records) for user, records in records_by_user.items()}
    return {
        "task": "task_05",
        "criterion": (
            "Both users have at least one instance with an entity or numeric value, "
            "and every entity instance they have is strictly before 2023-03-15."
        ),
        "cutoff_date": CUTOFF.isoformat(),
        "cutoff_is_strict": True,
        "records_count": len(items),
        "user_count": len(records_by_user),
        "qualifying_users": [int(user) for user, audit in users.items() if audit["qualifies"]],
        "pair_count": len(pairs),
        "computed_answer": answer,
        "compare_to": str(compare_to) if compare_to else None,
        "compare_to_present": existing is not None,
        "compare_to_matches": existing == answer if existing is not None else None,
        "users": users,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the task_05 pair answer from a standalone criterion encoding.")
    parser.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--compare-to", type=Path, default=DEFAULT_EXISTING_ANSWER)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit = build_audit(load_items(args.records), args.compare_to)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(audit["computed_answer"] + "\n", encoding="utf-8")
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"task_05 pair_count={audit['pair_count']}")
    print(f"qualifying_users={audit['qualifying_users']}")
    print(f"compare_to_matches={audit['compare_to_matches']}")
    if not args.output:
        print(audit["computed_answer"])
    if args.fail_on_mismatch and audit["compare_to_matches"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
