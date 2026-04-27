from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_RECORDS_PATH = Path(__file__).resolve().parents[1] / "synthetic_user_records.json"
DEFAULT_EXISTING_ANSWER = (
    Path(__file__).resolve().parents[1] / "results" / "oolong_pairs_verified_answers" / "paper_03.txt"
)
TARGET_CATEGORIES = {"description and abstract concept", "abbreviation"}


def load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a top-level 'items' list")
    return items


def group_records_by_user(items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    records_by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(items):
        for key in ("date", "user", "correct_answer", "correct_category"):
            if key not in item:
                raise ValueError(f"item[{index}] missing {key!r}")
        records_by_user[int(item["user"])].append(
            {
                "record_index": index,
                "date": item["date"],
                "correct_answer": item["correct_answer"],
                "correct_category": item["correct_category"],
            }
        )
    return {user: records_by_user[user] for user in sorted(records_by_user)}


def user_satisfies_paper_03(records: list[dict[str, Any]]) -> bool:
    return any(record["correct_category"] in TARGET_CATEGORIES for record in records)


def user_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["correct_category"] for record in records)
    target_entries = [
        record for record in records if record["correct_category"] in TARGET_CATEGORIES
    ]
    return {
        "category_counts": dict(sorted(counts.items())),
        "has_description_or_abbreviation": bool(target_entries),
        "qualifies": user_satisfies_paper_03(records),
        "target_entries": target_entries,
    }


def expected_pairs(records_by_user: dict[int, list[dict[str, Any]]]) -> list[tuple[int, int]]:
    qualifying_users = [
        user for user, records in records_by_user.items() if user_satisfies_paper_03(records)
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
        "task": "paper_03",
        "criterion": "Both users have at least one instance with a description and abstract concept or abbreviation.",
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
    parser = argparse.ArgumentParser(description="Generate the paper_03 pair answer from a standalone criterion encoding.")
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
    print(f"paper_03 pair_count={audit['pair_count']}")
    print(f"qualifying_users={audit['qualifying_users']}")
    print(f"compare_to_matches={audit['compare_to_matches']}")
    if not args.output:
        print(audit["computed_answer"])
    if args.fail_on_mismatch and audit["compare_to_matches"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
