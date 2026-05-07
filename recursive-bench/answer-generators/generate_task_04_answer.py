from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Any


CUTOFF = date(2023, 1, 6)
TARGET_CATEGORIES = {"human being", "location"}


def parse_date(raw: Any, index: int) -> date:
    try:
        return datetime.strptime(str(raw), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"item[{index}] has invalid date: {raw!r}") from exc


def group_records_by_user(items: list[dict[str, Any]]) -> dict[int, list[tuple[str, date]]]:
    records_by_user: dict[int, list[tuple[str, date]]] = defaultdict(list)
    for index, item in enumerate(items):
        for key in ("date", "user", "correct_category"):
            if key not in item:
                raise ValueError(f"item[{index}] missing {key!r}")
        records_by_user[int(item["user"])].append(
            (str(item["correct_category"]), parse_date(item["date"], index))
        )
    return {user: records_by_user[user] for user in sorted(records_by_user)}


def user_satisfies_task_04(records: list[tuple[str, date]]) -> bool:
    has_human_or_location = any(
        category in TARGET_CATEGORIES for category, _ in records
    )
    all_humans_after_cutoff = all(
        when > CUTOFF
        for category, when in records
        if category == "human being"
    )
    return has_human_or_location and all_humans_after_cutoff


def expected_pairs(records_by_user: dict[int, list[tuple[str, date]]]) -> list[tuple[int, int]]:
    qualifying_users = [
        user
        for user, records in records_by_user.items()
        if user_satisfies_task_04(records)
    ]
    return list(combinations(qualifying_users, 2))


def format_pairs(pairs: list[tuple[int, int]]) -> str:
    return "\n".join(f"({first}, {second})" for first, second in pairs) if pairs else "[]"


def build_audit(items: list[dict[str, Any]], _compare_to: Path | None) -> dict[str, Any]:
    pairs = expected_pairs(group_records_by_user(items))
    answer = format_pairs(pairs)
    return {
        "task": "task_04",
        "pair_count": len(pairs),
        "computed_answer": answer,
    }
