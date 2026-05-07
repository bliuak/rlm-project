from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


TARGET_CATEGORIES = {"numeric value", "location"}


def group_categories_by_user(items: list[dict[str, Any]]) -> dict[int, list[str]]:
    categories_by_user: dict[int, list[str]] = defaultdict(list)
    for index, item in enumerate(items):
        for key in ("user", "correct_category"):
            if key not in item:
                raise ValueError(f"item[{index}] missing {key!r}")
        categories_by_user[int(item["user"])].append(str(item["correct_category"]))
    return {user: categories_by_user[user] for user in sorted(categories_by_user)}


def user_satisfies_task_01(categories: list[str]) -> bool:
    return any(category in TARGET_CATEGORIES for category in categories)


def expected_pairs(categories_by_user: dict[int, list[str]]) -> list[tuple[int, int]]:
    qualifying_users = [
        user
        for user, categories in categories_by_user.items()
        if user_satisfies_task_01(categories)
    ]
    return list(combinations(qualifying_users, 2))


def format_pairs(pairs: list[tuple[int, int]]) -> str:
    return "\n".join(f"({first}, {second})" for first, second in pairs) if pairs else "[]"


def build_audit(items: list[dict[str, Any]], _compare_to: Path | None) -> dict[str, Any]:
    pairs = expected_pairs(group_categories_by_user(items))
    answer = format_pairs(pairs)
    return {
        "task": "task_01",
        "pair_count": len(pairs),
        "computed_answer": answer,
    }
