from __future__ import annotations

# Task 18: one user has abbreviation and exactly one human; the other has entity and numeric value.

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


def group_category_counts_by_user(items: list[dict[str, Any]]) -> dict[int, Counter[str]]:
    counts_by_user: dict[int, Counter[str]] = defaultdict(Counter)
    for index, item in enumerate(items):
        for key in ("user", "correct_category"):
            if key not in item:
                raise ValueError(f"item[{index}] missing {key!r}")
        counts_by_user[int(item["user"])][str(item["correct_category"])] += 1
    return {user: counts_by_user[user] for user in sorted(counts_by_user)}


def first_condition(counts: Counter[str]) -> bool:
    return counts["abbreviation"] >= 1 and counts["human being"] == 1


def second_condition(counts: Counter[str]) -> bool:
    return counts["entity"] >= 1 and counts["numeric value"] >= 1


def expected_pairs(counts_by_user: dict[int, Counter[str]]) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for first, second in combinations(sorted(counts_by_user), 2):
        first_counts = counts_by_user[first]
        second_counts = counts_by_user[second]
        if (
            first_condition(first_counts)
            and second_condition(second_counts)
        ) or (
            first_condition(second_counts)
            and second_condition(first_counts)
        ):
            pairs.add((first, second))
    return sorted(pairs)


def format_pairs(pairs: list[tuple[int, int]]) -> str:
    return "\n".join(f"({first}, {second})" for first, second in pairs) if pairs else "[]"


def build_audit(items: list[dict[str, Any]], _compare_to: Path | None) -> dict[str, Any]:
    pairs = expected_pairs(group_category_counts_by_user(items))
    answer = format_pairs(pairs)
    return {
        "task": "task_18",
        "pair_count": len(pairs),
        "computed_answer": answer,
    }
