from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generate_oolong_pairs_verified_answers import load_items
from oolong_pairs_tasks import (
    LABELS,
    TASK_SPECS,
    compute_expected_pairs,
    format_pairs,
    full_question,
    parse_records,
    records_by_user,
)


DEFAULT_RECORDS_PATH = Path(__file__).resolve().parents[1] / "synthetic_user_records.json"
DEFAULT_ANSWERS_DIR = Path(__file__).resolve().parents[1] / "results" / "oolong_pairs_verified_answers"


def validate_items(items: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, item in enumerate(items):
        for key in ("date", "user", "record", "correct_answer", "correct_category"):
            if key not in item:
                errors.append(f"item[{index}] missing {key!r}")
        category = item.get("correct_category")
        if category not in LABELS:
            errors.append(f"item[{index}] has unsupported correct_category: {category!r}")
        answer = item.get("correct_answer")
        if not isinstance(answer, str) or not answer.strip():
            errors.append(f"item[{index}] has empty/non-string correct_answer: {answer!r}")
    return errors


def record_evidence(items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    evidence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(items):
        evidence[int(item["user"])].append(
            {
                "record_index": index,
                "date": item["date"],
                "correct_answer": item["correct_answer"],
                "correct_category": item["correct_category"],
            }
        )
    return {user: rows for user, rows in sorted(evidence.items())}


def category_counts(items: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for item in items:
        counts[int(item["user"])][str(item["correct_category"])] += 1
    return {
        user: {label: counts[user][label] for label in LABELS}
        for user in sorted(counts)
    }


def qualifying_users(per_user: dict[int, list], task_name: str) -> dict[str, list[int]]:
    spec = TASK_SPECS[task_name]
    return {
        "first_condition_users": [
            user for user, records in sorted(per_user.items()) if spec.first_condition(records)
        ],
        "second_condition_users": [
            user for user, records in sorted(per_user.items()) if spec.second_condition(records)
        ],
    }


def pair_audit(per_user: dict[int, list], task_name: str) -> list[dict[str, Any]]:
    spec = TASK_SPECS[task_name]
    users = sorted(per_user)
    rows: list[dict[str, Any]] = []
    for index, first in enumerate(users):
        for second in users[index + 1 :]:
            first_a = spec.first_condition(per_user[first])
            first_b = spec.second_condition(per_user[first])
            second_a = spec.first_condition(per_user[second])
            second_b = spec.second_condition(per_user[second])
            included = (first_a and second_b) or (second_a and first_b)
            rows.append(
                {
                    "pair": [first, second],
                    "included": included,
                    "first_user_satisfies_first_condition": first_a,
                    "first_user_satisfies_second_condition": first_b,
                    "second_user_satisfies_first_condition": second_a,
                    "second_user_satisfies_second_condition": second_b,
                    "included_by_direction": (
                        "first->second"
                        if first_a and second_b
                        else "second->first"
                        if second_a and first_b
                        else None
                    ),
                }
            )
    return rows


def load_expected_answer_file(path: Path | None, task_name: str) -> str | None:
    if path is None:
        candidate = DEFAULT_ANSWERS_DIR / f"{task_name}.txt"
    elif path.is_dir():
        candidate = path / f"{task_name}.txt"
    else:
        candidate = path
    if not candidate.exists():
        return None
    return candidate.read_text(encoding="utf-8").strip()


def build_audit(
    items: list[dict[str, Any]],
    task_name: str,
    expected_answer_file: Path | None,
) -> dict[str, Any]:
    records = parse_records(items)
    per_user = records_by_user(records)
    pairs = compute_expected_pairs(per_user, task_name)
    answer = format_pairs(pairs)
    expected_answer_text = load_expected_answer_file(expected_answer_file, task_name)
    validation_errors = validate_items(items)
    pair_rows = pair_audit(per_user, task_name)
    included_from_audit = sorted(
        tuple(row["pair"]) for row in pair_rows if row["included"]
    )

    mismatches: list[str] = []
    if validation_errors:
        mismatches.extend(validation_errors)
    if included_from_audit != pairs:
        mismatches.append("pair audit inclusion set does not match compute_expected_pairs output")
    if expected_answer_text is not None and expected_answer_text != answer:
        mismatches.append("computed answer does not match expected answer file")

    return {
        "task": task_name,
        "question": full_question(task_name),
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "record_count": len(items),
        "user_count": len(per_user),
        "expected_pair_count": len(pairs),
        "computed_answer": answer,
        "expected_answer_file_present": expected_answer_text is not None,
        "expected_answer_file_matches": expected_answer_text == answer
        if expected_answer_text is not None
        else None,
        "qualifying_users": qualifying_users(per_user, task_name),
        "category_counts_by_user": category_counts(items),
        "record_evidence_by_user": record_evidence(items),
        "pair_audit": pair_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively verify an OOLONG-Pairs paper task answer from the "
            "stored correct_answer/correct_category fields."
        )
    )
    parser.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument(
        "--expected-answer-file",
        type=Path,
        help=(
            "Optional expected answer file or answer directory. Defaults to "
            "results/oolong_pairs_verified_answers/<task>.txt if present."
        ),
    )
    parser.add_argument("--audit-json", type=Path, help="Write full audit details as JSON.")
    parser.add_argument("--answer-out", type=Path, help="Write the computed answer text.")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit = build_audit(load_items(args.records), args.task, args.expected_answer_file)

    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    if args.answer_out:
        args.answer_out.parent.mkdir(parents=True, exist_ok=True)
        args.answer_out.write_text(audit["computed_answer"] + "\n", encoding="utf-8")

    print(f"{audit['task']}: {audit['status']}")
    print(f"records={audit['record_count']} users={audit['user_count']} pairs={audit['expected_pair_count']}")
    print(f"expected_answer_file_matches={audit['expected_answer_file_matches']}")
    if audit["mismatches"]:
        print("mismatches:")
        for mismatch in audit["mismatches"]:
            print(f"- {mismatch}")

    if args.fail_on_mismatch and audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
