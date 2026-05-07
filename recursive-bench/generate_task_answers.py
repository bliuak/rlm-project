from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ANSWER_GENERATORS_DIR = Path(__file__).resolve().parent / "answer-generators"
sys.path.insert(0, str(ANSWER_GENERATORS_DIR))

from generate_task_01_answer import build_audit as build_task_01_audit
from generate_task_02_answer import build_audit as build_task_02_audit
from generate_task_03_answer import build_audit as build_task_03_audit
from generate_task_04_answer import build_audit as build_task_04_audit
from generate_task_05_answer import build_audit as build_task_05_audit
from generate_task_06_answer import build_audit as build_task_06_audit
from generate_task_07_answer import build_audit as build_task_07_audit
from generate_task_08_answer import build_audit as build_task_08_audit
from generate_task_09_answer import build_audit as build_task_09_audit
from generate_task_10_answer import build_audit as build_task_10_audit
from oolong_pairs_tasks import (
    TASK_SPECS,
    compute_expected_pairs,
    format_pairs,
    full_question,
    parse_records,
    records_by_user,
    render_context,
    render_task_prompt,
)


DEFAULT_RECORDS_PATH = Path(__file__).resolve().parents[1] / "synthetic_user_records.json"

StandaloneAuditBuilder = Callable[[list[dict], Path | None], dict]
STANDALONE_ANSWER_BUILDERS: dict[str, tuple[str, StandaloneAuditBuilder]] = {
    "task_01": ("recursive-bench/answer-generators/generate_task_01_answer.py", build_task_01_audit),
    "task_02": ("recursive-bench/answer-generators/generate_task_02_answer.py", build_task_02_audit),
    "task_03": ("recursive-bench/answer-generators/generate_task_03_answer.py", build_task_03_audit),
    "task_04": ("recursive-bench/answer-generators/generate_task_04_answer.py", build_task_04_audit),
    "task_05": ("recursive-bench/answer-generators/generate_task_05_answer.py", build_task_05_audit),
    "task_06": ("recursive-bench/answer-generators/generate_task_06_answer.py", build_task_06_audit),
    "task_07": ("recursive-bench/answer-generators/generate_task_07_answer.py", build_task_07_audit),
    "task_08": ("recursive-bench/answer-generators/generate_task_08_answer.py", build_task_08_audit),
    "task_09": ("recursive-bench/answer-generators/generate_task_09_answer.py", build_task_09_audit),
    "task_10": ("recursive-bench/answer-generators/generate_task_10_answer.py", build_task_10_audit),
}


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a top-level 'items' list")
    return items


def expected_answer_for_task(
    items: list[dict],
    per_user: dict,
    task_name: str,
) -> tuple[str, int, str]:
    standalone = STANDALONE_ANSWER_BUILDERS.get(task_name)
    if standalone:
        script_path, build_audit = standalone
        audit = build_audit(items, None)
        return (
            str(audit["computed_answer"]),
            int(audit["pair_count"]),
            script_path,
        )

    expected_pairs = compute_expected_pairs(per_user, task_name)
    return (
        format_pairs(expected_pairs),
        len(expected_pairs),
        "recursive-bench/oolong_pairs_tasks.py",
    )


def generate_task_rows(items: list[dict], task_names: list[str]) -> list[dict]:
    records = parse_records(items)
    per_user = records_by_user(records)
    context = render_context(records)

    rows: list[dict] = []
    for task_name in task_names:
        expected_answer, expected_pair_count, answer_source = expected_answer_for_task(
            items,
            per_user,
            task_name,
        )
        rows.append(
            {
                "id": f"synthetic_{task_name}",
                "task_type": task_name,
                "context": context,
                "question": full_question(task_name),
                "prompt": render_task_prompt(records, task_name),
                "expected_answer": expected_answer,
                "expected_pair_count": expected_pair_count,
                "metadata": {
                    "source": "synthetic_user_records",
                    "record_count": len(records),
                    "user_count": len(per_user),
                    "answer_source": answer_source,
                },
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate verified OOLONG-Pairs benchmark task answers (task_01 through task_20)."
    )
    parser.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument(
        "--task",
        action="append",
        choices=sorted(TASK_SPECS),
        help="Only generate selected tasks. Defaults to all task_01..task_20.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("results/oolong_pairs_verified_tasks.jsonl"),
    )
    parser.add_argument(
        "--output-answers-dir",
        type=Path,
        default=Path("results/oolong_pairs_verified_answers"),
        help="Write one answer text file per task.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_names = args.task or sorted(TASK_SPECS)
    rows = generate_task_rows(load_items(args.records), task_names)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.output_answers_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (args.output_answers_dir / f"{row['task_type']}.txt").write_text(
            row["expected_answer"] + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {len(rows)} tasks to {args.output_jsonl}")
    print(f"Wrote per-task answers under {args.output_answers_dir}")


if __name__ == "__main__":
    main()
