from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pyarrow.ipc as ipc

try:
    import tiktoken
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".python-deps"))
    import tiktoken


LABELS = (
    "abbreviation",
    "entity",
    "human being",
    "numeric value",
    "location",
    "description and abstract concept",
)

LABEL_LIST = ", ".join(f"'{label}'" for label in LABELS)
CONTEXT_CACHE_ROOT = Path(
    ".hf-cache/datasets/oolongbench___oolong-synth/default/0.0.0"
)

LABELED_LINE_RE = re.compile(
    r"^Date: (?P<date>[A-Z][a-z]{2} \d{2}, \d{4}) \|\| "
    r"User: (?P<user>\d+) \|\| Instance: (?P<instance>.*?) "
    r"\|\| Label: (?P<label>.+)$"
)


@dataclass(frozen=True)
class Record:
    date: str
    user: str
    instance: str
    label: str

    @property
    def public_line(self) -> str:
        return f"Date: {self.date} || User: {self.user} || Instance: {self.instance}"


def user_has(entries: list[Record], label: str) -> bool:
    return any(entry.label == label for entry in entries)


def user_count(entries: list[Record], label: str) -> int:
    return sum(1 for entry in entries if entry.label == label)


PAIR_QUERY_SPECS: dict[str, dict[str, Any]] = {
    "paper_12": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, "
            "list lower ID first) such that one user has at least two instances "
            "with numeric value, and the other user has at least one instance "
            "with location and at least one instance with human being. Each of "
            "the questions can be labelled as one of the labels (the data does "
            "not provide the labels, you need to figure out the label from the "
            "semantics of the question): description and abstract concept, "
            "entity, human being, numeric value, location, abbreviation. In your "
            "answer, list all pairs in the format (user_id_1, user_id_2), "
            "separated by newlines. Your answer must be sorted by first user ID. "
            "If there is no answer, return an empty list []."
        ),
        "first_condition": lambda entries: user_count(entries, "numeric value") >= 2,
        "second_condition": lambda entries: user_has(entries, "location")
        and user_has(entries, "human being"),
    },
    "paper_14": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, "
            "list lower ID first) such that one user has at least one instance "
            "with human being and at least one instance with numeric value, and "
            "the other user has exactly two instances with location. Each of the "
            "questions can be labelled as one of the labels (the data does not "
            "provide the labels, you need to figure out the label from the "
            "semantics of the question): description and abstract concept, "
            "entity, human being, numeric value, location, abbreviation. In your "
            "answer, list all pairs in the format (user_id_1, user_id_2), "
            "separated by newlines. Your answer must be sorted by first user ID. "
            "If there is no answer, return an empty list []."
        ),
        "first_condition": lambda entries: user_has(entries, "human being")
        and user_has(entries, "numeric value"),
        "second_condition": lambda entries: user_count(entries, "location") == 2,
    },
    "paper_15": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, "
            "list lower ID first) such that one user has at least one instance "
            "with entity, at least one instance with location, and at least one "
            "instance with abbreviation, and the other user has exactly one "
            "instance with numeric value. Each of the questions can be labelled "
            "as one of the labels (the data does not provide the labels, you "
            "need to figure out the label from the semantics of the question): "
            "description and abstract concept, entity, human being, numeric "
            "value, location, abbreviation. In your answer, list all pairs in "
            "the format (user_id_1, user_id_2), separated by newlines. Your "
            "answer must be sorted by first user ID. If there is no answer, "
            "return an empty list []."
        ),
        "first_condition": lambda entries: user_has(entries, "entity")
        and user_has(entries, "location")
        and user_has(entries, "abbreviation"),
        "second_condition": lambda entries: user_count(entries, "numeric value") == 1,
    },
    "paper_16": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, "
            "list lower ID first) such that one user has at least one instance "
            "with description and abstract concept and at least one instance "
            "with human being, and the other user has at least two instances "
            "with entity and exactly one instance with abbreviation. Each of "
            "the questions can be labelled as one of the labels (the data does "
            "not provide the labels, you need to figure out the label from the "
            "semantics of the question): description and abstract concept, "
            "entity, human being, numeric value, location, abbreviation. In your "
            "answer, list all pairs in the format (user_id_1, user_id_2), "
            "separated by newlines. Your answer must be sorted by first user ID. "
            "If there is no answer, return an empty list []."
        ),
        "first_condition": lambda entries: user_has(
            entries, "description and abstract concept"
        )
        and user_has(entries, "human being"),
        "second_condition": lambda entries: user_count(entries, "entity") >= 2
        and user_count(entries, "abbreviation") == 1,
    },
    "paper_20": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, "
            "list lower ID first) such that one user has at least one instance "
            "with numeric value and at least one instance with human being, and "
            "the other user has at least one instance with location, at least "
            "one instance with entity, and exactly one instance with "
            "abbreviation. Each of the questions can be labelled as one of the "
            "labels (the data does not provide the labels, you need to figure "
            "out the label from the semantics of the question): description and "
            "abstract concept, entity, human being, numeric value, location, "
            "abbreviation. In your answer, list all pairs in the format "
            "(user_id_1, user_id_2), separated by newlines. Your answer must be "
            "sorted by first user ID. If there is no answer, return an empty "
            "list []."
        ),
        "first_condition": lambda entries: user_has(entries, "numeric value")
        and user_has(entries, "human being"),
        "second_condition": lambda entries: user_has(entries, "location")
        and user_has(entries, "entity")
        and user_count(entries, "abbreviation") == 1,
    },
}


def latest_cache_dir(root: Path) -> Path:
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No OOLONG cache directories found under {root}")
    return candidates[-1]


def scalar_row(batch: Any, index: int) -> dict[str, Any]:
    data = batch.slice(index, 1).to_pydict()
    return {key: values[0] for key, values in data.items()}


def iter_base_rows(args: argparse.Namespace):
    seen: set[tuple[str, int, int]] = set()
    cache_dir = latest_cache_dir(args.cache_root)
    files = sorted(cache_dir.glob(f"oolong-synth-{args.split}-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No cached {args.split!r} Arrow files found in {cache_dir}")

    for path in files:
        reader = ipc.open_stream(str(path))
        for batch in reader:
            metadata = batch.select(
                ["id", "context_len", "dataset", "context_window_id"]
            ).to_pydict()
            for index, dataset in enumerate(metadata["dataset"]):
                if dataset != args.dataset:
                    continue
                context_len = metadata["context_len"][index]
                if context_len < args.min_context_len:
                    continue
                context_window_id = metadata["context_window_id"][index]
                key = (dataset, context_len, context_window_id)
                if key in seen:
                    continue
                seen.add(key)
                yield scalar_row(batch, index)


def parse_labeled_records(labeled_context: str) -> list[Record]:
    records: list[Record] = []
    for line in labeled_context.splitlines():
        match = LABELED_LINE_RE.match(line.strip())
        if not match:
            continue
        label = match.group("label").strip()
        if label not in LABELS:
            raise ValueError(f"Unexpected label {label!r}")
        records.append(
            Record(
                date=match.group("date"),
                user=match.group("user"),
                instance=match.group("instance").rstrip(),
                label=label,
            )
        )
    return records


def build_public_context(records: list[Record]) -> str:
    count = len(records)
    preamble = (
        f"The following lines contain {count} general-knowledge questions, one per line. "
        f"Each question has an answer that can be described as one of 6 categories: "
        f"{LABEL_LIST}.\n\n"
        f"You will be asked to answer questions about the aggregate label statistics "
        f"across all {count} examples in this dataset. Do not try to guess, estimate, "
        f"or approximate the result. Calculate the exact answer given these datapoints.\n\n"
    )
    recall = (
        f"\nRecall: the preceding lines contain {count} general-knowledge questions, "
        f"one per line. Each question has an answer that can be described as one of "
        f"6 categories: {LABEL_LIST}.\n\n"
        f"You will be asked to answer questions about the aggregate label statistics "
        f"across all {count} examples in this dataset. Do not try to guess, estimate, "
        f"or approximate the result. Calculate the exact answer given these datapoints.\n\n"
    )
    return preamble + "\n".join(record.public_line for record in records) + recall


def trim_to_target(
    records: list[Record],
    encoder: Any,
    target_tokens: int,
) -> tuple[list[Record], str, int]:
    def token_count(n: int) -> tuple[str, int]:
        context = build_public_context(records[:n])
        return context, len(encoder.encode(context))

    low = 1
    high = len(records)
    below: tuple[int, str, int] | None = None
    above: tuple[int, str, int] | None = None

    while low <= high:
        mid = (low + high) // 2
        context, tokens = token_count(mid)
        if tokens <= target_tokens:
            below = (mid, context, tokens)
            low = mid + 1
        else:
            above = (mid, context, tokens)
            high = mid - 1

    candidates = [candidate for candidate in (below, above) if candidate is not None]
    if below is not None:
        for n in range(max(1, below[0] - 5), min(len(records), below[0] + 6) + 1):
            context, tokens = token_count(n)
            candidates.append((n, context, tokens))

    best = min(candidates, key=lambda item: abs(item[2] - target_tokens))
    n, context, tokens = best
    return records[:n], context, tokens


def group_by_user(records: list[Record]) -> dict[str, list[Record]]:
    users: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        users[record.user].append(record)
    return users


def pair_users(
    users: dict[str, list[Record]],
    first_condition: Callable[[list[Record]], bool],
    second_condition: Callable[[list[Record]], bool],
) -> list[tuple[str, str]]:
    first_users = {user for user, entries in users.items() if first_condition(entries)}
    second_users = {user for user, entries in users.items() if second_condition(entries)}

    pairs: set[tuple[str, str]] = set()
    for first in first_users:
        for second in second_users:
            if first == second:
                continue
            pairs.add(tuple(sorted((first, second), key=int)))
    return sorted(pairs, key=lambda pair: (int(pair[0]), int(pair[1])))


def task_record(task: dict[str, Any], include_context: bool) -> dict[str, Any]:
    record = {"task": dict(task)}
    if not include_context:
        record["task"].pop("context", None)
        record["task"].pop("expected_answer", None)
    return record


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    encoder = tiktoken.get_encoding(args.tokenizer)
    tasks: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for row in iter_base_rows(args):
        records = parse_labeled_records(row["context_window_text_with_labels"])
        if not records:
            continue

        trimmed_records, context, context_tokens = trim_to_target(
            records,
            encoder,
            args.target_tokens,
        )
        users = group_by_user(trimmed_records)

        for query_name, query in PAIR_QUERY_SPECS.items():
            expected_pairs = pair_users(
                users,
                query["first_condition"],
                query["second_condition"],
            )
            if args.min_expected_pairs is not None:
                if len(expected_pairs) < args.min_expected_pairs:
                    continue
            if args.max_expected_pairs is not None:
                if len(expected_pairs) > args.max_expected_pairs:
                    continue

            expected_answer = (
                "[]"
                if not expected_pairs
                else "\n".join(f"({first}, {second})" for first, second in expected_pairs)
            )
            task = {
                "id": (
                    f"{row['id']}_{query_name}_"
                    f"{round(context_tokens / 1000)}k"
                ),
                "source": "pairs",
                "dataset": row["dataset"],
                "context": context,
                "question": query["question"],
                "expected_answer": expected_answer,
                "answer_type": "PAIRS",
                "task_type": query_name,
                "context_length": len(context),
                "context_tokens": context_tokens,
                "expected_pair_count": len(expected_pairs),
                "metadata": {
                    "base_task_id": row["id"],
                    "base_context_len": row["context_len"],
                    "context_window_id": row["context_window_id"],
                    "query_name": query_name,
                    "record_count": len(trimmed_records),
                    "unique_users": len(users),
                    "target_tokens": args.target_tokens,
                    "tokenizer": args.tokenizer,
                    "source_split": args.split,
                },
            }
            tasks.append(task_record(task, include_context=True))
            manifest.append(task_record(task, include_context=False))

            if len(tasks) >= args.num_tasks:
                return tasks, manifest

    return tasks, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate OOLONG-Pairs-style tasks near a target token count."
    )
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--target-tokens", type=int, default=500_000)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset", default="trec_coarse")
    parser.add_argument("--min-context-len", type=int, default=1_048_576)
    parser.add_argument("--min-expected-pairs", type=int, default=1)
    parser.add_argument("--max-expected-pairs", type=int, default=None)
    parser.add_argument("--cache-root", type=Path, default=CONTEXT_CACHE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/oolong_pairs_500k_tasks.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/oolong_pairs_500k_manifest.jsonl"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks, manifest = generate(args)
    if len(tasks) < args.num_tasks:
        raise RuntimeError(f"Generated {len(tasks)} tasks, expected {args.num_tasks}")
    write_jsonl(args.output, tasks)
    write_jsonl(args.manifest, manifest)
    print(f"Wrote {len(tasks)} tasks to {args.output}")
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
