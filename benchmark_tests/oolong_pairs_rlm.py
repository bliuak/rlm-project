from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger


class AnswerType(str, Enum):
    NUMERIC = "NUMERIC"
    LABEL = "LABEL"
    COMPARISON = "COMPARISON"
    DATE = "DATE"
    PAIRS = "PAIRS"


@dataclass
class OolongTask:
    id: str
    source: str
    dataset: str
    context: str
    question: str
    expected_answer: str
    answer_type: AnswerType
    task_type: str = ""
    context_length: int = 0
    context_tokens: int | None = None
    expected_pairs: list[tuple[str, str]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = re.sub(r"[*_`]", "", normalized)
    return normalized.strip("\"'")


def stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return stringify_answer(value[0])
        return ", ".join(stringify_answer(item) for item in value)
    return str(value).strip().strip("[]").strip("\"'")


def parse_numeric(answer: str) -> float | None:
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", answer)
    if match is None:
        return None
    return float(match.group(0).replace(",", ""))


def label_score(expected: str, actual: str) -> float:
    return 1.0 if normalize_answer(expected) == normalize_answer(actual) else 0.0


def numeric_score(expected: str, actual: str) -> float:
    expected_number = parse_numeric(expected)
    actual_number = parse_numeric(actual)
    if expected_number is None or actual_number is None:
        return label_score(expected, actual)
    return 0.75 ** abs(expected_number - actual_number)


def comparison_score(expected: str, actual: str) -> float:
    more_variants = {"more", "more common", "greater", "higher", "larger"}
    less_variants = {"less", "less common", "smaller", "lower", "fewer"}
    same_variants = {"same", "equal", "same frequency", "tied"}

    def categorize(text: str) -> str | None:
        normalized = normalize_answer(text)
        if any(variant in normalized for variant in more_variants):
            return "more"
        if any(variant in normalized for variant in less_variants):
            return "less"
        if any(variant in normalized for variant in same_variants):
            return "same"
        return None

    expected_category = categorize(expected)
    actual_category = categorize(actual)
    if expected_category is None or actual_category is None:
        return label_score(expected, actual)
    return 1.0 if expected_category == actual_category else 0.0


def parse_pairs(answer: str) -> set[tuple[str, str]]:
    if normalize_answer(answer) in {"", "[]", "empty list", "none"}:
        return set()

    pairs: set[tuple[str, str]] = set()
    for first, second in re.findall(r"\((\d+)\s*,\s*(\d+)\)", answer):
        pair = tuple(sorted((first, second), key=int))
        pairs.add(pair)
    return pairs


def pair_f1_score(expected_pairs: list[tuple[str, str]], actual: str) -> float:
    expected = {tuple(sorted(pair, key=int)) for pair in expected_pairs}
    actual_pairs = parse_pairs(actual)

    if not expected and not actual_pairs:
        return 1.0
    if not expected or not actual_pairs:
        return 0.0

    true_positives = len(expected & actual_pairs)
    precision = true_positives / len(actual_pairs)
    recall = true_positives / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def map_answer_type(answer_type: Any, expected_answer: str) -> AnswerType:
    if answer_type:
        mapping = {
            "NUMERIC": AnswerType.NUMERIC,
            "NUMERIC_ONE_CLASS": AnswerType.NUMERIC,
            "LABEL": AnswerType.LABEL,
            "COMPARISON": AnswerType.COMPARISON,
            "DATE": AnswerType.DATE,
        }
        mapped = mapping.get(str(answer_type).upper())
        if mapped is not None:
            return mapped

    normalized = normalize_answer(expected_answer)
    if any(word in normalized for word in ("more", "less", "same", "greater", "fewer")):
        return AnswerType.COMPARISON
    if parse_numeric(expected_answer) is not None:
        return AnswerType.NUMERIC
    return AnswerType.LABEL


def score_answer(expected: str, actual: str, answer_type: AnswerType) -> float:
    if not actual.strip():
        return 0.0
    if answer_type == AnswerType.NUMERIC:
        return numeric_score(expected, actual)
    if answer_type == AnswerType.COMPARISON:
        return comparison_score(expected, actual)
    return label_score(expected, actual)


def score_task(task: OolongTask, actual: str) -> float:
    if task.answer_type == AnswerType.PAIRS:
        return pair_f1_score(task.expected_pairs or [], actual)
    return score_answer(task.expected_answer, actual, task.answer_type)


def clean_model_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    non_empty_lines = [line.strip() for line in answer.splitlines() if line.strip()]
    return non_empty_lines[0] if non_empty_lines else answer


def build_prompt(task: OolongTask) -> str:
    return f"""Analyze the following long context and answer the question.

Context:
{task.context}

Question:
{task.question}

Provide only the final answer. Do not include reasoning or explanation."""


def build_completion_payload(task: OolongTask, structured: bool) -> str | dict[str, Any]:
    if not structured:
        return build_prompt(task)

    return {
        "context": task.context,
        "question": task.question,
        "answer_format": (
            "Return only sorted pairs in the format (user_id_1, user_id_2), one per line. "
            "Return [] if there are no matching pairs."
        )
        if task.answer_type == AnswerType.PAIRS
        else "Return only the final answer.",
        "task_type": task.task_type,
    }


def task_record(task: OolongTask) -> dict[str, Any]:
    record = {
        key: value.value if isinstance(value, AnswerType) else value
        for key, value in asdict(task).items()
        if key != "context"
    }
    if task.expected_pairs is not None:
        record["expected_pair_count"] = len(task.expected_pairs)
        record.pop("expected_pairs", None)
    return record


def parse_backend_kwargs(values: list[str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Backend kwarg must be KEY=VALUE, got: {value}")
        key, raw_value = value.split("=", 1)
        try:
            kwargs[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            kwargs[key] = raw_value
    return kwargs


def resolve_optional_text_file(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text(encoding="utf-8").strip()


def resolve_root_prompt(args: argparse.Namespace) -> str:
    if args.root_prompt_file is None:
        return args.root_prompt
    return args.root_prompt_file.read_text(encoding="utf-8").strip()


def make_rlm(args: argparse.Namespace, logger: RLMLogger | None = None) -> RLM:
    backend_kwargs = parse_backend_kwargs(args.backend_kwarg)
    backend_kwargs.setdefault("model_name", args.model_name)

    return RLM(
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        max_timeout=args.max_timeout,
        custom_system_prompt=resolve_optional_text_file(args.custom_system_prompt_file),
        logger=logger,
        verbose=args.verbose,
    )


def should_count_tokens(args: argparse.Namespace) -> bool:
    return (
        args.show_tokens
        or args.min_context_tokens is not None
        or args.max_context_tokens is not None
    )


def build_token_counter(args: argparse.Namespace):
    if not should_count_tokens(args):
        return None

    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "Token-based length filtering requires tiktoken. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    encoding = tiktoken.get_encoding(args.tokenizer)
    return lambda text: len(encoding.encode(text))


def parse_oolong_labeled_context(context: str) -> dict[str, list[dict[str, Any]]]:
    users: dict[str, list[dict[str, Any]]] = {}
    pattern = re.compile(
        r"^Date: (?P<date>[A-Z][a-z]{2} \d{2}, \d{4}) \|\| "
        r"User: (?P<user>\d+) \|\| Instance: (?P<instance>.*?) \|\| "
        r"Label: (?P<label>.+)$"
    )

    for line in context.splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue

        label = match.group("label").strip()
        date_text = match.group("date")
        users.setdefault(match.group("user"), []).append(
            {
                "date": datetime.strptime(date_text, "%b %d, %Y").date(),
                "label": label,
                "instance": match.group("instance"),
            }
        )

    return users


def user_has(entries: list[dict[str, Any]], label: str) -> bool:
    return any(entry["label"] == label for entry in entries)


def user_count(entries: list[dict[str, Any]], label: str) -> int:
    return sum(1 for entry in entries if entry["label"] == label)


def user_all_label_dates_after(entries: list[dict[str, Any]], label: str, cutoff: str) -> bool:
    cutoff_date = datetime.strptime(cutoff, "%b %d, %Y").date()
    dates = [entry["date"] for entry in entries if entry["label"] == label]
    return bool(dates) and all(date > cutoff_date for date in dates)


def user_all_label_dates_before(entries: list[dict[str, Any]], label: str, cutoff: str) -> bool:
    cutoff_date = datetime.strptime(cutoff, "%b %d, %Y").date()
    dates = [entry["date"] for entry in entries if entry["label"] == label]
    return bool(dates) and all(date < cutoff_date for date in dates)


def pair_users(
    users: dict[str, list[dict[str, Any]]],
    first_condition,
    second_condition,
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


PAIR_QUERY_SPECS = {
    "paper_12": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
            "such that one user has at least two instances with numeric value, and the other user has "
            "at least one instance with location and at least one instance with human being. Each of "
            "the questions can be labelled as one of the labels (the data does not provide the labels, "
            "you need to figure out the label from the semantics of the question): description and "
            "abstract concept, entity, human being, numeric value, location, abbreviation. In your "
            "answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines. "
            "Your answer must be sorted by first user ID. If there is no answer, return an empty list []."
        ),
        "first_condition": lambda entries: user_count(entries, "numeric value") >= 2,
        "second_condition": lambda entries: user_has(entries, "location")
        and user_has(entries, "human being"),
    },
    "paper_14": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
            "such that one user has at least one instance with human being and at least one instance "
            "with numeric value, and the other user has exactly two instances with location. Each of "
            "the questions can be labelled as one of the labels (the data does not provide the labels, "
            "you need to figure out the label from the semantics of the question): description and "
            "abstract concept, entity, human being, numeric value, location, abbreviation. In your "
            "answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines. "
            "Your answer must be sorted by first user ID. If there is no answer, return an empty list []."
        ),
        "first_condition": lambda entries: user_has(entries, "human being")
        and user_has(entries, "numeric value"),
        "second_condition": lambda entries: user_count(entries, "location") == 2,
    },
    "paper_15": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
            "such that one user has at least one instance with entity, at least one instance with "
            "location, and at least one instance with abbreviation, and the other user has exactly one "
            "instance with numeric value. Each of the questions can be labelled as one of the labels "
            "(the data does not provide the labels, you need to figure out the label from the semantics "
            "of the question): description and abstract concept, entity, human being, numeric value, "
            "location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
            "separated by newlines. Your answer must be sorted by first user ID. If there is no answer, "
            "return an empty list []."
        ),
        "first_condition": lambda entries: user_has(entries, "entity")
        and user_has(entries, "location")
        and user_has(entries, "abbreviation"),
        "second_condition": lambda entries: user_count(entries, "numeric value") == 1,
    },
    "paper_16": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
            "such that one user has at least one instance with description and abstract concept and at "
            "least one instance with human being, and the other user has at least two instances with "
            "entity and exactly one instance with abbreviation. Each of the questions can be labelled "
            "as one of the labels (the data does not provide the labels, you need to figure out the "
            "label from the semantics of the question): description and abstract concept, entity, "
            "human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
            "format (user_id_1, user_id_2), separated by newlines. Your answer must be sorted by first "
            "user ID. If there is no answer, return an empty list []."
        ),
        "first_condition": lambda entries: user_has(entries, "description and abstract concept")
        and user_has(entries, "human being"),
        "second_condition": lambda entries: user_count(entries, "entity") >= 2
        and user_count(entries, "abbreviation") == 1,
    },
    "paper_20": {
        "question": (
            "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
            "such that one user has at least one instance with numeric value and at least one instance "
            "with human being, and the other user has at least one instance with location, at least one "
            "instance with entity, and exactly one instance with abbreviation. Each of the questions can "
            "be labelled as one of the labels (the data does not provide the labels, you need to figure "
            "out the label from the semantics of the question): description and abstract concept, entity, "
            "human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
            "format (user_id_1, user_id_2), separated by newlines. Your answer must be sorted by first "
            "user ID. If there is no answer, return an empty list []."
        ),
        "first_condition": lambda entries: user_has(entries, "numeric value")
        and user_has(entries, "human being"),
        "second_condition": lambda entries: user_has(entries, "location")
        and user_has(entries, "entity")
        and user_count(entries, "abbreviation") == 1,
    },
}


def build_pair_task(row: dict[str, Any], query_name: str, context_tokens: int | None) -> OolongTask:
    query = PAIR_QUERY_SPECS[query_name]
    users = parse_oolong_labeled_context(row["context_window_text_with_labels"])
    expected_pairs = pair_users(users, query["first_condition"], query["second_condition"])
    expected_answer = (
        "[]"
        if not expected_pairs
        else "\n".join(f"({first}, {second})" for first, second in expected_pairs)
    )

    return OolongTask(
        id=f"{row.get('id')}_{query_name}",
        source="pairs",
        dataset=str(row.get("dataset", "trec_coarse")),
        context=row.get("context_window_text", ""),
        question=query["question"],
        expected_answer=expected_answer,
        answer_type=AnswerType.PAIRS,
        task_type=query_name,
        context_length=len(row.get("context_window_text", "")),
        context_tokens=context_tokens,
        expected_pairs=expected_pairs,
        metadata={
            "base_task_id": row.get("id", ""),
            "context_window_id": row.get("context_window_id", ""),
            "query_name": query_name,
            "unique_users": len(users),
        },
    )


def iter_oolong_tasks(args: argparse.Namespace):
    token_counter = build_token_counter(args)

    if args.source in {"synth", "pairs"}:
        dataset = load_dataset("oolongbench/oolong-synth", split=args.split, streaming=args.streaming)
    else:
        dataset = load_dataset(
            "oolongbench/oolong-real",
            args.config,
            split=args.split,
            streaming=args.streaming,
        )

    yielded = 0
    for idx, row in enumerate(dataset):
        dataset_name = row.get("dataset", args.config if args.source == "real" else "unknown")
        if args.dataset_filter and dataset_name != args.dataset_filter:
            continue

        context = row.get("context_window_text", "")
        context_length = len(context)
        if context_length < args.min_context:
            continue
        if args.max_context is not None and context_length > args.max_context:
            continue

        context_tokens = token_counter(context) if token_counter else None
        if args.min_context_tokens is not None and context_tokens < args.min_context_tokens:
            continue
        if args.max_context_tokens is not None and context_tokens > args.max_context_tokens:
            continue

        expected_answer = stringify_answer(row.get("answer", ""))
        if args.source == "pairs":
            for query_name in args.pair_query:
                task = build_pair_task(row, query_name, context_tokens)
                if args.min_expected_pairs is not None and (
                    task.expected_pairs is None or len(task.expected_pairs) < args.min_expected_pairs
                ):
                    continue
                if args.max_expected_pairs is not None and (
                    task.expected_pairs is not None and len(task.expected_pairs) > args.max_expected_pairs
                ):
                    continue
                yield task

                yielded += 1
                if args.limit is not None and args.limit > 0 and yielded >= args.limit:
                    return
        else:
            task = OolongTask(
                id=str(row.get("id", idx)),
                source=args.source,
                dataset=str(dataset_name),
                context=context,
                question=str(row.get("question", "")),
                expected_answer=expected_answer,
                answer_type=map_answer_type(row.get("answer_type"), expected_answer),
                task_type=str(row.get("task", "")),
                context_length=context_length,
                context_tokens=context_tokens,
                metadata={
                    "task_group": row.get("task_group", ""),
                    "context_window_id": row.get("context_window_id", ""),
                    "input_subset": row.get("input_subset", ""),
                },
            )
            yield task

            yielded += 1
            if args.limit is not None and args.limit > 0 and yielded >= args.limit:
                break


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    root_prompt = resolve_root_prompt(args)
    rlm = None
    if not args.dry_run:
        rlm = make_rlm(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as output_file:
        for index, task in enumerate(iter_oolong_tasks(args), start=1):
            length_parts = [f"{task.context_length:,} chars"]
            if task.context_tokens is not None:
                length_parts.append(f"{task.context_tokens:,} tokens")
            print(
                f"[{index}] {task.id} "
                f"({task.dataset}, {', '.join(length_parts)}, {task.answer_type.value})"
            )

            if args.dry_run:
                result = {
                    "task": task_record(task),
                    "actual_answer": "",
                    "expected_answer": task.expected_answer,
                    "score": None,
                    "latency_ms": 0.0,
                    "error": None,
                    "dry_run": True,
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
                continue

            started_at = time.perf_counter()
            error = None
            actual_answer = ""
            log_path = None
            try:
                task_rlm = rlm
                if args.log_dir is not None:
                    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.id)
                    logger = RLMLogger(
                        log_dir=str(args.log_dir),
                        file_name=f"{safe_task_id}_depth{args.max_depth}",
                    )
                    log_path = logger.log_file_path
                    task_rlm = make_rlm(args, logger=logger)

                if task_rlm is None:
                    raise RuntimeError("RLM client was not initialized")
                if rlm is None:
                    raise RuntimeError("RLM client was not initialized")
                completion = task_rlm.completion(
                    build_completion_payload(task, args.structured_prompt),
                    root_prompt=root_prompt,
                )
                actual_answer = clean_model_answer(completion.response)
            except Exception as exc:  # Keep long runs moving and record the failed task.
                error = str(exc)

            latency_ms = (time.perf_counter() - started_at) * 1000
            score = 0.0 if error else score_task(task, actual_answer)
            result = {
                "task": task_record(task),
                "actual_answer": actual_answer,
                "expected_answer": task.expected_answer,
                "score": score,
                "latency_ms": latency_ms,
                "error": error,
                "log_path": log_path,
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            results.append(result)

            status = "ERROR" if error else f"score={score:.4f}"
            print(f"    expected={task.expected_answer!r} actual={actual_answer!r} {status}")

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    dry_run = any(result.get("dry_run") for result in results)
    completed = [result for result in results if not result["error"]]
    failed = len(results) - len(completed)
    average_score = (
        sum(float(result["score"]) for result in completed if result["score"] is not None)
        / len([result for result in completed if result["score"] is not None])
        if completed and not dry_run
        else 0.0
    )
    print()
    print("Summary")
    print(f"  tasks: {len(results)}")
    if dry_run:
        print("  dry_run: true")
    else:
        print(f"  completed: {len(completed)}")
        print(f"  failed: {failed}")
        print(f"  average_score: {average_score:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an OOLONG/OOLONG-Pairs-style benchmark with the Python RLM package."
    )
    parser.add_argument("--source", choices=["synth", "real", "pairs"], default="synth")
    parser.add_argument("--config", default="dnd", help="Config for oolong-real.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset-filter", default="trec_coarse")
    parser.add_argument(
        "--pair-query",
        action="append",
        choices=sorted(PAIR_QUERY_SPECS),
        default=None,
        help="Pair query to synthesize when --source pairs. Can be passed multiple times.",
    )
    parser.add_argument("--min-expected-pairs", type=int, default=None)
    parser.add_argument("--max-expected-pairs", type=int, default=None)
    parser.add_argument("--min-context", type=int, default=100_000)
    parser.add_argument("--max-context", type=int, default=None)
    parser.add_argument("--min-context-tokens", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--show-tokens", action="store_true")
    parser.add_argument("--limit", type=int, default=5, help="Max tasks to run. Use 0 for no cap.")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--model-name", default="gpt-5.4-mini")
    parser.add_argument("--backend-kwarg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-timeout", type=float, default=None)
    parser.add_argument(
        "--structured-prompt",
        action="store_true",
        help="Pass context/question as a dict instead of one long string.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print matching tasks without calling the RLM backend.",
    )
    parser.add_argument(
        "--root-prompt",
        default=(
            "You are solving OOLONG long-context benchmark questions. "
            "Use the full provided context and return only the answer."
        ),
    )
    parser.add_argument(
        "--root-prompt-file",
        type=Path,
        help="Read the RLM root/system prompt from a text file.",
    )
    parser.add_argument(
        "--custom-system-prompt-file",
        type=Path,
        help="Override the internal RLM system prompt from a text file.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for per-task RLM trajectory JSONL logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/oolong_pairs_rlm.jsonl"),
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    if args.source == "pairs" and args.pair_query is None:
        args.pair_query = ["paper_16"]
    elif args.pair_query is None:
        args.pair_query = []
    if args.source == "pairs":
        args.structured_prompt = True
    results = run_benchmark(args)
    print_summary(results)
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
