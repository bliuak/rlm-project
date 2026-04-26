from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger


PAIR_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")

# Edit these defaults for local benchmark runs. CLI flags still override them.
BENCHMARK_BACKEND = "openrouter"
BENCHMARK_MODEL_NAME = "openai/gpt-5.4-mini"

# None means RLM uses its default root/user prompt behavior.
BENCHMARK_ROOT_PROMPT: str | None = None


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc


def load_tasks(path: Path, limit: int | None = None, start: int = 0) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(read_jsonl(path)):
        if index < start:
            continue
        task = row.get("task", row)
        if "id" not in task:
            raise ValueError(f"Task at index {index} has no id")
        if "context" not in task:
            raise ValueError(f"Task {task['id']} has no context")
        if "question" not in task:
            raise ValueError(f"Task {task['id']} has no question")
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def parse_pairs(text: str | None) -> set[tuple[str, str]]:
    if not text:
        return set()
    normalized = text.strip()
    if normalized in {"[]", "empty list", "none"}:
        return set()

    pairs: set[tuple[str, str]] = set()
    for first, second in PAIR_RE.findall(normalized):
        if first == second:
            continue
        pairs.add(tuple(sorted((first, second), key=int)))
    return pairs


def pair_metrics(actual: str | None, expected: str | None) -> dict[str, Any]:
    actual_pairs = parse_pairs(actual)
    expected_pairs = parse_pairs(expected)
    true_positives = len(actual_pairs & expected_pairs)
    precision = true_positives / len(actual_pairs) if actual_pairs else (1.0 if not expected_pairs else 0.0)
    recall = true_positives / len(expected_pairs) if expected_pairs else (1.0 if not actual_pairs else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "score": f1,
        "exact_match": actual_pairs == expected_pairs,
        "actual_pair_count": len(actual_pairs),
        "expected_pair_count": len(expected_pairs),
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "context": task["context"],
        "question": task["question"],
        "answer_format": (
            "Return only sorted pairs in the format (user_id_1, user_id_2), "
            "one per line. Return [] if there are no matching pairs."
        ),
        "task_type": task.get("task_type"),
        "metadata": task.get("metadata", {}),
    }


def sanitize_file_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:160]


def make_rlm(args: argparse.Namespace, logger: RLMLogger | None) -> RLM:
    backend_kwargs: dict[str, Any] = dict(args.backend_kwarg)
    backend_kwargs.setdefault("model_name", args.model_name)

    return RLM(
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        max_timeout=args.max_timeout,
        max_budget=args.max_budget,
        max_tokens=args.max_tokens,
        max_errors=args.max_errors,
        logger=logger,
        verbose=args.verbose,
        compaction=args.compaction,
    )


def build_result_record(
    task: dict[str, Any],
    actual_answer: str,
    completion: Any | None,
    latency_ms: float,
    error: str | None,
    log_path: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected_answer = task.get("expected_answer")
    metrics = pair_metrics(actual_answer, expected_answer)

    task_summary = {
        "id": task.get("id"),
        "source": task.get("source"),
        "dataset": task.get("dataset"),
        "question": task.get("question"),
        "answer_type": task.get("answer_type"),
        "task_type": task.get("task_type"),
        "context_length": task.get("context_length", len(task.get("context", ""))),
        "context_tokens": task.get("context_tokens"),
        "expected_pair_count": task.get("expected_pair_count"),
        "metadata": task.get("metadata", {}),
        "context_sha256": sha256_text(task.get("context")),
        "expected_answer_sha256": sha256_text(expected_answer),
    }
    if args.include_context:
        task_summary["context"] = task.get("context")
    if args.include_expected_answer:
        task_summary["expected_answer"] = expected_answer

    record: dict[str, Any] = {
        "task": task_summary,
        "actual_answer": actual_answer,
        "actual_answer_sha256": sha256_text(actual_answer),
        "score": metrics["score"],
        "metrics": metrics,
        "latency_ms": latency_ms,
        "error": error,
        "log_path": log_path,
        "dry_run": args.dry_run,
    }

    if completion is not None:
        record["rlm"] = {
            "root_model": completion.root_model,
            "execution_time": completion.execution_time,
            "usage_summary": completion.usage_summary.to_dict(),
        }
    return record


def parse_backend_kwargs(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Backend kwarg must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def require_runtime_config(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    if not args.model_name:
        raise RuntimeError("Set --model-name or RLM_MODEL_NAME/OPENROUTER_MODEL.")

    required_key = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(args.backend)
    if required_key and not os.getenv(required_key):
        raise RuntimeError(f"{required_key} is required for backend={args.backend!r}.")


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    args.backend_kwarg = parse_backend_kwargs(args.backend_kwarg)
    if args.no_log:
        args.log_dir = None
    require_runtime_config(args)

    tasks = load_tasks(args.tasks, limit=args.limit, start=args.start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    with args.output.open("a" if args.append else "w", encoding="utf-8") as output:
        for index, task in enumerate(tasks, args.start + 1):
            task_id = str(task["id"])
            print(f"[{index}] running {task_id}", flush=True)
            logger = (
                RLMLogger(
                    log_dir=str(args.log_dir),
                    file_name=sanitize_file_name(task_id),
                )
                if args.log_dir and not args.dry_run
                else None
            )
            completion = None
            actual_answer = ""
            error = None
            start = time.perf_counter()
            try:
                if args.dry_run:
                    actual_answer = ""
                else:
                    rlm = make_rlm(args, logger)
                    completion = rlm.completion(
                        build_payload(task),
                        root_prompt=args.root_prompt,
                    )
                    actual_answer = completion.response
            except Exception as exc:  # Keep benchmark sweeps moving.
                error = f"{type(exc).__name__}: {exc}"
                print(f"[{index}] error {task_id}: {error}", flush=True)

            latency_ms = (time.perf_counter() - start) * 1000
            record = build_result_record(
                task=task,
                actual_answer=actual_answer,
                completion=completion,
                latency_ms=latency_ms,
                error=error,
                log_path=logger.log_file_path if logger else None,
                args=args,
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            print(
                f"[{index}] done {task_id}: score={record['score']:.4f} "
                f"latency_ms={latency_ms:.0f}",
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RLM on generated OOLONG-Pairs tasks.")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("results/oolong_pairs_500k_tasks.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/oolong_pairs_500k_rlm_results.jsonl"),
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="Zero-based task offset.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default=BENCHMARK_BACKEND)
    parser.add_argument(
        "--model-name",
        default=BENCHMARK_MODEL_NAME,
    )
    parser.add_argument(
        "--backend-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-timeout", type=float, default=None)
    parser.add_argument("--max-budget", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-errors", type=int, default=None)
    parser.add_argument("--compaction", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--include-expected-answer", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/oolong_pairs_500k"))
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--root-prompt", default=BENCHMARK_ROOT_PROMPT)
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
