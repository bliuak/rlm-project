from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from generate_oolong_pairs_verified_answers import generate_task_rows, load_items
from oolong_pairs_tasks import TASK_SPECS

PAIR_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
DEFAULT_RECORDS_PATH = Path(__file__).resolve().parents[1] / "synthetic_user_records.json"


def parse_backend_kwargs(values: list[str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid backend kwarg: {value}")
        key, raw = value.split("=", 1)
        kwargs[key] = raw
    return kwargs


def completion_response_text(completion: Any) -> str | None:
    response = completion.get("response") if isinstance(completion, dict) else getattr(completion, "response", None)
    if response is None:
        return None
    return str(response).strip()


def completion_debug_summary(completion: Any) -> Any:
    if isinstance(completion, dict):
        return json_safe(completion)
    data = getattr(completion, "__dict__", None)
    if data:
        return json_safe(data)
    return repr(completion)


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return repr(value)


def parse_pairs(text: str) -> set[tuple[str, str]]:
    normalized = text.strip().lower()
    if normalized in {"", "[]", "empty list", "none"}:
        return set()
    pairs: set[tuple[str, str]] = set()
    for a, b in PAIR_RE.findall(text):
        if a == b:
            continue
        pairs.add(tuple(sorted((a, b), key=int)))
    return pairs


def f1(actual: str, expected: str) -> float:
    a = parse_pairs(actual)
    e = parse_pairs(expected)
    if not a and not e:
        return 1.0
    if not a or not e:
        return 0.0
    tp = len(a & e)
    precision = tp / len(a)
    recall = tp / len(e)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OOLONG-Pairs paper tasks on RLM backends.")
    parser.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--task", action="append", choices=sorted(TASK_SPECS))
    parser.add_argument("--backend", default="openrouter")
    parser.add_argument("--model-name", default="openai/gpt-5.4-mini")
    parser.add_argument("--backend-kwarg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/oolong_pairs_rlm_eval.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_names = args.task or sorted(TASK_SPECS)
    rows = generate_task_rows(load_items(args.records), task_names)

    rlm = None
    if not args.dry_run:
        try:
            from dotenv import load_dotenv
            from rlm import RLM
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RLM execution requires optional dependencies. "
                "Install with `pip install -r requirements.txt`."
            ) from exc
        load_dotenv()
        backend_kwargs = parse_backend_kwargs(args.backend_kwarg)
        backend_kwargs.setdefault("model_name", args.model_name)
        rlm = RLM(
            backend=args.backend,
            backend_kwargs=backend_kwargs,
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0.0
    run_started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            started = time.perf_counter()
            actual = ""
            error = None
            completion = None
            if not args.dry_run:
                try:
                    completion = rlm.completion(row["prompt"])
                    response_text = completion_response_text(completion)
                    if response_text is None:
                        error = "RuntimeError: RLM completion returned no response"
                        actual = ""
                    else:
                        actual = response_text
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"

            score = 0.0 if error else f1(actual, row["expected_answer"])
            latency_ms = (time.perf_counter() - started) * 1000
            total += score
            out = {
                "task": row["task_type"],
                "score": score,
                "expected_answer": row["expected_answer"],
                "expected_pair_count": row["expected_pair_count"],
                "expected_answer_source": row.get("metadata", {}).get("answer_source"),
                "actual_answer": actual,
                "error": error,
                "latency_ms": latency_ms,
                "latency_seconds": latency_ms / 1000,
            }
            if error and completion is not None:
                out["completion"] = completion_debug_summary(completion)
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(
                f"{row['task_type']}: score={score:.4f} "
                f"runtime={latency_ms / 1000:.2f}s error={error}"
            )

    avg = total / len(rows) if rows else 0.0
    total_runtime = time.perf_counter() - run_started
    print(f"\nCompleted {len(rows)} tasks in {total_runtime:.2f}s. Average score: {avg:.4f}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
