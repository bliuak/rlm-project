from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from generate_task_answers import generate_task_rows, load_items
from oolong_pairs_tasks import TASK_SPECS

PAIR_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
DEFAULT_RECORDS_PATH = Path(__file__).resolve().parents[1] / "synthetic_user_records.json"
FULL_RUN_DEPTHS = (1, 2)


def completion_response_text(completion: Any) -> str | None:
    response = completion.get("response") if isinstance(completion, dict) else getattr(completion, "response", None)
    if response is None:
        return None
    return str(response).strip()


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


def build_rlm(args: argparse.Namespace, max_depth: int) -> Any:
    try:
        from dotenv import load_dotenv
        from rlm import RLM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RLM execution requires optional dependencies. "
            "Install with `pip install -r requirements.txt`."
        ) from exc

    load_dotenv()
    return RLM(
        backend="openrouter",
        backend_kwargs={"model_name": args.model_name},
        max_depth=max_depth,
        max_iterations=args.max_iterations,
    )


def run_task(
    row: dict[str, Any],
    rlm: Any | None,
    max_depth: int,
    model_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    actual = ""
    error = None

    if rlm is not None:
        try:
            response_text = completion_response_text(rlm.completion(row["prompt"]))
            if response_text is None:
                error = "RuntimeError: RLM completion returned no response"
            else:
                actual = response_text
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    score = 0.0 if error else f1(actual, row["expected_answer"])
    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "task": row["task_type"],
        "model": model_name,
        "max_depth": max_depth,
        "score": score,
        "expected_answer": row["expected_answer"],
        "expected_pair_count": row["expected_pair_count"],
        "expected_answer_source": row.get("metadata", {}).get("answer_source"),
        "actual_answer": actual,
        "error": error,
        "latency_ms": latency_ms,
        "latency_seconds": latency_ms / 1000,
    }


def output_for_depth(output: Path, max_depth: int) -> Path:
    return output.with_name(f"{output.stem}_max_depth_{max_depth}{output.suffix}")


def summary_output_for(output: Path) -> Path:
    return output.with_suffix(".txt")


def run_rows(
    rows: list[dict[str, Any]],
    rlm: Any | None,
    output: Path,
    max_depth: int,
    model_name: str,
) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output = summary_output_for(output)
    total = 0.0
    run_started = time.perf_counter()
    with output.open("w", encoding="utf-8") as handle, summary_output.open(
        "w",
        encoding="utf-8",
    ) as summary:
        summary.write(f"model: {model_name}\n")
        summary.write(f"max_depth: {max_depth}\n\n")
        summary.write("task\tmodel\tmax_depth\tscore\ttime_seconds\terror\n")
        for row in rows:
            out = run_task(row, rlm, max_depth, model_name)
            total += out["score"]
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            summary.write(
                f"{out['task']}\t{out['model']}\t{out['max_depth']}\t"
                f"{out['score']:.4f}\t{out['latency_seconds']:.2f}\t"
                f"{out['error'] or ''}\n"
            )
            print(
                f"{out['task']}: max_depth={max_depth} score={out['score']:.4f} "
                f"runtime={out['latency_seconds']:.2f}s error={out['error']}"
            )

    avg = total / len(rows) if rows else 0.0
    total_runtime = time.perf_counter() - run_started
    with summary_output.open("a", encoding="utf-8") as summary:
        summary.write("\n")
        summary.write(f"completed_tasks: {len(rows)}\n")
        summary.write(f"total_runtime_seconds: {total_runtime:.2f}\n")
        summary.write(f"average_score: {avg:.4f}\n")
    print(f"\nCompleted {len(rows)} tasks in {total_runtime:.2f}s. Average score: {avg:.4f}")
    print(f"Results: {output}")
    print(f"Text summary: {summary_output}")
    return avg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OOLONG-Pairs benchmark tasks on OpenRouter via RLM.")
    parser.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--task", action="append", choices=sorted(TASK_SPECS))
    parser.add_argument("--model-name", default="openai/gpt-5.4-mini")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Run all tasks twice, once with max-depth 1 and once with max-depth 2.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/oolong_pairs_rlm_eval.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.full_run and args.task:
        raise SystemExit("--full-run runs all tasks; remove --task to use it.")

    task_names = sorted(TASK_SPECS) if args.full_run else args.task or sorted(TASK_SPECS)
    rows = generate_task_rows(load_items(args.records), task_names)
    if args.full_run:
        for max_depth in FULL_RUN_DEPTHS:
            print(f"\nRunning full benchmark with max_depth={max_depth}")
            rlm = None if args.dry_run else build_rlm(args, max_depth)
            run_rows(
                rows,
                rlm,
                output_for_depth(args.output, max_depth),
                max_depth,
                args.model_name,
            )
        return

    rlm = None if args.dry_run else build_rlm(args, args.max_depth)
    run_rows(rows, rlm, args.output, args.max_depth, args.model_name)


if __name__ == "__main__":
    main()
