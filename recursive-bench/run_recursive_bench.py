from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

from generate_task_answers import generate_task_rows, load_items
from oolong_pairs_tasks import TASK_SPECS

PAIR_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
DEFAULT_RECORDS_PATH = Path(__file__).resolve().parents[1] / "synthetic_user_records.json"
FULL_RUN_DEPTHS = (1, 2)
MAX_COMPLETION_RETRIES = 2


def default_output_path() -> Path:
    return Path("results") / f"results_{time.strftime('%H%M%S')}.txt"


def completion_response_text(completion: Any) -> str | None:
    response = completion.get("response") if isinstance(completion, dict) else getattr(completion, "response", None)
    if response is None:
        return None
    return str(response).strip()


def completion_usage_summary(completion: Any) -> Any | None:
    if isinstance(completion, dict):
        return completion.get("usage_summary")
    return getattr(completion, "usage_summary", None)


def usage_total(usage_summary: Any | None, attr_name: str) -> int | None:
    if usage_summary is None:
        return None

    value = usage_summary.get(attr_name) if isinstance(usage_summary, dict) else getattr(usage_summary, attr_name, None)
    if value is not None:
        return int(value)

    summaries = (
        usage_summary.get("model_usage_summaries")
        if isinstance(usage_summary, dict)
        else getattr(usage_summary, "model_usage_summaries", None)
    )
    if not summaries:
        return None

    model_attr = attr_name.removeprefix("total_")
    total = 0
    found = False
    for model_usage in summaries.values():
        model_value = (
            model_usage.get(model_attr)
            if isinstance(model_usage, dict)
            else getattr(model_usage, model_attr, None)
        )
        if model_value is None:
            model_value = (
                model_usage.get(attr_name)
                if isinstance(model_usage, dict)
                else getattr(model_usage, attr_name, None)
            )
        if model_value is not None:
            total += int(model_value)
            found = True
    return total if found else None


def usage_cost(usage_summary: Any | None) -> float | None:
    if usage_summary is None:
        return None

    value = usage_summary.get("total_cost") if isinstance(usage_summary, dict) else getattr(usage_summary, "total_cost", None)
    if value is not None:
        return float(value)

    summaries = (
        usage_summary.get("model_usage_summaries")
        if isinstance(usage_summary, dict)
        else getattr(usage_summary, "model_usage_summaries", None)
    )
    if not summaries:
        return None

    total = 0.0
    found = False
    for model_usage in summaries.values():
        model_cost = (
            model_usage.get("total_cost")
            if isinstance(model_usage, dict)
            else getattr(model_usage, "total_cost", None)
        )
        if model_cost is not None:
            total += float(model_cost)
            found = True
    return total if found else None


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
    usage_summary = None

    if rlm is not None:
        for attempt in range(MAX_COMPLETION_RETRIES + 1):
            try:
                completion = rlm.completion(row["prompt"])
                response_text = completion_response_text(completion)
                if response_text is None:
                    raise RuntimeError("RLM completion returned no response")
                actual = response_text
                usage_summary = completion_usage_summary(completion)
                error = None
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < MAX_COMPLETION_RETRIES:
                    time.sleep(1)

    output_pair_count = len(parse_pairs(actual))
    score = 0.0 if error else f1(actual, row["expected_answer"])
    latency_ms = (time.perf_counter() - started) * 1000
    total_input_tokens = usage_total(usage_summary, "total_input_tokens")
    total_output_tokens = usage_total(usage_summary, "total_output_tokens")
    total_tokens = (
        total_input_tokens + total_output_tokens
        if total_input_tokens is not None and total_output_tokens is not None
        else None
    )
    return {
        "task": row["task_type"],
        "model": model_name,
        "max_depth": max_depth,
        "score": round(score, 4),
        "expected_pair_count": row["expected_pair_count"],
        "output_pair_count": output_pair_count,
        "error": error,
        "attempts": 0 if rlm is None else attempt + 1,
        "latency_ms": int(round(latency_ms)),
        "latency_seconds": round(latency_ms / 1000, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_cost": usage_cost(usage_summary),
    }


def run_rows(
    rows: list[dict[str, Any]],
    rlm: Any | None,
    max_depth: int,
    model_name: str,
    output_handle: Any,
) -> float:
    total = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    has_cost = False
    run_started = time.perf_counter()
    output_handle.write(f"model: {model_name}\n")
    output_handle.write(f"max_depth: {max_depth}\n\n")
    output_handle.write(
        "task\tmodel\tmax_depth\tscore\texpected_pairs\toutput_pairs\t"
        "time_seconds\tinput_tokens\toutput_tokens\ttotal_tokens\tcost_usd\terror\n"
    )
    for row in rows:
        out = run_task(row, rlm, max_depth, model_name)
        total += out["score"]
        if out["total_input_tokens"] is not None:
            total_input_tokens += out["total_input_tokens"]
        if out["total_output_tokens"] is not None:
            total_output_tokens += out["total_output_tokens"]
        if out["total_tokens"] is not None:
            total_tokens += out["total_tokens"]
        if out["total_cost"] is not None:
            total_cost += out["total_cost"]
            has_cost = True
        cost_text = f"{out['total_cost']:.6f}" if out["total_cost"] is not None else ""
        output_handle.write(
            f"{out['task']}\t{out['model']}\t{out['max_depth']}\t"
            f"{out['score']:.4f}\t{out['expected_pair_count']}\t"
            f"{out['output_pair_count']}\t{out['latency_seconds']:.2f}\t"
            f"{out['total_input_tokens'] if out['total_input_tokens'] is not None else ''}\t"
            f"{out['total_output_tokens'] if out['total_output_tokens'] is not None else ''}\t"
            f"{out['total_tokens'] if out['total_tokens'] is not None else ''}\t"
            f"{cost_text}\t"
            f"{out['error'] or ''}\n"
        )
        print(
            f"{out['task']}: max_depth={max_depth} score={out['score']:.4f} "
            f"runtime={out['latency_seconds']:.2f}s "
            f"output_tokens={out['total_output_tokens']} error={out['error']}"
        )

    avg = total / len(rows) if rows else 0.0
    total_runtime = time.perf_counter() - run_started
    output_handle.write("\n")
    output_handle.write(f"completed_tasks: {len(rows)}\n")
    output_handle.write(f"total_runtime_seconds: {total_runtime:.2f}\n")
    output_handle.write(f"average_score: {avg:.4f}\n")
    output_handle.write(f"total_input_tokens: {total_input_tokens}\n")
    output_handle.write(f"total_output_tokens: {total_output_tokens}\n")
    output_handle.write(f"total_tokens: {total_tokens}\n")
    if has_cost:
        output_handle.write(f"total_cost_usd: {total_cost:.6f}\n")
    print(f"\nCompleted {len(rows)} tasks in {total_runtime:.2f}s. Average score: {avg:.4f}")
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
    parser.add_argument(
        "--output",
        type=Path,
        help="Text results path. Defaults to results/results_HHMMSS.txt.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.full_run and args.task:
        raise SystemExit("--full-run runs all tasks; remove --task to use it.")

    task_names = sorted(TASK_SPECS) if args.full_run else args.task or sorted(TASK_SPECS)
    rows = generate_task_rows(load_items(args.records), task_names)
    output = args.output or default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        if args.full_run:
            for index, max_depth in enumerate(FULL_RUN_DEPTHS):
                if index:
                    handle.write("\n")
                print(f"\nRunning full benchmark with max_depth={max_depth}")
                rlm = None if args.dry_run else build_rlm(args, max_depth)
                run_rows(rows, rlm, max_depth, args.model_name, handle)
        else:
            rlm = None if args.dry_run else build_rlm(args, args.max_depth)
            run_rows(rows, rlm, args.max_depth, args.model_name, handle)

    print(f"Results: {output}")


if __name__ == "__main__":
    main()
