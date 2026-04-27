from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from oolong_pairs_tasks import TASK_SPECS
from generate_oolong_pairs_verified_answers import generate_task_rows, load_items


DEFAULT_RECORDS_PATH = Path(__file__).resolve().parents[1] / "synthetic_user_records.json"
ANSWER_FORMAT = (
    "Return only sorted pairs in the format (user_id_1, user_id_2), one per line. "
    "Return [] if there are no matching pairs."
)


def parse_backend_kwargs(values: list[str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid backend kwarg: {value}")
        key, raw = value.split("=", 1)
        kwargs[key] = raw
    return kwargs


def render_prompt(payload: dict[str, Any]) -> str:
    return str(payload["prompt"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a single OOLONG-Pairs task payload and optionally run it through an LLM/RLM backend."
    )
    parser.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--payload-out", type=Path, default=Path("results/oolong_pairs_single_payload.json"))
    parser.add_argument(
        "--prompt-out",
        type=Path,
        help="Also write a single plain-text prompt that can be pasted into a regular LLM.",
    )
    parser.add_argument("--run", action="store_true", help="Call the model backend with the generated payload.")
    parser.add_argument("--backend", default="openrouter")
    parser.add_argument("--model-name", default="openai/gpt-5.4-mini")
    parser.add_argument("--backend-kwarg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    row = generate_task_rows(load_items(args.records), [args.task])[0]

    payload = {
        "task": row["task_type"],
        "prompt": row["prompt"],
        "context": row["context"],
        "question": row["question"],
        "answer_format": ANSWER_FORMAT,
        "entries_have_no_labels": True,
        "metadata": row["metadata"],
    }

    args.payload_out.parent.mkdir(parents=True, exist_ok=True)
    args.payload_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Payload saved to {args.payload_out}")

    if args.prompt_out:
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_out.write_text(render_prompt(payload), encoding="utf-8")
        print(f"Prompt saved to {args.prompt_out}")

    if not args.run:
        return

    try:
        from dotenv import load_dotenv
        from rlm import RLM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Running the model requires optional dependencies. "
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
    completion = rlm.completion(payload["prompt"])
    print("\nModel answer:\n")
    print(completion.response)


if __name__ == "__main__":
    main()
