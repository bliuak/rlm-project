from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from synthetic_records import DEFAULT_RECORDS_PATH, get_llm_payload  # noqa: E402


def write_json(payload: dict[str, object], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export synthetic records for an LLM prompt without correct_answer "
            "or correct_category fields."
        )
    )
    parser.add_argument(
        "records",
        nargs="?",
        type=Path,
        default=DEFAULT_RECORDS_PATH,
        help="Path to the synthetic user records JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write the sanitized JSON payload to.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_json(get_llm_payload(args.records), args.output)


if __name__ == "__main__":
    main()
