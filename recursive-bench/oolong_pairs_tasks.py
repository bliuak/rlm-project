from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


LABELS = (
    "description and abstract concept",
    "entity",
    "human being",
    "numeric value",
    "location",
    "abbreviation",
)

LABEL_LIST = ", ".join(LABELS)
BENCHMARK_TASKS_PATH = Path(__file__).with_name("benchmark_tasks.json")


@dataclass(frozen=True)
class Record:
    user: int
    when: date
    instance: str
    label: str

    @property
    def public_line(self) -> str:
        return (
            f"Date: {self.when.strftime('%b %d, %Y')} || User: {self.user} "
            f"|| Instance: {self.instance}"
        )


def parse_records(items: list[dict]) -> list[Record]:
    records: list[Record] = []
    for index, item in enumerate(items):
        label = item.get("correct_category")
        if label not in LABELS:
            raise ValueError(f"Record {index} has unsupported category: {label!r}")
        when = datetime.strptime(str(item["date"]), "%Y-%m-%d").date()
        records.append(
            Record(
                user=int(item["user"]),
                when=when,
                instance=str(item["record"]).strip(),
                label=str(label),
            )
        )
    return records


def records_by_user(records: list[Record]) -> dict[int, list[Record]]:
    grouped: dict[int, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.user].append(record)
    return dict(grouped)


TASK_SPECS = tuple(f"task_{number:02d}" for number in range(1, 21))


def load_benchmark_task_prompts() -> dict[str, str]:
    prompts = json.loads(BENCHMARK_TASKS_PATH.read_text(encoding="utf-8"))
    missing = sorted(set(TASK_SPECS) - set(prompts))
    extra = sorted(set(prompts) - set(TASK_SPECS))
    if missing or extra:
        raise ValueError(
            f"{BENCHMARK_TASKS_PATH} must match TASK_SPECS. "
            f"Missing: {missing}. Extra: {extra}."
        )
    return {name: str(prompt).strip() for name, prompt in prompts.items()}


BENCHMARK_TASK_PROMPTS = load_benchmark_task_prompts()


def render_instructions(records: list[Record]) -> str:
    return (
        f"The following dataset contains {len(records)} entries. Each entry has a date, a user ID, "
        "and a small reasoning problem ending with the question \"What should be reported?\" "
        "The reported value is not shown. To use an entry, first solve its reasoning problem and "
        f"determine the reported value, then classify that reported value into one of 6 categories: {LABEL_LIST}.\n\n"
        "The category is based on the reported value, not on every noun or number mentioned in the prompt. "
        "For example, a reported city is a location, a reported person name is a human being, a reported "
        "number is a numeric value, a reported code or acronym is an abbreviation, a reported organization "
        "or named object is an entity, and a reported concept or descriptive phrase is a description and "
        "abstract concept.\n\n"
        "Users may have multiple entries. Pair questions ask about aggregate counts or existence of these "
        f"inferred answer categories per user across all {len(records)} entries. Do not guess. Calculate "
        "the exact answer from the entries."
    )


def render_entries(records: list[Record]) -> str:
    return "\n".join(record.public_line for record in records)


def render_context(records: list[Record]) -> str:
    return f"{render_instructions(records)}\n\nEntries:\n{render_entries(records)}"


def full_question(task_name: str) -> str:
    return BENCHMARK_TASK_PROMPTS[task_name]


def render_task_prompt(records: list[Record], task_name: str) -> str:
    return (
        f"Instructions:\n{render_instructions(records)}\n\n"
        f"Prompt:\n{full_question(task_name)}\n\n"
        f"Entries/data:\n{render_entries(records)}"
    )
