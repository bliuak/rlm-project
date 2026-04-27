from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from typing import Callable


LABELS = (
    "description and abstract concept",
    "entity",
    "human being",
    "numeric value",
    "location",
    "abbreviation",
)

LABEL_LIST = ", ".join(f"'{label}'" for label in LABELS)


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


def counts(entries: list[Record]) -> Counter[str]:
    return Counter(record.label for record in entries)


def has(entries: list[Record], label: str) -> bool:
    return any(record.label == label for record in entries)


def has_any(entries: list[Record], labels: set[str]) -> bool:
    return any(record.label in labels for record in entries)


def count(entries: list[Record], label: str) -> int:
    return sum(1 for record in entries if record.label == label)


def all_dates_after(entries: list[Record], label: str, cutoff: date) -> bool:
    dates = [record.when for record in entries if record.label == label]
    return bool(dates) and all(when > cutoff for when in dates)


def all_dates_before(entries: list[Record], label: str, cutoff: date) -> bool:
    dates = [record.when for record in entries if record.label == label]
    return bool(dates) and all(when < cutoff for when in dates)


Predicate = Callable[[list[Record]], bool]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    question: str
    first_condition: Predicate
    second_condition: Predicate


def symmetric(question: str, predicate: Predicate) -> TaskSpec:
    return TaskSpec(
        name="",
        question=question,
        first_condition=predicate,
        second_condition=predicate,
    )


TASK_SPECS: dict[str, TaskSpec] = {
    "paper_01": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with a numeric value or location.",
        lambda e: has_any(e, {"numeric value", "location"}),
    ),
    "paper_02": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with an entity or human being.",
        lambda e: has_any(e, {"entity", "human being"}),
    ),
    "paper_03": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with a description and abstract concept or abbreviation.",
        lambda e: has_any(e, {"description and abstract concept", "abbreviation"}),
    ),
    "paper_04": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with a human being or location, and all instances "
        "that are a human being for both users must be after January 6, 2023.",
        lambda e: has_any(e, {"human being", "location"}) and all_dates_after(e, "human being", date(2023, 1, 6)),
    ),
    "paper_05": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with an entity or numeric value, and all instances "
        "that are an entity for both users must be before March 15, 2023.",
        lambda e: has_any(e, {"entity", "numeric value"}) and all_dates_before(e, "entity", date(2023, 3, 15)),
    ),
    "paper_06": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with a location or abbreviation.",
        lambda e: has_any(e, {"location", "abbreviation"}),
    ),
    "paper_07": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with a description and abstract concept or numeric value, "
        "and all instances that are a numeric value for both users must be after February 1, 2023.",
        lambda e: has_any(e, {"description and abstract concept", "numeric value"})
        and all_dates_after(e, "numeric value", date(2023, 2, 1)),
    ),
    "paper_08": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with a human being or description and abstract concept.",
        lambda e: has_any(e, {"human being", "description and abstract concept"}),
    ),
    "paper_09": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with an entity or location, and all instances that "
        "are a location for both users must be after April 10, 2023.",
        lambda e: has_any(e, {"entity", "location"}) and all_dates_after(e, "location", date(2023, 4, 10)),
    ),
    "paper_10": symmetric(
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
        "where both users have at least one instance with a numeric value or abbreviation, and all instances "
        "that are an abbreviation for both users must be before May 20, 2023.",
        lambda e: has_any(e, {"numeric value", "abbreviation"}) and all_dates_before(e, "abbreviation", date(2023, 5, 20)),
    ),
    "paper_11": TaskSpec(
        name="paper_11",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with entity and one with abbreviation, and the other user has exactly one instance with entity.",
        first_condition=lambda e: has(e, "entity") and has(e, "abbreviation"),
        second_condition=lambda e: count(e, "entity") == 1,
    ),
    "paper_12": TaskSpec(
        name="paper_12",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with numeric value, and the other user has at least one instance with location and at least one instance with human being.",
        first_condition=lambda e: count(e, "numeric value") >= 2,
        second_condition=lambda e: has(e, "location") and has(e, "human being"),
    ),
    "paper_13": TaskSpec(
        name="paper_13",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with description and abstract concept, and the other user has at least one instance with abbreviation and at least one instance with entity.",
        first_condition=lambda e: count(e, "description and abstract concept") == 1,
        second_condition=lambda e: has(e, "abbreviation") and has(e, "entity"),
    ),
    "paper_14": TaskSpec(
        name="paper_14",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with human being and at least one instance with numeric value, and the other user has exactly two instances with location.",
        first_condition=lambda e: has(e, "human being") and has(e, "numeric value"),
        second_condition=lambda e: count(e, "location") == 2,
    ),
    "paper_15": TaskSpec(
        name="paper_15",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with entity, at least one instance with location, and at least one instance with abbreviation, and the other user has exactly one instance with numeric value.",
        first_condition=lambda e: has(e, "entity") and has(e, "location") and has(e, "abbreviation"),
        second_condition=lambda e: count(e, "numeric value") == 1,
    ),
    "paper_16": TaskSpec(
        name="paper_16",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with description and abstract concept and at least one instance with human being, and the other user has at least two instances with entity and exactly one instance with abbreviation.",
        first_condition=lambda e: has(e, "description and abstract concept") and has(e, "human being"),
        second_condition=lambda e: count(e, "entity") >= 2 and count(e, "abbreviation") == 1,
    ),
    "paper_17": TaskSpec(
        name="paper_17",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with numeric value, and the other user has at least one instance with location and at least one instance with description and abstract concept.",
        first_condition=lambda e: count(e, "numeric value") == 1,
        second_condition=lambda e: has(e, "location") and has(e, "description and abstract concept"),
    ),
    "paper_18": TaskSpec(
        name="paper_18",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with abbreviation and exactly one instance with human being, and the other user has at least one instance with entity and at least one instance with numeric value.",
        first_condition=lambda e: has(e, "abbreviation") and count(e, "human being") == 1,
        second_condition=lambda e: has(e, "entity") and has(e, "numeric value"),
    ),
    "paper_19": TaskSpec(
        name="paper_19",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with location and at least one instance with entity, and the other user has exactly one instance with description and abstract concept and exactly one instance with abbreviation.",
        first_condition=lambda e: count(e, "location") >= 2 and has(e, "entity"),
        second_condition=lambda e: count(e, "description and abstract concept") == 1 and count(e, "abbreviation") == 1,
    ),
    "paper_20": TaskSpec(
        name="paper_20",
        question="In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with numeric value and at least one instance with human being, and the other user has at least one instance with location, at least one instance with entity, and exactly one instance with abbreviation.",
        first_condition=lambda e: has(e, "numeric value") and has(e, "human being"),
        second_condition=lambda e: has(e, "location") and has(e, "entity") and count(e, "abbreviation") == 1,
    ),
}

for key, spec in list(TASK_SPECS.items()):
    if not spec.name:
        TASK_SPECS[key] = TaskSpec(
            name=key,
            question=spec.question,
            first_condition=spec.first_condition,
            second_condition=spec.second_condition,
        )


def compute_expected_pairs(
    per_user: dict[int, list[Record]],
    task_name: str,
) -> list[tuple[int, int]]:
    spec = TASK_SPECS[task_name]
    pairs: set[tuple[int, int]] = set()
    for first, second in combinations(sorted(per_user), 2):
        first_entries = per_user[first]
        second_entries = per_user[second]
        if (
            spec.first_condition(first_entries)
            and spec.second_condition(second_entries)
        ) or (
            spec.first_condition(second_entries)
            and spec.second_condition(first_entries)
        ):
            pairs.add((first, second))
    return sorted(pairs)


def format_pairs(pairs: list[tuple[int, int]]) -> str:
    if not pairs:
        return "[]"
    return "\n".join(f"({a}, {b})" for a, b in pairs)


def render_context(records: list[Record]) -> str:
    preamble = (
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
        "the exact answer from the entries below.\n\n"
    )
    return preamble + "\n".join(record.public_line for record in records)


def full_question(task_name: str) -> str:
    return (
        TASK_SPECS[task_name].question
        + " For each entry, solve the entry's question and classify the reported value into one of these labels "
        + f"(the data does not provide the reported values or labels, you need to infer them): {LABEL_LIST}. "
        + "In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines. "
        + "Your answer must be sorted by first user ID. If there is no answer, return an empty list []."
    )
