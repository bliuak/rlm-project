You are generating synthetic user records for a benchmark that tests recursive reasoning agents.

The benchmark is inspired by OOLONG-style aggregation tasks. Each record belongs to one user and has a date. The tested model will later need to solve each record, infer the semantic category of the correct answer, aggregate category counts per user, and answer pair-style questions across users.

Generate synthetic records where each record requires TWO-STEP REASONING:

Step 1: Solve a mini problem.
The mini problem may involve arithmetic, date/time reasoning, table lookup, unit conversion, set logic, ordering, conditional logic, or multi-hop deduction.

Step 2: Use the solved result to decide what answer should be reported.

The correct answer must belong to exactly one of these six categories:

1. abbreviation
   Examples: "NASA", "SNCF", "HTML", "UNESCO"

2. entity
   Examples: "Saturn V", "Aurora Beacon", "The Odyssey", "Toyota Prius", "World Health Organization"

3. human being
   Examples: "Ada Lovelace", "Grace Hopper", "Maya Chen"

4. numeric value
   Examples: "38", "165 minutes", "$42.50", "7.2 kilometers"

5. location
   Examples: "Lisbon", "Kyoto", "Lake Erie", "Nairobi"

6. description and abstract concept
   Examples: "supply shortage", "photosynthesis", "inventory overflow", "vendor dependency"

Important category rule:
The category is determined by the final correct answer, not by the wording of the question. For example, a record about boxes might have correct answer "Lisbon", whose category is "location", if the decision rule says to report the destination city.

Good example 1:
{
  "date": "2021-06-14",
  "user": 1042,
  "record": "A warehouse starts with 18 crates. Each crate contains 6 sensors.\nA technician removes 17 damaged sensors.\n\nDecision rule:\nIf the number of usable sensors is greater than 90, report the destination city: Lisbon.\nOtherwise, report the number of usable sensors.\n\nQuestion: What should be reported?",
  "correct_answer": "Lisbon",
  "correct_category": "location"
}

Why this is good:
The model must compute 18 * 6 - 17 = 91, then apply the threshold rule. The answer is Lisbon, so the category is location.

Good example 2:
{
  "date": "2018-11-03",
  "user": 2881,
  "record": "A train leaves at 09:20 and arrives at 12:05.\n\nDecision rule:\nIf the trip lasted more than 160 minutes, report the rail company code: SNCF.\nOtherwise, report the arrival city: Lyon.\n\nQuestion: What should be reported?",
  "correct_answer": "SNCF",
  "correct_category": "abbreviation"
}

Good example 3:
{
  "date": "2024-02-09",
  "user": 7310,
  "record": "A library shelf list says:\n- shelf A has books by Ada Lovelace, Grace Hopper, and Alan Turing\n- shelf B has books by Grace Hopper, Katherine Johnson, and Donald Knuth\n- shelf C has books by Grace Hopper, Mary Jackson, and Alan Turing\n\nDecision rule:\nIf exactly one person appears on all three shelves, report that person.\nOtherwise, report the concept \"catalog inconsistency\".\n\nQuestion: What should be reported?",
  "correct_answer": "Grace Hopper",
  "correct_category": "human being"
}

Now generate records.

Generation requirements:
- Generate exactly {10} records.
- Use user ID {1000}.
- Each record must include a date between 2010-01-01 and 2026-12-31.
- Use ISO date format: YYYY-MM-DD.
- Every record must require two-step reasoning.
- Every record must include at least two possible reportable outcomes with different categories.
- The correct answer must depend on solving the mini problem.
- Do not reveal the correct answer or category inside the public record except as one possible outcome in the decision rule.
- Use varied subproblem types across records.
- Include a mix of correct categories across the records.
- Do not copy the examples.
- Avoid ambiguous facts.
- Avoid records where the answer can be guessed from keywords alone.
- Do not include hidden reasoning in the output.

Each output item must include exactly:
- date
- user
- record
- correct_answer
- correct_category

The correct_category must be exactly one of:
- abbreviation
- entity
- human being
- numeric value
- location
- description and abstract concept

Return JSON only. No markdown. No commentary.

Output format:
{
  "items": [
    {
      "date": "2024-03-19",
      "user": 12345,
      "record": "...\n Decision rule:..\n Question: What should be reported?",
      "correct_answer": "...",
      "correct_category": "numeric value"
    }
  ]
}