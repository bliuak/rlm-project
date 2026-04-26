# RLM System Prompt Iteration Notes

These notes summarize the OOLONG-Pairs trajectory issues that motivated the generic depth-2 prompt variants. The prompts should remain task-agnostic; the benchmark is only the diagnostic workload.

## Observed Issues

- Some runs spent several iterations saying they would inspect the context, but produced no REPL code. This wastes iterations and can lead to empty final answers.
- The RLM sometimes searched for task wording in the long raw context instead of first checking structured fields like `context["question"]`.
- A depth-2 run launched 39 recursive child RLM calls over user chunks in one wave. That single iteration took about 890 seconds and hit the run timeout before finalization.
- Depth-1 and depth-2 runs printed long pair lists instead of storing them in `final_answer` and returning `FINAL_VAR(final_answer)`. Printed REPL output was truncated, so the model later believed the result was lost.
- Some runs replaced semantic classification with broad heuristic classifiers inferred from samples. This produced noisy user sets and incorrect pair counts.
- One trajectory hit `TypeError: 'NoneType' object is not callable`, likely from polluted REPL state or a shadowed callable, then repeatedly restarted instead of isolating and avoiding the polluted name.
- The benchmark runner was preserving only the first non-empty output line. That is wrong for pair-list answers, which can validly span many lines.

## Prompt Changes In `rlm_system_prompt_depth2_v3.txt`

- Requires the first substantive response to execute REPL code.
- Makes the root RLM the orchestrator and child RLMs bounded workers.
- Adds a pilot-before-scale rule for recursive batches and caps recursive child waves to 10 prompts by default.
- Directs simple high-volume semantic work to `llm_query_batched` instead of `rlm_query_batched`.
- Requires stable IDs, machine-parseable schemas, coverage validation, and targeted retries.
- Warns against unvalidated heuristic shortcuts for required semantic analysis.
- Emphasizes assigning long final results to `final_answer` and using `FINAL_VAR(final_answer)`.
- Warns against shadowing Python built-ins and RLM tool names.

## Runner Change

`clean_model_answer()` now preserves all non-empty lines for `PAIRS` answers so pair-F1 scoring can see the complete predicted pair list.