# RLM Recursive Benchmark

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Add a `.env` file to the repo root with the OpenRouter key used by the benchmark runner:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Generate Expected Answers

Regenerate the verified answer key and task JSONL for all 20 benchmark tasks:

```bash
python recursive-bench/generate_task_answers.py
```

This writes:

- `results/oolong_pairs_verified_tasks.jsonl`
- `results/oolong_pairs_verified_answers/task_01.txt` through `task_20.txt`

To generate only selected tasks, repeat `--task`:

```bash
python recursive-bench/generate_task_answers.py \
  --task task_01 \
  --task task_03
```

## Inspect Task Prompts

Create one task payload:

```bash
python recursive-bench/task_prompt.py --task task_20
```

Write the exact prompt text for a regular LLM:

```bash
python recursive-bench/task_prompt.py \
  --task task_20 \
  --prompt-out results/task_prompts/task_20.txt
```

Print the prompt and expected answer:

```bash
python recursive-bench/task_prompt.py --task task_20 --run
```

The task wording lives in `recursive-bench/benchmark_tasks.json`.

## Run The Benchmark

Run one benchmark task:

```bash
python recursive-bench/run_recursive_bench.py \
  --task task_03 \
  --max-depth 2 \
  --output results/task_03_d2.txt
```

Run task 1 with Kimi K2.6 at max depth 2:

```bash
python recursive-bench/run_recursive_bench.py \
  --task task_01 \
  --max-depth 2 \
  --model-name moonshotai/kimi-k2.6 \
  --sub-model-name moonshotai/kimi-k2.6 \
  --output results/task_01_kimi_k2_6_d2.txt
```

Run all tasks once:

```bash
python recursive-bench/run_recursive_bench.py \
  --max-depth 2 \
  --model-name openai/gpt-5.5 \
  --sub-model-name openai/gpt-5.5 \
  --output results/all_tasks_d2.txt
```

Run the full benchmark at both depth 1 and depth 2:

```bash
python recursive-bench/run_recursive_bench.py \
  --full-run \
  --model-name openai/gpt-5.5 \
  --sub-model-name openai/gpt-5.5 \
  --output results/full_bench_d1_d2.txt
```

## Benchmark Flags

- `records`: Optional positional path to the synthetic records file. Defaults to `synthetic_user_records.json`.
- `--task task_03`: Run one task. Repeat the flag to run multiple tasks. Omit it to run all tasks once.
- `--max-depth 2`: Set the RLM recursion depth for a normal run. Defaults to `1`.
- `--model-name openai/gpt-5.5`: Set the root RLM model used through OpenRouter.
- `--sub-model-name openai/gpt-5.5`: Set the model used for `llm_query`, child RLMs, and max-depth fallback calls.
- `--max-iterations 30`: Set the RLM iteration budget.
- `--depth2-system-prompt recursive-bench/prompts/rlm_system_prompt_depth2.txt`: Override the system prompt used for depth-2 runs.
- `--dry-run`: Build and score task rows without making model calls.
- `--output results/task_03_d2.txt`: Write the tab-separated results report to a specific file. If omitted, the runner writes `results/results_HHMMSS.txt`.
- `--full-run`: Run all tasks twice, once at max depth 1 and once at max depth 2. Do not combine this with `--task`.

## Scoring

The runner extracts answer pairs from the model response with the pattern `(number, number)` and computes pair-set F1 against the expected answer. Extra explanatory text is okay as long as it does not contain additional parenthesized number pairs.

Each result row includes the task, model, max depth, score, expected/output pair counts, runtime, token usage, cost when available, subcall counts, and any error.
