# Dependencies

```bash
pip install -r requirements.txt
```

Add a `.env` file to the repo root with any model provider keys your RLM backend needs.

For OpenRouter-backed runs, add:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

RLM has a first-class `openrouter` backend. The 500k benchmark runner hardcodes
the backend/model near the top of `benchmark_tests/run_rlm_benchmark_tasks.py`;
edit `BENCHMARK_MODEL_NAME` there if you want a different model.

## HuggingFace

Some benchmark datasets require Hugging Face access.

1. Create a Hugging Face account.
2. Run `hf auth login`.
3. Paste your access token.

## LongCoT Smoke Test

```bash
python benchmark_tests/longcot_test.py
```

## OOLONG-Pairs-Style RLM Benchmark

Run a small OOLONG synthetic validation benchmark through the Python `rlms` package:

```bash
python benchmark_tests/oolong_pairs_rlm.py --limit 5 --verbose
```

Check dataset access without spending model calls:

```bash
python benchmark_tests/oolong_pairs_rlm.py --limit 1 --dry-run
```

Edit the task-level root prompt in `prompts/oolong_root_prompt.txt`, then run:

```bash
python benchmark_tests/oolong_pairs_rlm.py \
  --limit 5 \
  --root-prompt-file prompts/oolong_root_prompt.txt
```

To test a generic RLM control prompt derived from the package's original system prompt, use `--custom-system-prompt-file` instead:

```bash
python benchmark_tests/oolong_pairs_rlm.py \
  --source pairs \
  --pair-query task_20 \
  --limit 1 \
  --max-depth 2 \
  --custom-system-prompt-file prompts/rlm_system_prompt_depth2_v3.txt \
  --log-dir logs/depth2_v3
```

`--root-prompt-file` is a task hint passed with each query. `--custom-system-prompt-file` replaces the RLM's generic REPL/controller policy, including when it should use `llm_query`, `llm_query_batched`, `rlm_query`, and `rlm_query_batched`. The `prompts/rlm_system_prompt_depth2_v*.txt` files are intentionally not OOLONG-specific; they are meant to improve depth-2 recursive planning and aggregation across tasks.

See `prompts/rlm_system_prompt_iteration_notes.md` for the trajectory issues that motivated the latest generic system prompt.

By default this uses:

- Hugging Face dataset: `oolongbench/oolong-synth`
- Split: `validation`
- Dataset filter: `trec_coarse`
- Minimum context length: `100000` characters
- RLM backend: `openai`
- RLM model name: `gpt-5.4-mini`
- Output: `results/oolong_pairs_rlm.jsonl`

Useful variants:

```bash
# Run more synthetic tasks.
python benchmark_tests/oolong_pairs_rlm.py --limit 25

# Run every task in a character-length bin.
python benchmark_tests/oolong_pairs_rlm.py \
  --limit 0 \
  --min-context 100000 \
  --max-context 200000

# Use another OpenAI-compatible model name.
python benchmark_tests/oolong_pairs_rlm.py --model-name gpt-5.4-mini --max-depth 3

# Run OOLONG-real instead of OOLONG-synth.
python benchmark_tests/oolong_pairs_rlm.py \
  --source real \
  --config dnd \
  --dataset-filter "" \
  --limit 5
```

To approximate the paper's length sweeps, use token-length bins. For example, the paper discusses scaling prompts from `2^13` to `2^18` tokens; you can run one bin at a time:

```bash
# 2^13 to 2^14 tokens.
python benchmark_tests/oolong_pairs_rlm.py \
  --limit 0 \
  --min-context 0 \
  --min-context-tokens 8192 \
  --max-context-tokens 16384 \
  --show-tokens

# 2^14 to 2^15 tokens.
python benchmark_tests/oolong_pairs_rlm.py \
  --limit 0 \
  --min-context 0 \
  --min-context-tokens 16384 \
  --max-context-tokens 32768 \
  --show-tokens
```

Token counts use `tiktoken` with the `cl100k_base` encoding by default. Override this with `--tokenizer` if you want a different installed `tiktoken` encoding.

The runner writes one JSON object per task with the task metadata, expected answer, RLM answer, latency, score, and any error. Scoring follows the oolong-pairs conventions: exact match for labels/dates, `0.75^abs(error)` for numeric answers, and normalized `more`/`less`/`same` matching for comparisons.

## OOLONG-Pairs 500k Tasks

Generate local 500k-token OOLONG-Pairs-style tasks:

```bash
python benchmark_tests/generate_oolong_pairs_500k.py
```

Run RLM on those tasks:

```bash
python benchmark_tests/run_rlm_benchmark_tasks.py \
  --tasks results/oolong_pairs_500k_tasks.jsonl \
  --output results/oolong_pairs_500k_rlm_results.jsonl \
  --limit 1
```

Use a model with a context window larger than the benchmark context size.

## Trajectory Viewer

Open `trajectory_viewer/index.html` in a browser, then drop one of the JSONL
files from `logs/` into the viewer.


## OOLONG-Pairs Benchmark Tasks (1-20) on Local Synthetic Records

Generate verified answers and task JSONL (all 20 benchmark tasks):

```bash
python recursive-bench/generate_task_answers.py
```

For tasks 1-6, this generator reads the answer key from the standalone
criterion scripts in `recursive-bench/answer-generators/`
(`generate_task_01_answer.py` through `generate_task_06_answer.py`). The
generated JSONL stores the script path in `metadata.answer_source`, and the RLM
result JSONL copies it to `expected_answer_source`.

Create one task payload. The JSON includes structured fields plus a `prompt` field
ordered as instructions, task prompt, then entries/data. Use `--prompt-out` to
write that exact prompt as plain text for regular LLMs:

```bash
python recursive-bench/task_prompt.py --task task_20
python recursive-bench/task_prompt.py --task task_20 --prompt-out results/sample_task_20_prompt.txt
python recursive-bench/task_prompt.py --task task_20 --run
```

The benchmark task prompt text is stored verbatim in
`recursive-bench/benchmark_tasks.json`. Edit that file when the
task wording changes; the Python module keeps only the answer-verification
predicates.

For tasks 1-6, there are also standalone answer scripts that encode the
prompt criterion directly, without using the shared task predicate:

```bash
python recursive-bench/answer-generators/generate_task_01_answer.py --output results/task_01_independent_answer.txt --audit-json results/audits/task_01_independent.json --fail-on-mismatch
python recursive-bench/answer-generators/generate_task_02_answer.py --output results/task_02_independent_answer.txt --audit-json results/audits/task_02_independent.json --fail-on-mismatch
python recursive-bench/answer-generators/generate_task_03_answer.py --output results/task_03_independent_answer.txt --audit-json results/audits/task_03_independent.json --fail-on-mismatch
python recursive-bench/answer-generators/generate_task_04_answer.py --output results/task_04_independent_answer.txt --audit-json results/audits/task_04_independent.json --fail-on-mismatch
python recursive-bench/answer-generators/generate_task_05_answer.py --output results/task_05_independent_answer.txt --audit-json results/audits/task_05_independent.json --fail-on-mismatch
python recursive-bench/answer-generators/generate_task_06_answer.py --output results/task_06_independent_answer.txt --audit-json results/audits/task_06_independent.json --fail-on-mismatch
```

Run all benchmark tasks against OpenRouter:

```bash
python recursive-bench/run_recursive_bench.py --model-name openai/gpt-5.4-mini
```

Run one benchmark task:

```bash
python recursive-bench/run_recursive_bench.py \
  --task task_03 \
  --max-depth 2 \
  --output results/task_03_d2.txt
```

Useful flags:

- `records`: Optional positional path to the synthetic records file. Defaults to `synthetic_user_records.json`.
- `--task task_03`: Run a single task. Repeat the flag to run multiple tasks, for example `--task task_03 --task task_04`. Omit it to run all tasks once.
- `--max-depth 2`: Set the RLM recursion depth for a normal run. Defaults to `1`.
- `--model-name openai/gpt-5.5`: Set the root RLM model used through OpenRouter.
- `--sub-model-name openai/gpt-5.5`: Set the model used for `llm_query`, child RLMs, and max-depth fallback calls.
- `--max-iterations 30`: Set the RLM iteration budget.
- `--depth2-system-prompt recursive-bench/prompts/rlm_system_prompt_depth2.txt`: Override the system prompt used for depth-2 runs.
- `--dry-run`: Build and score the task rows without making model calls.
- `--output results/task_03_d2.txt`: Write the tab-separated results report to a specific file. If omitted, the runner writes `results/results_HHMMSS.txt`.
- `--full-run`: Run the full benchmark twice, once at max depth 1 and once at max depth 2. Do not combine this with `--task`.

Run the full benchmark:

```bash
python recursive-bench/run_recursive_bench.py \
  --full-run \
  --model-name openai/gpt-5.5 \
  --sub-model-name openai/gpt-5.5 \
  --output results/full_bench_d1_d2.txt
```
