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
## Inspect Task Prompts
Individual task prompts are located in the task_prompts folder. 

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

- `--task task_03`: Run one task. Repeat the flag to run multiple tasks. Omit it to run all tasks once.
- `--max-depth 2`: Set the RLM recursion depth for a normal run. Defaults to `1`.
- `--model-name openai/gpt-5.5`: Set the root RLM model used through OpenRouter.
- `--sub-model-name openai/gpt-5.5`: Set the model used for `llm_query`, child RLMs, and max-depth fallback calls.
- `--max-iterations 30`: Set the RLM iteration budget.
- `--output results/task_03_d2.txt`: Write the tab-separated results report to a specific file. If omitted, the runner writes `results/results_HHMMSS.txt`.
- `--full-run`: Run all tasks twice, once at max depth 1 and once at max depth 2. Do not combine this with `--task`.
