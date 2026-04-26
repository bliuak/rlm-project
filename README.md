# Dependencies

```bash
pip install -r requirements.txt
```

Add a `.env` file to the repo root with any model provider keys your RLM backend needs.

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

Edit the RLM root/system prompt in `prompts/oolong_root_prompt.txt`, then run:

```bash
python benchmark_tests/oolong_pairs_rlm.py \
  --limit 5 \
  --root-prompt-file prompts/oolong_root_prompt.txt
```

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
