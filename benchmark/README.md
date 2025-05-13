# Benchmark
**`benchmark.py`**
A straightforward Spider‐style SQL benchmark for testing your NL→SQL model in isolation. It loads a JSON “test suite” of questions and corresponding gold SQL, spins up temporary SQLite databases from schema/data scripts, then uses your configured GGUF (or API-backed) SQL generator (`_generate_sql_query`) to produce queries. Each generated SQL is executed (with one retry for errors) and compared against the gold results, yielding execution and set‐match accuracy metrics. You can swap in any model via `aux_models`, tweak sampling size, or point it at your own dataset folder structure to benchmark on custom data.

---

**`benchmark_nl.py`**
An end-to-end “dual‐model” benchmark that evaluates the full `process_natural_language_query` pipeline—including database selection, routing, SQL generation, and optional semantic fallbacks. It randomly samples questions, injects both the target and distractor schemas (with sample rows) into the orchestrating LLM, then measures (a) whether the correct DB was chosen, (b) SQL generation success, and (c) execution & result‐matching against gold SQL. Results are logged as JSONL, with handy metrics for analysis accuracy, end-to-end execution rate, and match accuracy. Easily adapt the schema context size, sampling count, or model endpoints in your `.env`, `prompts.yaml`, and `utils.py` to benchmark any custom NL→SQL + orchestration setup.
