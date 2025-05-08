# Testing setup
## Base benchmark
```bash
python benchmark/benchmark.py     app_data/spider_data/spider_data/test.json     app_data/spider_data/spider_data/test_gold.sql     app_data/spider_data/spider_data/test_database/     --limit 50     --results_out spider_test_results.json     --log_level DEBUG
```
[Results](#test-1)
## Natural language benchmark
Run 1 (fast)
```bash
python benchmark/benchmark_nl.py \
    app_data/spider_data/spider_data/test.json \
    app_data/spider_data/spider_data/test_gold.sql \
    app_data/spider_data/spider_data/test_database/ \
    app_data/spider_data/spider_data/tables.json \
    --num_samples 20 \
    --results_out nl_random_test_results.json \
    --log_level INFO
```
[Results](#test-2)

```bash
python benchmark/benchmark_nl.py \
    app_data/spider_data/spider_data/test.json \
    app_data/spider_data/spider_data/test_gold.sql \
    app_data/spider_data/spider_data/test_database/ \
    app_data/spider_data/spider_data/test_tables.json \
    --num_samples 100 \
    --num_schemas 15 \
    --results_out spider_random_context20_results.json \
    --log_level INFO
```
[Results](#test-3)

# Results

## [Test 1](r1)
```bat
2025-05-07 16:22:51,189 - INFO - Reached benchmark limit of 50 examples processed.
2025-05-07 16:22:51,189 - INFO - 
--- Benchmark Summary (Simple SQL Generation Test) ---
2025-05-07 16:22:51,189 - INFO - Total examples in dataset provided: 2147
2025-05-07 16:22:51,189 - INFO - Examples processed (or attempted up to limit): 50
2025-05-07 16:22:51,189 - INFO - Examples skipped (DB setup failed): 0
2025-05-07 16:22:51,189 - INFO - SQL Generation Failures (by LLM): 0
2025-05-07 16:22:51,189 - INFO - Generated SQL Execution Errors: 2
2025-05-07 16:22:51,189 - INFO - Gold SQL Execution Errors: 0
2025-05-07 16:22:51,189 - INFO - Successfully executed generated SQL: 48
2025-05-07 16:22:51,189 - INFO - Generated SQL matched gold results: 39
2025-05-07 16:22:51,189 - INFO - Execution Accuracy (Exec Success / (Processed - Skipped DB Setup)): 96.00%
2025-05-07 16:22:51,189 - INFO - Exact Set Match Accuracy (Match Success / Successful Executions): 81.25%
2025-05-07 16:22:51,189 - INFO - Simple SQL generation benchmarking finished. Results appended to spider_test_results.json
2025-05-07 16:22:51,189 - INFO - Overall Execution Accuracy: 96.00%
2025-05-07 16:22:51,189 - INFO - Overall Exact Set Match Accuracy: 81.25%
    --log_level INFO
```
## [Test 2](r2)
```bat 
--- Randomized Benchmark Summary (Schema Subset) ---
2025-04-10 17:51:19,076 - INFO - Schema Subset Size for Selection Context: 15
2025-04-10 17:51:19,076 - INFO - Total examples processed: 100
2025-04-10 17:51:19,076 - INFO - Examples skipped (DB setup failed): 11
2025-04-10 17:51:19,076 - INFO - Pipeline Analysis/Selection/Generation Errors: 3
2025-04-10 17:51:19,076 - INFO - Pipeline Analysis Successful (Correct DB, SQL Route & Gen OK): 86
2025-04-10 17:51:19,076 - INFO - Successfully executed generated SQL: 53
2025-04-10 17:51:19,076 - INFO - Generated SQL matched gold results: 38
2025-04-10 17:51:19,076 - INFO - Analysis & Generation Accuracy (vs Processed): 86.00%
2025-04-10 17:51:19,076 - INFO - Execution Accuracy (vs Pipeline Success): 61.63%
2025-04-10 17:51:19,076 - INFO - Exact Set Match Accuracy (vs Exec Success): 71.70%
2025-04-10 17:51:19,078 - INFO - Randomized benchmarking finished.
    --log_level INFO
```
## [Test 3](r3)
```bat
--- Randomized Benchmark Summary (Schema Subset) ---
2025-05-07 11:54:22,878 - INFO - Schema Subset Size for Selection Context: 15
2025-05-07 11:54:22,878 - INFO - Total examples processed: 300
2025-05-07 11:54:22,878 - INFO - Examples skipped (DB setup failed): 0
2025-05-07 11:54:22,878 - INFO - Pipeline Analysis/Selection/Generation Errors: 53
2025-05-07 11:54:22,878 - INFO - Pipeline Analysis Successful (Correct DB, SQL Route & Gen OK): 246
2025-05-07 11:54:22,878 - INFO - Successfully executed generated SQL: 105
2025-05-07 11:54:22,878 - INFO - Generated SQL matched gold results: 63
2025-05-07 11:54:22,878 - INFO - Analysis & Generation Accuracy (vs Processed): 82.00%
2025-05-07 11:54:22,878 - INFO - Execution Accuracy (vs Pipeline Success): 42.68%
2025-05-07 11:54:22,878 - INFO - Exact Set Match Accuracy (vs Exec Success): 60.00%
2025-05-07 11:54:22,882 - INFO - Detailed results saved to nl_random_test_results_250507-2.json
2025-05-07 11:54:22,882 - INFO - Randomized benchmarking finished.
    --log_level INFO
```
