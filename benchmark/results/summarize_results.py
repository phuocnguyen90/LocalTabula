import json

def summarize_benchmark_file(filepath):
    """
    Reads a benchmark result from either a JSON array or a JSONL file,
    and prints a summary:
      - Total examples processed
      - Examples skipped due to DB setup failure
      - Pipeline analysis/generation errors
      - Pipeline analysis successes (correct DB selected, SQL route & generation OK)
      - Successfully executed generated SQL
      - Generated SQL matched gold results
      - Analysis accuracy, execution accuracy, and match accuracy percentages
    """
    # Load records from JSON or JSONL
    try:
        with open(filepath, 'r') as f:
            text = f.read()
        text_stripped = text.lstrip()
        if text_stripped.startswith('['):
            # JSON array
            records = json.loads(text)
        else:
            # JSONL: one JSON object per non-empty line
            records = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except Exception as e:
        print(f"Failed to load data from {filepath}: {e}")
        return

    total = len(records)
    skipped_db = 0
    pipeline_analysis_success = 0
    exec_success = 0
    match_success = 0
    analysis_errors = 0

    for rec in records:
        status = rec.get("pipeline_status", "")
        # Count skipped DB
        if "DB Setup Failed" in status or "Schema Introspection Failed" in status or "Empty DB Schema" in status:
            skipped_db += 1
        # Count pipeline analysis success
        if status.startswith("Analysis & Gen Success"):
            pipeline_analysis_success += 1
        # Count analysis errors
        if status.startswith("Analysis Failed") or status.startswith("Outer Loop Error"):
            analysis_errors += 1
        # Execution & match flags
        if rec.get("exec_success"):
            exec_success += 1
        if rec.get("match_success"):
            match_success += 1

    # Compute accuracies
    analysis_accuracy = (pipeline_analysis_success / total * 100) if total else 0.0
    execution_accuracy = (exec_success / pipeline_analysis_success * 100) if pipeline_analysis_success else 0.0
    match_accuracy = (match_success / exec_success * 100) if exec_success else 0.0

    # Print summary
    print("\n--- Benchmark Summary ---")
    print(f"Total examples processed:            {total}")
    print(f"Examples skipped (DB setup failed):  {skipped_db}")
    print(f"Pipeline analysis/generation errors: {analysis_errors}")
    print(f"Pipeline analysis successes:         {pipeline_analysis_success}")
    print(f"Successfully executed SQL:           {exec_success}")
    print(f"Generated SQL matched gold results:  {match_success}")
    print(f"Analysis Accuracy:   {analysis_accuracy:.2f}%")
    print(f"Execution Accuracy:  {execution_accuracy:.2f}%")
    print(f"Exact Match Accuracy:{match_accuracy:.2f}%")

    # Return the summary dict if you need to programmatically inspect it
    return {
        "total_processed": total,
        "examples_skipped_db_setup": skipped_db,
        "pipeline_analysis_errors": analysis_errors,
        "pipeline_analysis_successes": pipeline_analysis_success,
        "exec_success_count": exec_success,
        "match_success_count": match_success,
        "analysis_accuracy_pct": round(analysis_accuracy, 2),
        "execution_accuracy_pct": round(execution_accuracy, 2),
        "match_accuracy_pct": round(match_accuracy, 2),
    }

# Example usage:
# summary = summarize_benchmark_file("benchmark_randomized_results.jsonl")
summary = summarize_benchmark_file("benchmark/results/nl_random_test_results_250507-20.jsonl")
