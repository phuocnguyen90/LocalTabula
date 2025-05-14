# --- START OF FILE benchmark.py ---
import os
import json
import sqlite3
import pandas as pd
import argparse
import logging
import time
import shutil
import tempfile
from typing import List, Dict, Any, Set, Tuple, Optional

# Assuming utils.py is in the parent directory's utils folder
import sys
project_root_for_benchmark_v2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root_for_benchmark_v2 not in sys.path:
    sys.path.insert(0, project_root_for_benchmark_v2)

try:
    from utils.utils import (
        # setup_environment, get_db_connection, init_qdrant_client, # Not strictly needed for this script's core
        get_llm_wrapper, get_cached_aux_models,
        get_schema_info, _generate_sql_query, # Core SQL generator
        get_table_sample_data # For sample data helper
    )
    # Import helpers from this file or define them if they were meant to be local
    # from benchmark.benchmark_v2 import ( # This would be circular if it's the same file
    #      load_test_suite_questions, load_test_suite_gold_sql,
    #      combine_test_suite_data, setup_temp_database,
    #      robust_execute_sql, compare_results
    # )
except ImportError as e:
    print(f"Error importing from utils: {e}")
    print("Please ensure benchmark.py is in the correct directory relative to utils.py (project_root/benchmark/benchmark.py)")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# --- Helper Functions from previous benchmark script (ensure they are defined or imported) ---
# These are often part of benchmark_v2 or a shared benchmark utility module.
# For this exercise, I'll assume they are available or defined in this file if not imported.

def load_test_suite_questions(filepath: str) -> List[Dict[str, Any]]:
    """Loads the questions and db_ids from the test suite JSON file."""
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        logging.info(f"Loaded {len(data)} questions/db_ids from {filepath}")
        for i, item in enumerate(data): item['original_index'] = i
        return data
    except Exception as e: logging.error(f"Error loading questions from {filepath}: {e}"); return []

def load_test_suite_gold_sql(filepath: str) -> List[str]:
    """Loads the gold SQL queries from the .sql file (one per line)."""
    try:
        with open(filepath, 'r') as f: gold_queries = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(gold_queries)} gold SQL queries from {filepath}")
        return gold_queries
    except Exception as e: logging.error(f"Error loading gold SQL from {filepath}: {e}"); return []

def combine_test_suite_data(questions_data: List[Dict[str, Any]], gold_queries: List[str]) -> List[Dict[str, Any]]:
    """Combines question data with gold queries based on order."""
    if not questions_data or not gold_queries or len(questions_data) != len(gold_queries):
        logging.error(f"Mismatch or empty questions/gold queries. Q: {len(questions_data)}, G: {len(gold_queries)}.")
        return []
    combined_data = []
    for i, question_item in enumerate(questions_data):
        gold_sql_full = gold_queries[i]
        actual_gold_sql = gold_sql_full.split('\t')[0].strip() if '\t' in gold_sql_full else gold_sql_full
        question_item['query'] = actual_gold_sql
        combined_data.append(question_item)
    logging.info(f"Successfully combined {len(combined_data)} test examples.")
    return combined_data

def create_db_from_sql_script(script_path: str, db_path: str) -> bool:
    try:
        with open(script_path, 'r', encoding='utf-8') as f: sql_script = f.read()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor(); cursor.executescript(sql_script); conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error creating DB from script '{script_path}' on '{db_path}': {e}")
        if os.path.exists(db_path): os.remove(db_path)
        return False

def setup_temp_database(db_id: str, db_base_dir: str, temp_dir: str) -> Optional[str]:
    db_subdir = os.path.join(db_base_dir, db_id)
    sqlite_src = os.path.join(db_subdir, f"{db_id}.sqlite")
    temp_db_path = os.path.join(temp_dir, f"{db_id}.sqlite")

    if os.path.exists(sqlite_src):
        logging.debug(f"Found existing SQLite DB for '{db_id}' at '{sqlite_src}'. Copying to temp.")
        try:
            shutil.copyfile(sqlite_src, temp_db_path)
            return temp_db_path
        except Exception as e:
            logging.error(f"Failed to copy SQLite DB '{sqlite_src}' to '{temp_db_path}': {e}")
            return None
    
    schema_sql_path = os.path.join(db_subdir, "schema.sql")
    data_sql_path = os.path.join(db_subdir, f"{db_id}.sql")

    if not os.path.exists(schema_sql_path):
        logging.error(f"Schema script '{schema_sql_path}' not found for DB '{db_id}'.")
        return None
    
    if os.path.exists(temp_db_path): os.remove(temp_db_path) # Clear old temp file

    if create_db_from_sql_script(schema_sql_path, temp_db_path):
        if os.path.exists(data_sql_path):
            logging.debug(f"Found data script '{data_sql_path}'. Populating data...")
            if not create_db_from_sql_script(data_sql_path, temp_db_path):
                logging.warning(f"Failed to populate data from '{data_sql_path}'. DB may be schema-only.")
        else:
            logging.debug(f"Data script for '{db_id}' not found at '{data_sql_path}'. DB is schema-only.")
        return temp_db_path
    else:
        logging.error(f"Failed to create temp DB for '{db_id}' from schema script.")
        return None

def robust_execute_sql(conn: sqlite3.Connection, query: str) -> Tuple[Optional[Set[Tuple]], Optional[str], float]:
    results_set: Optional[Set[Tuple]] = None; error_message: Optional[str] = None
    start_time = 0.0; duration = 0.0
    try:
        query = query.strip().rstrip(';')
        if not query: return set(), None, 0.0
        start_time = time.perf_counter()
        cursor = conn.cursor(); cursor.execute(query); rows = cursor.fetchall()
        duration = time.perf_counter() - start_time
        results_set = set(tuple(map(str, row)) for row in rows)
    except Exception as e:
        if start_time > 0: duration = time.perf_counter() - start_time
        error_message = f"{type(e).__name__}: {e}"
        logging.warning(f"Query failed ({duration:.4f}s): {query} | Error: {error_message}")
    return results_set, error_message, duration

def compare_results(gold_results: Optional[Set[Tuple]], gen_results: Optional[Set[Tuple]]) -> bool:
    if gold_results is None or gen_results is None: return False
    return gold_results == gen_results

# Helper for multi-table sampling (consistent with benchmark_nl.py's helper)
def _get_multi_table_sample_str_for_benchmark( 
    db_id_for_log: str, # Used for logging context
    db_schema_dict: Dict[str, List[str]], # Schema for this specific db_id {table: [cols]}
    db_conn: sqlite3.Connection,
    num_tables_to_sample: int = 3,
    num_rows_per_table: int = 2
) -> str:
    """Helper to get a concatenated sample string from multiple tables of a DB."""
    sample_parts = []
    if not db_schema_dict:
        logging.warning(f"[{db_id_for_log}] No schema provided for sampling.")
        return "N/A (schema not found for sampling)"

    tables_in_schema = list(db_schema_dict.keys())
    if not tables_in_schema:
        return "N/A (no tables in schema for sampling)"
    
    # Shuffle to get varied tables if num_tables_to_sample is less than total tables
    import random
    random.shuffle(tables_in_schema)
    tables_to_sample_names = tables_in_schema[:num_tables_to_sample]

    for table_name in tables_to_sample_names:
        s_data = get_table_sample_data(db_conn, table_name, limit=num_rows_per_table) # from utils.utils
        if s_data:
            if "Error" not in s_data and "exists but is empty" not in s_data:
                sample_parts.append(f"-- Sample rows from table {table_name} (in DB {db_id_for_log})\n{s_data}")
            elif "exists but is empty" in s_data:
                sample_parts.append(f"-- Table {table_name} (in DB {db_id_for_log}) is empty.")
            # else: Error in s_data, perhaps log it but don't append
        else:
            logging.debug(f"[{db_id_for_log}] No sample data returned for table '{table_name}'.")


    if not sample_parts:
        return f"N/A (no data found in sampled tables for DB {db_id_for_log} or tables had errors during sampling)"
    return "\n\n".join(sample_parts)


# --- Main Benchmarking Logic for benchmark.py ---
def run_benchmark(
    dataset: List[Dict[str, Any]],
    db_dir: str,
    llm_wrapper, # General LLM (not used in this version for retry, but kept for consistency)
    aux_models,  # Contains SQL LLM
    results_filepath: str = "benchmark_simple_results.jsonl", # For JSONL output
    limit: Optional[int] = None
):
    if not llm_wrapper or not llm_wrapper.is_ready: # Though not used for retry here, check for completeness
        logging.error("LLM Wrapper (general) is not ready. Aborting benchmark.")
        return None, 0, 0
    if not aux_models or aux_models.get('status') != 'loaded' or not aux_models.get('sql_gguf_model'):
        logging.error("Auxiliary models (specifically SQL GGUF) not ready. Aborting benchmark.")
        return None, 0, 0

    # results_summary_list = [] # If you want to return a list of dicts from function
    processed_count = 0
    success_exec_count = 0
    success_match_count = 0
    skipped_db_setup_count = 0
    generation_errors_count = 0
    gen_sql_exec_errors_count = 0
    gold_sql_exec_errors_count = 0


    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Using temporary directory for databases: {temp_dir}")

        for i, example in enumerate(dataset):
            if limit is not None and processed_count >= limit: # Check against processed_count
                logging.info(f"Reached benchmark limit of {limit} examples processed.")
                break

            processed_count += 1
            logging.info(f"\n--- Processing Example {processed_count}/{limit if limit else len(dataset)} (DB: {example['db_id']}, Orig Index: {example.get('original_index', 'N/A')}) ---")
            logging.info(f"Question: {example['question']}")
            logging.info(f"Gold SQL: {example['query']}")

            # Initialize per-sample results
            current_status = "Initializing"
            generated_sql_str = None
            gen_sql_exec_error_msg = None
            gold_sql_exec_error_msg = None
            exec_success_flag = False
            match_success_flag = False
            multi_table_sample_for_log = "N/A"
            schema_for_db_dict = None # Store the schema dict for logging

            # 1. Setup Temporary Database
            temp_db_path = setup_temp_database(example['db_id'], db_dir, temp_dir)
            if not temp_db_path:
                logging.warning(f"Skipping example due to DB setup failure for '{example['db_id']}'.")
                skipped_db_setup_count += 1
                current_status = "DB Setup Failed"
                # Append error result and continue
                append_result_to_jsonl({
                    "original_index": example.get('original_index', 'N/A'), "db_id": example['db_id'],
                    "question": example['question'], "gold_sql": example['query'],
                    "status": current_status, "generated_sql": None,
                    "exec_success": False, "match_success": False,
                    "sample_data_used": multi_table_sample_for_log
                }, results_filepath)
                continue

            try:
                with sqlite3.connect(temp_db_path, timeout=10) as temp_conn:
                    # 2. Get Schema and Sample Data
                    try:
                        schema_for_db_dict = get_schema_info(temp_conn) # This is {table_name: [cols]}
                        if not schema_for_db_dict:
                            raise ValueError("Failed to retrieve schema from temporary database.")
                        
                        multi_table_sample_for_log = _get_multi_table_sample_str_for_benchmark(
                            example['db_id'], schema_for_db_dict, temp_conn
                        )
                        logging.debug(f"Multi-table sample for {example['db_id']}:\n{multi_table_sample_for_log[:300]}...")
                        current_status = "Schema and Sample Obtained"
                    except Exception as schema_err:
                        logging.error(f"Failed to get schema/sample for temp DB '{temp_db_path}': {schema_err}")
                        current_status = "Schema/Sample Error"
                        append_result_to_jsonl({
                            "original_index": example.get('original_index', 'N/A'), "db_id": example['db_id'],
                            "question": example['question'], "gold_sql": example['query'],
                            "status": current_status, "generated_sql": None,
                            "exec_success": False, "match_success": False,
                            "sample_data_used": multi_table_sample_for_log # Might still be N/A
                        }, results_filepath)
                        continue # Skip to next example

                    # 3. Generate SQL
                    start_time_gen = time.time()
                    generated_sql_str = _generate_sql_query(
                        user_query=example['question'],
                        schema=schema_for_db_dict, # Pass the {table:cols} schema
                        sample_data_str=multi_table_sample_for_log, # Pass multi-table sample
                        aux_models=aux_models, # Contains SQL LLM
                        # No retry mechanism in this simple benchmark version
                        previous_sql=None, feedback=None
                    )
                    gen_time_secs = time.time() - start_time_gen
                    logging.info(f"Generated SQL ({gen_time_secs:.2f}s): {generated_sql_str}")

                    if generated_sql_str.startswith("-- Error"):
                        gen_sql_exec_error_msg = generated_sql_str # Store the generation error
                        generation_errors_count +=1
                        current_status = "SQL Generation Failed by LLM"
                    else:
                        current_status = "SQL Generation Succeeded by LLM"
                        # 4. Execute Generated SQL
                        gen_results_set, gen_sql_exec_error_msg, _ = robust_execute_sql(temp_conn, generated_sql_str)
                        if gen_sql_exec_error_msg:
                            gen_sql_exec_errors_count += 1
                            current_status = "Generated SQL Execution Failed"
                        else:
                            exec_success_flag = True
                            success_exec_count += 1
                            current_status = "Generated SQL Executed Successfully"
                            logging.debug("Generated SQL executed successfully.")

                            # 5. Execute Gold SQL
                            gold_results_set, gold_sql_exec_error_msg, _ = robust_execute_sql(temp_conn, example['query'])
                            if gold_sql_exec_error_msg:
                                gold_sql_exec_errors_count += 1
                                current_status = "Gold SQL Execution Failed"
                                logging.warning(f"Gold SQL failed execution! Error: {gold_sql_exec_error_msg}")
                            else:
                                current_status = "Both SQL Executed"
                                logging.debug("Gold SQL executed successfully.")
                                # 6. Compare Results
                                match_success_flag = compare_results(gold_results_set, gen_results_set)
                                if match_success_flag:
                                    success_match_count += 1
                                    current_status = "Result Match: SUCCESS"
                                    logging.info("Result Match: SUCCESS")
                                else:
                                    current_status = "Result Match: FAILED"
                                    logging.warning("Result Match: FAILED")
                                    # Add match_details if needed, similar to benchmark_nl.py
            except Exception as outer_err:
                current_status = f"Outer Loop Error: {type(outer_err).__name__}"
                logging.error(f"Error during processing example for DB '{example['db_id']}': {outer_err}", exc_info=True)

            # Store results for this sample
            sample_result_dict = {
                "original_index": example.get('original_index', 'N/A'),
                "db_id": example['db_id'],
                "question": example['question'],
                "gold_sql": example['query'],
                "generated_sql": generated_sql_str,
                "status": current_status,
                "generation_error_msg": gen_sql_exec_error_msg if current_status == "SQL Generation Failed by LLM" else None,
                "generated_sql_exec_error": gen_sql_exec_error_msg if current_status == "Generated SQL Execution Failed" else None,
                "gold_sql_exec_error": gold_sql_exec_error_msg,
                "exec_success": exec_success_flag,
                "match_success": match_success_flag,
                "sample_data_used": multi_table_sample_for_log
            }
            append_result_to_jsonl(sample_result_dict, results_filepath)
            # results_summary_list.append(sample_result_dict)

    # --- Calculate Metrics ---
    # Denominator for exec_accuracy should be examples where generation didn't fail and DB setup was OK
    # total_candidates_for_execution = processed_count - skipped_db_setup_count - generation_errors_count
    
    # A simpler interpretation: how many of the *processed* samples led to successful execution?
    total_fully_processed = processed_count - skipped_db_setup_count
    
    exec_accuracy = (success_exec_count / total_fully_processed) * 100 if total_fully_processed > 0 else 0.0
    # Match accuracy: of those that executed successfully, how many matched.
    match_accuracy = (success_match_count / success_exec_count) * 100 if success_exec_count > 0 else 0.0

    logging.info("\n--- Benchmark Summary (Simple SQL Generation Test) ---")
    logging.info(f"Total examples in dataset provided: {len(dataset)}")
    logging.info(f"Examples processed (or attempted up to limit): {processed_count}")
    logging.info(f"Examples skipped (DB setup failed): {skipped_db_setup_count}")
    logging.info(f"SQL Generation Failures (by LLM): {generation_errors_count}")
    logging.info(f"Generated SQL Execution Errors: {gen_sql_exec_errors_count}")
    logging.info(f"Gold SQL Execution Errors: {gold_sql_exec_errors_count}")
    logging.info(f"Successfully executed generated SQL: {success_exec_count}")
    logging.info(f"Generated SQL matched gold results: {success_match_count}")
    logging.info(f"Execution Accuracy (Exec Success / (Processed - Skipped DB Setup)): {exec_accuracy:.2f}%")
    logging.info(f"Exact Set Match Accuracy (Match Success / Successful Executions): {match_accuracy:.2f}%")

    return None, exec_accuracy, match_accuracy # Return None for summary list as it's written to JSONL


def append_result_to_jsonl(result_dict: dict, filepath: str):
    """Appends a dictionary as a JSON line to the specified file."""
    try:
        json_string = json.dumps(result_dict)
        with open(filepath, 'a') as f:
            f.write(json_string + '\n')
    except Exception as e:
        logging.error(f"Failed to append result to {filepath}: {e} | Data: {result_dict}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark NL-to-SQL generation (simple version).")
    parser.add_argument("test_suite_file", help="Path to the test suite JSON file (e.g., dev.json).")
    parser.add_argument("gold_sql_file", help="Path to the corresponding gold SQL file (e.g., dev_gold.sql).")
    parser.add_argument("database_dir", help="Path to the directory containing database folders (e.g., 'database/').")
    # No tables.json needed here as schema is derived from live DB connection
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to process.")
    parser.add_argument("--results_out_jsonl", default="benchmark_simple_results.jsonl", help="Path to save detailed benchmark results (JSONL).")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())
    logging.info(f"Logging level set to {args.log_level.upper()}")

    # Clear results file if it exists
    if os.path.exists(args.results_out_jsonl):
        os.remove(args.results_out_jsonl)
        logging.info(f"Removed existing results file: {args.results_out_jsonl}")

    logging.info("Initializing LLM Wrapper and Auxiliary Models...")
    llm_wrapper = get_llm_wrapper() # General LLM
    aux_models = get_cached_aux_models() # Contains SQL LLM
    logging.info("Initialization complete.")

    questions_data = load_test_suite_questions(args.test_suite_file)
    gold_queries = load_test_suite_gold_sql(args.gold_sql_file)
    if not questions_data or not gold_queries:
        logging.error("Failed to load questions or gold queries. Exiting.")
        exit(1)

    dataset = combine_test_suite_data(questions_data, gold_queries)
    if not dataset:
        logging.error("Failed to combine dataset. Exiting.")
        exit(1)

    _, exec_acc, match_acc = run_benchmark(
        dataset=dataset,
        db_dir=args.database_dir,
        llm_wrapper=llm_wrapper,
        aux_models=aux_models,
        results_filepath=args.results_out_jsonl,
        limit=args.limit
    )

    logging.info(f"Simple SQL generation benchmarking finished. Results appended to {args.results_out_jsonl}")
    logging.info(f"Overall Execution Accuracy: {exec_acc:.2f}%")
    logging.info(f"Overall Exact Set Match Accuracy: {match_acc:.2f}%")

if __name__ == "__main__":
    main()
# --- END OF FILE benchmark.py ---