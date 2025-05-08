import os
import json
import sqlite3
import pandas as pd
import argparse
import logging
import time
import shutil
import tempfile
import random # Import random
from typing import List, Dict, Any, Set, Tuple, Optional
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import necessary components from your existing utils ---
try:
    from utils.utils import (
        setup_environment, get_db_connection, init_qdrant_client, get_llm_wrapper,
        get_cached_aux_models, get_schema_info, get_table_sample_data,
        process_natural_language_query, # <--- Import the main pipeline function
        QDRANT_COLLECTION_PREFIX # If needed by dependencies, though not used directly here
    )
    # Import helper functions from the previous benchmark script (or define them here)
    from benchmark.benchmark_v2 import ( # Assuming benchmark.py exists in the same dir
         load_test_suite_questions, load_test_suite_gold_sql,
         combine_test_suite_data, setup_temp_database, # We'll use the version creating from SQL
         robust_execute_sql, compare_results
    )
except ImportError as e:
    print(f"Error importing from utils/benchmark: {e}")
    print("Please ensure benchmark_randomized.py is in the correct directory relative to utils.py and benchmark.py")
    exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# --- Helper Functions ---

def append_result_to_jsonl(result_dict: dict, filepath: str):
    """Appends a dictionary as a JSON line to the specified file."""
    try:
        json_string = json.dumps(result_dict)
        with open(filepath, 'a') as f: # Use 'a' for append mode
            f.write(json_string + '\n')
    except Exception as e:
        logging.error(f"Failed to append result to {filepath}: {e} | Data: {result_dict}")

def load_schemas_from_tables_json(tables_file: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Loads database schemas directly from a Spider-like tables.json file.
    Now used for loading ALL relevant schemas (e.g., from test_tables.json).
    """
    schemas: Dict[str, Dict[str, List[str]]] = {}
    logging.info(f"Attempting to load schemas from: {tables_file}") # Log which file is used
    try:
        with open(tables_file, 'r') as f: tables_data = json.load(f)
        for db_info in tables_data:
            db_id = db_info['db_id']; db_schema = {}
            table_names = db_info.get('table_names_original', db_info.get('table_names', []))
            column_data = db_info.get('column_names_original', db_info.get('column_names', []))
            table_idx_to_name = {i: name for i, name in enumerate(table_names)}
            for col_info in column_data:
                table_idx, col_name = col_info[0], col_info[1]
                if table_idx == -1: continue
                table_name = table_idx_to_name.get(table_idx)
                if table_name:
                    if table_name not in db_schema: db_schema[table_name] = []
                    # Store column name, maybe add type later if needed
                    db_schema[table_name].append(col_name.strip()) # Strip potential whitespace
            schemas[db_id] = db_schema
        logging.info(f"Loaded schemas for {len(schemas)} databases from {tables_file}")
        return schemas
    except FileNotFoundError:
        logging.error(f"Schema file not found: {tables_file}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from schema file: {tables_file}")
        return {}
    except Exception as e:
        logging.error(f"Error processing schema file {tables_file}: {e}", exc_info=True)
        return {}

# --- Main Benchmarking Logic ---
# --- START OF FILE benchmark_nl.py ---
# ... (imports and other functions remain largely the same) ...

# --- Helper Functions ---
# ... (append_result_to_jsonl, load_schemas_from_tables_json are fine) ...

def _get_multi_table_sample_str(
    db_id_to_sample: str,
    db_schema_dict: Dict[str, List[str]], # Schema for this specific db_id {table: [cols]}
    db_conn: sqlite3.Connection,
    num_tables_to_sample: int = 3,
    num_rows_per_table: int = 2
) -> str:
    """Helper to get a concatenated sample string from multiple tables of a DB."""
    sample_parts = []
    if not db_schema_dict:
        logging.warning(f"No schema provided for DB '{db_id_to_sample}' for sampling.")
        return "N/A (schema not found)"

    tables_to_sample = list(db_schema_dict.keys())[:num_tables_to_sample]
    if not tables_to_sample:
        return "N/A (no tables in schema)"

    for table_name in tables_to_sample:
        s_data = get_table_sample_data(db_conn, table_name, limit=num_rows_per_table)
        if s_data and "Error" not in s_data and "exists but is empty" not in s_data : # Basic check for valid sample
            sample_parts.append(f"-- Sample rows from table {table_name}\n{s_data}")
        elif s_data and "exists but is empty" in s_data:
             sample_parts.append(f"-- Table {table_name} is empty.")


    if not sample_parts:
        return "N/A (no data found in sampled tables or tables had errors)"
    return "\n\n".join(sample_parts)


# --- Main Benchmarking Logic ---
def run_randomized_benchmark(
    full_dataset: List[Dict[str, Any]],
    num_samples: int,
    db_base_dir: str,
    relevant_schemas: Dict[str, Dict[str, List[str]]], # {db_id: {table: [cols]}}
    llm_wrapper,
    aux_models,
    qdrant_client,
    results_filepath: str = "benchmark_randomized_results.jsonl", # Removed conversation_history, num_schema_subset from here
                                                                  # as they are effectively replaced by context_schemas_for_llm logic
    num_schema_context_size: int = 20 # Renamed from num_schema_subset for clarity
):
    """Runs the benchmark evaluation loop on randomly sampled examples."""
    # ... (Readiness Checks remain similar) ...

    all_available_db_ids = list(relevant_schemas.keys())
    if not all_available_db_ids:
        logging.error("No schemas found in 'relevant_schemas'. Cannot proceed.")
        return None, 0,0,0

    # --- Select Random Samples (Questions) ---
    if num_samples >= len(full_dataset):
        sampled_dataset = full_dataset
        logging.info(f"Using all {len(full_dataset)} examples from dataset.")
    else:
        sampled_dataset = random.sample(full_dataset, num_samples)
        logging.info(f"Selected {num_samples} random examples (questions) for benchmarking.")

    # ... (Initialize Results Storage remains similar) ...
    exec_success = 0
    match_success = 0
    skipped_db = 0
    analysis_errors = 0
    pipeline_analysis_success_sql = 0
    count=0
    
    # Initialize results summary list (moved outside loop for appending, or handle inside if preferred)
    results_summary_list = [] # If you want to return a list of dicts

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Using temporary directory for databases: {temp_dir}")

        for i, example in enumerate(sampled_dataset):
            count += 1
            actual_db_id = example['db_id']
            logging.info(f"\n--- Processing Sample {i+1}/{num_samples} (Actual DB: {actual_db_id}, Orig Index: {example.get('original_index', 'N/A')}) ---")

            pipeline_status = "Processing Initial Setup"
            pipeline_error_msg = None
            determined_route = None
            selected_db_id_by_llm = None # Renamed for clarity
            generated_sql = None
            gen_sql_exec_error = None
            gold_sql_exec_error = None
            exec_success_flag = False
            match_success_flag = False
            sample_data_for_actual_db_logging = "N/A" # For logging this specific sample
            pipeline_time = None
            gen_sql_exec_time = 0.0
            gold_sql_exec_time = 0.0

            # --- 1. Setup ACTUAL DB temporarily for execution and initial sampling ---
            temp_db_path_actual = setup_temp_database(actual_db_id, db_base_dir, temp_dir)
            if not temp_db_path_actual:
                logging.warning(f"Skipping sample due to DB setup failure for actual DB '{actual_db_id}'.")
                skipped_db += 1
                pipeline_status = "DB Setup Failed (Actual DB)"
                results_summary_list.append({ # Or append_result_to_jsonl
                    "original_index": example.get('original_index', 'N/A'), "actual_db_id": actual_db_id,
                    "question": example['question'], "gold_sql": example['query'],
                    "pipeline_status": pipeline_status, "selected_db_id_by_llm": None,
                    # ... other null fields
                })
                append_result_to_jsonl(results_summary_list[-1], results_filepath)
                continue

            # --- 2. Prepare Schema and Sample Context for LLM (Actual DB + Distractors) ---
            # This context will be passed to `process_natural_language_query`
            context_schemas_for_llm = {}
            db_samples_for_llm_selection_prompt = {}

            # A. Include Actual DB's schema and sample
            actual_db_schema_dict = relevant_schemas.get(actual_db_id)
            if not actual_db_schema_dict:
                logging.error(f"Schema for actual_db_id '{actual_db_id}' not found in relevant_schemas. Skipping.")
                # ... (handle skip, append error to results) ...
                append_result_to_jsonl({
                     "original_index": example.get('original_index', 'N/A'), "actual_db_id": actual_db_id,
                     "question": example['question'], "gold_sql": example['query'],
                     "pipeline_status": "Schema Missing for Actual DB", "selected_db_id_by_llm": None,
                }, results_filepath)
                continue

            context_schemas_for_llm[actual_db_id] = actual_db_schema_dict
            try:
                with sqlite3.connect(temp_db_path_actual) as actual_conn:
                    sample_str = _get_multi_table_sample_str(actual_db_id, actual_db_schema_dict, actual_conn)
                    db_samples_for_llm_selection_prompt[actual_db_id] = sample_str
                    sample_data_for_actual_db_logging = sample_str # For logging
            except Exception as e_actual_sample:
                logging.error(f"Error getting samples for actual DB {actual_db_id}: {e_actual_sample}")
                db_samples_for_llm_selection_prompt[actual_db_id] = "Sample data fetch error."
                sample_data_for_actual_db_logging = "Error fetching samples."


            # B. Add Distractor DBs' schemas and samples
            num_distractors_needed = num_schema_context_size - 1
            if num_distractors_needed > 0:
                other_db_ids = [db_id for db_id in all_available_db_ids if db_id != actual_db_id]
                num_distractors_to_select = min(num_distractors_needed, len(other_db_ids))

                if num_distractors_to_select > 0:
                    distractor_ids_selected = random.sample(other_db_ids, num_distractors_to_select)
                    for dist_db_id in distractor_ids_selected:
                        dist_schema_dict = relevant_schemas.get(dist_db_id)
                        if not dist_schema_dict:
                            logging.warning(f"Schema for distractor_db_id '{dist_db_id}' not found. Skipping for context.")
                            continue
                        context_schemas_for_llm[dist_db_id] = dist_schema_dict

                        # Setup temp DB for distractor to get samples
                        temp_db_path_dist = setup_temp_database(dist_db_id, db_base_dir, temp_dir)
                        if temp_db_path_dist:
                            try:
                                with sqlite3.connect(temp_db_path_dist) as dist_conn:
                                    sample_str_dist = _get_multi_table_sample_str(dist_db_id, dist_schema_dict, dist_conn)
                                    db_samples_for_llm_selection_prompt[dist_db_id] = sample_str_dist
                            except Exception as e_dist_sample:
                                logging.warning(f"Could not get samples for distractor DB {dist_db_id}: {e_dist_sample}")
                                db_samples_for_llm_selection_prompt[dist_db_id] = "Sample data fetch error."
                            # temp_db_path_dist will be cleaned up by TemporaryDirectory
                        else:
                            logging.warning(f"Could not setup temp DB for distractor {dist_db_id} to get samples.")
                            db_samples_for_llm_selection_prompt[dist_db_id] = "Sample data unavailable (DB setup failed)."
            logging.info(f"LLM context schema size: {len(context_schemas_for_llm)}. Sample map size: {len(db_samples_for_llm_selection_prompt)}")
            logging.debug(f"Context DB IDs: {list(context_schemas_for_llm.keys())}")
            logging.debug(f"Sample map keys for LLM: {list(db_samples_for_llm_selection_prompt.keys())}")



            # --- 3. Establish connection for pipeline (to ACTUAL DB) ---
            # This connection is used if the pipeline needs to execute SQL against the *true* DB.
            # If the pipeline *selects* a different DB, this connection is "wrong" for that selected DB,
            # which is part of testing the selection accuracy. SQL execution will likely fail, correctly.
            conn_for_pipeline_execution = sqlite3.connect(temp_db_path_actual, timeout=10)

            # --- 4. Run Analysis & Generation ---
            try:
                start_time_pipe = time.time()
                pipeline_result = process_natural_language_query(
                    original_query=example['question'],
                    conn=conn_for_pipeline_execution, # Connection to the *actual* database
                    full_schema=context_schemas_for_llm,                    
                    llm_wrapper=llm_wrapper,
                    aux_models=aux_models,
                    qdrant_client=qdrant_client,
                    db_samples_for_selection=db_samples_for_llm_selection_prompt, # Pass prepared samples
                    conversation_history=[] # Assuming fresh for each benchmark item
                )
                pipeline_time = time.time() - start_time_pipe
                logging.info(f"Pipeline processing finished in {pipeline_time:.2f}s")

                analysis_status = pipeline_result.get("status", "error_unknown")
                pipeline_error_msg = pipeline_result.get("message")
                determined_route = pipeline_result.get("determined_route")
                selected_db_id_by_llm = pipeline_result.get("selected_db_id") # Capture LLM's choice
                generated_sql = pipeline_result.get("generated_sql")

                analysis_succeeded_for_sql_route = False
                if selected_db_id_by_llm != actual_db_id:
                    analysis_errors += 1
                    pipeline_status = f"Analysis Failed (Incorrect DB: LLM chose '{selected_db_id_by_llm}', actual '{actual_db_id}')"
                    logging.error(pipeline_status)
                    if not pipeline_error_msg: pipeline_error_msg = f"Incorrect DB Selected: Got '{selected_db_id_by_llm}', Expected '{actual_db_id}'"
                elif analysis_status.startswith("error"):
                    analysis_errors += 1
                    pipeline_status = f"Analysis Failed ({analysis_status})"
                    logging.error(f"Pipeline analysis/generation error: {pipeline_error_msg}")
                elif determined_route != "SQL":
                    pipeline_status = f"Analysis Success (Route: {determined_route})"
                    logging.info(f"Route determined as '{determined_route}'. Skipping SQL benchmark for this sample.")
                elif not generated_sql or generated_sql.startswith("-- Error"):
                    analysis_errors += 1
                    pipeline_status = "SQL Generation Failed in Pipeline"
                    pipeline_error_msg = generated_sql if generated_sql else "No SQL generated by pipeline"
                    logging.error(f"SQL Generation Failed in Pipeline: {pipeline_error_msg}")
                else:
                    analysis_succeeded_for_sql_route = True
                    pipeline_analysis_success_sql += 1
                    pipeline_status = "Analysis & SQL Gen Success (SQL Route)"
                    logging.info(f"Pipeline Analysis successful. Route: SQL. Selected DB: {selected_db_id_by_llm}. SQL: {generated_sql}")

                # --- 5. Execute and Compare (only if analysis for SQL route was good AND LLM selected the CORRECT DB) ---
                if analysis_succeeded_for_sql_route and selected_db_id_by_llm == actual_db_id:
                    # Execute generated SQL on the actual DB (conn_for_pipeline_execution is already for actual_db_id)
                    try:
                        start_gen_exec = time.time()
                        gen_results_set, gen_sql_exec_error_msg, _ = robust_execute_sql(conn_for_pipeline_execution, generated_sql)
                        gen_sql_exec_time = time.time() - start_gen_exec # Capture actual exec time
                        gen_sql_exec_error = gen_sql_exec_error_msg # Store error message string

                        if gen_sql_exec_error:
                            pipeline_status = "Generated SQL Execution Failed"
                            logging.warning(f"Generated SQL execution failed: {gen_sql_exec_error}")
                        else:
                            exec_success_flag = True
                            exec_success += 1
                            pipeline_status = "Generated SQL Executed Successfully"
                            logging.debug("Generated SQL executed successfully.")

                            # Execute Gold SQL
                            start_gold_exec = time.time()
                            gold_results_set, gold_sql_exec_error_msg, _ = robust_execute_sql(conn_for_pipeline_execution, example['query'])
                            gold_sql_exec_time = time.time() - start_gold_exec # Capture actual exec time
                            gold_sql_exec_error = gold_sql_exec_error_msg # Store error message string

                            if gold_sql_exec_error:
                                pipeline_status = "Gold SQL Execution Failed"
                                logging.warning(f"Gold SQL failed execution! Error: {gold_sql_exec_error}")
                            else:
                                pipeline_status = "Both SQL Executed"
                                logging.debug("Gold SQL executed successfully.")
                                match_success_flag = compare_results(gold_results_set, gen_results_set)
                                if match_success_flag:
                                    match_success += 1
                                    pipeline_status = "Result Match: SUCCESS"
                                    logging.info("Result Match: SUCCESS")
                                else:
                                    pipeline_status = "Result Match: FAILED"
                                    logging.warning("Result Match: FAILED")
                    except Exception as exec_err:
                        pipeline_status = f"Execution Loop Error: {type(exec_err).__name__}"
                        pipeline_error_msg = str(exec_err)
                        logging.error(f"Error during execution phase for DB '{actual_db_id}': {exec_err}", exc_info=True)
                elif analysis_succeeded_for_sql_route and selected_db_id_by_llm != actual_db_id:
                    # Analysis was fine, SQL generated, but for the wrong DB. We don't execute.
                    # This case is already covered by the `selected_db_id_by_llm != actual_db_id` check earlier.
                    # `pipeline_status` would already be set to "Analysis Failed (Incorrect DB...)"
                    logging.info(f"SQL generated for '{selected_db_id_by_llm}' but actual is '{actual_db_id}'. No execution attempted on actual DB.")


            except Exception as outer_pipeline_err:
                analysis_errors += 1
                pipeline_status = f"Outer Pipeline Error: {type(outer_pipeline_err).__name__}"
                pipeline_error_msg = str(outer_pipeline_err)
                logging.error(f"Error processing pipeline for example {example.get('original_index', i)+1}: {outer_pipeline_err}", exc_info=True)
            finally:
                if conn_for_pipeline_execution:
                    conn_for_pipeline_execution.close()


            # --- Store results for this sample ---
            current_sample_result = {
                "original_index": example.get('original_index', 'N/A'),
                "actual_db_id": actual_db_id,
                "question": example['question'],
                "gold_sql": example['query'],
                "pipeline_status": pipeline_status,
                "pipeline_error": pipeline_error_msg,
                "determined_route": determined_route,
                "selected_db_id_by_llm": selected_db_id_by_llm,
                "generated_sql": generated_sql,
                "generated_sql_exec_error": gen_sql_exec_error,
                "gold_sql_exec_error": gold_sql_exec_error,
                "exec_success": exec_success_flag,
                "match_success": match_success_flag,
                "sample_data_for_actual_db": sample_data_for_actual_db_logging, # Log the multi-table sample
                "pipeline_time_seconds": pipeline_time,
                "generated_sql_exec_time_seconds": gen_sql_exec_time,
                "gold_sql_exec_time_seconds": gold_sql_exec_time,
                "context_db_ids_for_llm": list(context_schemas_for_llm.keys())
            }
            # results_summary_list.append(current_sample_result) # If returning a list
            append_result_to_jsonl(current_sample_result, results_filepath)
            # --- End Example Loop ---

    # --- Calculate Metrics ---
    total_processed = count
    # total_candidates_for_execution: SQL generated for the *correct* DB.
    # pipeline_analysis_success_sql counts SQL generated, regardless of DB selection.
    # We need a count where: route=SQL, gen_sql_ok, AND selected_db==actual_db
    # This is implicitly `pipeline_analysis_success_sql` MINUS cases where selected_db was wrong but SQL was still generated.
    # Or, more directly, `exec_success` is already based on this condition.
    
    # For accuracy, consider the denominator for exec_accuracy and match_accuracy.
    # Analysis Accuracy: % of (processed where pipeline route=SQL, gen SQL OK, selected DB correct)
    # This is `pipeline_analysis_success_sql` if we assume it also implies correct DB selection for "success".
    # Let's refine `pipeline_analysis_success_sql` to mean: correct DB selected, SQL route, SQL gen OK.
    # The current logic for `pipeline_analysis_success_sql` increments if route=SQL and gen_sql_ok,
    # *before* checking if `selected_db_id_by_llm == actual_db_id`.

    # Recalculate `pipeline_analysis_success_sql` based on results file or re-evaluate logic.
    # For now, let's use `exec_success` as the base for `match_success`.
    # And `pipe_an_acc` based on `pipeline_analysis_success_sql` which means "SQL generated for selected DB".
    # A more stringent "Correct Analysis Accuracy" would be (count of (correct_db_selected AND sql_route AND gen_ok)) / total_processed

    # Using the existing metric names:
    if total_processed == 0: pipe_an_acc = 0.0; exec_acc = 0.0; match_acc = 0.0
    else:
        # `pipeline_analysis_success_sql` = SQL generated (for *any* selected DB)
        # `exec_success` = SQL executed (implies correct DB was selected and SQL generated for it)
        pipe_an_acc = (pipeline_analysis_success_sql / total_processed) * 100 if total_processed > 0 else 0.0
        exec_acc = (exec_success / pipeline_analysis_success_sql) * 100 if pipeline_analysis_success_sql > 0 else 0.0 #This might be too broad if pipeline_analysis_success_sql includes wrong DBs
                                                                                                                      # Better: exec_acc = (exec_success / num_where_correct_db_and_sql_route_and_gen_ok)
                                                                                                                      # For now, let's interpret `exec_success` as a raw count.
        # Let's define Execution Accuracy as: (exec_success / number of times SQL was generated FOR THE CORRECT DB)
        # This `number of times SQL was generated FOR THE CORRECT DB` is implicitly `exec_success` + `gen_sql_exec_error where correct DB was selected`.
        # Simpler: `exec_acc` = (number of successful executions) / (number of attempts to execute generated SQL for the correct DB)
        # The current loop structure means `exec_success` is the count of successful executions.
        # The denominator for exec_acc should be number of times we *tried* to execute SQL generated for the *correct* DB.
        # This is `pipeline_analysis_success_sql` IF we filter it for `selected_db_id_by_llm == actual_db_id`.
        
        # Let's count `attempted_execution_on_correct_db` from the results for better accuracy calc
        # This would require reading the results file or accumulating differently.
        # For now, use simpler interpretation:
        
        # `pipe_an_acc`: Percentage of samples where the pipeline successfully analyzed, selected a DB (any DB),
        # determined SQL route, and generated SQL.
        
        # `exec_acc`: Of those cases where SQL was generated for the *correct* DB, what percentage executed.
        # `exec_success` is the count of successful executions.
        # The number of times SQL was generated for the correct DB and execution was attempted is effectively the number of items
        # that reached the `if analysis_succeeded_for_sql_route and selected_db_id_by_llm == actual_db_id:` block.
        # Let this count be `num_candidates_for_correct_db_execution`.
        # We need to sum this up.
        
        # For simplicity here, let's use the counts we have:
        # `exec_success` is the number of times SQL executed successfully.
        # `match_success` is the number of times results matched.
        
        # To calculate `exec_acc` properly, we need the number of times SQL execution was *attempted* on the correct DB.
        # This is `exec_success` (succeeded) + count of (gen_sql_exec_error AND correct DB was selected).
        # This isn't directly available without iterating results.
        
        # Redefine metrics based on what's easily countable:
        # Analysis & SQL Generation Accuracy (pipeline produced SQL for *some* DB):
        #   (pipeline_analysis_success_sql / total_processed)
        # Execution Accuracy (of SQL generated for the *correct* DB, how many executed):
        #   This is hard to get denominator without parsing results.
        #   Let's use (exec_success / total_processed) as a "End-to-End Execution Success Rate"
        # Match Accuracy (of successfully executed SQL, how many matched):
        #   (match_success / exec_success)
        
        e2e_exec_rate = (exec_success / total_processed) * 100 if total_processed > 0 else 0.0
        match_acc_strict = (match_success / exec_success) * 100 if exec_success > 0 else 0.0

    logging.info("\n--- Randomized Benchmark Summary (Schema Subset) ---")
    logging.info(f"Schema Context Size for Selection: {num_schema_context_size}")
    logging.info(f"Total examples processed: {total_processed}")
    logging.info(f"Examples skipped (DB setup failed for actual DB): {skipped_db}")
    logging.info(f"Pipeline Analysis/Selection/Generation Errors (before SQL exec phase): {analysis_errors}")
    logging.info(f"Pipeline Analysis Successful (SQL Route & Gen OK, any DB selected): {pipeline_analysis_success_sql}")
    logging.info(f"Successfully executed generated SQL (on correct DB): {exec_success}")
    logging.info(f"Generated SQL matched gold results: {match_success}")

    logging.info(f"Overall Analysis & SQL Generation Accuracy (SQL generated for any selected DB / Processed): {pipe_an_acc:.2f}%")
    # This is a bit tricky. Let's make exec_acc relative to when SQL was generated for the correct DB.
    # We don't have a direct counter for "SQL generated for correct DB".
    # Let's report exec_success directly.
    # logging.info(f"Execution Accuracy (vs Pipeline Success for SQL route): {exec_acc:.2f}%") # Old interpretation
    logging.info(f"End-to-End Execution Rate (Exec Success on Correct DB / Total Processed): {e2e_exec_rate:.2f}%")
    logging.info(f"Exact Set Match Accuracy (vs Successful Executions on Correct DB): {match_acc_strict:.2f}%")

    # Return `results_summary_list` if you want to process it further, otherwise it's written to JSONL.
    return None, pipe_an_acc, e2e_exec_rate, match_acc_strict # Returning overall rates


# --- Main Execution ---
def main():
    
    parser = argparse.ArgumentParser(description="Benchmark NL-to-SQL FULL PIPELINE on random samples.")
    parser.add_argument("test_suite_file", help="Path to the test suite JSON file (e.g., test.json or dev.json).")
    parser.add_argument("gold_sql_file", help="Path to the corresponding gold SQL file (e.g., test_gold.sql or dev_gold.sql).")
    parser.add_argument("database_dir", help="Path to the directory containing database folders (e.g., 'test_database/' or 'database/').")
    parser.add_argument("tables_json_file", help="Path to the tables.json file containing schemas for all databases.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random examples to sample from the dataset.") # Default to 10 for quick test
    parser.add_argument("--num_schemas", type=int, default=5, help="Number of DB schemas (actual + distractors) for LLM context.") # Default to 5 for quick test
    parser.add_argument("--results_out_jsonl", default="benchmark_nl_results.jsonl", help="Path to save detailed benchmark results (JSONL format).")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())
    logging.info(f"Logging level set to {args.log_level.upper()}")
    
    # Clear results file if it exists
    if os.path.exists(args.results_out_jsonl):
        os.remove(args.results_out_jsonl)
        logging.info(f"Removed existing results file: {args.results_out_jsonl}")

    logging.info("Initializing LLM Wrapper, Auxiliary Models, and Qdrant...")
    llm_wrapper = get_llm_wrapper()
    aux_models = get_cached_aux_models()
    qdrant_client = init_qdrant_client()
    logging.info("Initialization complete.")

    questions_data = load_test_suite_questions(args.test_suite_file)
    gold_queries = load_test_suite_gold_sql(args.gold_sql_file)
    # `test_schemas` will be Dict[db_id, Dict[table_name, List[col_name]]]
    test_schemas  = load_schemas_from_tables_json(args.tables_json_file)
    if not test_schemas:
         logging.error("Test schemas failed to load or is empty. Exiting.") # More direct error
         exit(1)
         # ---- ADD DEBUG PRINT ----
    logging.info(f"Number of schemas loaded: {len(test_schemas)}")
    if test_schemas:
        logging.info(f"Example DB IDs loaded into test_schemas: {list(test_schemas.keys())[:10]}") # Print some keys
    # ---- END DEBUG PRINT ---
    if not questions_data or not gold_queries or not test_schemas:
         logging.error("Failed to load test questions, gold queries, or test schemas. Exiting.")
         exit(1)
    full_dataset = combine_test_suite_data(questions_data, gold_queries)
    if not full_dataset: exit(1)

    _, overall_analysis_acc, overall_exec_rate, overall_match_acc = run_randomized_benchmark(
        full_dataset=full_dataset,
        num_samples=args.num_samples,
        db_base_dir=args.database_dir,
        relevant_schemas=test_schemas,
        llm_wrapper=llm_wrapper,
        aux_models=aux_models,
        qdrant_client=qdrant_client,
        results_filepath=args.results_out_jsonl,
        num_schema_context_size=args.num_schemas # Pass the context size
    )
    logging.info(f"Randomized benchmarking finished. Results appended to {args.results_out_jsonl}")
    logging.info(f"Overall Analysis & SQL Generation Accuracy: {overall_analysis_acc:.2f}%")
    logging.info(f"Overall End-to-End Execution Rate: {overall_exec_rate:.2f}%")
    logging.info(f"Overall Exact Set Match Accuracy: {overall_match_acc:.2f}%")

if __name__ == "__main__":
    main()
# --- END OF FILE benchmark_nl.py ---