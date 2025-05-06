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

# --- Import necessary components from your existing utils ---
try:
    from utils import (
        setup_environment, get_db_connection, init_qdrant_client, get_llm_wrapper,
        get_cached_aux_models, get_schema_info, get_table_sample_data,
        process_natural_language_query, # <--- Import the main pipeline function
        QDRANT_COLLECTION_PREFIX # If needed by dependencies, though not used directly here
    )
    # Import helper functions from the previous benchmark script (or define them here)
    from benchmark import ( # Assuming benchmark.py exists in the same dir
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

def run_randomized_benchmark(
    full_dataset: List[Dict[str, Any]], 
    num_samples: int,
    db_base_dir: str, 
    relevant_schemas: Dict[str, Dict[str, List[str]]], 
    llm_wrapper,
    aux_models,
    qdrant_client,
    conversation_history=[], 
    num_schema_subset: int = 20,
    results_filepath: str = "benchmark_randomized_results.jsonl",
    sample_result_dict: Optional[Dict[str, Any]] = None

    ):
    """Runs the benchmark evaluation loop on randomly sampled examples."""
    # --- Readiness Checks ---
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.error("LLM Wrapper not ready.")
        return None, 0, 0, 0
    if not aux_models or aux_models.get('status') != 'loaded':
        logging.error(f"Auxiliary models not ready (status: {aux_models.get('status', 'N/A')}).")
        return None, 0, 0, 0
    if not qdrant_client:
        logging.warning("Qdrant client not provided/ready. Semantic routes may fail if selected.")
        # Proceeding as NL-to-SQL is primary focus

    # --- Select Random Samples ---

    # 1. Find all unique DB IDs present in the *full loaded dataset* (e.g., test set)

    unique_db_ids_in_relevant_schemas = sorted(list(relevant_schemas.keys()))
    logging.info(f"Found {len(unique_db_ids_in_relevant_schemas)} unique database IDs in the provided dataset.")

    # 2. Select a random subset of these DB IDs for the context
    if len(unique_db_ids_in_relevant_schemas) <= num_schema_subset:
        target_db_ids = unique_db_ids_in_relevant_schemas
        logging.info(f"Using all {len(target_db_ids)} unique DB IDs from dataset for selection context (<= {num_schema_subset}).")
    else:
        target_db_ids = sorted(random.sample(unique_db_ids_in_relevant_schemas, num_schema_subset))
        logging.info(f"Randomly selected {num_schema_subset} DB IDs for selection context: {target_db_ids}")

    # 3. Create the schema dictionary containing only the target schemas
    target_schemas_subset = {db_id: relevant_schemas[db_id] for db_id in target_db_ids if db_id in relevant_schemas}

    # Check if the subset creation worked (it should if target_db_ids came from relevant_schemas)
    if len(target_schemas_subset) == 0 and len(target_db_ids) > 0:
         logging.error("Failed to create target schema subset. Check schema loading.")
         # Cannot proceed without schemas for the LLM
         return None, 0,0,0
    elif len(target_schemas_subset) < len(target_db_ids):
         logging.warning("Created schema subset has fewer entries than target IDs. This shouldn't happen.")
    logging.info(f"Prepared subset schema dictionary with {len(target_schemas_subset)} entries for LLM context.")
    
    # --- End Scope Determination ---


    
    # --- Get list of all available DB IDs from the relevant schemas ---
    all_available_db_ids = list(relevant_schemas.keys())
    if not all_available_db_ids:
        logging.error("No schemas found in the 'relevant_schemas' dictionary. Cannot proceed.")
        return None, 0,0,0

    # --- Select Random Samples (Questions) ---
    if num_samples >= len(full_dataset):
        sampled_dataset = full_dataset
        logging.info(f"Using all {len(full_dataset)} examples from dataset.")
    else:
        sampled_dataset = random.sample(full_dataset, num_samples)
        logging.info(f"Selected {num_samples} random examples (questions) for benchmarking.")


    # --- Initialize Results Storage ---
    
    exec_success = 0 # Count successful executions of generated SQL
    match_success = 0 # Count successful matches with gold results
    skipped_db = 0
    analysis_errors = 0 # Count analysis/generation errors
    results_summary = []
    count = 0
    pipeline_analysis_success_sql = 0 # Succeeded analysis & SQL Generation
    analysis_errors = 0    
    # --- Lists to store execution times for aggregation ---
    gen_sql_exec_times = []
    gold_sql_exec_times = []
    # ---          

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Using temporary directory for databases: {temp_dir}")

        for i, example in enumerate(sampled_dataset):
            count += 1
            actual_db_id = example['db_id'] # The *true* DB for this question
            logging.info(f"\n--- Processing Sample {i+1}/{num_samples} (Actual DB: {actual_db_id}, Orig Index: {example.get('original_index', 'N/A')}) ---")
            
            # --- Init results for this sample ---
            pipeline_status = "Processing Analysis"
            pipeline_error_msg = None
            determined_route = None
            selected_db_id = None
            generated_sql = None
            gen_sql_exec_error = None
            gold_sql_exec_error = None
            exec_success_flag = False
            match_success_flag = False
            sample_data_str_actual = None # To store sample data for the actual DB
            gen_exec_time = 0.0 # Initialize execution times
            gold_exec_time = 0.0 # Initialize execution times

            # --- Pre-fetch Sample Data & Prepare LLM Context ---
            # 1. Setup ACTUAL DB temporarily to get sample data
            temp_db_path = setup_temp_database(actual_db_id, db_base_dir, temp_dir)
            if not temp_db_path:
                logging.warning(f"Skipping sample due to DB setup failure for actual DB '{actual_db_id}'.")
                skipped_db += 1
                pipeline_status = "DB Setup Failed (Initial)"
                # Append error result and continue
                results_summary.append({ ... }) # Populate error details
                continue # Skip to next sample

            # 2. Get sample data from the actual DB
            try:
                with sqlite3.connect(temp_db_path) as temp_conn_sample:
                    first_table_actual = next(iter(relevant_schemas.get(actual_db_id, {})), None)
                    if first_table_actual:
                        sample_data_str_actual = get_table_sample_data(temp_conn_sample, first_table_actual, limit=3)
                    if not sample_data_str_actual:
                        logging.warning(f"Could not fetch sample data for DB '{actual_db_id}', table '{first_table_actual}'.")
                        sample_data_str_actual = "N/A" # Use placeholder
            except Exception as sample_err:
                logging.error(f"Error getting sample data for '{actual_db_id}': {sample_err}")
                sample_data_str_actual = f"Error: {sample_err}" # Pass error info

            # --- Now temp_db_path points to the correctly setup DB for LATER execution ---

            # 3. Prepare schema context for LLM selection (including sample for GT)
            context_schemas_for_llm = {}
            if actual_db_id in relevant_schemas:
                 pass
            else:
                 # This case should ideally not happen if dataset uses schemas correctly
                 logging.error(f"Actual DB ID '{actual_db_id}' for sample {i+1} not found in loaded relevant schemas! Skipping.")
                 analysis_errors += 1
                 pipeline_status = "Analysis Failed (Actual Schema Missing)"
                 # Append error result and continue
                 results_summary.append({ ... }) # Populate with error details
                 continue

            # 4. Add random distractors up to the desired context size
            num_distractors_needed = num_schema_subset - 1
            context_schemas_for_llm = { 
                 actual_db_id: relevant_schemas[actual_db_id]
            }
            if num_distractors_needed > 0:
                 other_db_ids = [db for db in all_available_db_ids if db != actual_db_id]
                 # Select distractors, making sure not to request more than available
                 num_distractors_to_select = min(num_distractors_needed, len(other_db_ids))
                 if num_distractors_to_select > 0:
                     distractor_ids = random.sample(other_db_ids, num_distractors_to_select)
                     for db_id in distractor_ids:
                          if db_id in relevant_schemas: # Should always be true
                               context_schemas_for_llm[db_id] = relevant_schemas[db_id]
                 logging.debug(f"Added {len(context_schemas_for_llm)-1} distractor schemas for LLM context.")

            logging.info(f"LLM context schema size: {len(context_schemas_for_llm)} (Target size: {num_schema_subset})")
            logging.debug(f"Context DB IDs: {list(context_schemas_for_llm.keys())}")
            # --- End Schema Context Prep ---

            # --- 1. Run Analysis & Generation (conn=None) ---
            try:
                pipeline_time = None
                gen_sql_exec_time = None
                gold_sql_exec_time = None
                start_time = time.time()
                pipeline_result = process_natural_language_query(
                    original_query=example['question'],
                    conn=None, # <--- Pass None here
                    schema=context_schemas_for_llm, # Use the subset schemas
                    llm_wrapper=llm_wrapper,
                    aux_models=aux_models,
                    qdrant_client=qdrant_client,
                    conversation_history=[]
                    # max_retries not needed here
                )
                pipeline_time = time.time() - start_time
                logging.info(f"Pipeline analysis/generation finished in {pipeline_time:.2f}s")

                # --- 2. Process Analysis Results ---
                analysis_status = pipeline_result.get("status", "error_unknown")
                pipeline_error_msg = pipeline_result.get("message")
                determined_route = pipeline_result.get("determined_route")
                selected_db_id = pipeline_result.get("selected_db_id")
                generated_sql = pipeline_result.get("generated_sql") # Important: get the generated SQL string

                analysis_succeeded = False # Flag if analysis phase was ok for SQL route
                # Check for correct DB selection first
                if selected_db_id != actual_db_id:
                     # Even if the pipeline didn't report an "error", selecting the wrong DB is a failure for this benchmark sample
                     analysis_errors += 1
                     pipeline_status = f"Analysis Failed (Incorrect DB Selected: LLM chose '{selected_db_id}' for actual '{actual_db_id}')"
                     logging.error(pipeline_status)
                     # Ensure pipeline_error_msg reflects this core issue if not already set
                     if not pipeline_error_msg or not pipeline_error_msg.startswith("Could not determine"):
                          pipeline_error_msg = f"Incorrect DB Selected: Got '{selected_db_id}', Expected '{actual_db_id}'"
                # Now check other failure conditions if DB selection was correct (or if selected_db_id was None initially)
                elif analysis_status.startswith("error"):
                    analysis_errors += 1
                    pipeline_status = f"Analysis Failed ({analysis_status})"
                    logging.error(f"Pipeline analysis/generation failed: {pipeline_error_msg}")
                elif determined_route != "SQL":
                    pipeline_status = f"Analysis Success (Route: {determined_route})"
                    logging.info(f"Route determined as '{determined_route}'. Skipping SQL benchmark.")
                elif not generated_sql or generated_sql.startswith("-- Error"):
                    analysis_errors += 1 # Count SQL gen failure as analysis error
                    pipeline_status = "SQL Generation Failed"
                    pipeline_error_msg = generated_sql if generated_sql else "No SQL generated"
                    logging.error(f"SQL Generation Failed: {pipeline_error_msg}")
                else:
                    # --- Analysis, DB Selection, & SQL Generation Successful ---
                    analysis_succeeded = True
                    pipeline_analysis_success_sql += 1
                    pipeline_status = "Analysis & Gen Success (SQL)"
                    logging.info(f"Analysis successful. Route: SQL. Selected DB: {selected_db_id}. SQL: {generated_sql}")

                # --- 3. Execute and Compare (only if analysis succeeded) ---
                if analysis_succeeded:
                    # Execute the generated SQL on the actual DB
                    
                    try:
                        with sqlite3.connect(temp_db_path, timeout=10) as temp_conn:
                            # Execute Generated SQL
                            start_gen_exec = time.time()
                            gen_results, gen_sql_exec_error, sql_exec_duration = robust_execute_sql(temp_conn, generated_sql)
                            gen_sql_exec_time = time.time() - start_gen_exec
                            if gen_sql_exec_error:
                                pipeline_status = "Generated SQL Execution Failed"
                                logging.warning(f"Generated SQL execution failed: {gen_sql_exec_error}")
                            else:
                                exec_success_flag = True
                                exec_success += 1
                                pipeline_status = "Generated SQL Executed"
                                logging.debug("Generated SQL executed successfully.")

                                # Execute Gold SQL
                                start_gold_exec = time.time()
                                gold_results, gold_sql_exec_error, sql_exec_duration = robust_execute_sql(temp_conn, example['query'])
                                gold_sql_exec_time = time.time() - start_gold_exec
                                if gold_sql_exec_error:
                                    pipeline_status = "Gold SQL Execution Failed"
                                    logging.warning(f"Gold SQL failed execution! Error: {gold_sql_exec_error}")
                                else:
                                    pipeline_status = "Both Executed"
                                    logging.debug("Gold SQL executed successfully.")
                                    # Compare Results
                                    match_success_flag = compare_results(gold_results, gen_results)
                                    if match_success_flag:
                                        match_success += 1
                                        pipeline_status = "Match Success"
                                        logging.info("Result Match: SUCCESS")
                                    else:
                                        pipeline_status = "Match Failed"
                                        logging.warning("Result Match: FAILED")
                    except Exception as exec_err:
                            pipeline_status = f"Execution Loop Error: {type(exec_err).__name__}"
                            pipeline_error_msg = str(exec_err)
                            logging.error(f"Error during execution phase for DB '{actual_db_id}': {exec_err}", exc_info=True)


            except Exception as outer_err:
                 # Catch errors during the main pipeline call itself
                 analysis_errors += 1 # Count as analysis failure
                 pipeline_status = f"Outer Loop Error: {type(outer_err).__name__}"
                 pipeline_error_msg = str(outer_err)
                 logging.error(f"Error processing example analysis {example.get('original_index', i)+1}: {outer_err}", exc_info=True)

            # Store results
            results_summary.append({
                "original_index": example.get('original_index', 'N/A'),
                "actual_db_id": actual_db_id, # Store the true DB ID for the question
                "question": example['question'], "gold_sql": example['query'],
                "pipeline_status": pipeline_status, "pipeline_error": pipeline_error_msg,
                "determined_route": determined_route,
                "selected_db_id_by_llm": selected_db_id, # Store the ID selected by the LLM
                "generated_sql": generated_sql, "generated_sql_exec_error": gen_sql_exec_error,
                "gold_sql_exec_error": gold_sql_exec_error, "exec_success": exec_success_flag,
                "match_success": match_success_flag
            })

             # --- Prepare result dictionary for this sample ---
            sample_result_dict = {
                "original_index": example.get('original_index', 'N/A'),
                "actual_db_id": actual_db_id,
                "question": example['question'],
                "gold_sql": example['query'],
                "pipeline_status": pipeline_status,
                "pipeline_error": pipeline_error_msg,
                "determined_route": determined_route,
                "selected_db_id_by_llm": selected_db_id,
                "generated_sql": generated_sql,
                "generated_sql_exec_error": gen_sql_exec_error,
                "gold_sql_exec_error": gold_sql_exec_error,
                "exec_success": exec_success_flag,
                "match_success": match_success_flag,
                "sample_data": sample_data_str_actual,
                "pipeline_time": pipeline_time,
                "generated_sql_exec_time": gen_sql_exec_time,
                "gold_sql_exec_time": gold_sql_exec_time
            }


            # --- Append result to file ---
            append_result_to_jsonl(sample_result_dict, results_filepath)
            # ---
            # --- End Example Loop ---

    # --- Calculate Metrics ---
    total_processed = count
    # Total where analysis didn't error AND route was SQL AND SQL gen didn't error AND selected DB matched actual DB
    total_candidates_for_execution = pipeline_analysis_success_sql
    total_attempted_execution = total_candidates_for_execution - skipped_db

    if total_processed == 0: pipe_an_acc = 0.0; exec_acc = 0.0; match_acc = 0.0
    else:
        # Analysis Accuracy: % of processed where analysis didn't error, route=SQL, gen SQL ok, selected DB correct
        pipe_an_acc = (pipeline_analysis_success_sql / total_processed) * 100
        # Execution Accuracy: % of successful analysis runs where generated SQL executed
        exec_acc = (exec_success / pipeline_analysis_success_sql) * 100 if pipeline_analysis_success_sql > 0 else 0.0
        # Match Accuracy: % of successfully executed generated SQLs that matched gold
        match_acc = (match_success / exec_success) * 100 if exec_success > 0 else 0.0

    logging.info("\n--- Randomized Benchmark Summary (Schema Subset) ---")
    logging.info(f"Schema Subset Size for Selection Context: {len(target_schemas_subset)}")
    logging.info(f"Total examples processed: {total_processed}")
    logging.info(f"Examples skipped (DB setup failed): {skipped_db}")
    logging.info(f"Pipeline Analysis/Selection/Generation Errors: {analysis_errors}")
    logging.info(f"Pipeline Analysis Successful (Correct DB, SQL Route & Gen OK): {pipeline_analysis_success_sql}")
    logging.info(f"Successfully executed generated SQL: {exec_success}")
    logging.info(f"Generated SQL matched gold results: {match_success}")
    logging.info(f"Analysis & Generation Accuracy (vs Processed): {pipe_an_acc:.2f}%")
    logging.info(f"Execution Accuracy (vs Pipeline Success): {exec_acc:.2f}%")
    logging.info(f"Exact Set Match Accuracy (vs Exec Success): {match_acc:.2f}%")


    return results_summary, pipe_an_acc, exec_acc, match_acc


# --- Main Execution ---
def main():
    # ... (argparse setup remains the same - needs test_suite_file, gold_sql_file, database_dir, tables_json_file) ...
    parser = argparse.ArgumentParser(description="Benchmark NL-to-SQL FULL PIPELINE on random samples.")
    parser.add_argument("test_suite_file", help="Path to the test suite JSON file (e.g., test.json or dev.json).")
    parser.add_argument("gold_sql_file", help="Path to the corresponding gold SQL file (e.g., test_gold.sql or dev_gold.sql).")
    parser.add_argument("database_dir", help="Path to the directory containing database folders (e.g., 'test_database/' or 'database/').")
    parser.add_argument("tables_json_file", help="Path to the tables.json file containing schemas for all databases.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of random examples to sample from the dataset.")
    parser.add_argument("--num_schemas", type=int, default=20, help="Number of random schemas to provide as context for DB selection.") # <-- New Arg
    parser.add_argument("--results_out", default="benchmark_randomized_results.json", help="Path to save detailed benchmark results.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())
    logging.info(f"Logging level set to {args.log_level.upper()}")

    logging.info("Initializing LLM Wrapper, Auxiliary Models, and Qdrant...")
    llm_wrapper = get_llm_wrapper()
    aux_models = get_cached_aux_models()
    qdrant_client = init_qdrant_client()
    logging.info("Initialization complete.")

    questions_data = load_test_suite_questions(args.test_suite_file)
    gold_queries = load_test_suite_gold_sql(args.gold_sql_file)
    test_schemas  = load_schemas_from_tables_json(args.tables_json_file)
    if not questions_data or not gold_queries or not test_schemas:
         logging.error("Failed to load test questions, gold queries, or test schemas. Exiting.")
         exit(1)
    full_dataset = combine_test_suite_data(questions_data, gold_queries)
    if not full_dataset: exit(1)

    # --- Run Randomized Benchmark ---
    results, pipe_acc, exec_acc, match_acc = run_randomized_benchmark(
        full_dataset=full_dataset,
        num_samples=args.num_samples,
        db_base_dir=args.database_dir, # Pass the specific dir for test DBs
        relevant_schemas=test_schemas, # Pass test schemas for selection stage
        llm_wrapper=llm_wrapper,
        aux_models=aux_models,
        qdrant_client=qdrant_client,
        num_schema_subset=args.num_schemas # <-- Pass the subset size
    )

    # ... (Save results) ...
    if results:
        try:
            with open(args.results_out, 'w') as f: json.dump(results, f, indent=2)
            logging.info(f"Detailed results saved to {args.results_out}")
        except Exception as e: logging.error(f"Failed to save results: {e}")

    logging.info("Randomized benchmarking finished.")

if __name__ == "__main__":
    main()