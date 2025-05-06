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
try:
    from utils import (
    setup_environment, get_db_connection, init_qdrant_client, get_llm_wrapper,
    get_cached_aux_models, get_schema_info, _generate_sql_query, # Use the core generator
    # We need a way to execute SQL and get results robustly for comparison
    )

except ImportError as e:
    print(f"Error importing from utils: {e}")
    print("Please ensure benchmark.py is in the correct directory relative to utils.py")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# --- Helper Functions ---

def load_test_suite_questions(filepath: str) -> List[Dict[str, Any]]:
    """Loads the questions and db_ids from the test suite JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} questions/db_ids from {filepath}")
            # Add an index to preserve order for matching with gold SQL
            for i, item in enumerate(data):
                item['original_index'] = i
        return data
    except FileNotFoundError:
        logging.error(f"Test suite questions file not found: {filepath}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from file: {filepath}")
        return []

def load_test_suite_gold_sql(filepath: str) -> List[str]:
    """Loads the gold SQL queries from the .sql file (one per line)."""
    try:
        with open(filepath, 'r') as f:
            # Read lines, strip whitespace, and filter empty lines
            gold_queries = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(gold_queries)} gold SQL queries from {filepath}")
        return gold_queries
    except FileNotFoundError:
        logging.error(f"Gold SQL file not found: {filepath}")   
    return []



def combine_test_suite_data(questions_data: List[Dict[str, Any]], gold_queries: List[str]) -> List[Dict[str, Any]]:
    """Combines question data with gold queries based on order."""
    if len(questions_data) != len(gold_queries):
        logging.error(f"Mismatch questions ({len(questions_data)}) vs gold queries ({len(gold_queries)}).")
        return []

    combined_data = []
    for i, question_item in enumerate(questions_data):
        gold_sql = gold_queries[i]
        actual_gold_sql = gold_sql.split('\t', 1)[0].strip() if '\t' in gold_sql else gold_sql
        question_item['query'] = actual_gold_sql
        combined_data.append(question_item)
    logging.info(f"Successfully combined {len(combined_data)} test examples.")
    return combined_data


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Loads the NL-to-SQL dataset from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} examples from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {filepath}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from file: {filepath}")
        return []

def create_db_from_sql_script(script_path: str, db_path: str) -> bool:
    """Creates and populates an SQLite database from a .sql script."""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript(sql_script)
            conn.commit()
        logging.debug(f"Executed script '{script_path}' on db '{db_path}'.")
        return True
    except sqlite3.Error as e:
        logging.error(f"SQLite error executing script '{script_path}' on db '{db_path}': {e}")
        if os.path.exists(db_path): os.remove(db_path)
        return False
    except FileNotFoundError:
        logging.error(f"SQL script file not found: '{script_path}'")
        return False
    except Exception as e:
        logging.error(f"Unexpected error creating db from script '{script_path}': {e}")
        if os.path.exists(db_path): os.remove(db_path)
        return False

def setup_temp_database(db_id: str, db_base_dir: str, temp_dir: str) -> Optional[str]:
    """
    Sets up a temporary database specifically by creating it from schema.sql
    and potentially data.sql found within the db_base_dir/db_id/ directory.
    Args:
        db_id: The database identifier (e.g., 'e_commerce').
        db_base_dir: The path to the directory containing db_id subfolders
                     (e.g., path/to/test_database).
        temp_dir: The path to the temporary directory for the new .sqlite file.
    Returns:
        The path to the created temporary database file or None on failure.
    """
    # Construct relative paths first
    db_subdir_rel = os.path.join(db_base_dir, db_id)
    schema_sql_path_rel = os.path.join(db_subdir_rel, "schema.sql")
    data_sql_path_rel = os.path.join(db_subdir_rel, f"{db_id}.sql") # Relative path for data script
    temp_db_path = os.path.join(temp_dir, f"{db_id}.sqlite") # Temp path is fine relative to temp_dir

    # --- Get and Log Absolute Paths for Checks ---
    # os.path.abspath resolves paths relative to the Current Working Directory (CWD)
    cwd = os.getcwd()
    abs_db_subdir = os.path.abspath(db_subdir_rel)
    abs_schema_sql_path = os.path.abspath(schema_sql_path_rel)
    abs_data_sql_path = os.path.abspath(data_sql_path_rel) # Absolute path for data script check

    logging.debug(f"Current Working Directory: {cwd}")
    logging.debug(f"Checking for schema.sql at absolute path: {abs_schema_sql_path}")
    logging.debug(f"Base DB subdirectory absolute path: {abs_db_subdir}")
    # --- End Debug Prints ---

    # Ensure the temporary directory structure exists
    try:
        # Use absolute path for temp dir base? Usually fine relative.
        os.makedirs(os.path.dirname(temp_db_path), exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create temporary directory '{os.path.dirname(temp_db_path)}' for DB '{db_id}': {e}")
        return None

    # --- Check existence using the ABSOLUTE path ---
    schema_exists = os.path.exists(abs_schema_sql_path)
    logging.debug(f"os.path.exists result for schema ({abs_schema_sql_path}): {schema_exists}")
    # ---

    if schema_exists:
        logging.info(f"Found schema.sql for '{db_id}'. Creating temporary database...")

        # Remove existing temp file first (use relative/temp path)
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
                logging.debug(f"Removed existing temp db file: {temp_db_path}")
            except OSError as e:
                 logging.error(f"Failed to remove existing temp db file '{temp_db_path}': {e}")
                 return None # Abort if we can't clear the old file


        # Create DB from schema (use relative path as it worked before for reading)
        if create_db_from_sql_script(schema_sql_path_rel, temp_db_path):
            # Check for and execute data population script (use absolute path for exists check)
            data_script_exists = os.path.exists(abs_data_sql_path)
            logging.debug(f"os.path.exists result for data script ({abs_data_sql_path}): {data_script_exists}")

            if data_script_exists:
                logging.info(f"Found data script '{data_sql_path_rel}'. Populating data...")
                # Use relative path for reading the script
                if not create_db_from_sql_script(data_sql_path_rel, temp_db_path):
                     logging.error(f"Failed to populate data from '{data_sql_path_rel}'. Database may be empty.")
                     # Decide if an empty DB is acceptable or should be an error
                     # return None # Uncomment if data population failure should skip the example
            else:
                logging.warning(f"Data population script ('{os.path.basename(data_sql_path_rel)}') not found in '{db_subdir_rel}'. Database will only have schema.")

            logging.debug(f"Successfully prepared temp db: {temp_db_path}")
            return temp_db_path # Return path
        else:
            logging.error(f"Failed to create temporary database for '{db_id}' from schema script '{schema_sql_path_rel}'.")
            return None
    else:
        # If schema.sql doesn't exist using absolute path
        logging.error(f"Cannot set up database for '{db_id}'. Schema script '{abs_schema_sql_path}' not found.")
        # Add check for directory existence using absolute path
        if not os.path.isdir(abs_db_subdir):
             logging.error(f"Underlying directory also not found: {abs_db_subdir}")
        return None

def robust_execute_sql(conn: sqlite3.Connection, query: str) -> Tuple[Optional[Set[Tuple]], Optional[str], float]:
    """
    Executes SQL, returns results set or error message, and execution time in seconds.
    """
    results_set: Optional[Set[Tuple]] = None
    error_message: Optional[str] = None
    start_time = 0.0
    end_time = 0.0
    duration = 0.0

    try:
        query = query.strip().rstrip(';')
        if not query:
            return set(), None, 0.0 # Empty query takes no time

        start_time = time.perf_counter() # Start timer just before execution

        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        end_time = time.perf_counter() # Stop timer immediately after execution
        duration = end_time - start_time

        # Process results after timing
        results_set = set(tuple(map(str, row)) for row in rows)

    except sqlite3.Error as db_err:
        end_time = time.perf_counter() # Still record time if error occurs during/after exec
        duration = end_time - start_time if start_time > 0 else 0.0
        error_message = f"SQLite Execution Error: {db_err}"
        logging.warning(f"Query failed ({duration:.4f}s): {query} | Error: {error_message}")
    except Exception as e:
        end_time = time.perf_counter() # Record time on general error too
        duration = end_time - start_time if start_time > 0 else 0.0
        error_message = f"General Execution Error: {type(e).__name__}: {e}"
        logging.error(f"Unexpected error executing query ({duration:.4f}s): {query} | Error: {error_message}", exc_info=False)
    # No finally needed as duration is updated in except blocks

    if error_message is None:
        logging.debug(f"Query executed successfully ({duration:.4f}s)")

    return results_set, error_message, duration # Return duration

def compare_results(gold_results: Optional[Set[Tuple]], gen_results: Optional[Set[Tuple]]) -> bool:
    """
    Compares two sets of results (handles None cases).
    Returns True if they match exactly (order-insensitive).
    """
    if gold_results is None or gen_results is None:
        # Cannot match if one or both failed execution or returned None
        return False
    # Strict equality check between the two sets
    return gold_results == gen_results
    
def run_benchmark(dataset: List[Dict[str, Any]], db_dir: str, llm_wrapper, aux_models, limit: Optional[int] = None):
    """
    Runs the benchmark evaluation loop.
    """
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.error("LLM Wrapper is not ready. Aborting benchmark.")
        return None
    if not aux_models or aux_models.get('status') != 'loaded' or not aux_models.get('sql_gguf_model'):
        logging.error("Auxiliary models (specifically SQL GGUF) not ready. Aborting benchmark.")
        return None
    results_summary = []
    count = 0
    success_exec = 0
    success_match = 0
    skipped_db = 0

    # Create a temporary directory for isolated DB copies
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Using temporary directory for databases: {temp_dir}")

        for i, example in enumerate(dataset):
            if limit is not None and count >= limit:
                logging.info(f"Reached benchmark limit of {limit} examples.")
                break

            count += 1
            logging.info(f"\n--- Processing Example {i+1}/{len(dataset)} (DB: {example['db_id']}) ---")
            logging.info(f"Question: {example['question']}")
            logging.info(f"Gold SQL: {example['query']}")

            # 1. Setup Temporary Database
            temp_db_path = setup_temp_database(example['db_id'], db_dir, temp_dir)
            
            
            if not temp_db_path:
                logging.warning(f"Skipping example {i+1} due to missing/failed database setup for '{example['db_id']}'.")
                skipped_db += 1
                results_summary.append({
                    "original_index": example.get('original_index', i), "db_id": example['db_id'], "question": example['question'],
                    "gold_sql": example['query'], "generated_sql": None,
                    "status": "DB Setup Failed", "exec_success": False, "match_success": False
                })
                continue

            # 2. Connect to Temp DB & Get Schema
            gen_sql = None
            gen_results = None
            gen_error = None
            gold_results = None
            gold_error = None
            exec_success = False
            match_success = False
            status = "Processing"
            schema = None # Initialize schema
            match_details = None # Store detailed diff if match fails

            try:
                with sqlite3.connect(temp_db_path, timeout=10) as temp_conn: 
                    try:
                        schema = get_schema_info(temp_conn)
                        if not schema:
                            raise ValueError("Failed to retrieve schema from temporary database.")
                        # Get sample data (optional, but _generate_sql_query might use it)
                        first_table = next(iter(schema), None)
                        sample_data_str = ""
                        if first_table:
                            try:
                                sample_df_pd = pd.read_sql_query(f'SELECT * FROM "{first_table}" LIMIT 3', temp_conn)
                                sample_data_str = sample_df_pd.to_markdown(index=False)
                            except Exception as pd_err: logging.warning(f"Could not get sample data: {pd_err}")
                            
                        
                            

                    except Exception as schema_err:
                        logging.error(f"Failed to get schema for temp DB '{temp_db_path}': {schema_err}")
                        status = "Schema Error"
                        # Add to summary and continue to next example
                        results_summary.append({
                            "original_index": example.get('original_index', i), "db_id": example['db_id'], "question": example['question'],
                            "gold_sql": example['query'], "generated_sql": None,
                            "status": status, "exec_success": False, "match_success": False
                        })
                        continue # Skip to next example if schema fails
                    # Ensure schema is available before proceeding
                    if not schema:
                        logging.error("Schema is unexpectedly None after try block. Skipping generation.")
                        status="Schema Error Post Check"
                        # Append result and continue... (similar to above)
                        continue

                    # 3. Generate SQL using the system
                    start_time = time.time()
                    # Ensure aux_models is passed correctly
                    if not isinstance(aux_models, dict): raise TypeError("aux_models is not dict")
                    gen_sql = _generate_sql_query(
                        refined_query_input=example['question'],
                        schema=schema,
                        sample_data_str=sample_data_str,
                        aux_models=aux_models, # Pass the whole dict
                        # No feedback in benchmark mode initially
                        previous_sql=None,
                        feedback=None
                    )
                    gen_time = time.time() - start_time
                    logging.info(f"Generated SQL ({gen_time:.2f}s): {gen_sql}")

                    if gen_sql.startswith("-- Error"):
                        gen_error = gen_sql # Store the generation error message
                        logging.warning("SQL generation failed.")
                        status = "Generation Failed"
                    else:
                        # 4. Execute Generated SQL
                        logging.debug("Executing Generated SQL...")
                        gen_results, gen_error, sql_exec_duration = robust_execute_sql(temp_conn, gen_sql)
                        if gen_error:
                            status = "Generated SQL Execution Failed"
                        else:
                            exec_success = True
                            success_exec += 1 # Count successful executions
                            status = "Generated SQL Executed"
                            logging.debug("Generated SQL executed successfully.")

                            # 5. Execute Gold SQL (only if generated SQL executed)
                            logging.debug("Executing Gold SQL...")
                            gold_results, gold_error, sql_exec_duration = robust_execute_sql(temp_conn, example['query'])
                            if gold_error:
                                status = "Gold SQL Execution Failed"
                                logging.warning(f"Gold SQL failed execution! Error: {gold_error}")
                                # Note: We might still count exec_success for generated SQL
                            else:
                                status = "Both Executed"
                                logging.debug("Gold SQL executed successfully.")

                                # 6. Compare Results
                                match_success_bool = compare_results(gold_results, gen_results) # Get True/False
                                if match_success_bool:
                                    success_match += 1 # Increment counter ONLY if True
                                    status = "Match Success"
                                    logging.info("Result Match: SUCCESS")
                                else:
                                    status = "Match Failed"
                                    logging.warning("Result Match: FAILED")
                                    # Store difference details if comparison was possible
                                    if gold_results is not None and gen_results is not None:
                                         missing_in_gen = gold_results - gen_results
                                         extra_in_gen = gen_results - gold_results
                                         # Limit size of logged diffs
                                         max_diff_items = 5
                                         missing_str = str(list(missing_in_gen)[:max_diff_items]) + ('...' if len(missing_in_gen) > max_diff_items else '')
                                         extra_str = str(list(extra_in_gen)[:max_diff_items]) + ('...' if len(extra_in_gen) > max_diff_items else '')
                                         match_details = f"Results differ. Missing in generated: {missing_str}, Extra in generated: {extra_str}"
                                         logging.debug(match_details) # Log details here
                                    else:
                                         match_details = "Comparison not possible (one query failed execution)."
                                


            except Exception as outer_err:
                status = f"Outer Loop Error: {type(outer_err).__name__}"
                logging.error(f"Error during processing example {i+1}: {outer_err}", exc_info=True)


            # Store results
            results_summary.append({
                "original_index": example.get('original_index', i),
                "db_id": example['db_id'],
                "question": example['question'],
                "gold_sql": example['query'],
                "generated_sql": gen_sql,
                "status": status,
                "generation_error": gen_error if status == "Generation Failed" else None,
                "generated_sql_exec_error": gen_error if status == "Generated SQL Execution Failed" else None,
                "gold_sql_exec_error": gold_error,
                "exec_success": exec_success, # Based on generated query execution
                "match_success": match_success_bool, # Only if both executed successfully
                "match_details": match_details if not match_success_bool and match_details else None # Store diff only on failure
            })

    # --- Calculate Metrics ---
    total_attempted = count - skipped_db
    if total_attempted == 0:
        logging.warning("No examples were successfully attempted (DB setup failed for all?).")
        exec_accuracy = 0.0
        match_accuracy = 0.0
    else:
        exec_accuracy = (success_exec / total_attempted) * 100 if total_attempted > 0 else 0
        # Match accuracy is calculated only on those queries that *executed successfully*
        match_accuracy = (success_match / success_exec) * 100 if success_exec > 0 else 0

    logging.info("\n--- Benchmark Summary ---")
    logging.info(f"Total examples in dataset: {len(dataset)}")
    logging.info(f"Examples attempted (limit): {count}")
    logging.info(f"Examples skipped (DB setup failed): {skipped_db}")
    logging.info(f"Total examples evaluated: {total_attempted}")
    logging.info(f"Successfully executed generated SQL: {success_exec}")
    logging.info(f"Successfully matched gold results: {success_match}")
    logging.info(f"Execution Accuracy (Exec Success / Total Evaluated): {exec_accuracy:.2f}%")
    logging.info(f"Exact Set Match Accuracy (Match Success / Exec Success): {match_accuracy:.2f}%")

    return results_summary, exec_accuracy, match_accuracy


def main():
    parser = argparse.ArgumentParser(description="Benchmark NL-to-SQL generation using Spider Test Suite format.")
    # Arguments specific to Test Suite
    parser.add_argument("test_suite_file", help="Path to the test suite JSON file (e.g., test.json).")
    parser.add_argument("gold_sql_file", help="Path to the corresponding gold SQL file (e.g., test_gold.sql).")
    parser.add_argument("test_database_dir", help="Path to the directory containing TEST database folders (e.g., 'test_database/'). Should contain db_id subfolders with schema.sql.")
    # General arguments
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to process.")
    parser.add_argument("--results_out", default="benchmark_test_suite_results.json", help="Path to save detailed benchmark results.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")

    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())
    logging.info(f"Logging level set to {args.log_level.upper()}")

    logging.info("Initializing LLM Wrapper and Auxiliary Models...")
    llm_wrapper = get_llm_wrapper()
    aux_models = get_cached_aux_models()
    logging.info("Initialization complete.")

    questions_data = load_test_suite_questions(args.test_suite_file)
    gold_queries = load_test_suite_gold_sql(args.gold_sql_file)
    if not questions_data or not gold_queries: exit(1)

    dataset = combine_test_suite_data(questions_data, gold_queries)
    if not dataset: exit(1)

    # --- Pass the specific test_database_dir to run_benchmark ---
    results, exec_acc, match_acc = run_benchmark(
        dataset=dataset,
        db_dir=args.test_database_dir, # Use the test database directory path
        llm_wrapper=llm_wrapper,
        aux_models=aux_models,
        limit=args.limit
    )
    # ---

    if results:
        try:
            output_results = [{k: v for k, v in item.items()} for item in results]
            with open(args.results_out, 'w') as f:
                json.dump(output_results, f, indent=2)
            logging.info(f"Detailed results saved to {args.results_out}")
        except Exception as e:
            logging.error(f"Failed to save results to {args.results_out}: {e}")

    logging.info("Benchmarking finished.")

if __name__ == "__main__":
    main()