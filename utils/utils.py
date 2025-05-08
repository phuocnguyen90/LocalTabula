# utils.py
import streamlit as st
import pandas as pd
import sqlite3
from qdrant_client import QdrantClient, models
import time
import os
from llama_cpp import Llama
import uuid
import json
import dotenv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
import re
from dotenv import load_dotenv
import yaml # YAML for prompts
import logging
import traceback # For detailed error logging in query processing
from collections import Counter # For detecting duplicate columns
import shutil


# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dotenv.load_dotenv('.env', override=True)
SQLITE_TIMEOUT_SECONDS = 15
QDRANT_COLLECTION_PREFIX = "table_data_" # Using prefix convention
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
PROMPT_PATH = CONFIG_DIR / "prompts.yaml"

ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(dotenv_path=str(ENV_PATH))  
MAX_SCHEMA_CHARS_ESTIMATE = 20000 # Rough estimate for prompt length control

# --- Load Prompts ---
def load_prompts():
    """
    Returns a dict (or list) from your prompt YAML file.
    """
    prompts = yaml.safe_load(PROMPT_PATH.open())
    return prompts
    

PROMPTS = load_prompts() # Load prompts globally for the module

if PROMPTS is None:
    logging.fatal("Prompts failed to load. Stopping application.")
    st.stop() # Halt execution if prompts failed to load
else:
    logging.info('Prompts loaded successfully.')
    logging.info(f"Loaded prompts keys: {list(PROMPTS.keys())}")


# --- Environment Setup ---
def setup_environment():
    """
    Determines DB path based on DEVELOPMENT_MODE env variable.
    Creates necessary directories. Returns absolute database path or None.
    """
    
    dev_mode_str = os.getenv('DEVELOPMENT_MODE', 'false').lower()
    is_dev_mode = dev_mode_str in ('true', '1', 't', 'yes', 'y')
    db_filename = "chat_database.db"

    try:
        if is_dev_mode:
            home_dir = os.path.expanduser("~")
            app_data_dir = os.path.join(home_dir, ".streamlit_chat_dev_data")
            logging.info("Running in DEVELOPMENT mode.")
        else:
            current_working_dir = os.getcwd()
            app_data_dir = os.path.join(current_working_dir, "app_data")
            logging.info("Running in PRODUCTION mode.")

        logging.info(f"App data directory set to: {app_data_dir}")
        os.makedirs(app_data_dir, exist_ok=True)
        logging.info(f"Ensured app data directory exists.")
        db_path = os.path.join(app_data_dir, db_filename)
        logging.info(f"Database path set to: {db_path}")
        return db_path

    except OSError as e:
        logging.error(f"Failed to create data directory '{app_data_dir}': {e}. Check permissions.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during environment setup: {e}", exc_info=True)
        return None

# --- Imports from other local modules ---
# (Keep these as they were in your provided file)
from llm_interface import LLMWrapper
from model import load_aux_models

# --- Cached Resource Initializers ---
@st.cache_resource
def get_llm_wrapper():
    """Initializes and caches the LLMWrapper."""
    logging.info("Attempting to get/initialize LLMWrapper...")
    wrapper = LLMWrapper()
    if not wrapper.is_ready:
        logging.error("LLMWrapper initialization failed.")
    return wrapper

@st.cache_resource
def get_cached_aux_models():
    """Loads and caches auxiliary models (SQL, Embedder)."""
    logging.info("Cache miss or first run: Calling load_aux_models...")
    models_dict_from_loader = load_aux_models()
    if not isinstance(models_dict_from_loader, dict):
        logging.error(f"CRITICAL: load_aux_models returned type {type(models_dict_from_loader)} instead of dict!")
        return {"status": "error", "error_message": "Internal Error: Model loader returned invalid type."}
    if models_dict_from_loader.get('status') != 'loaded':
         logging.warning(f"Aux model loading status: {models_dict_from_loader.get('status')}. Error: {models_dict_from_loader.get('error_message', 'N/A')}")
    else:
        logging.info("Aux models loaded and cached successfully.")
    return models_dict_from_loader

@st.cache_resource
def get_db_connection(db_path):
    """Establishes and caches SQLite connection."""
    logging.info(f"Attempting DB connection to: {db_path}")
    if not db_path:
         logging.error("Invalid database path provided to get_db_connection.")
         return None
    try:
        conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_SECONDS, check_same_thread=False)
        try: conn.execute("PRAGMA journal_mode=WAL;") ; logging.info("WAL mode enabled.")
        except Exception as wal_e: logging.warning(f"Could not enable WAL mode: {wal_e}")
        logging.info("DB Connection Successful.")
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to DB at {db_path}: {e}", exc_info=True)
        return None

QDRANT_FILENAME = "qdrant_storage"

@st.cache_resource
def init_qdrant_client() -> QdrantClient | None:
    """Initializes and caches a Qdrant client (persistent fallback to memory)."""
    # 1) Decide base directory
    dev = os.getenv('DEVELOPMENT_MODE', 'false').lower() in ('true','1','t','yes','y')
    root = os.path.expanduser("~") if dev else os.getcwd()
    storage_dir = os.path.join(root, "app_data", QDRANT_FILENAME) if not dev \
                  else os.path.join(root, f".streamlit_chat_dev_data/{QDRANT_FILENAME}")
    os.makedirs(storage_dir, exist_ok=True)

    def _try_persistent(path: str):
        client = QdrantClient(path=path)
        # sanity check
        client.get_collections()
        return client

    # 2) Try persistent, catch stale-lock error, retry once
    try:
        logging.info(f"Attempting persistent Qdrant client at {storage_dir}")
        return _try_persistent(storage_dir)
    except RuntimeError as e:
        msg = str(e).lower()
        if "already accessed by another instance" in msg:
            logging.warning(f"Stale lock detected at {storage_dir}, wiping directory and retrying...")
            try:
                shutil.rmtree(storage_dir)
                os.makedirs(storage_dir, exist_ok=True)
                time.sleep(0.2)
                client = _try_persistent(storage_dir)
                logging.info(f"Persistent Qdrant re-initialized at {storage_dir}")
                return client
            except Exception as e2:
                logging.error(f"Retry persistent init failed: {e2}", exc_info=True)
        else:
            logging.error(f"Persistent init failed: {e}", exc_info=True)

    # 3) Fallback to in-memory
    try:
        logging.warning("Falling back to in-memory Qdrant client.")
        client = QdrantClient(":memory:")
        client.get_collections()
        logging.info("In-memory Qdrant client initialized.")
        return client
    except Exception as e:
        logging.fatal(f"Could not initialize in-memory Qdrant: {e}", exc_info=True)
        return None


# --- Data Reading / Writing ---
def read_google_sheet(sheet_url):
    """Reads data from a published Google Sheet URL. Returns a DataFrame."""
    # (Implementation remains the same as provided)
    if not sheet_url.startswith("https://docs.google.com/spreadsheets/d/"):
        raise ValueError("Invalid Google Sheet URL. Ensure it's a published link.")
    sheet_id = sheet_url.split("/d/")[1].split("/")[0]
    url= f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    try: df = pd.read_csv(url)
    except Exception as e: raise ValueError(f"Error reading Google Sheet: {e}")
    if df.empty: raise ValueError("The Google Sheet is empty or not accessible.")
    return df

def table_exists(conn, table_name):
    """Checks if a table exists in the SQLite database."""
    if not conn or not table_name: return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        exists = cursor.fetchone() is not None
        cursor.close(); return exists
    except Exception: return False

def get_sqlite_table_row_count(conn: sqlite3.Connection, table_name: str) -> int | None:
    """Gets the row count for a specific SQLite table."""
    if not conn or not table_exists(conn, table_name): return None
    try:
        cursor = conn.cursor(); cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"') # Quote table name
        count = cursor.fetchone()[0]; cursor.close(); return count
    except Exception as e:
        logging.error(f"Error getting row count for table '{table_name}': {e}", exc_info=True)
        return None

# --- Schema and Sample Data Helpers ---

def get_schema_info(conn) -> Dict[str, List[str]]:
    """
    Fetches table names and columns from the SQLite DB.
    Assumes the DB contains potentially multiple tables representing different 'db_ids'.
    Returns a dictionary where keys are table names (acting as db_ids) and values are lists of column names.
    """
    if not conn: return {}
    schema = {}
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            # Filter out internal SQLite tables
            if table_name.startswith("sqlite_"): continue
            # Fetch columns for this table
            cursor.execute(f'PRAGMA table_info("{table_name}");') # Quote table name
            columns = [info[1] for info in cursor.fetchall()] # Column name is the second element
            if columns: # Only add tables that have columns
                schema[table_name] = columns
            else:
                 logging.warning(f"Table '{table_name}' found but has no columns defined?")
        if not schema:
             logging.warning("No user tables found in the database.")
        return schema
    except sqlite3.Error as e:
        logging.error(f"SQLite error fetching schema: {e}", exc_info=True)
        st.error(f"Error fetching database schema: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error fetching schema: {e}", exc_info=True)
        st.error(f"Unexpected error fetching database schema: {e}")
        return {}

def get_table_sample_data(conn: sqlite3.Connection, table_name: str, limit: int = 3) -> Optional[str]:
    """Fetches the first few rows of a table as a Markdown sample string."""
    # (Implementation remains the same as provided - includes table check)
    if not conn or not table_name: return None
    if not table_exists(conn, table_name):
        logging.warning(f"Table '{table_name}' not found in DB during sample fetching.")
        return None
    try:
        query = f'SELECT * FROM "{table_name}" LIMIT {limit}' # Quote table name
        df_sample = pd.read_sql_query(query, conn)
        if df_sample.empty:
             return f"Table '{table_name}' exists but is empty."
        return df_sample.to_markdown(index=False)
    except pd.io.sql.DatabaseError as e:
        logging.error(f"Pandas SQL Error fetching sample data for table '{table_name}': {e}")
        return f"Error fetching sample data for table '{table_name}': {e}"
    except Exception as e:
        logging.error(f"Unexpected error fetching sample data for table '{table_name}': {e}", exc_info=True)
        return f"Unexpected error fetching sample for table '{table_name}'."

# <-- NEW HELPER -->
def format_context_for_db_selection(
    available_schemas: Dict[str, List[str]],      # { db_id: [col1, col2, …], … }
    db_samples: Dict[str, Optional[str]],         # { db_id: sample_string or None, … }
    max_entries: Optional[int] = None             # truncate by number of DBs, not chars
) -> Tuple[Dict[str, Any], bool]:
    """
    Build the raw schema+sample context for DB selection.

    Returns:
      - context_data: { db_id: { "columns": […], "sample_data_snippet": str|None }, … }
      - truncated: True if we dropped entries to respect max_entries
    """
    # 1) Build full context
    context_data = {
        db_id: {
            "columns": cols,
            "sample_data_snippet": db_samples.get(db_id)
        }
        for db_id, cols in available_schemas.items()
    }

    # 2) Truncate by number of DBs if requested
    truncated = False
    if max_entries is not None and len(context_data) > max_entries:
        truncated = True
        # keep only the first max_entries keys
        keep = list(context_data)[:max_entries]
        context_data = {db_id: context_data[db_id] for db_id in keep}

    return context_data, truncated

# <-- NEW HELPER -->
def select_database_id(
    user_query: str,
    schema_and_samples: dict,    
    available_db_ids: list, # Pass original keys for validation
    llm_wrapper: LLMWrapper,
    conversation_history: Optional[list[dict]] = None # Keep if prompt uses it
) -> Optional[str]:
    """ Uses LLM to select the most relevant DB ID (table name) using formatted schema+sample context. """
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for DB ID selection.")
        return None
    if not PROMPTS:
        logging.warning("Prompts not loaded for DB ID selection.")
        return None
    if 'select_database_id' not in PROMPTS:
        logging.error("V3 Prompt 'select_database_id' not found in prompts.yaml!")
        logging.info("V3 Available prompts:", list(PROMPTS.keys()))
        return None
    template = PROMPTS.get("select_database_id")
    if not template:
        logging.error("No select_database_id template loaded")
        return None
    schemas_json = json.dumps(schema_and_samples, indent=2)
    db_id_list  = ", ".join(available_db_ids)

    try:
        prompt = template.format(
                    user_query=user_query,
                    schemas_json=schemas_json,
                    db_id_list=db_id_list,
                )
        
        prompt_preview = prompt[:500] + ("..." if len(prompt) > 500 else "")
        logging.debug(f"Prompt Preview:\n{prompt_preview}")

        selected_id_raw = llm_wrapper.generate_response(prompt, max_tokens=75, temperature=0.1) # Short response needed, allow slightly more tokens
        selected_id = selected_id_raw.strip().strip('`"\' ') # Remove common delimiters and whitespace

        logging.info(f"LLM suggested DB ID: '{selected_id}' (Raw: '{selected_id_raw}')")

        # --- Validation ---
        if selected_id in available_db_ids:
            return selected_id
        else:
            logging.warning(f"LLM selection '{selected_id}' invalid or not found in available DB IDs {available_db_ids}. Attempting fallback.")
            # --- Fallback Strategy (Simple Keyword Match) ---
            query_lower = user_query.lower()
            # Prioritize matching db_id (table name) first
            for db_id in available_db_ids:
                 # Match whole word if possible, or substring
                 if re.search(r'\b' + re.escape(db_id.lower()) + r'\b', query_lower):
                     logging.info(f"Fallback: Matched DB ID '{db_id}' (whole word) in query.")
                     return db_id
            # If no whole word match, try substring
            for db_id in available_db_ids:
                 if db_id.lower() in query_lower:
                      logging.info(f"Fallback: Matched DB ID '{db_id}' (substring) in query.")
                      return db_id

            logging.error("LLM selection failed and fallback keyword match also failed.")
            return None # Indicate failure

    except KeyError as e:
        logging.error(f"Couldn’t format 'select_database_id' prompt: missing placeholder {e!r}")
        return None
    except Exception as e:
        logging.error(f"V3 Error during DB ID selection LLM call: {e}", exc_info=True)
        return None

# --- LLM Interaction Helpers ---

def _ensure_english_query(user_query: str, llm_wrapper: LLMWrapper) -> tuple[str, bool]:
    """Checks if query is English. If not, attempts translation using LLM."""
    # (Implementation remains the same as provided, using prompts from PROMPTS)
    logging.info("Checking query language...")
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for language check/translation. Assuming English.")
        return user_query, False
    if not PROMPTS or 'check_language' not in PROMPTS or 'translate_to_english' not in PROMPTS:
         logging.error("Language check/translation prompts not loaded.")
         return user_query, False

    try:
        lang_prompt = PROMPTS['check_language'].format(user_query=user_query)
        lang_response = llm_wrapper.generate_response(lang_prompt, max_tokens=10, temperature=0.1).strip().upper() # Allow slightly more tokens

        # More robust checking
        if lang_response.startswith("YES") or "ENGLISH" in lang_response:
            logging.info("Query identified as English.")
            return user_query, False
        elif lang_response.startswith("NO") or any(kw in lang_response for kw in ["NON-ENGLISH", "NOT ENGLISH"]):
            logging.info("Query identified as non-English. Attempting translation...")
            trans_prompt = PROMPTS['translate_to_english'].format(user_query=user_query)
            translated_query = llm_wrapper.generate_response(trans_prompt, max_tokens=len(user_query) + 80, temperature=0.5).strip()

            # Basic validation of translation
            if translated_query and translated_query.lower() != user_query.lower() and len(translated_query) > 5 and len(translated_query.split()) > 1:
                logging.info(f"Translated query to English: \"{translated_query}\"")
                return translated_query, True
            else:
                logging.warning(f"Translation failed or unusable: '{translated_query}'. Using original.")
                return user_query, False
        else:
            logging.warning(f"Language check ambiguous: '{lang_response}'. Assuming English.")
            return user_query, False

    except Exception as e:
        logging.error(f"Error during language check/translation: {e}", exc_info=True)
        return user_query, False # Fail safe


# <-- MODIFIED HELPER -->
def refine_and_select(
    user_query: str,
    conversation_history: Optional[list[dict]],
    selected_db_schema: dict,        # { table_name: [cols] }
    sample_data_str: Optional[str],  # raw sample snippet or None
    llm_wrapper: LLMWrapper
) -> dict:
    """ Refines query, picks route, and augments keywords for a single DB context. """
    default_result = {
        "refined_query": user_query,
        "route": "SQL",
        "augmented_keywords": []
    }

    # Quick bailouts
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM not ready for refine_and_select; using defaults.")
        return default_result
    if not selected_db_schema:
        logging.warning("No schema provided to refine_and_select; using defaults.")
        return default_result

    # 1) Fetch the template
    template = PROMPTS.get("refine_and_select")
    if template is None:
        logging.error("Prompt 'refine_and_select' not found in PROMPTS.")
        return default_result

    # 2) Build the prompt inputs
    recent_context = (
        "\n".join(f"{m['role'].capitalize()}: {m['content']}"
                  for m in (conversation_history or [])[-10:])
        or "No conversation history."
    )
    schema_json = json.dumps(selected_db_schema, indent=2)
    sample_context = (
        f"Sample Data Snippet:\n{sample_data_str}"
        if sample_data_str else
        "Sample data not available."
    )

    # 3) Format the prompt, catching missing‐placeholder errors
    try:
        prompt = template.format(
            recent_context=recent_context,
            user_query=user_query,
            schema=schema_json,
            sample_data=sample_context
        )
    except KeyError as e:
        logging.error(
            f"refine_and_select prompt is missing placeholder {e!r}. "
            "Check your YAML template keys."
        )
        return default_result

    # 4) Call the LLM
    raw = llm_wrapper.generate_response(prompt, max_tokens=350, temperature=0.2)
    cleaned = raw.strip()
    # strip code fences if present
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # 5) Parse JSON (with fallback)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Not a dict")
    except Exception as e:
        logging.error(f"Failed to parse refine_and_select JSON: {e}; raw='{cleaned}'")
        # Try to extract just the route
        m = re.search(r"\b(SQL|SEMANTIC)\b", raw.upper())
        if m:
            return {**default_result, "route": m.group(1)}
        return default_result

    # 6) Build the final result with type checks
    result = {
        "refined_query": str(data.get("refined_query", user_query)) or user_query,
        "route": data.get("route", "SQL").upper() if isinstance(data.get("route"), str) else "SQL",
        "augmented_keywords": []
    }
    kws = data.get("augmented_keywords", [])
    if isinstance(kws, list):
        result["augmented_keywords"] = [str(k) for k in kws if k]
    else:
        logging.warning("refine_and_select returned non‐list for augmented_keywords; dropping it.")

    logging.info(f"Refine_and_select result: {result}")
    return result


# <-- MODIFIED HELPER -->
def _generate_sql_query(
    user_query: str,
    schema: dict, # <-- Changed: Schema for the single selected DB {table_name: [cols]}
    sample_data_str: Optional[str], # <-- Changed: Sample data string for selected DB
    aux_models,
    augmented_keywords: Optional[list[str]] = None,
    previous_sql: str | None = None,
    feedback: str | None = None
) -> str:
    """Generates SQL query using refined query and the SINGLE selected DB's schema/sample."""
    sql_llm: Llama = aux_models.get('sql_gguf_model')
    if not sql_llm: return "-- Error: SQL GGUF model object missing."
    if not PROMPTS or 'generate_sql_base' not in PROMPTS: return "-- Error: SQL generation prompt not loaded."

    # Format schema/sample for the selected DB only
    formatted_schema = json.dumps(schema, indent=2)
    sample_context = f"Sample Data Snippet:\n{sample_data_str}\n" if sample_data_str else "Sample data not available.\n"

    # --- Construct Prompt using loaded components ---
    prompt_context = {
        "schema": formatted_schema,
        "sample_data": sample_context,
        "user_query": user_query,
        "augmented_keywords": ", ".join(augmented_keywords) if augmented_keywords else "N/A" # Format keywords
    }
    prompt_parts = [PROMPTS['generate_sql_base'].format(**prompt_context)]

    if previous_sql and feedback:
        # Frame the feedback as direct guidance for the SQL LLM's next attempt
        feedback_header = "IMPORTANT GUIDANCE FOR YOUR NEXT SQL GENERATION ATTEMPT:\n" \
                        "A previous attempt to generate SQL for this request resulted in an issue.\n" \
                        "Please carefully analyze the following information and generate a corrected SQL query.\n"
        prompt_parts.append(feedback_header)

        if "syntax error" in feedback.lower(): # Or other specific error checks you might add
            # For syntax errors, the focus is on correcting the provided SQL
            prompt_parts.append(PROMPTS['generate_sql_feedback_syntax_correction'].format(
                previous_sql=previous_sql, 
                error_message=feedback # Pass the raw error as 'error_message'
            ))
        elif "no such table" in feedback.lower() or "no such column" in feedback.lower():
            # For schema-related errors, re-emphasize using the provided schema and highlight the issue
            prompt_parts.append(PROMPTS['generate_sql_feedback_schema_issue'].format(
                previous_sql=previous_sql,
                error_message=feedback,
                # We might need to pass schema_json and sample_data again here if they are not sticky enough
                # from the base prompt, or trust the LLM remembers it from generate_sql_base.
                # For now, let's assume the base prompt's schema is still in its "attention".
                # The key is to point out *what kind* of error it was.
                schema_context_reminder="Remember to strictly adhere to the tables and columns defined in the provided database schema."
            ))
        else: # General "other" errors (e.g., validation failure from general LLM)
            prompt_parts.append(PROMPTS['generate_sql_feedback_other_improvement'].format(
                previous_sql=previous_sql, 
                reason_for_improvement=feedback # 'feedback' here might be "results not satisfactory"
            ))

    prompt_parts.append(PROMPTS['generate_sql_response_format']) # "SQL Query:"
    prompt = "\n".join(prompt_parts)
    # --- End Prompt Construction ---

    logging.debug(f"Sending prompt to SQL GGUF model (Retry? {'Yes' if previous_sql else 'No'})...")
    prompt_preview = prompt[:500] + ("..." if len(prompt) > 500 else "")
    logging.debug(f"Prompt Preview:\n{prompt_preview}")

    try:
        output = sql_llm(prompt, max_tokens=500, temperature=0.1, top_p=0.9, stop=[";", "\n\n", "```", "---"], echo=False) # Added --- as stop token
        if output and 'choices' in output and len(output['choices']) > 0:
            generated_sql = output['choices'][0]['text'].strip()
            logging.info(f"Raw SQL GGUF output: {generated_sql}")

            # --- Robust Cleaning ---
            # Remove markdown
            cleaned_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
            # Find first SELECT or WITH
            cleaned_sql_upper = cleaned_sql.upper()
            select_pos = cleaned_sql_upper.find("SELECT")
            with_pos = cleaned_sql_upper.find("WITH")

            start_pos = -1
            if select_pos != -1 and with_pos != -1: start_pos = min(select_pos, with_pos)
            elif select_pos != -1: start_pos = select_pos
            elif with_pos != -1: start_pos = with_pos

            if start_pos > 0: # If text precedes SELECT/WITH, remove it
                cleaned_sql = cleaned_sql[start_pos:]
            elif start_pos == -1 and cleaned_sql: # No SELECT/WITH found
                 logging.warning(f"Generated SQL lacks SELECT/WITH keywords: {cleaned_sql}")
                 return "-- Error: Model did not generate a valid SQL query structure (no SELECT/WITH)."

            # Handle semicolon
            if ';' in cleaned_sql: cleaned_sql = cleaned_sql.split(';')[0] + ';'
            elif cleaned_sql.strip(): cleaned_sql += ';'; logging.debug("Added missing semicolon.")

            cleaned_sql = cleaned_sql.strip()
            logging.info(f"Cleaned SQL query: {cleaned_sql}")

            if not cleaned_sql: return "-- Error: Model generated an empty SQL query after cleaning."
            return cleaned_sql
        else:
            logging.error(f"SQL GGUF model returned empty/invalid output structure: {output}")
            return "-- Error: SQL model returned invalid output structure."
    except KeyError as e:
        logging.error(f"Prompt key missing for SQL generation: {e}")
        return f"-- Error: Prompt template key missing ({e})."
    except Exception as e:
        logging.error(f"Error during SQL GGUF generation: {e}", exc_info=True)
        return f"-- Error generating SQL: {type(e).__name__}"


def _execute_sql_query(conn, sql_query):
    """Executes the SQL query and returns results as a DataFrame."""
    # (Implementation remains the same as provided)
    if not conn: return pd.DataFrame(), "Database connection is not available."
    try:
        logging.info(f"Executing SQL: {sql_query}")
        start_time = time.perf_counter()
        # Ensure connection is valid before query
        conn.execute("SELECT 1") # Ping connection
        result_df = pd.read_sql_query(sql_query, conn)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.info(f"SQL execution time: {duration:.2f} seconds")
        return result_df, None
    except pd.io.sql.DatabaseError as e:
         logging.error(f"SQL Execution Error: {e}")
         # Improve error message parsing
         err_str = str(e).lower()
         if "no such table" in err_str: msg = f"Error: Table referenced in query does not exist. ({e})"
         elif "no such column" in err_str: msg = f"Error: Column referenced in query does not exist. ({e})"
         elif "ambiguous column name" in err_str: msg = f"Error: Ambiguous column name. Qualify with table name. ({e})"
         elif "syntax error" in err_str: msg = f"Error: Syntax error in the generated SQL. ({e})"
         else: msg = f"SQL Database Error: {e}"
         return pd.DataFrame(), msg
    except sqlite3.OperationalError as e:
         # Catch specific OperationalErrors like "database is locked"
         logging.error(f"SQLite Operational Error: {e}")
         msg = f"SQLite Operational Error: {e}"
         if "locked" in str(e): msg = "Database is locked. Please wait and retry."
         return pd.DataFrame(), msg
    except Exception as e:
        logging.error(f"Unexpected SQL Execution Error: {e}", exc_info=True)
        return pd.DataFrame(), f"An unexpected error occurred during SQL execution: {type(e).__name__}"

# <-- MODIFIED HELPER -->
def _validate_sql_results(user_query: str, executed_sql: str, result_df: pd.DataFrame, llm_wrapper: LLMWrapper) -> tuple[bool, str | None]:
    """Uses LLM to validate if SQL results satisfy the user query."""
    # (Implementation remains the same as provided - uses prompts, parses JSON/text)
    logging.info("Validating SQL results using LLM...")
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for validation. Assuming results are satisfactory.")
        return True, None
    if not PROMPTS or 'validate_sql_results' not in PROMPTS:
         logging.error("Validation prompt 'validate_sql_results' not loaded.")
         return True, None # Assume valid if prompt missing

    max_rows_for_context = 5 # Increase slightly
    context_str = "The query returned zero rows." if result_df.empty else result_df.head(max_rows_for_context).to_markdown(index=False)
    if not result_df.empty and len(result_df) > max_rows_for_context:
         context_str += f"\n\n...(and {len(result_df) - max_rows_for_context} more rows)"

    try:
        prompt = PROMPTS['validate_sql_results'].format(
            user_query=user_query, # Use the query that was used for SQL generation (refined query)
            executed_sql=executed_sql,
            context_str=context_str
        )
        logging.debug(f"Sending validation prompt to LLM...")
        prompt_preview = prompt[:500] + ("..." if len(prompt) > 500 else "")
        logging.debug(f"Prompt Preview:\n{prompt_preview}")


        # Attempt structured response first if available, fallback to text
        response_text = llm_wrapper.generate_response(prompt + ' Respond in JSON: {"satisfactory": boolean, "reason": "string_or_null"}', max_tokens=150, temperature=0.1) # Request JSON
        logging.debug(f"Raw validation response: {response_text}")

        # --- Robust Parsing ---
        cleaned_text = response_text.strip()
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL) # Find first JSON object

        if json_match:
            try:
                parsed_json = json.loads(json_match.group(0))
                if isinstance(parsed_json, dict):
                    is_satisfactory = parsed_json.get("satisfactory")
                    reason = parsed_json.get("reason")
                    if isinstance(is_satisfactory, bool):
                        feedback = reason if not is_satisfactory and isinstance(reason, str) and reason else None
                        logging.info(f"LLM validation (parsed JSON): Satisfactory={is_satisfactory}, Feedback='{feedback}'")
                        return is_satisfactory, feedback
                    else: logging.warning(f"Parsed validation JSON invalid 'satisfactory': {parsed_json}. Assuming satisfactory.")
                else: logging.warning("Parsed validation response not dict. Assuming satisfactory.")
            except json.JSONDecodeError:
                 logging.warning(f"Could not parse JSON from LLM validation response: {json_match.group(0)}. Trying text fallback.")
                 # Fallback to simple text check if JSON parsing fails
                 if "YES" in response_text.upper() or "SATISFACTORY" in response_text.upper():
                     logging.info("LLM validation (text fallback): Assumed satisfactory.")
                     return True, None
                 else:
                     logging.info("LLM validation (text fallback): Assumed unsatisfactory (no positive confirmation).")
                     # Extract potential reason if possible
                     reason_match = re.search(r'(?:reason|feedback|issue|problem)[:\s]*(.*)', response_text, re.IGNORECASE | re.DOTALL)
                     feedback = reason_match.group(1).strip() if reason_match else "LLM indicated results might not be satisfactory (based on text)."
                     return False, feedback
        else:
             # No JSON found, rely on text interpretation
             logging.warning(f"No JSON object found in LLM validation response: '{response_text}'. Using text check.")
             if "YES" in response_text.upper() or "SATISFACTORY" in response_text.upper():
                 logging.info("LLM validation (text only): Assumed satisfactory.")
                 return True, None
             else:
                 logging.info("LLM validation (text only): Assumed unsatisfactory.")
                 reason_match = re.search(r'(?:reason|feedback|issue|problem)[:\s]*(.*)', response_text, re.IGNORECASE | re.DOTALL)
                 feedback = reason_match.group(1).strip() if reason_match else "LLM indicated results might not be satisfactory (text only)."
                 return False, feedback

    except KeyError:
         logging.error("Prompt key 'validate_sql_results' not found.")
         return True, None # Assume valid
    except Exception as e:
        logging.error(f"Error during LLM validation call: {e}", exc_info=True)
        return True, None # Fail safely: assume satisfactory


# --- Semantic Search and Qdrant Helpers ---
# (Functions: _perform_semantic_search, get_qdrant_collection_info,
#  _suggest_semantic_columns, prepare_collection, prepare_documents,
#  compute_vector_size, create_qdrant_collection_with_retry,
#  create_or_update_collection, _embed_and_index_data, reindex_table,
#  delete_table_data)
# These generally remain the same, but ensure logging includes table_name/db_id
# Make sure collection names consistently use the QDRANT_COLLECTION_PREFIX + table_name (db_id)

def _suggest_semantic_columns(df_head: pd.DataFrame, schema: dict, table_name: str, llm_wrapper: LLMWrapper) -> list[str]:
    """Uses LLM to suggest semantic columns. Needs the schema for the specific table."""
    # (Implementation mostly the same, ensure it uses the correct prompt key)
    logging.info(f"[{table_name}] Requesting LLM suggestion for semantic columns...")
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning(f"[{table_name}] LLM wrapper not ready for semantic column suggestion. Falling back.")
        return df_head.select_dtypes(include=['object', 'string']).columns.tolist()
    if not PROMPTS or 'suggest_semantic_columns' not in PROMPTS:
        logging.error(f"[{table_name}] Prompt 'suggest_semantic_columns' not loaded.")
        return df_head.select_dtypes(include=['object', 'string']).columns.tolist()

    # Use only the columns for the specific table_name from the schema dict
    table_columns = schema.get(table_name, [])
    if not table_columns:
         logging.warning(f"[{table_name}] No schema columns found for table '{table_name}'.")
         return df_head.select_dtypes(include=['object', 'string']).columns.tolist()

    schema_str = json.dumps({table_name: table_columns}, indent=2) # Format schema for this table only
    df_head_str = df_head.to_string(index=False, max_rows=5)

    prompt = PROMPTS['suggest_semantic_columns'].format(
        table_name=table_name,
        schema_str=schema_str,
        df_head_str=df_head_str
    )
    logging.debug(f"[{table_name}] Sending semantic column suggestion prompt...")

    suggested_columns = []
    try:
        # Generate response (expecting JSON list as string)
        raw_llm_output = llm_wrapper.generate_response(prompt + ' Respond as JSON list: []', max_tokens=200) # Guide LLM
        logging.debug(f"[{table_name}] Raw LLM suggestion response: {raw_llm_output}")

        # Clean and parse
        cleaned_text = raw_llm_output.strip()
        json_match = re.search(r'\[.*?\]', cleaned_text, re.DOTALL) # Find JSON list

        if json_match:
             try:
                 parsed_list = json.loads(json_match.group(0))
                 if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                     suggested_columns = parsed_list
                     logging.info(f"[{table_name}] LLM suggested (parsed JSON): {suggested_columns}")
                 else:
                     logging.warning(f"[{table_name}] Parsed LLM response was not a valid list of strings: {parsed_list}")
             except json.JSONDecodeError as e:
                 logging.error(f"[{table_name}] Failed to parse JSON response for suggestions: {e}")
                 logging.debug(f"Invalid JSON received: {json_match.group(0)}")
        else:
            logging.warning(f"[{table_name}] No JSON list found in LLM suggestion response: {cleaned_text}")
            # Fallback: try splitting if it looks like a comma-separated list
            if ',' in cleaned_text and len(cleaned_text) < 100:
                 suggested_columns = [col.strip().strip("'\"") for col in cleaned_text.split(',') if col.strip()]
                 logging.info(f"[{table_name}] LLM suggested (parsed comma-separated fallback): {suggested_columns}")


    except KeyError:
        logging.error(f"[{table_name}] Prompt key 'suggest_semantic_columns' not found.")
    except Exception as e:
        logging.error(f"[{table_name}] Error during LLM semantic column suggestion: {e}", exc_info=True)

    # Final fallback & validation against actual DataFrame columns
    if not suggested_columns:
        logging.warning(f"[{table_name}] LLM suggestion failed/empty. Selecting all object/string columns.")
        suggested_columns = df_head.select_dtypes(include=['object', 'string']).columns.tolist()

    final_columns = [col for col in suggested_columns if col in df_head.columns]
    if len(final_columns) != len(suggested_columns):
         logging.warning(f"[{table_name}] LLM suggested columns not present in DataFrame were filtered out. Original: {suggested_columns}, Final: {final_columns}")

    logging.info(f"[{table_name}] Final semantic columns selected: {final_columns}")
    return final_columns


def prepare_documents(df: pd.DataFrame, valid_cols: list[str], table_name: str) -> tuple[list[str], list[dict], list[str]]:
    """Prepares documents, payloads, and UUIDs for Qdrant."""
    # (Implementation remains the same as provided)
    documents, payloads, point_ids = [], [], []
    potential_pk = next((col for col in df.columns if col.lower() == 'id'), None)
    if potential_pk is None and len(df.columns) > 0 and df.columns[0].lower() in ['id', 'pk', 'key']: potential_pk = df.columns[0]
    logging.debug(f"[{table_name}] Potential primary key for payload ID: {potential_pk}")

    for idx, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in valid_cols if pd.notna(row[col]) and str(row[col]).strip()]
        if not parts: continue
        doc_text = " | ".join(parts)
        documents.append(doc_text)
        payload = row.to_dict()
        for k, v in payload.items():
            if pd.isna(v): payload[k] = None # Convert NaN/NaT to None for JSON/Qdrant
        payload["_table_name"] = table_name # Add metadata

        original_id = f"{table_name}_idx_{idx}" # Default ID
        if potential_pk and potential_pk in row and pd.notna(row[potential_pk]):
            try: original_id = f"{table_name}_{str(row[potential_pk])}"
            except Exception as ex: logging.warning(f"[{table_name}] Error converting PK for row {idx}: {ex}")
        payload["_original_id"] = original_id
        payload["_source_text"] = doc_text # Store the text used for embedding

        point_ids.append(str(uuid.uuid4())) # Always use UUID for Qdrant ID
        payloads.append(payload)

    logging.info(f"[{table_name}] Prepared {len(documents)} documents for embedding.")
    return documents, payloads, point_ids


def compute_vector_size(embeddings: list, table_name: str) -> tuple[int, str]:
    """Determines vector size, checks against ENV var."""
    # (Implementation remains the same as provided)
    if not embeddings or not isinstance(embeddings[0], (np.ndarray, list)):
        return 0, "Invalid or empty embeddings list."
    try:
        computed_size = len(embeddings[0])
        expected_size = int(os.getenv("EMBEDDING_VECTOR_SIZE", "768")) # Default to common size
        if computed_size != expected_size:
            logging.warning(f"[{table_name}] Computed vector size {computed_size} differs from EMBEDDING_VECTOR_SIZE {expected_size}. Using computed value.")
        return computed_size, ""
    except IndexError: return 0, "Embeddings list is empty."
    except TypeError: return 0, "Embeddings elements are not lists or arrays."
    except Exception as e:
        logging.error(f"[{table_name}] Error determining vector size: {e}", exc_info=True)
        return 0, f"Error determining vector size: {e}"


def create_or_update_collection(qdrant_client: QdrantClient, collection_name: str, vector_size: int, table_name: str) -> tuple[bool, str, list[str]]:
    """Creates or recreates a Qdrant collection."""
    # (Implementation remains the same as provided)
    status_messages = []
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logging.info(f"[{table_name}] Collection '{collection_name}' exists. Deleting to update.")
        qdrant_client.delete_collection(collection_name=collection_name, timeout=10)
        status_messages.append(f"Deleted existing collection '{collection_name}'.")
        time.sleep(0.5)
    except Exception: # Catches specific "Not Found" errors implicitly
        logging.info(f"[{table_name}] Collection '{collection_name}' not found or error checking.")

    try:
        logging.info(f"[{table_name}] Creating collection '{collection_name}' with vector size {vector_size}...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            # Consider adding HNSW indexing config here if needed
            # hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
        )
        logging.info(f"[{table_name}] Collection '{collection_name}' created successfully.")
        status_messages.append(f"Created collection '{collection_name}'.")
    except Exception as e:
        logging.error(f"[{table_name}] Failed to create collection '{collection_name}': {e}", exc_info=True)
        return False, f"Failed to create Qdrant collection: {e}", status_messages
    return True, "", status_messages


def _embed_and_index_data(df: pd.DataFrame, table_name: str, semantic_columns: list, aux_models: dict,
                          qdrant_client: QdrantClient) -> tuple[bool, str]:
    """Embeds data and indexes in Qdrant."""
    # (Implementation remains the same as provided, uses helpers defined above)
    logging.info(f"[{table_name}] Starting embedding and indexing...")
    if not qdrant_client: return False, "Qdrant client not available."
    if not aux_models or aux_models.get('status') != 'loaded' or not aux_models.get('embedding_model'): return False, "Embedding model not loaded."
    if df.empty: return True, "DataFrame is empty."
    valid_cols = [col for col in semantic_columns if col in df.columns]
    if not valid_cols: return True, "No valid semantic columns found."

    documents, payloads, point_ids = prepare_documents(df, valid_cols, table_name)
    if not documents: return True, "No non-empty documents prepared."

    embedding_model = aux_models.get("embedding_model")
    logging.info(f"[{table_name}] Generating embeddings for {len(documents)} documents...")
    try:
        # Assuming embed returns a list or ndarray
        embeddings_result = embedding_model.embed(documents)
        # Ensure it's a list for consistent handling
        embeddings = list(embeddings_result) if not isinstance(embeddings_result, list) else embeddings_result
        if len(embeddings) != len(documents): return False, "Embedding count mismatch."
    except Exception as embed_e: return False, f"Embedding generation failed: {embed_e}"

    vector_size, size_error = compute_vector_size(embeddings, table_name)
    if vector_size == 0: return False, size_error
    logging.info(f"[{table_name}] Using vector size: {vector_size}")

    collection_name = f"{QDRANT_COLLECTION_PREFIX}{table_name}"
    coll_ok, coll_error, _ = create_or_update_collection(qdrant_client, collection_name, vector_size, table_name)
    if not coll_ok: return False, coll_error

    batch_size = 100 # Adjust as needed
    num_batches = (len(point_ids) + batch_size - 1) // batch_size
    logging.info(f"[{table_name}] Upserting {len(point_ids)} points into '{collection_name}' in {num_batches} batches...")
    for i in range(num_batches):
        start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, len(point_ids))
        batch_ids = point_ids[start_idx:end_idx]
        # Ensure vectors are lists of floats
        batch_vectors = [[float(v) for v in vec] for vec in embeddings[start_idx:end_idx]]
        batch_payloads = payloads[start_idx:end_idx]
        logging.debug(f"[{table_name}] Upserting batch {i+1}/{num_batches}...")
        try:
            qdrant_client.upsert(collection_name=collection_name, points=models.Batch(ids=batch_ids, vectors=batch_vectors, payloads=batch_payloads), wait=True)
        except Exception as upsert_e: return False, f"Error during Qdrant upsert (batch {i+1}): {upsert_e}"

    logging.info(f"[{table_name}] Successfully indexed {len(point_ids)} points into '{collection_name}'.")
    return True, f"Indexed {len(point_ids)} points into collection '{collection_name}'."


def get_qdrant_collection_info(qdrant_client: QdrantClient, collection_name: str) -> dict | None:
    """Gets point count and vector dimension for a Qdrant collection."""
    # (Implementation remains the same as provided - robust attribute access)
    if not qdrant_client: return None
    try:
        info = qdrant_client.get_collection(collection_name=collection_name)
        points_count = getattr(info, 'points_count', 0)
        vectors_count = getattr(info, 'vectors_count', 0) # Might differ if named vectors used
        status = getattr(info, 'status', 'Unknown')
        vec_size = "N/A"
        vectors_config_obj = getattr(info, 'vectors_config', None)
        config_obj = getattr(info, 'config', None)

        # Try standard locations first
        if isinstance(vectors_config_obj, models.VectorParams): vec_size = getattr(vectors_config_obj, 'size', 'N/A')
        elif isinstance(vectors_config_obj, dict): # Named vectors
            first_vec_name = next(iter(vectors_config_obj), None)
            if first_vec_name and isinstance(vectors_config_obj[first_vec_name], models.VectorParams): vec_size = getattr(vectors_config_obj[first_vec_name], 'size', 'N/A')
        # Try nested location (less common now)
        elif config_obj and hasattr(config_obj, 'params') and hasattr(config_obj.params, 'vectors'):
             vectors_param = getattr(config_obj.params, 'vectors', None)
             if isinstance(vectors_param, models.VectorParams): vec_size = getattr(vectors_param, 'size', 'N/A')
             elif isinstance(vectors_param, dict): # Named vectors under config.params
                  first_vec_name = next(iter(vectors_param), None)
                  if first_vec_name and isinstance(vectors_param[first_vec_name], models.VectorParams): vec_size = getattr(vectors_param[first_vec_name], 'size', 'N/A')

        if vec_size == "N/A": logging.warning(f"[{collection_name}] Could not determine vector size from CollectionInfo.")

        return {"points_count": points_count, "vectors_count": vectors_count, "vector_size": vec_size, "status": str(status)}
    except Exception as e:
        if "not found" in str(e).lower() or "status_code=404" in str(e): logging.info(f"Qdrant collection '{collection_name}' not found.")
        elif isinstance(e, AttributeError): logging.warning(f"Attribute error getting Qdrant info for '{collection_name}': {e}.")
        else: logging.warning(f"Could not get Qdrant info for '{collection_name}': {type(e).__name__}: {e}")
        return None

def _perform_semantic_search(user_query: str, aux_models: dict, qdrant_client: QdrantClient, schema: dict) -> tuple[list[str], str | None]:
    """Performs semantic search across all relevant Qdrant collections."""
    # (Implementation remains the same as provided - searches multiple collections)
    logging.info("Performing semantic search...")
    if not qdrant_client: return [], "Qdrant client not initialized."
    if not aux_models or aux_models.get('status') != 'loaded' or not aux_models.get('embedding_model'): return [], "Embedding model not loaded."

    embedding_model = aux_models.get('embedding_model')
    try:
        logging.debug(f"Embedding user query: '{user_query[:100]}...'")
        query_vector = list(embedding_model.embed([user_query]))[0]
        logging.debug("Query embedded successfully.")

        all_collections = qdrant_client.get_collections()
        target_collections = [c.name for c in all_collections.collections if c.name.startswith(QDRANT_COLLECTION_PREFIX)]
        if not target_collections: return ["No vector data collections found for searching."], None
        logging.info(f"Searching in Qdrant collections: {target_collections}")

        all_hits = []
        search_limit_per_collection = 5
        for collection_name in target_collections:
            try:
                search_result = qdrant_client.search(collection_name=collection_name, query_vector=query_vector, limit=search_limit_per_collection)
                logging.info(f"Found {len(search_result)} hits in '{collection_name}'.")
                all_hits.extend(search_result)
            except Exception as search_e: logging.error(f"Error searching collection '{collection_name}': {search_e}")

        if not all_hits: return ["No relevant matches found in the vector data."], None
        all_hits.sort(key=lambda x: x.score, reverse=True)
        formatted_results = []
        max_total_results = 10

        logging.info(f"Processing top {min(len(all_hits), max_total_results)} hits...")
        for hit in all_hits[:max_total_results]:
            payload = hit.payload 
            if not payload: continue
            table_name = payload.get("_table_name", "Unknown")
            display_text = payload.get("_source_text", str(payload))[:200] + ("..." if len(payload.get("_source_text", "")) > 200 else "")
            display_id = payload.get("_original_id", str(hit.id))
            if isinstance(display_id, str) and display_id.startswith(f"{table_name}_"): display_id = display_id.split('_', 1)[1]
            result_str = f"**Table:** `{table_name}` | **ID:** `{display_id}` | **Score:** {hit.score:.3f}\n> {display_text}"
            formatted_results.append(result_str)

        return formatted_results, None
    except Exception as e:
        logging.error(f"Unexpected error during semantic search: {e}", exc_info=True)
        return [], f"Error during semantic search: {e}"

# --- Pipeline Execution Helpers ---

# <-- MODIFIED HELPER -->
def execute_sql_pipeline(
    user_query: str,
    selected_db_schema: dict, 
    sample_data_str: Optional[str], 
    aux_models,
    conn,
    llm_wrapper,
    augmented_keywords: Optional[list[str]] = None,
    max_retries: int = 1
) -> dict:
    """Generates and executes SQL for the selected DB, with retries and validation."""
    sql_result = { "sql_success": False, "generated_sql": None, "sql_data": None, "sql_error": None }
    current_sql = None
    previous_feedback = None

    if not isinstance(aux_models, dict):
        logging.error("ERROR: aux_models is not a dict in execute_sql_pipeline!")
        sql_result["sql_error"] = "Internal Error: Model configuration invalid."
        return sql_result
    if not conn:
        sql_result["sql_error"] = "Internal Error: DB connection is invalid for execution."
        return sql_result

    for attempt in range(max_retries + 1):
        logging.info(f"SQL Generation/Execution Attempt #{attempt + 1}")
        current_sql = _generate_sql_query(
            user_query=user_query,
            schema=selected_db_schema, # Pass the selected DB schema
            sample_data_str=sample_data_str, # Pass the selected DB sample
            aux_models=aux_models,
            augmented_keywords=augmented_keywords,
            previous_sql=current_sql,
            feedback=previous_feedback
        )
        sql_result["generated_sql"] = current_sql
        previous_feedback = None # Reset feedback for next potential retry

        if current_sql.startswith("-- Error"):
            feedback = f"SQL generation failed: {current_sql}"
            if attempt < max_retries: previous_feedback = feedback; logging.warning(feedback + " Retrying..."); continue
            else: sql_result["sql_error"] = feedback; break # Max retries hit or unrecoverable generation error

        sql_df, exec_error = _execute_sql_query(conn, current_sql)
        if exec_error:
            feedback = f"SQL execution error: {exec_error}"
            if attempt < max_retries: previous_feedback = feedback; logging.warning(feedback + " Retrying..."); continue
            else: sql_result["sql_error"] = feedback; break # Max retries hit

        # Validate results using the refined query (which led to this SQL)
        is_valid, validation_feedback = _validate_sql_results(user_query, current_sql, sql_df, llm_wrapper)
        if is_valid:
            sql_result["sql_success"] = True; sql_result["sql_data"] = sql_df; sql_result["sql_error"] = None
            logging.info("SQL attempt successful and validated.")
            break # Success!
        else:
            feedback = f"Result validation failed: {validation_feedback}"
            if attempt < max_retries: previous_feedback = feedback; logging.warning(feedback + " Retrying..."); continue
            else: sql_result["sql_error"] = feedback; break # Max retries hit on validation fail

    return sql_result


# <-- MODIFIED HELPER -->
def execute_semantic_pipeline(
    english_query: str,
    # schema: dict, # Schema might not be directly needed if searching all collections
    aux_models: dict,
    qdrant_client
) -> dict:
    """Executes semantic search across all relevant collections."""
    semantic_result = { "semantic_success": False, "semantic_data": None, "semantic_error": None }
    # Pass empty schema dict as _perform_semantic_search doesn't strictly need it currently
    semantic_data, error_msg = _perform_semantic_search(english_query, aux_models, qdrant_client, {})
    if not error_msg:
        semantic_result["semantic_success"] = True
        semantic_result["semantic_data"] = semantic_data
    else:
        semantic_result["semantic_error"] = error_msg
    return semantic_result


# <-- MODIFIED HELPER -->
def run_secondary_route(
    user_query: str,
    primary_route: str,
    sql_result: dict, # Result from primary SQL attempt
    semantic_result: dict, # Result from primary Semantic attempt
    aux_models,
    conn,
    selected_db_schema: dict, # Changed: Schema for the single selected DB
    selected_db_sample: Optional[str], # Changed: Sample for the single selected DB
    llm_wrapper,
    qdrant_client,
    augmented_keywords: Optional[list[str]] = None # Added keywords
) -> tuple[dict, dict]:
    """If primary route failed/empty, runs the other route once."""
    # Determine if secondary run is needed
    primary_sql_failed = (primary_route == "SQL" and not sql_result.get("sql_success"))
    primary_semantic_failed = (primary_route == "SEMANTIC" and not semantic_result.get("semantic_success"))

    if primary_sql_failed:
        logging.info("Primary SQL failed/empty, running secondary Semantic search...")
        # Execute semantic pipeline as secondary
        sec_semantic_result = execute_semantic_pipeline(user_query, aux_models, qdrant_client)
        # Update semantic_result only if the secondary attempt was successful
        if sec_semantic_result.get("semantic_success"):
            semantic_result = sec_semantic_result # Replace primary result

    elif primary_semantic_failed:
        logging.info("Primary Semantic failed/empty, running secondary SQL generation/execution...")
        # Execute SQL pipeline as secondary (only 1 attempt, no retry loop here)
        sec_sql_result = execute_sql_pipeline(
            user_query=user_query, 
            selected_db_schema=selected_db_schema,
            sample_data_str=selected_db_sample,
            aux_models=aux_models,
            conn=conn,
            llm_wrapper=llm_wrapper,
            augmented_keywords=augmented_keywords,
            max_retries=0 # Only one attempt for secondary
        )
        # Update sql_result only if the secondary attempt was successful
        if sec_sql_result.get("sql_success"):
            sql_result = sec_sql_result # Replace primary result

    return sql_result, semantic_result


# --- Final Summary Generation ---
def generate_final_summary(
    original_query: str,
    english_query: str,
    was_translated: bool,
    sql_data: Optional[pd.DataFrame], # Accept DataFrame or None
    semantic_data: Optional[list], # Accept list or None
    llm_wrapper
) -> str:
    """Aggregates results into a final natural language summary."""
    # (Implementation remains the same as provided - uses prompts)
    summary_parts = []
    if sql_data is not None: # Check if df exists
        if not sql_data.empty:
            max_rows = 5
            summary_parts.append("SQL Query Results (Sample):")
            summary_parts.append(sql_data.head(max_rows).to_markdown(index=False))
            if len(sql_data) > max_rows: summary_parts.append(f"... ({len(sql_data) - max_rows} more rows)")
        else: summary_parts.append("The SQL query returned no matching rows.")

    # Check if semantic_data is a non-empty list and first item isn't a 'not found' message
    if semantic_data and isinstance(semantic_data, list) and not ("No relevant matches found" in semantic_data[0] or "No vector data found" in semantic_data[0]):
        max_snippets = 5
        summary_parts.append("\nSemantic Search Results (Top Snippets):")
        summary_parts.append("\n\n".join(semantic_data[:max_snippets])) # Add spacing
        if len(semantic_data) > max_snippets: summary_parts.append(f"... ({len(semantic_data) - max_snippets} more snippets)")

    summary_context = "\n".join(summary_parts).strip()
    translated_context = f'Query Interpreted as (English): "{english_query}"' if was_translated else ""

    if not summary_context:
        return "I searched the data based on your query, but couldn't find specific information matching your request."
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for summary generation. Returning raw results.")
        return f"Summary LLM unavailable.\n{translated_context}\n{summary_context}" # Return formatted raw data
    if not PROMPTS or 'generate_final_summary' not in PROMPTS:
         logging.error("Summary prompt 'generate_final_summary' not loaded.")
         return f"Summary prompt missing.\n{translated_context}\n{summary_context}"

    try:
        summary_prompt = PROMPTS['generate_final_summary'].format(
            original_query=original_query,
            translated_context=translated_context,
            summary_context=summary_context
        )
        logging.info("Requesting final LLM summary...")
        logging.debug(f"Summary prompt:\n{summary_prompt}")
        nl_summary = llm_wrapper.generate_response(summary_prompt, max_tokens=400) # Increase token limit
        return nl_summary.strip()
    except KeyError:
        logging.error("Prompt key 'generate_final_summary' not found.")
        return f"Summary prompt key missing.\n{translated_context}\n{summary_context}"
    except Exception as e:
        logging.error(f"Error during summary generation: {e}", exc_info=True)
        return f"Error generating summary. Raw results:\n{translated_context}\n{summary_context}"

# --- MAIN PIPELINE FUNCTION ---

def process_natural_language_query(
    original_query: str,
    conn: Optional[sqlite3.Connection], # Connection can be None if only analysis is needed
    full_schema: dict, # Expects {db_id: [cols], ...} or {table_name: [cols], ...}
    llm_wrapper, aux_models, qdrant_client, max_retries: int = 1,
    conversation_history: Optional[list] = None,
    db_samples_for_selection: Optional[Dict[str, str]] = None, # New: {db_id: "concatenated_sample_str_for_db_id"}
    pre_selected_db_id: Optional[str] = None, # Optional: Force use of a specific db_id/table
    semantic_search: Optional[bool] = False
) -> dict:
    """
    Main pipeline for processing NL queries against multiple potential DBs/tables.
    1. Validate inputs. Check language.
    2. Select relevant db_id (table) using schemas and samples.
    3. Refine query, get keywords, determine route based on selected context.
    4. If route is SQL, generate SQL string for the selected table.
    5. If conn provided: Execute SQL/Semantic, run secondary route, summarize.
    """
    # --- Initial Checks ---
    start_time_total = time.perf_counter()
    result = {
        "status": "processing", "message": "Starting query processing...",
        "query_was_translated": False, "processed_query": original_query,
        "selected_db_id": None, "refined_query": None, "augmented_keywords": [],
        "determined_route": None, "generated_sql": None,
        "sql_success": False, "sql_data": None, "sql_error": None,
        "semantic_success": False, "semantic_data": None, "semantic_error": None,
        "natural_language_summary": None, "raw_display_data": None, "timing": {}
    }

    # ... (Initial Checks remain the same) ...
    if not full_schema: return {**result, "status": "error", "message": "No database schema provided."}
    available_db_ids_from_full_schema = list(full_schema.keys()) # These are the DB_IDs in context
    if not available_db_ids_from_full_schema: return {**result, "status": "error", "message": "Schema dictionary is empty."}


    # --- 1. Language Check ---
    start_time_lang = time.perf_counter()
    processed_query, was_translated = _ensure_english_query(original_query, llm_wrapper)
    result["query_was_translated"] = was_translated
    result["processed_query"] = processed_query
    result["timing"]["language_check"] = time.perf_counter() - start_time_lang

    # --- 2. Select Database ID ---
    start_time_select = time.perf_counter()
    selected_db_id_by_llm = None # LLM's choice of DB
    sample_data_for_sql_gen = "Sample data not available for the selected DB for SQL generation." # Default

    if pre_selected_db_id and pre_selected_db_id in full_schema:
        selected_db_id_by_llm = pre_selected_db_id
        logging.info(f"Using pre-selected DB ID: {selected_db_id_by_llm}")
        if db_samples_for_selection and selected_db_id_by_llm in db_samples_for_selection:
            sample_data_for_sql_gen = db_samples_for_selection[selected_db_id_by_llm]
        elif conn: # Fallback: try to get sample using conn if it's for this pre_selected_db_id
            logging.warning(f"Samples for pre-selected DB '{selected_db_id_by_llm}' not in db_samples_for_selection. Attempting fetch with 'conn'.")
            current_db_schema_dict = full_schema.get(selected_db_id_by_llm)
            if current_db_schema_dict:
                # Re-using _get_multi_table_sample_str logic concept
                s_parts = []
                tables_to_sample = list(current_db_schema_dict.keys())[:3]
                for t_name in tables_to_sample:
                    s_data = get_table_sample_data(conn, t_name, limit=2)
                    if s_data and "Error" not in s_data: s_parts.append(f"-- Sample rows from table {t_name}\n{s_data}")
                if s_parts: sample_data_for_sql_gen = "\n\n".join(s_parts)
    else:
        logging.info("Attempting to select relevant Database ID (Table)...")
        
        effective_db_samples_for_select_prompt = {}
        if db_samples_for_selection:
            logging.info(f"Received db_samples_for_selection with {len(db_samples_for_selection)} entries.") # ADD DEBUG
            effective_db_samples_for_select_prompt = db_samples_for_selection
            # Ensure samples only for DBs present in full_schema to avoid issues
            effective_db_samples_for_select_prompt = {
                k: v for k, v in effective_db_samples_for_select_prompt.items() if k in full_schema
            }
            if not effective_db_samples_for_select_prompt:
                logging.warning("`db_samples_for_selection` was filtered to empty after checking against `full_schema` keys.")
        else:
            logging.warning("`db_samples_for_selection` not provided to `process_natural_language_query`. Sample quality for DB selection might be reduced.")
            # LLM will select based on schemas only if samples are missing.

        # Format context (schemas + samples) for the selection prompt
        # full_schema is {db_id: {table:cols}, ...}
        # format_context_for_db_selection expects {db_id: [cols]} for its `available_schemas`
        # We need to adapt or pass a simplified schema view for selection if it expects flat [cols] per db_id.
        # The current `format_context_for_db_selection` takes {db_id: {"columns": [cols], "sample": str}} which is fine.
        # It seems `select_database_id` expects `schemas_json` to be string dump of this.
        # `full_schema` is already structured as {db_id: schema_dict_for_db_id}.
        # `format_context_for_db_selection` is more about formatting a string.
        # `select_database_id` takes `schema_and_samples` (a dict) and `available_db_ids`.
        # Let's construct `schema_and_samples_for_select_prompt` directly for `select_database_id`
        
        schema_and_samples_for_select_prompt_dict = {}
        for db_id_key, single_db_schema_val in full_schema.items():
            # single_db_schema_val is {table_name: [cols], ...}
            # We need to represent this in a way `select_database_id` prompt understands.
            # The prompt `select_database_id` expects `schemas_json` which is a dump of
            # { db_id: { "columns": [col1, col2, ...], "sample_data_snippet": str|None }, ... }
            # The "columns" here should ideally be a representation of all tables and columns.
            
            # Let's simplify: the prompt for `select_database_id` likely just wants a string representation
            # of the schema for each db_id.
            
            # The `format_context_for_db_selection` creates this structure from a simpler input.
            # Let's use it as intended:
            # `available_schemas_for_formatter = {db_id: list_of_all_cols_in_db_id_for_simplicity}`
            # Or, `available_schemas_for_formatter = full_schema` if format_context understands it.
            # Looking at format_context_for_db_selection, it expects `available_schemas: Dict[str, List[str]]`
            # which means db_id -> list of columns (flat, not per table). This might be a simplification in that helper.
            
            # For now, let's assume `select_database_id`'s prompt handles `schemas_json` derived from
            # `full_schema` (which is `{db_id: {table:cols}}`) and `effective_db_samples_for_select_prompt`.
            # The prompt might iterate `db_id: schema_info_dict` from `schemas_json`.
            # Let's build what `select_database_id` expects for its `schema_and_samples` argument.
            db_context_for_selection = {}
            for db_id_sel, schema_dict_sel in full_schema.items():
                # Create a string summary of tables and columns for "columns" field
                col_summary_parts = []
                for table_n, table_c in schema_dict_sel.items():
                    col_summary_parts.append(f"Table {table_n}: {', '.join(table_c)}")
                db_context_for_selection[db_id_sel] = {
                    "schema_summary": "; ".join(col_summary_parts), # Or just pass schema_dict_sel
                    "tables_and_columns": schema_dict_sel, # More structured
                    "sample_data_snippet": effective_db_samples_for_select_prompt.get(db_id_sel, "N/A")
                }

        selected_db_id_by_llm = select_database_id(
            user_query=processed_query,
            schema_and_samples=db_context_for_selection, # Pass the constructed context
            available_db_ids=available_db_ids_from_full_schema, # Pass original full list for validation
            llm_wrapper=llm_wrapper,
            conversation_history=conversation_history
        )
        
        # After LLM selects a DB, get its specific sample for SQL generation
        if selected_db_id_by_llm and db_samples_for_selection and selected_db_id_by_llm in db_samples_for_selection:
            sample_data_for_sql_gen = db_samples_for_selection[selected_db_id_by_llm]
        elif selected_db_id_by_llm:
            logging.warning(f"Sample for LLM-selected DB '{selected_db_id_by_llm}' not in db_samples_for_selection map (keys: {list(db_samples_for_selection.keys() if db_samples_for_selection else [])}).")
            # Fallback to trying with `conn` if it's for this selected DB (benchmark case)
            if conn:
                current_db_schema_dict = full_schema.get(selected_db_id_by_llm)
                if current_db_schema_dict:
                    logging.info(f"Attempting to fetch sample for '{selected_db_id_by_llm}' using provided 'conn'.")
                    s_parts_fallback = []
                    tables_to_sample_fb = list(current_db_schema_dict.keys())[:3]
                    for t_name_fb in tables_to_sample_fb:
                        s_data_fb = get_table_sample_data(conn, t_name_fb, limit=2) # This might fail if conn is not for selected_db_id_by_llm
                        if s_data_fb and "Error" not in s_data_fb: s_parts_fallback.append(f"-- Sample rows from table {t_name_fb}\n{s_data_fb}")
                    if s_parts_fallback: sample_data_for_sql_gen = "\n\n".join(s_parts_fallback)


    result["timing"]["db_selection"] = time.perf_counter() - start_time_select

    if not selected_db_id_by_llm:
        msg = "Could not determine the relevant table/database for your query via LLM selection."
        logging.error(msg)
        # If running in a benchmark where a correct DB exists, this is an error.
        # If running live, might offer general search or ask for clarification.
        return {**result, "status": "error_db_selection", "message": msg, "timing": {**result["timing"], "total": time.perf_counter() - start_time_total}}

    result["selected_db_id"] = selected_db_id_by_llm
    # Schema for the LLM-selected DB, for SQL generation
    schema_for_selected_db_sql_gen = {selected_db_id_by_llm: full_schema[selected_db_id_by_llm]}
    logging.info(f"LLM Selected DB ID: {selected_db_id_by_llm}")
    logging.debug(f"Sample for SQL Gen for '{selected_db_id_by_llm}':\n{sample_data_for_sql_gen[:300]}...")

    # --- 3. Refine Query & Determine Route (using selected context) ---
    start_time_refine = time.perf_counter()
    refinement_data = refine_and_select(
        user_query=processed_query, # Use language-processed query
        conversation_history=conversation_history,
        selected_db_schema=schema_for_selected_db_sql_gen, # Schema of the DB chosen by LLM
        sample_data_str=sample_data_for_sql_gen,           # Sample from the DB chosen by LLM
        llm_wrapper=llm_wrapper,
    )
    result["timing"]["refine_route"] = time.perf_counter() - start_time_refine

    result["refined_query"] = refinement_data.get("refined_query", processed_query)
    result["determined_route"] = refinement_data.get("route", "SQL").upper()
    result["augmented_keywords"] = refinement_data.get("augmented_keywords", [])
    logging.info(f"Refined Query='{result['refined_query']}', Route='{result['determined_route']}', Keywords={result['augmented_keywords']}")

    # --- 4. Generate SQL (if route is SQL) ---
    # Initialize results dicts for SQL and Semantic parts
    sql_pipeline_outcome = { "sql_success": False, "generated_sql": None, "sql_data": None, "sql_error": None }
    semantic_pipeline_outcome = { "semantic_success": False, "semantic_data": None, "semantic_error": None }


    if result["determined_route"] == "SQL":
        start_time_gen_sql = time.perf_counter()
        logging.info("Pipeline: Generating SQL Query...")
        # _generate_sql_query is called within execute_sql_pipeline, so no direct call here needed if using the full pipeline.
        # However, for benchmark_nl.py, we need generated_sql even if conn is None or if execution fails.
        # So, we might need a distinct generation step.
        # The current `process_natural_language_query` was structured to generate SQL first, then execute.
        
        # Let's keep the structure: generate SQL first, then decide on execution.
        # This `generated_sql_string` is the attempt before any retries from `execute_sql_pipeline`.
        generated_sql_string = _generate_sql_query(
            user_query=result["refined_query"],
            schema=schema_for_selected_db_sql_gen,
            sample_data_str=sample_data_for_sql_gen,
            aux_models=aux_models,
            augmented_keywords=result["augmented_keywords"]
        )
        sql_pipeline_outcome["generated_sql"] = generated_sql_string # Store initial generation attempt
        result["generated_sql"] = generated_sql_string # Update main result dict
        result["timing"]["sql_generation"] = time.perf_counter() - start_time_gen_sql

        if generated_sql_string.startswith("-- Error"):
            msg = f"SQL Generation Failed (Initial Attempt): {generated_sql_string}"
            logging.error(msg)
            result.update({"status": "error_sql_generation", "message": msg})
            # If conn is None (analysis only), we return here.
            if conn is None:
                result["timing"]["total"] = time.perf_counter() - start_time_total
                return result
            # If conn is present, execute_sql_pipeline will handle this error again, but this provides early exit insight.
            sql_pipeline_outcome["sql_error"] = msg # Store error for execute_sql_pipeline to potentially see
        else:
            logging.info("Pipeline: Initial SQL Generation successful.")
            result["status"] = "analysis_complete_sql_generated"
            result["message"] = "Analysis & SQL Generation complete. Execution pending or next."


    elif result["determined_route"] == "SEMANTIC":
        result["status"] = "analysis_complete_semantic_route"
        result["message"] = "Analysis complete, SEMANTIC route determined."
        logging.info("Pipeline: Semantic route determined.")
    else: # Invalid route
        msg = f"Analysis failed: Unknown route '{result['determined_route']}'."
        logging.error(msg)
        result.update({"status": "error_unknown_route", "message": msg})
        result["timing"]["total"] = time.perf_counter() - start_time_total
        return result

    # --- 5. Execute Pipelines & Summarize (Requires Connection for SQL exec or Semantic Search) ---
    if conn is None and result["determined_route"] == "SQL": # SQL generated but no conn to execute
        logging.info("DB connection is None. Skipping SQL execution phase. Returning generated SQL.")
        result["timing"]["total"] = time.perf_counter() - start_time_total
        return result
    if qdrant_client is None and result["determined_route"] == "SEMANTIC":
        logging.warning("Qdrant client is None. Skipping Semantic execution phase.")
        result.update({"status": "error_semantic_no_client", "message": "Semantic route chosen but Qdrant client unavailable."})
        result["timing"]["total"] = time.perf_counter() - start_time_total
        return result
    if conn is None and qdrant_client is None : # Nothing to execute
         logging.info("Neither DB connection nor Qdrant client available. Skipping execution.")
         result["timing"]["total"] = time.perf_counter() - start_time_total
         return result


    logging.info("Pipeline: Executing chosen primary pipeline...")
    start_time_exec = time.perf_counter()

    if result["determined_route"] == "SQL":
        # `conn` here is crucial. For benchmark_nl.py, it's the connection to the *actual* DB.
        # If `selected_db_id_by_llm` is different, `execute_sql_pipeline` will try to run SQL
        # (generated for `selected_db_id_by_llm`'s schema) on `conn` (for `actual_db_id`).
        # This will likely fail, which is the desired behavior to penalize wrong DB selection.
        sql_pipeline_outcome = execute_sql_pipeline(
            user_query=result["refined_query"],
            selected_db_schema=schema_for_selected_db_sql_gen, # Schema for LLM's choice
            sample_data_str=sample_data_for_sql_gen,         # Sample for LLM's choice
            aux_models=aux_models,
            conn=conn, # Connection (e.g., to actual_db_id in benchmark)
            llm_wrapper=llm_wrapper,
            augmented_keywords=result["augmented_keywords"],
            max_retries=max_retries
        )
    elif result["determined_route"] == "SEMANTIC":
        semantic_pipeline_outcome = execute_semantic_pipeline(
            result["refined_query"],
            aux_models,
            qdrant_client # Assumed to be ready if this route is taken seriously
        )

    # Execute secondary route if primary failed
    # Note: Pass the *results* from the primary attempt to run_secondary_route
    sql_final_result, semantic_final_result = run_secondary_route(
        user_query=result["refined_query"], # refined query for secondary route too
        primary_route=result["determined_route"],
        sql_result=sql_pipeline_outcome, # Pass primary SQL result
        semantic_result=semantic_pipeline_outcome, # Pass primary Semantic result
        aux_models=aux_models,
        conn=conn, # Use same conn for secondary SQL if needed
        selected_db_schema=schema_for_selected_db_sql_gen, # Schema of originally LLM-selected DB
        selected_db_sample=sample_data_for_sql_gen,       # Sample of originally LLM-selected DB
        llm_wrapper=llm_wrapper,
        qdrant_client=qdrant_client,
        augmented_keywords=result["augmented_keywords"]
    )
    result["timing"]["execution_pipelines"] = time.perf_counter() - start_time_exec

    # Update main result dict with final outcomes from execution pipelines
    result["sql_success"] = sql_final_result.get("sql_success", False)
    result["generated_sql"] = sql_final_result.get("generated_sql", result["generated_sql"]) # Keep latest SQL
    result["sql_data"] = sql_final_result.get("sql_data")
    result["sql_error"] = sql_final_result.get("sql_error")

    result["semantic_success"] = semantic_final_result.get("semantic_success", False)
    result["semantic_data"] = semantic_final_result.get("semantic_data")
    result["semantic_error"] = semantic_final_result.get("semantic_error")

    # --- 6. Generate Summary ---
    start_time_summary = time.perf_counter()
    if llm_wrapper and llm_wrapper.is_ready: # Only generate summary if LLM is available
        nl_summary = generate_final_summary(
            original_query,
            result["processed_query"],
            was_translated,
            result.get("sql_data"),
            result.get("semantic_data"),
            llm_wrapper
        )
        result["natural_language_summary"] = nl_summary
        if not result["sql_error"] and not result["semantic_error"]: # If no errors, message is summary
             result["message"] = nl_summary
        else: # Append errors to summary if any
             result["message"] = nl_summary + f"\n\nErrors encountered: SQL: {result['sql_error']}, Semantic: {result['semantic_error']}"
    else:
        result["natural_language_summary"] = "Summary LLM not available."
        result["message"] = "LLM for summary was not available. Raw results provided."
        if result.get("sql_data") is not None: result["message"] += f"\nSQL Data: Present"
        if result.get("semantic_data") is not None: result["message"] += f"\nSemantic Data: Present"


    result["timing"]["summary_generation"] = time.perf_counter() - start_time_summary

    # Determine final status
    if result["sql_success"] or result["semantic_success"]:
        result["status"] = "success"
    elif result["sql_error"] or result["semantic_error"]: # Check specific errors from execution
        result["status"] = "error_execution"
        # Message already updated above
    elif result.get("status") not in ["error_db_selection", "error_sql_generation", "error_unknown_route", "error_semantic_no_client"]:
        # If no specific error status set earlier, and no success, then it's no results found.
        result["status"] = "success_no_results"
        if not result["message"] or result["message"] == "Starting query processing...": # Ensure message is updated
            result["message"] = "Query processed, but no specific data found matching your request."


    result["raw_display_data"] = {"sql": result.get("sql_data"), "semantic": result.get("semantic_data")}
    result["timing"]["total"] = time.perf_counter() - start_time_total
    logging.info(f"Query processing finished. Status: {result['status']}. Total time: {result['timing']['total']:.2f}s")
    return result


# --- Data Processing Function (Handles Upload/GSheet -> SQLite -> Qdrant) ---
def process_uploaded_data(uploaded_file, gsheet_published_url, table_name, conn, llm_wrapper, aux_models, qdrant_client, replace_confirmed=False):
    """Reads data, writes to SQLite, analyzes, embeds, indexes."""
    # (Implementation mostly the same as provided - includes robust column cleaning and detailed logging)
    # Ensure it uses the MODIFIED _suggest_semantic_columns and _embed_and_index_data
    # which now expect the table_name/db_id for context/collection naming.

    # --- Initial checks ---
    start_time = time.time()
    logging.info(f"[{table_name}] Starting data processing...")
    if not conn: return False, "Database connection lost."
    if not aux_models or aux_models.get('status') != 'loaded': return False, "Models not loaded."
    if not qdrant_client: return False, "Qdrant client not available."
    if not table_name or not table_name.isidentifier(): return False, f"Invalid table name: '{table_name}'."
    if not uploaded_file and not gsheet_published_url: return False, "No data source provided."
    if uploaded_file and gsheet_published_url: return False, "Provide only one data source."

    status_messages = []
    df = None
    error_msg = None

    table_already_exists = table_exists(conn, table_name)
    if table_already_exists and not replace_confirmed:
        return False, f"Table `{table_name}` already exists. Confirmation required to replace."
    elif table_already_exists:
        status_messages.append(f"Replacing existing table `{table_name}` as confirmed.")

    try:
        # --- Read Data ---
        status_messages.append(f"Processing data for table: `{table_name}`...")
        read_success = False
        if uploaded_file:
            status_messages.append(f"Reading file: {uploaded_file.name}...")
            try:
                # Try Excel first, fallback to CSV
                if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                     df = pd.read_excel(uploaded_file)
                else: # Assume CSV or similar text format
                     df = pd.read_csv(uploaded_file)
                status_messages.append(f"File read. Shape: {df.shape}")
                read_success = True
            except Exception as e:
                error_msg = f"Error reading file '{uploaded_file.name}': {e}"; logging.error(error_msg, exc_info=True); st.error(error_msg)
        elif gsheet_published_url:
            status_messages.append(f"Attempting to read Published Google Sheet URL...")
            try:
                df = read_google_sheet(gsheet_published_url)
                status_messages.append(f"Read {len(df)} rows from URL. Shape: {df.shape}")
                read_success = True
            except ValueError as ve:
                error_msg = f"Error reading Google Sheet: {ve}"; logging.error(error_msg); st.error(error_msg)
            except Exception as e:
                 error_msg = f"Unexpected error reading Google Sheet URL: {e}"; logging.error(error_msg, exc_info=True); st.error(error_msg)

        if not read_success or df is None or df.empty:
            final_error = error_msg if error_msg else "Failed to read data source or data source is empty."
            return False, "\n".join(status_messages + [final_error])

        initial_shape = df.shape
        original_columns = df.columns.tolist()
        expected_columns = len(original_columns)
        status_messages.append(f"Initial DataFrame shape: {initial_shape}. Columns: {expected_columns}")
        logging.debug(f"[{table_name}] Initial columns read ({expected_columns}): {original_columns}")

        # --- Column Sanitization ---
        status_messages.append("Sanitizing column names...")
        sanitized_columns = []
        col_map = {} # Store original -> sanitized mapping
        try:
            # Use Counter for efficient duplicate detection *after* initial sanitization
            temp_sanitized = []
            for i, col in enumerate(original_columns):
                col_str = str(col).strip() if pd.notna(col) else f"col_{i}_unnamed"
                if not col_str: col_str = f"col_{i}_blank" # Handle empty string columns
                # Aggressive sanitization, make lowercase first
                sanitized = col_str.lower()
                sanitized = re.sub(r'\s+', '_', sanitized) # Replace spaces with underscores
                sanitized = re.sub(r'[^\w_]', '', sanitized) # Keep only word chars and underscores
                sanitized = re.sub(r'^[^a-zA-Z_]+', '', sanitized) # Remove leading non-alpha/underscore
                if not sanitized: sanitized = f'col_{i}' # If all chars removed
                if not sanitized[0].isalpha() and sanitized[0] != '_': sanitized = '_' + sanitized # Ensure starts with letter or _
                # Check against SQL keywords
                reserved_sql = {'select', 'from', 'where', 'index', 'table', 'insert', 'update', 'delete', 'order', 'group', 'alter', 'drop', 'create', 'primary', 'foreign', 'key', 'add', 'all', 'and', 'or', 'not', 'null', 'is', 'in', 'like', 'between', 'exists', 'case', 'when', 'then', 'else', 'end', 'join', 'on', 'left', 'right', 'inner', 'outer', 'full', 'cross', 'union', 'distinct', 'limit', 'offset', 'as', 'asc', 'desc', 'by', 'having', 'set', 'values', 'default', 'constraint', 'check', 'unique', 'references', 'with'}
                if sanitized.lower() in reserved_sql: sanitized = f'_{sanitized}' # Prepend underscore if keyword
                if not sanitized.isidentifier(): # Final check if still not valid identifier
                    safe_original = ''.join(c for c in col_str if c.isalnum() or c == '_')[:20]
                    sanitized = f'col_{i}_{safe_original}'
                    if not sanitized.isidentifier(): sanitized = f'col_{i}' # Absolute fallback
                temp_sanitized.append(sanitized)
                col_map[str(col)] = sanitized # Map original to temporary sanitized

            # Handle duplicates by appending sequence number
            counts = Counter(temp_sanitized)
            final_sanitized = []
            seen_counts = Counter()
            for san_col in temp_sanitized:
                 if counts[san_col] > 1:
                     suffix_num = seen_counts[san_col] + 1
                     final_sanitized.append(f"{san_col}_{suffix_num}")
                     seen_counts[san_col] += 1
                 else:
                     final_sanitized.append(san_col)

            if len(final_sanitized) != expected_columns:
                error_msg = f"[{table_name}] CRITICAL SANITIZE ERROR: Length mismatch AFTER sanitizing. Expected {expected_columns}, Got {len(final_sanitized)}."
                logging.error(error_msg); logging.error(f"Original: {original_columns}, Temp Sanitized: {temp_sanitized}, Final: {final_sanitized}")
                st.error(error_msg)
                return False, "\n".join(status_messages + [error_msg])
            else:
                df.columns = final_sanitized
                status_messages.append("Column names sanitized.")
                # Log changes if any occurred
                if original_columns != final_sanitized:
                     changed_cols_log = "\n".join([f"  - '{orig}' -> '{san}'" for orig, san in zip(original_columns, final_sanitized) if orig != san])
                     logging.info(f"[{table_name}] Column names adjusted:\n{changed_cols_log}")
                     st.info("Note: Some column names were adjusted for compatibility.")

        except Exception as sanitize_e:
            error_msg = f"Error during column name sanitization: {sanitize_e}"
            logging.error(error_msg, exc_info=True); st.error(error_msg)
            return False, "\n".join(status_messages + [error_msg])

        # --- Check DataFrame state BEFORE writing ---
        current_shape = df.shape
        current_columns_count = len(df.columns)
        status_messages.append(f"DataFrame shape before write: {current_shape}. Columns: {current_columns_count}")
        logging.debug(f"[{table_name}] DEBUG: Columns before write ({current_columns_count}): {df.columns.tolist()}")
        if current_columns_count != expected_columns:
             error_msg = f"[{table_name}] CRITICAL ERROR: Column count changed unexpectedly before write. Expected {expected_columns}, Found {current_columns_count}."
             logging.error(error_msg); st.error(error_msg)
             return False, "\n".join(status_messages + [error_msg])

        # --- Write to SQLite ---
        status_messages.append(f"Writing data to SQLite table `{table_name}` (replace)...")
        write_successful = False
        try:
            with conn: # Use context manager for transaction
                conn.execute(f"PRAGMA busy_timeout = {SQLITE_TIMEOUT_SECONDS * 1000};")
                # Quote table name in DROP and during write
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                df.to_sql(table_name, conn, if_exists='replace', index=False) # to_sql handles quoting internally if needed
            status_messages.append("Data written successfully to SQLite.")
            write_successful = True
        except sqlite3.OperationalError as e: error_msg = f"SQLite error: {'Database locked' if 'locked' in str(e) else e}"; logging.error(f"[{table_name}] {error_msg}"); st.error(error_msg)
        except ValueError as e: error_msg = f"ValueError writing data (check types/lengths): {e}"; logging.error(f"[{table_name}] {error_msg}"); st.error(error_msg)
        except Exception as e: error_msg = f"Unexpected error writing data: {e}"; logging.error(f"[{table_name}] {error_msg}", exc_info=True); st.error(error_msg)

        if write_successful:
            # --- Analyze, Embed, Index ---
            status_messages.append("Analyzing columns for embedding (using LLM)...")
            # Pass the schema for the specific table being processed
            single_table_schema = {table_name: df.columns.tolist()}
            semantic_cols_to_embed = _suggest_semantic_columns(df.head(), single_table_schema, table_name, llm_wrapper)
            status_messages.append(f"LLM suggested for embedding: {semantic_cols_to_embed}")

            if not semantic_cols_to_embed:
                 status_messages.append("Embedding skipped: No columns suggested or available.")
                 embed_success, embed_msg = True, "Embedding skipped: No columns suggested." # Consider this a success if no columns needed embedding
            else:
                 status_messages.append("Starting embedding & indexing...")
                 embed_success, embed_msg = _embed_and_index_data(df, table_name, semantic_cols_to_embed, aux_models, qdrant_client)
                 status_messages.append(f"Embedding/Indexing result: {embed_msg}")

            if embed_success:
                processing_time = time.time() - start_time
                status_messages.append(f"Processing complete for `{table_name}` in {processing_time:.2f} seconds.")
                return True, "\n".join(status_messages)
            else:
                 status_messages.append(f"SQL write OK, but embedding/indexing failed.")
                 return False, "\n".join(status_messages)
        else: # Write failed
             return False, "\n".join(status_messages + [f"Failed during SQLite write: {error_msg}"])

    except Exception as e:
        critical_error_msg = f"Critical error during data processing for '{table_name}': {type(e).__name__} - {e}"
        st.error(critical_error_msg); logging.error(critical_error_msg, exc_info=True)
        return False, critical_error_msg


# --- Management Functions ---
def reindex_table(conn: sqlite3.Connection, table_name: str, llm_wrapper: LLMWrapper, aux_models: dict, qdrant_client: QdrantClient) -> tuple[bool, str]:
    """Reads data from SQLite, re-suggests columns, re-embeds, re-indexes."""
    # (Implementation remains the same as provided, uses updated helpers)
    logging.info(f"[{table_name}] Attempting to re-index...")
    if not conn: return False, "DB connection unavailable."
    if not table_exists(conn, table_name): return False, f"SQLite table '{table_name}' not found."
    if not qdrant_client: return False, "Qdrant client unavailable."
    if not aux_models or aux_models.get('status') != 'loaded': return False, "Aux models not loaded."
    if not llm_wrapper or not llm_wrapper.is_ready: logging.warning(f"[{table_name}] LLM wrapper not ready for re-suggesting columns.")

    try:
        logging.info(f"[{table_name}] Reading data from SQLite for re-indexing...")
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn) # Quote table name
        if df.empty: return True, f"Table '{table_name}' is empty, re-indexing skipped." # Return True, empty is not error
        logging.info(f"[{table_name}] Read {len(df)} rows.")

        single_table_schema = {table_name: df.columns.tolist()}
        semantic_cols_to_embed = _suggest_semantic_columns(df.head(), single_table_schema, table_name, llm_wrapper)
        if not semantic_cols_to_embed:
             return True, "Re-embedding skipped: No semantic columns identified or suggested." # Success if no columns

        logging.info(f"[{table_name}] Re-embedding with columns: {semantic_cols_to_embed}")
        embed_success, embed_msg = _embed_and_index_data(df, table_name, semantic_cols_to_embed, aux_models, qdrant_client)

        if embed_success: msg = f"Successfully re-indexed '{table_name}'. {embed_msg}"; logging.info(msg); return True, msg
        else: msg = f"Re-indexing failed for '{table_name}'. Reason: {embed_msg}"; logging.error(msg); return False, msg

    except Exception as e:
        logging.error(f"Unexpected error during re-indexing of '{table_name}': {e}", exc_info=True)
        return False, f"Unexpected error during re-indexing: {e}"


def delete_table_data(conn: sqlite3.Connection, table_name: str, qdrant_client: QdrantClient) -> tuple[bool, str]:
    """Deletes table from SQLite and corresponding collection from Qdrant."""
    # (Implementation remains the same as provided)
    logging.info(f"[{table_name}] Attempting to delete data...")
    if not conn: return False, "DB connection unavailable."
    if not qdrant_client: return False, "Qdrant client unavailable."

    sqlite_success, qdrant_success = False, False
    sqlite_msg, qdrant_msg = "Not Attempted", "Not Attempted"

    # Delete SQLite table
    try:
        with conn: conn.execute(f'DROP TABLE IF EXISTS "{table_name}"') # Quote table name
        sqlite_success = True; sqlite_msg = f"SQLite table '{table_name}' dropped."
        logging.info(sqlite_msg)
    except Exception as e:
        sqlite_msg = f"Error dropping SQLite table '{table_name}': {e}"
        logging.error(sqlite_msg, exc_info=True)

    # Delete Qdrant collection
    collection_name = f"{QDRANT_COLLECTION_PREFIX}{table_name}"
    try:
        try: qdrant_client.get_collection(collection_name=collection_name); collection_exists = True
        except Exception: collection_exists = False; qdrant_msg = f"Qdrant collection '{collection_name}' did not exist."; qdrant_success = True # Not existing isn't failure
        if collection_exists:
            qdrant_client.delete_collection(collection_name=collection_name, timeout=10)
            qdrant_success = True; qdrant_msg = f"Qdrant collection '{collection_name}' deleted."
            logging.info(qdrant_msg)
    except Exception as e:
        qdrant_msg = f"Error deleting Qdrant collection '{collection_name}': {e}"
        logging.error(qdrant_msg, exc_info=True)

    overall_success = sqlite_success and qdrant_success
    final_message = f"SQLite: {sqlite_msg} | Qdrant: {qdrant_msg}"
    return overall_success, final_message


def derive_requirements_from_history(conversation_history: list[dict], llm_wrapper: LLMWrapper, max_turns: int = 10) -> str:
    """
    Takes conversation history and asks LLM to summarize requirements.
    Uses prompt from PROMPTS.
    """
    if not conversation_history: return "" # No history, no requirements derived
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for deriving requirements.")
        return "" # Cannot derive if LLM unavailable

    recent_turns = conversation_history[-max_turns:]
    context = "\n".join(f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}" for msg in recent_turns)

    # --- Use loaded prompt ---
    prompt = PROMPTS['derive_requirements_from_history'].format(context=context)
    # --- End Use loaded prompt ---

    try:
        refined_requirements = llm_wrapper.generate_response(prompt, max_tokens=150, temperature=0.3)
        logging.info(f"Derived requirements from history: {refined_requirements.strip()}")
        return refined_requirements.strip()
    except Exception as e:
        logging.error(f"Error deriving requirements from history: {e}", exc_info=True)
        return "" # Return empty string on error

