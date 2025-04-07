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
# --- NEW: Load environment variables from .env file ---

dotenv.load_dotenv('.env', override=True) # Searches for .env file in current dir or parent dirs and loads variables

# --- Imports for Google Auth ---
# from google.colab import userdata # NO LONGER NEEDED
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import pickle
import gspread # Moved import here to ensure it's available for read_google_sheet
import logging
import traceback # For detailed error logging in query processing

# Import the new LLM wrapper and aux model loader
from llm_interface import LLMWrapper
from model import load_aux_models # Renamed import


SQLITE_TIMEOUT_SECONDS = 15 # Increased timeout (default is 5)
SHEETS_SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
ALL_SCOPES = SHEETS_SCOPES + DRIVE_SCOPES
QDRANT_COLLECTION_PREFIX = "table_data_"

# --- NEW: Environment Setup Function ---
def setup_environment():
    """
    Determines DB path based on DEVELOPMENT_MODE env variable.
    Creates the necessary directories.
    Returns the absolute database path or None on failure.
    """
    # Check DEVELOPMENT_MODE environment variable (default to False)
    # Accepts true, 1, t, yes, y (case-insensitive)
    dev_mode_str = os.getenv('DEVELOPMENT_MODE', 'false').lower()
    is_dev_mode = dev_mode_str in ('true', '1', 't', 'yes', 'y')

    app_data_dir = None
    db_filename = "chat_database.db"

    try:
        if is_dev_mode:
            # Development mode (e.g., WSL): Use user's home directory
            home_dir = os.path.expanduser("~")
            # Use a hidden directory for tidiness
            app_data_dir = os.path.join(home_dir, ".streamlit_chat_dev_data")
            print("INFO: Running in DEVELOPMENT mode.")
        else:
            # Production mode: Use a directory relative to the project root
            # Assuming app.py is run from the project root directory
            # Place data in a subdirectory named 'app_data'
            current_working_dir = os.getcwd() # Where streamlit run was executed
            app_data_dir = os.path.join(current_working_dir, "app_data")
            print("INFO: Running in PRODUCTION mode.")

        print(f"INFO: App data directory set to: {app_data_dir}")

        # Ensure the directory exists
        os.makedirs(app_data_dir, exist_ok=True)
        print(f"INFO: Ensured app data directory exists.")

        # Construct the full database path
        db_path = os.path.join(app_data_dir, db_filename)
        print(f"INFO: Database path set to: {db_path}")
        return db_path

    except OSError as e:
        print(f"ERROR: Failed to create data directory '{app_data_dir}': {e}. Check permissions.")
        # Cannot proceed without data directory
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during environment setup: {e}")
        return None

# read data from published Google Sheet
def read_google_sheet(sheet_url):
    """
    Reads data from a published Google Sheet URL.
    Returns a DataFrame.
    """
    # Check if the URL is valid
    if not sheet_url.startswith("https://docs.google.com/spreadsheets/d/"):
        raise ValueError("Invalid Google Sheet URL. Ensure it's a published link.")

    # Extract the sheet ID from the URL
    sheet_id = sheet_url.split("/d/")[1].split("/")[0]
    
    # Construct the CSV export URL
    url= f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    # Read the CSV data from the URL
    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise ValueError(f"Error reading Google Sheet: {e}")
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The Google Sheet is empty or not accessible.")
    # Optionally, you can sanitize the DataFrame columns here
    # Example: df.columns = [col.replace(' ', '_') for col in df.columns]
    # Return the DataFrame

    
    return df


# --- Helper to check if table exists ---
def table_exists(conn, table_name):
    """Checks if a table exists in the SQLite database."""
    if not conn or not table_name: return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        exists = cursor.fetchone() is not None
        cursor.close(); return exists
    except Exception: return False

# ---Cached LLM Wrapper Initialization ---
@st.cache_resource
def get_llm_wrapper():
    """Initializes and caches the LLMWrapper."""
    logging.info("Attempting to get/initialize LLMWrapper...")
    wrapper = LLMWrapper()
    if not wrapper.is_ready:
        logging.error("LLMWrapper initialization failed.")
        # Return wrapper anyway, app.py should check wrapper.is_ready
    return wrapper

# --- Cached Auxiliary Model Loading ---

@st.cache_resource
def get_cached_aux_models():
    """Loads and caches auxiliary models (SQL, Embedder)."""
    logging.info("Cache miss or first run: Calling load_aux_models...")
    models_dict_from_loader = load_aux_models() # Call function from model.py

    # --- Add Type and Value Debugging ---
    print(f"DEBUG [utils.py]: Type returned by load_aux_models: {type(models_dict_from_loader)}")
    print(f"DEBUG [utils.py]: Value returned by load_aux_models: {models_dict_from_loader}")
    # --- End Debugging ---

    # Explicit check if it's NOT a dict
    if not isinstance(models_dict_from_loader, dict):
        logging.error("CRITICAL [utils.py]: load_aux_models did NOT return a dictionary!")
        # Return a consistent error dictionary structure
        return {"status": "error", "error_message": f"Internal Error: Model loader returned type {type(models_dict_from_loader)} instead of dict."}

    # Check status within the dictionary
    if models_dict_from_loader.get("status") != "loaded":
         logging.warning(f"Auxiliary model loading status: {models_dict_from_loader.get('status')}. Error: {models_dict_from_loader.get('error_message', 'N/A')}")
         # Return the dictionary containing the error status
         return models_dict_from_loader
    else:
        logging.info("Aux models loaded and cached successfully.")
        return models_dict_from_loader # Return the valid dictionary


# --- Helper function to suggest semantic columns ---
def _suggest_semantic_columns(df_head: pd.DataFrame, schema: dict, table_name: str, llm_wrapper: LLMWrapper) -> list[str]:
    """
    Uses the LLM to suggest columns suitable for semantic search embedding.
    """
    logging.info(f"Requesting LLM suggestion for semantic columns in table '{table_name}'...")
    # Basic check if wrapper is ready
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for semantic column suggestion. Falling back to simple text column selection.")
        # Fallback: Select object/string columns (same as placeholder)
        return df_head.select_dtypes(include=['object', 'string']).columns.tolist()

    # Prepare prompt for the LLM
    schema_str = json.dumps(schema.get(table_name, "Schema unavailable"), indent=2)
    # Get df.head() as string, limit rows/columns shown if needed
    df_head_str = df_head.to_string(index=False, max_rows=5) # Show first 5 rows

    prompt = f"""Analyze the following table schema and sample data to identify columns containing free-form text suitable for semantic search (embedding). Semantic search is used for finding similar descriptions, understanding meaning, or searching based on concepts rather than exact values.

            Table Name: {table_name}

            Schema (Columns and Types - Inferred):
            {schema_str}

            Sample Data (First 5 Rows):
            {df_head_str}

            Identify the column names from the schema list that contain natural language text, descriptions, notes, or comments that would be valuable for semantic search. Exclude IDs, numerical values, dates, codes, categories (unless the categories themselves have descriptive names), or URLs unless the URL text itself is descriptive.

            Respond ONLY with a valid JSON list of strings containing the suggested column names. Example: ["product_description", "customer_review_text"]
            If no columns seem suitable, respond with an empty JSON list: []

            Suggested Columns JSON:""" # Let the LLM complete this

    logging.debug(f"Sending semantic column suggestion prompt to LLM:\n{prompt}")

    suggested_columns = []
    raw_llm_output = None # Store raw output for debugging
    try:
        # Use structured output if available (preferred)
        if llm_wrapper.mode == 'openrouter':
            structured_result = llm_wrapper.generate_structured_response(prompt)
            if structured_result and isinstance(structured_result, list):
                 # Basic validation: Are items strings?
                 if all(isinstance(item, str) for item in structured_result):
                      suggested_columns = structured_result
                      logging.info(f"LLM suggested (structured): {suggested_columns}")
                 else:
                      logging.warning(f"LLM structured response was a list, but contained non-strings: {structured_result}. Falling back.")
            else:
                logging.warning(f"LLM structured response failed or returned non-list: {structured_result}. Attempting text fallback.")
                # Attempt text generation as fallback if structured failed
                response_text = llm_wrapper.generate_response(prompt + " []", max_tokens=150) # Add default if empty
                try:
                    # Try parsing the text response as JSON list
                    parsed_list = json.loads(response_text)
                    if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                        suggested_columns = parsed_list
                        logging.info(f"LLM suggested (text fallback parsed): {suggested_columns}")
                    else:
                         logging.warning(f"LLM text response could not be parsed into a valid list of strings: {response_text}")
                except json.JSONDecodeError:
                     logging.warning(f"LLM text response was not valid JSON: {response_text}")

        # If not OpenRouter or structured failed, use standard text generation
        if not suggested_columns:
             # Add default example to prompt for text generation
             response_text = llm_wrapper.generate_response(prompt + ' []', max_tokens=150)
             raw_llm_output = response_text # Store text output
             logging.debug(f"Raw text suggestion response: {response_text}")
             # *** Clean Markdown before parsing ***
             cleaned_text = response_text.strip()
             if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
             if cleaned_text.startswith("```"): cleaned_text = cleaned_text[3:]
             if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
             cleaned_text = cleaned_text.strip()

             try:
                 parsed_list = json.loads(cleaned_text)
                 if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                     suggested_columns = parsed_list
                     logging.info(f"LLM suggested (parsed from text): {suggested_columns}")
                 else: logging.warning(f"Parsed text response was not a valid list of strings: {parsed_list}")
             except json.JSONDecodeError:
                  logging.warning(f"Could not parse LLM text response as JSON: '{cleaned_text}'")

    except Exception as e:
        logging.error(f"Error during LLM semantic column suggestion: {e}", exc_info=True)
        # Fallback on error

    # Final fallback if LLM fails completely
    if not suggested_columns:
        logging.warning("LLM suggestion failed. Falling back to selecting all object/string columns.")
        suggested_columns = df_head.select_dtypes(include=['object', 'string']).columns.tolist()

    # Validate suggested columns against actual df columns
    final_columns = [col for col in suggested_columns if col in df_head.columns]
    if len(final_columns) != len(suggested_columns):
         logging.warning(f"LLM suggested columns not present in DataFrame were filtered out. Original: {suggested_columns}, Final: {final_columns}")

    logging.info(f"Final semantic columns selected for embedding: {final_columns}")
    return final_columns

# --- Placeholders for embedding ---
def _analyze_columns_for_embedding(df, table_name, models): # Placeholder
    print(f"Simulating analysis for {table_name}")
    time.sleep(0.1); return df.select_dtypes(include=['object']).columns[:1].tolist()


def _embed_and_index_data(df: pd.DataFrame, table_name: str, semantic_columns: list, aux_models: dict, qdrant_client: QdrantClient):
    """
    Embeds specified semantic columns and indexes them in Qdrant.
    """
    # --- Readiness Checks ---
    if not qdrant_client:
        logging.error("Qdrant client not available for indexing.")
        return False, "Qdrant client not initialized."
    if not aux_models or aux_models.get('status') in ['error', 'partial']:
        if not aux_models or not aux_models.get('embedding_model'):
             logging.error("Embedding model not available for indexing.")
             return False, "Embedding model not loaded."
        if aux_models.get('status') == 'error':
             return False, "Aux models failed." # Fail if overall status is error

    embedding_model = aux_models.get('embedding_model')
    if not embedding_model:
         return False, "Embedding model object missing." # Should be caught above, but double-check

    if df.empty:
        logging.info(f"DataFrame for table '{table_name}' is empty. Skipping embedding.")
        return True, "DataFrame is empty." # Not an error
    if not semantic_columns:
        logging.info(f"No semantic columns identified for table '{table_name}'. Skipping embedding.")
        return True, "No semantic columns specified." # Not an error

    # Ensure semantic columns actually exist in the dataframe
    valid_semantic_cols = [col for col in semantic_columns if col in df.columns]
    if not valid_semantic_cols:
        logging.warning(f"Specified semantic columns {semantic_columns} not found in DataFrame for table '{table_name}'. Skipping.")
        return True, "Specified semantic columns not found in DataFrame."

    logging.info(f"Starting embedding and indexing for table '{table_name}', columns: {valid_semantic_cols}")
    collection_name = f"{QDRANT_COLLECTION_PREFIX}{table_name}"
    status_messages = []
    is_in_memory = qdrant_client.connection_params.host is None and qdrant_client.connection_params.port is None and qdrant_client.connection_params.path == ":memory:"
    if is_in_memory:
        logging.info("Qdrant client is in-memory: Point IDs will be generated as UUIDs.")
    else:
        logging.info("Qdrant client is on-disk/remote: Point IDs will attempt to use primary key.")

    try:
        # --- Prepare Text Documents for Embedding ---
        # Combine text from semantic columns for each row
        # Also, identify a reliable ID for each row (primary key if available, otherwise index)
        documents_to_embed = []
        payloads = []
        point_ids = []

        # Try to guess a primary key (often 'id', 'ID', or first column) for better payload linking
        potential_pk = None
        if 'id' in df.columns.str.lower():
             potential_pk = df.columns[df.columns.str.lower() == 'id'][0]
        elif df.columns[0].lower() in ['id', 'pk', 'key']:
             potential_pk = df.columns[0]
        logging.info(f"Potential primary key column identified: {potential_pk}")

        for index, row in df.iterrows():
            # Create text by concatenating valid semantic columns
            text_parts = [f"{col}: {row[col]}" for col in valid_semantic_cols if pd.notna(row[col]) and str(row[col]).strip()]
            if not text_parts:
                continue # Skip rows with no text in semantic columns

            doc_text = " | ".join(text_parts)
            documents_to_embed.append(doc_text)
                        # *** ID Generation Logic ***
            point_id_value = None # This will be the UUID if needed
            original_id_value = None # Store the potentially non-UUID ID here

            if potential_pk and potential_pk in row and pd.notna(row[potential_pk]):
                original_id_value = f"{table_name}_{row[potential_pk]}"
            else:
                original_id_value = f"{table_name}_idx_{index}" # Fallback original ID using index

            payload["_original_id"] = original_id_value # Store original ID in payload

            if is_in_memory:
                point_id_value = str(uuid.uuid4()) # Generate UUID for in-memory
            else:
                
                point_id_value = original_id_value
            if point_id_value is None: # Should not happen with UUID fallback, but safety check
                logging.warning(f"Failed to generate point ID for row index {index}. Skipping.")
                continue

            point_ids.append(point_id_value)
            payloads.append(payload)
            documents_to_embed.append(doc_text) # Append doc_text only if ID/payload were successful

            # Create payload - store original data for retrieval
            # Convert row to dict, handle potential non-serializable types if needed
            payload = row.to_dict()
            # Ensure payload values are JSON serializable (Qdrant requirement)
            for k, v in payload.items():
                 if pd.isna(v):
                     payload[k] = None # Replace Pandas NaT/NaN with None
                 # Add checks for other types like Timestamps if necessary
                 # elif isinstance(v, pd.Timestamp): payload[k] = v.isoformat()

            # Add metadata to payload
            payload["_table_name"] = table_name
            payload["_source_text"] = doc_text # Store the text that was embedded

            payloads.append(payload)

            # Generate Point ID: Use potential primary key or fallback to index/UUID
            if potential_pk and potential_pk in row and pd.notna(row[potential_pk]):
                 # Prefix with table name for potential cross-table uniqueness if PKs overlap
                 point_id = f"{table_name}_{row[potential_pk]}"
            else:
                 # Fallback to using DataFrame index (can clash if multiple uploads use index)
                 # point_id = index
                 # Safer fallback: UUID - ensures uniqueness but less meaningful
                 point_id = str(uuid.uuid4())
            point_ids.append(point_id)

        if not documents_to_embed:
            logging.info(f"No documents found to embed for table '{table_name}'.")
            return True, "No non-empty documents to embed."

        # --- Get Embeddings ---
        logging.info(f"Generating {len(documents_to_embed)} embeddings...")
        # Make sure embedding model handles lists correctly and returns list/np.array
        embeddings = list(embedding_model.embed(documents_to_embed))
        if not embeddings or len(embeddings) != len(documents_to_embed):
            logging.error("Embedding generation failed or returned incorrect number of vectors.")
            return False, "Embedding generation failed."
        logging.info("Embeddings generated.")

        # --- Interact with Qdrant ---
        vector_size = len(embeddings[0]) # Dynamically get vector size
        logging.info(f"Vector size: {vector_size}")

        # Check if collection exists and recreate it (simple strategy)
        try:
            # Note: QdrantClient methods might raise specific exceptions
            qdrant_client.get_collection(collection_name=collection_name)
            logging.warning(f"Qdrant collection '{collection_name}' already exists. Recreating.")
            qdrant_client.delete_collection(collection_name=collection_name, timeout=10)
            status_messages.append(f"Recreated collection '{collection_name}'.")
            # Wait briefly after delete before create
            time.sleep(1)
        except Exception as get_coll_e:
             # Handle case where collection doesn't exist (expected path for first time)
             # Check specific exception type if possible (e.g., qdrant_client.http.exceptions.UnexpectedResponse)
             logging.info(f"Qdrant collection '{collection_name}' not found or error checking: {get_coll_e}. Creating new.")
             status_messages.append(f"Creating collection '{collection_name}'.")


        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE) # Or DOT / EUCLID
        )
        logging.info(f"Qdrant collection '{collection_name}' created/ensured.")

        # Upsert data in batches (Qdrant recommendation)
        batch_size = 100 # Adjust batch size based on data size/memory
        num_batches = (len(point_ids) + batch_size - 1) // batch_size
        logging.info(f"Upserting {len(point_ids)} points to Qdrant in {num_batches} batches...")

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(point_ids))
            batch_ids = point_ids[start_idx:end_idx]
            batch_vectors = embeddings[start_idx:end_idx]
            batch_payloads = payloads[start_idx:end_idx]

            logging.debug(f"Upserting batch {i+1}/{num_batches} (size {len(batch_ids)})")
            qdrant_client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    payloads=batch_payloads
                ),
                wait=True # Wait for operation to complete for more predictable results
            )
        logging.info("Qdrant upsert complete.")
        status_messages.append(f"Indexed {len(point_ids)} points.")
        return True, "\n".join(status_messages)

    except Exception as e:
        logging.error(f"Error during embedding/indexing for table {table_name}: {e}", exc_info=True)
        return False, f"Error during indexing: {e}"


# --- REVISED process_uploaded_data ---
def process_uploaded_data(uploaded_file, gsheet_published_url, table_name, conn, llm_model, aux_models, qdrant_client, replace_confirmed=False):
    """
    Reads data, writes to SQLite (with confirmation), analyzes, embeds, indexes.
    Includes detailed debugging for Length mismatch errors.
    """
    # --- Initial checks ---
    if not conn: return False, "Database connection lost."
    if not models: return False, "Models not loaded."
    if not qdrant_client: return False, "Qdrant client not available."
    if not table_name or not table_name.isidentifier(): return False, f"Invalid table name: '{table_name}'."
    if not uploaded_file and not gsheet_published_url: return False, "No data source provided."
    if uploaded_file and gsheet_published_url: return False, "Provide only one data source."

    status_messages = []
    df = None
    error_msg = None
    expected_columns = 0 # To track expected column count

    # --- Check if table exists BEFORE reading data ---
    table_already_exists = table_exists(conn, table_name)
    if table_already_exists and not replace_confirmed:
        return False, f"Table `{table_name}` already exists. Confirmation required to replace."

    try:
        # --- Read Data ---
        status_messages.append(f"Processing data for table: `{table_name}`...")
        read_success = False
        if uploaded_file:
            status_messages.append(f"Reading Excel file: {uploaded_file.name}...")
            try:
                df = pd.read_excel(uploaded_file)
                status_messages.append(f"Excel file read. Shape: {df.shape}")
                read_success = True
            except Exception as e:
                error_msg = f"Error reading Excel file: {e}"
                st.error(error_msg)
        elif gsheet_published_url:
            status_messages.append(f"Attempting to read Published Google Sheet CSV URL...")
            
            try:
                df = read_google_sheet(gsheet_published_url)
                status_messages.append(f"Read {len(df)} rows from URL. Shape: {df.shape}")
                read_success = True
            except ValueError as ve:
                error_msg = f"Error reading Google Sheet: {ve}"
                st.error(error_msg)


        if not read_success or df is None:
            final_error = error_msg if error_msg else "Failed to read data source or data source is empty."
            return False, "\n".join(status_messages + [final_error])

        # --- Store initial shape and columns for debugging ---
        initial_shape = df.shape
        original_columns = df.columns.tolist()
        expected_columns = len(original_columns)
        status_messages.append(f"Initial DataFrame shape: {initial_shape}. Columns: {expected_columns}")
        print(f"DEBUG: Initial columns read ({expected_columns}): {original_columns}")

        # --- Column Sanitization (Robust Implementation) ---
        status_messages.append("Sanitizing column names for SQL compatibility...")
        sanitized_columns = []
        sanitization_errors = []
        try:
            for i, col in enumerate(original_columns):
                col_str = str(col).strip() # Convert to string and strip whitespace
                if not col_str: # Handle empty column names
                    sanitized = f'col_{i}_blank'
                    sanitization_errors.append(f"Original column index {i} was blank, named '{sanitized}'.")
                else:
                    # Replace invalid characters
                    sanitized = col_str.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'perc')
                    sanitized = sanitized.replace('/', '_').replace('.', '').replace(':', '').replace('-', '_')
                    # Remove any remaining non-alphanumeric characters (except underscore)
                    sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
                    # Ensure starts with letter or underscore
                    if not sanitized[0].isalpha() and sanitized[0] != '_':
                        sanitized = '_' + sanitized

                # Check if valid identifier or reserved SQL word
                reserved_sql = ['select', 'from', 'where', 'index', 'table', 'insert', 'update', 'delete', 'order', 'group']
                if not sanitized.isidentifier() or sanitized.lower() in reserved_sql:
                    original_prefix = f'col_{i}_'
                    safe_original = ''.join(c for c in col_str if c.isalnum() or c == '_')[:20] # Keep part of original name safely
                    sanitized = original_prefix + safe_original
                    # Final check after prefixing
                    if not sanitized.isidentifier():
                        sanitized = f'col_{i}' # Absolute fallback

                sanitized_columns.append(sanitized)

            # --- Crucial Length Check AFTER Sanitization Loop ---
            if len(sanitized_columns) != expected_columns:
                error_msg = f"CRITICAL SANITIZE ERROR: Length mismatch AFTER sanitizing. Expected {expected_columns}, Got {len(sanitized_columns)}."
                print(error_msg)
                print("DEBUG: Original Columns:", original_columns)
                print("DEBUG: Sanitized Columns Attempt:", sanitized_columns)
                st.error(error_msg)
                # Add details about potential duplicate sanitized names
                from collections import Counter
                dupes = [item for item, count in Counter(sanitized_columns).items() if count > 1]
                if dupes:
                    st.warning(f"Potential cause: Duplicate column names after sanitization: {dupes}")
                return False, "\n".join(status_messages + [error_msg])
            else:
                # Lengths match, assign the new columns
                df.columns = sanitized_columns
                renamed_columns = df.columns.tolist() # Get assigned columns
                status_messages.append("Column names sanitized.")
                if original_columns != renamed_columns:
                    st.info("Note: Some column names were adjusted for compatibility.")
                    # Optionally display mapping in expander
                    # with st.expander("Column Name Changes"):
                    #    for orig, new in zip(original_columns, renamed_columns):
                    #        if orig != new: st.write(f"`{orig}` -> `{new}`")

        except Exception as sanitize_e:
            error_msg = f"Error occurred during column name sanitization phase: {sanitize_e}"
            print(f"ERROR: {error_msg}")
            st.error(error_msg)
            return False, "\n".join(status_messages + [error_msg])

        # --- Check DataFrame state BEFORE writing ---
        current_shape = df.shape
        current_columns_count = len(df.columns)
        status_messages.append(f"DataFrame shape before write: {current_shape}. Columns: {current_columns_count}")
        print(f"DEBUG: Columns before write ({current_columns_count}): {df.columns.tolist()}")

        if current_columns_count != expected_columns:
             error_msg = f"CRITICAL ERROR: Column count changed unexpectedly before write. Expected {expected_columns}, Found {current_columns_count}."
             print(error_msg)
             st.error(error_msg)
             return False, "\n".join(status_messages + [error_msg])


        # --- Write to SQLite ---
        status_messages.append(f"Writing data to SQLite table `{table_name}` (replacing)...")
        write_successful = False
        try:
            with conn:
                conn.execute(f"PRAGMA busy_timeout = {SQLITE_TIMEOUT_SECONDS * 1000};")
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            status_messages.append("Data written successfully to SQLite.")
            write_successful = True
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e): error_msg = f"Database lock persisted..."
            else: error_msg = f"SQLite operational error during write: {e}"
            print(f"ERROR: {error_msg}"); st.error(error_msg)
        except ValueError as e: # Catch specific ValueError which sometimes includes length issues
             error_msg = f"ValueError during data writing (check data types/lengths): {e}"
             print(f"ERROR: {error_msg}"); st.error(error_msg)
        except Exception as e:
             error_msg = f"Unexpected error during data writing: {e}"
             print(f"ERROR: {error_msg}"); st.error(error_msg)
             # Check if the error message contains the specific length mismatch text
             if "Length mismatch" in str(e):
                  error_msg += f". DF Shape: {df.shape}. Expected cols based on read: {expected_columns}."
                  print(f"DEBUG: Mismatch during write. DF Columns: {df.columns.tolist()}")
                  st.info("Length mismatch occurred during the write phase. This is unusual if checks passed before. Check data types or potential hidden issues.")

        if write_successful:
            # --- Analyze, Embed, Index ---
            
            status_messages.append("Analyzing columns for embedding (using LLM)...")
            current_schema = {table_name: df.columns.tolist()} # Schema based on sanitized columns
            # Pass df.head() for context
            semantic_cols_to_embed = _suggest_semantic_columns(
                df.head(), current_schema, table_name, llm_model
            )
            status_messages.append(f"LLM suggested for embedding: {semantic_cols_to_embed}")

            # --- Embed and Index ---
            status_messages.append("Starting embedding & indexing...")
            embed_success, embed_msg = _embed_and_index_data(
                df, table_name, semantic_cols_to_embed, aux_models, qdrant_client
            )
            status_messages.append(f"Embedding/Indexing result: {embed_msg}")
            # *** NEED A WAY TO DETERMINE semantic_cols ***
            # Placeholder: Assume text columns are semantic for now
            # In a real scenario, this might involve user input or LLM analysis earlier
            potential_semantic_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            logging.info(f"Potential semantic columns: {potential_semantic_cols}")
            # Let's just embed all text columns found for simplicity in this example:
            semantic_cols_to_embed = potential_semantic_cols
            status_messages.append(f"Identified for embedding: {semantic_cols_to_embed}")

            # *** Call the updated embedding function ***
            status_messages.append("Starting embedding & indexing...")
            embed_success, embed_msg = _embed_and_index_data(
                df, table_name, semantic_cols_to_embed, aux_models, qdrant_client
            )
            status_messages.append(f"Embedding/Indexing result: {embed_msg}")

            if embed_success:
                status_messages.append(f"Processing complete for `{table_name}`.")
                return True, "\n".join(status_messages)
            else:
                 status_messages.append(f"SQL write OK, but embedding/indexing failed.")
                 return False, "\n".join(status_messages) # Return messages even on partial failure
        if not write_successful:
             return False, "\n".join(status_messages + [f"Failed during SQLite write: {error_msg}"])

        # --- Analyze, Embed, Index (only if write successful) ---
        status_messages.append("Analyzing columns..."); semantic_cols = _analyze_columns_for_embedding(df, table_name, models)
        status_messages.append("Embedding & Indexing data..."); embed_success = _embed_and_index_data(df, table_name, semantic_cols, models, qdrant_client)

        if embed_success:
            status_messages.append(f"Processing complete for `{table_name}`.")
            return True, "\n".join(status_messages)
        else:
             status_messages.append(f"SQL OK, embedding/indexing failed for `{table_name}`.")
             return False, "\n".join(status_messages)

    except Exception as e:
        # Catch unexpected errors in the overall process
        critical_error_msg = f"Critical error during data processing: {e}"
        # Add context if possible
        critical_error_msg += f" (Expected Columns: {expected_columns}, Current DF: {'Exists' if df is not None else 'None'})"
        st.error(critical_error_msg)
        print(f"CRITICAL ERROR in process_uploaded_data: {e}")
        # You might want to print the df.head() here if df exists for more context
        # if df is not None: print("DEBUG: df.head() at critical error:", df.head())
        return False, critical_error_msg

# --- Database Connection (remains mostly the same, uses path argument) ---
@st.cache_resource
def get_db_connection(db_path):
    """Establishes and caches SQLite connection using the provided path."""
    print(f"Attempting DB connection to: {db_path}")
    if not db_path: # Check if path is valid
         print("ERROR: Invalid database path provided to get_db_connection.")
         return None
    try:
        # No need for os.makedirs here, setup_environment handles it
        conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_SECONDS, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            print("INFO: Write-Ahead Logging (WAL) mode enabled.")
        except Exception as wal_e:
            print(f"WARNING: Could not enable WAL mode: {wal_e}")
        print("INFO: DB Connection Successful.")
        return conn
    except Exception as e:
        print(f"ERROR: Failed to connect to DB at {db_path}: {e}")
        return None

@st.cache_resource
def init_qdrant_client():
    """
    Initializes and caches an in-memory Qdrant client.
    """
    print("Initializing in-memory Qdrant client...")
    try:
        client = QdrantClient(":memory:")
        print("Qdrant client initialized.")
        return client
    except Exception as e:
        print(f"FATAL QDRANT ERROR: {e}")
        return None

@st.cache_resource
def load_models():
    """
    Loads and caches the required models (LLMs, Embedding).
    Replace simulation with actual model loading using Transformers/FastEmbed.
    """
    print("Simulating the loading of LLM and Embedding models...")
    # --- Placeholder for actual model loading ---
    # Requires: transformers, torch, accelerate, bitsandbytes, fastembed
    # try:
    #     bnb_config = BitsAndBytesConfig(...)
    #     gemma_tokenizer = AutoTokenizer.from_pretrained(...)
    #     gemma_model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config, device_map="auto")
    #     sql_tokenizer = AutoTokenizer.from_pretrained(...)
    #     sql_model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config, device_map="auto")
    #     embedding_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    #     print("Models loaded successfully.")
    #     # Return models and tokenizers in a dictionary or tuple
    #     # return {"gemma_model": gemma_model, "gemma_tokenizer": gemma_tokenizer, ... , "embed_model": embedding_model}
    # except Exception as e:
    #     st.error(f"Fatal Error: Failed to load models: {e}")
    #     return None # Indicate failure
    # --- End Placeholder ---

    time.sleep(1) # Simulate loading time
    print("Model loading simulation complete.")
    # Return placeholder indication of success
    return {"status": "loaded", "embed_model": "simulated_embedder", "gemma": "simulated_gemma", "sql_model": "simulated_sql_gen"}


# --- Data Processing Functions ---

def get_schema_info(conn):
    """Fetches table names and columns from the SQLite DB."""
    if not conn:
        return {}
    schema = {}
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            if table_name.startswith("sqlite_"): # Skip internal tables
                 continue
            cursor.execute(f"PRAGMA table_info('{table_name}');") # Use quotes for table names
            columns = [info[1] for info in cursor.fetchall()] # Get column names
            schema[table_name] = columns
        return schema
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return {} # Return empty dict on error

def _analyze_columns_for_embedding(df, table_name, models):
    """
    Placeholder: Simulates using Gemma-3 to find semantic columns.
    In reality, this would involve prompting Gemma with schema + sample data.
    """
    print(f"Simulating Gemma-3 analysis for semantic columns in table '{table_name}'...")
    time.sleep(1)
    # Basic heuristic for simulation: find object/string columns
    potential_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Simulate picking one or two based on name (e.g., 'description', 'text', 'comment')
    semantic_cols = [col for col in potential_cols if any(kw in col.lower() for kw in ['desc', 'text', 'comment', 'note', 'detail'])]
    if not semantic_cols and potential_cols: # Fallback if no keywords match
        semantic_cols = [potential_cols[0]] # Just take the first object column
    print(f"Simulated semantic columns identified: {semantic_cols}")
    time.sleep(0.5); return df.select_dtypes(include=['object']).columns[:1].tolist()



# --- Query Processing Functions ---

def _route_query(user_query: str, schema: dict, llm_wrapper: LLMWrapper) -> str:
    """
    Determines query route ('SQL' or 'SEMANTIC') using LLM structured output if available,
    otherwise falls back to keywords.
    """
    default_route = "SQL" # Default assumption if LLM fails
    logging.info("Routing query...")

    # Check if we can use the advanced method
    if not llm_wrapper or not llm_wrapper.is_ready or llm_wrapper.mode != 'openrouter':
        logging.warning(f"LLM wrapper not ready or not in OpenRouter mode ({getattr(llm_wrapper, 'mode', 'N/A')}). Falling back to keyword routing.")
        # --- Keyword Fallback Logic ---
        query_lower = user_query.lower()
        # More specific keywords might help fallback accuracy
        sql_kws = ["how many", "list", "total", "average", "maximum", "minimum", "count", "sum", "max", "min", "avg", "group by", "order by", " where ", " id ", " date ", " year ", " month ", " price", " number of ", " value of "]
        sem_kws = ["describe", "description", "details about", "similar to", "meaning", "related to", "find products like", "tell me about", "explain"]
        if any(kw in query_lower for kw in sem_kws):
            logging.info("Keyword fallback route: SEMANTIC")
            return "SEMANTIC"
        elif any(kw in query_lower for kw in sql_kws):
             logging.info("Keyword fallback route: SQL")
             return "SQL"
        else:
            logging.info(f"Keyword fallback couldn't classify strongly. Defaulting to: {default_route}")
            return default_route # Default if keywords are ambiguous

    # --- Structured Output Logic (OpenRouter Mode) ---
    logging.info("Attempting routing via OpenRouter structured output...")

    # Define the desired JSON structure for the model
    json_structure_description = """Respond ONLY with a valid JSON object containing a single key "query_type" whose value is either the string "SQL" or the string "SEMANTIC". Example: {"query_type": "SQL"}"""

    # Construct the prompt
    prompt = f"""Analyze the user query based on the database schema to determine the best approach.
        Database Schema:
        {schema}

        User Query:
        "{user_query}"

        Carefully consider the query's intent:
        - Choose "SQL" if the query requires precise data retrieval, filtering, aggregation (counts, sums, averages), or specific values matching the schema columns.
        - Choose "SEMANTIC" if the query asks for descriptions, meaning, similarity, or information likely found in unstructured text columns (like product descriptions, notes, etc.).

        {json_structure_description}""" # Append the JSON instructions

    logging.debug(f"Sending structured prompt to LLM:\n{prompt}")

    structured_result = llm_wrapper.generate_structured_response(prompt)

    # Process the structured result
    if structured_result and isinstance(structured_result, dict):
        query_type = structured_result.get("query_type") # Case-sensitive key lookup
        if query_type in ["SQL", "SEMANTIC"]:
            logging.info(f"Structured routing successful. Determined type: {query_type}")
            return query_type
        else:
            logging.warning(f"LLM returned valid JSON but with unexpected 'query_type' value: '{query_type}'. Falling back.")
    elif structured_result is not None: # It returned something, but not a dict we could parse/use
        logging.warning(f"LLM returned non-dictionary or invalid structured result: {structured_result}. Falling back.")
    else: # It returned None (API error, JSON parse error, etc.)
        logging.warning("LLM structured response failed (returned None). Falling back.")

    # Fallback if structured output failed
    logging.info(f"Structured routing failed. Defaulting route to: {default_route}")
    return default_route

def _generate_sql_query(user_query: str, schema: dict, sample_data_str: str | None, aux_models, previous_sql: str | None = None, feedback: str | None = None) -> str:
    """
    Generates SQL query, incorporating feedback from previous attempts if provided.
    """
    sql_llm: Llama = aux_models.get('sql_gguf_model')
    if not sql_llm:
        return "-- Error: SQL GGUF model object missing."
        
    sample_context = f"Data Sample (representative table):\n{sample_data_str}\n" if sample_data_str else "Sample data not available.\n"

    # --- Construct Prompt with Feedback ---
    prompt = f"""Given the database schema:
{schema}

{sample_context}
Generate an SQLite query to find the data relevant to the user's question. Follow these rules:
1. Select `*` or key columns plus relevant value columns.
2. Use correct numerical comparisons (e.g., `Count < 5`, not `Count < 5 units`).
3. Handle extremes (min/max/top) by retrieving all ties (e.g., using subqueries like `WHERE col = (SELECT MAX(col) ...)` or `ORDER BY ... LIMIT 5` as fallback).
4. Avoid unnecessary `ORDER BY` or `LIMIT` otherwise.

User question: "{user_query}"
"""
    if previous_sql and feedback:
        if "syntax error" in feedback.lower():
            prompt += f"""
PREVIOUS ATTEMPT FAILED due to a syntax error.
Incorrect SQL: `{previous_sql}`
Error: "{feedback}"
Please correct the syntax.
"""
        else:
            prompt += f"""
PREVIOUS ATTEMPT was unsatisfactory.
SQL Executed: `{previous_sql}`
Reason: "{feedback}"
Please generate an improved query incorporating this feedback.
"""
    prompt += """
Respond ONLY with the SQLite query, ending with a semicolon.
SQLite Query:
"""
    logging.info(f"Sending prompt to SQL GGUF model (Retry Attempt? {'Yes' if previous_sql else 'No'}):\n{prompt}")
    try:
        output = sql_llm(
            prompt, max_tokens=500, temperature=0.1, top_p=0.9,
            stop=[";", "\n\n", "```"], echo=False
        )
        if output and 'choices' in output and len(output['choices']) > 0:
            generated_sql = output['choices'][0]['text'].strip()
            logging.info(f"Raw SQL GGUF output: {generated_sql}")
            if "```sql" in generated_sql:
                generated_sql = generated_sql.split("```sql")[1]
            if "```" in generated_sql:
                generated_sql = generated_sql.split("```")[0]
            sql_keywords = ["SELECT", "WITH"]
            first_kw_pos = -1
            for keyword in sql_keywords:
                pos = generated_sql.upper().find(keyword)
                if pos != -1:
                    if first_kw_pos == -1 or pos < first_kw_pos:
                        first_kw_pos = pos
            if first_kw_pos != -1:
                cleaned_sql = generated_sql[first_kw_pos:]
            else:
                logging.warning(f"Generated text might not contain SELECT/WITH: {generated_sql}")
                cleaned_sql = generated_sql
            if ';' in cleaned_sql:
                cleaned_sql = cleaned_sql.split(';')[0] + ';'
            elif cleaned_sql.strip():
                cleaned_sql += ';'
                logging.warning("Added missing semicolon to generated SQL.")
            cleaned_sql = cleaned_sql.strip()
            logging.info(f"Cleaned SQL query: {cleaned_sql}")
            if not cleaned_sql or not any(kw in cleaned_sql.upper() for kw in ["SELECT", "WITH"]):
                logging.warning(f"Generated text doesn't look like SQL: {cleaned_sql}")
                return "-- Error: Model did not generate a valid SQL query structure."
            return cleaned_sql
        else:
            logging.error(f"SQL GGUF model returned empty or invalid output structure: {output}")
            return "-- Error: SQL model returned empty or invalid output."
    except Exception as e:
        logging.error(f"Error during SQL GGUF generation: {e}", exc_info=True)
        return f"-- Error generating SQL: {e}"




def _execute_sql_query(conn, sql_query):
    """Executes the SQL query and returns results as a DataFrame."""
    if not conn:
        return pd.DataFrame(), "Database connection is not available."
    try:
        print(f"Executing SQL: {sql_query}")
        result_df = pd.read_sql_query(sql_query, conn)
        print("SQL execution successful.")
        return result_df, None # Data, No error message
    except pd.io.sql.DatabaseError as e:
         print(f"SQL Execution Error: {e}")
         # Try to provide a more helpful message for common errors
         if "no such table" in str(e):
             return pd.DataFrame(), f"Error: Table mentioned in the query might not exist. Please check the schema. ({e})"
         elif "no such column" in str(e):
              return pd.DataFrame(), f"Error: Column mentioned in the query might not exist in the table. Please check the schema. ({e})"
         else:
              return pd.DataFrame(), f"SQL Error: {e}"
    except Exception as e:
        print(f"Unexpected SQL Execution Error: {e}")
        return pd.DataFrame(), f"An unexpected error occurred during SQL execution: {e}"


# --- Semantic Search Function ---
def _perform_semantic_search(user_query: str, aux_models: dict, qdrant_client: QdrantClient, schema: dict) -> tuple[list[str], str | None]:
    """
    Performs semantic search using the embedding model and Qdrant.
    Returns a list of formatted result strings and an optional error message.
    """
    logging.info("Performing semantic search...")
    # --- Readiness Checks ---
    if not qdrant_client: return [], "Qdrant client not initialized."
    if not aux_models or aux_models.get('status') in ['error', 'partial']:
        if not aux_models or not aux_models.get('embedding_model'):
             return [], "Embedding model not loaded."
        if aux_models.get('status') == 'error': return [], "Aux models failed."

    embedding_model = aux_models.get('embedding_model')
    if not embedding_model: return [], "Embedding model object missing."

    try:
        # --- Embed the Query ---
        logging.info(f"Embedding user query: '{user_query[:100]}...'")
        query_embedding_result = list(embedding_model.embed([user_query]))
        if not query_embedding_result:
            logging.error("Query embedding failed.")
            return [], "Failed to generate embedding for the query."
        query_vector = query_embedding_result[0]
        logging.info("Query embedded successfully.")

        # --- Find Relevant Qdrant Collections ---
        # Search all collections associated with table data
        all_collections = qdrant_client.get_collections()
        target_collections = [
            coll.name for coll in all_collections.collections
            if coll.name.startswith(QDRANT_COLLECTION_PREFIX)
        ]

        if not target_collections:
            logging.warning("No Qdrant collections found matching prefix.")
            return ["No vector data found for searching. Please process data first."], None

        logging.info(f"Searching in Qdrant collections: {target_collections}")

        # --- Perform Search and Collect Hits ---
        all_hits = []
        search_limit_per_collection = 5 # Retrieve top N from each collection

        for collection_name in target_collections:
            try:
                search_result = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=search_limit_per_collection,
                    # with_payload=True # Default is usually True, but explicit doesn't hurt
                    # with_vector=False # Usually don't need the vector back
                )
                logging.info(f"Found {len(search_result)} hits in '{collection_name}'.")
                all_hits.extend(search_result)
            except Exception as search_e:
                 logging.error(f"Error searching collection '{collection_name}': {search_e}")
                 # Continue searching other collections

        if not all_hits:
            logging.info("Semantic search returned no hits across all collections.")
            return ["No relevant matches found in the vector data."], None

        # --- Sort and Format Results ---
        # Sort by score (descending for Cosine similarity)
        all_hits.sort(key=lambda x: x.score, reverse=True)

        formatted_results = []
        max_total_results = 10 # Limit the final number of results shown

        logging.info(f"Processing top {min(len(all_hits), max_total_results)} hits...")
        for hit in all_hits[:max_total_results]:
            payload = hit.payload
            if not payload: continue # Skip hits with no payload

            # Extract useful info from payload (customize based on what you stored)
            table_name = payload.get("_table_name", "Unknown Table")
            # Try to find original text or a descriptive field
            display_text = payload.get("_source_text", str(payload)) # Fallback to full payload string
            # Truncate long text
            if len(display_text) > 200:
                display_text = display_text[:200] + "..."

            # Try to get a meaningful ID (might need adjustment based on point_id format)
            hit_id_str = str(hit.id)
            if hit_id_str.startswith(f"{table_name}_"):
                 display_id = hit_id_str.split('_', 1)[1] # Show original PK if used
            else:
                 display_id = hit_id_str # Show index or UUID

            # Format the result string
            result_str = f"**Table:** `{table_name}` | **ID:** `{display_id}` | **Score:** {hit.score:.3f}\n> {display_text}"
            formatted_results.append(result_str)

        if not formatted_results:
             return ["No relevant matches found after processing hits."], None

        return formatted_results, None # Success: return list and None for error

    except Exception as e:
        logging.error(f"Unexpected error during semantic search: {e}", exc_info=True)
        return [], f"Error during semantic search: {e}"

def validate_inputs(query: str, conn, schema: dict, llm_wrapper, qdrant_client) -> dict:
    """Validates inputs and returns an initial result dict or an error dict."""
    if not query:
        return {"status": "error", "message": "Query empty."}
    if not conn:
        return {"status": "error", "message": "Database connection is not available."}
    if not schema:
        return {"status": "error", "message": "No database schema found. Please load data first."}
    if not llm_wrapper:
        return {"status": "error", "message": "Models not loaded."}
    if not qdrant_client:
        return {"status": "error", "message": "Qdrant client not available."}
    return {"status": "ok"}

def get_sample_data(conn, schema: dict, limit: int = 3) -> str:
    """
    Retrieves a sample from the first table in the schema and returns it as a markdown string.
    """
    first_table = next(iter(schema), None)
    if not first_table:
        return "No tables found in schema."
    sample_df = _get_table_sample_df(conn, first_table, limit=limit)
    if sample_df is None:
        return f"Could not retrieve sample for table '{first_table}'."
    elif sample_df.empty:
        return f"Table '{first_table}' exists but is empty."
    return sample_df.to_markdown(index=False)

def ensure_query_in_english(original_query: str, llm_wrapper) -> tuple[str, bool]:
    """
    Ensures that the query is in English, translating if necessary.
    Returns (english_query, was_translated).
    """
    # Assume _ensure_english_query is implemented elsewhere.
    return _ensure_english_query(original_query, llm_wrapper)

def route_query_wrapper(english_query: str, schema: dict, llm_wrapper, sample_data_str: str) -> str:
    """
    Routes the query using the LLM by providing extra context from the database schema and sample data.
    Returns the determined route (SQL or SEMANTIC).
    """
    # Construct an enhanced prompt that includes the sample data
    prompt = f"""Analyze the following context and user query to determine the best approach:
    
Database Schema:
{schema}

Sample Data:
{sample_data_str}

User Query (in English):
"{english_query}"

Respond ONLY with a valid JSON object containing a single key "query_type" whose value is either "SQL" or "SEMANTIC".
Example: {{"query_type": "SQL"}}
"""
    structured_result = llm_wrapper.generate_structured_response(prompt)
    if structured_result and isinstance(structured_result, dict):
        query_type = structured_result.get("query_type")
        if query_type in ["SQL", "SEMANTIC"]:
            logging.info(f"Structured routing successful. Determined type: {query_type}")
            return query_type
    logging.info("Fallback: Using default SQL routing.")
    return "SQL"


def execute_sql_pipeline(
    english_query: str, 
    schema: dict,
    sample_data_str:str, 
    aux_models, 
    conn, 
    llm_wrapper, 
    max_retries: int = 1
) -> dict:
    """
    Attempts to generate and execute an SQL query with retries.
    Returns a dictionary containing keys: sql_success, generated_sql, sql_data, sql_error.
    """
    sql_result = {
        "sql_success": False,
        "generated_sql": None,
        "sql_data": None,
        "sql_error": None,
    }
    current_sql = None
    previous_feedback = None
    # debugging aux_models dtype

    if not isinstance(aux_models, dict): print("ERROR: aux_models IS STRING before process_natural_language_query!")

    for attempt in range(max_retries + 1):
        logging.info(f"SQL Generation/Execution Attempt #{attempt + 1}")
        current_sql = _generate_sql_query(
            user_query=english_query,
            schema=schema,
            sample_data_str=sample_data_str,
            aux_models=aux_models,
            previous_sql=current_sql,
            feedback=previous_feedback
        )

        sql_result["generated_sql"] = current_sql  # Store latest attempt
        previous_feedback = None  # Reset feedback

        if "-- Error" in current_sql:
            feedback = f"SQL generation failed: {current_sql}"
            if attempt < max_retries:
                previous_feedback = feedback
                continue
            else:
                sql_result["sql_error"] = feedback
                break

        sql_df, exec_error = _execute_sql_query(conn, current_sql)
        if exec_error:
            feedback = f"SQL execution error: {exec_error}"
            if attempt < max_retries:
                previous_feedback = feedback
                continue
            else:
                sql_result["sql_error"] = feedback
                break

        is_valid, validation_feedback = _validate_sql_results(english_query, current_sql, sql_df, llm_wrapper)
        if is_valid:
            sql_result["sql_success"] = True
            sql_result["sql_data"] = sql_df
            logging.info("SQL attempt successful and validated.")
            break
        else:
            feedback = f"Result validation failed: {validation_feedback}"
            if attempt < max_retries:
                previous_feedback = validation_feedback
                continue
            else:
                sql_result["sql_error"] = feedback
                break

    return sql_result


def execute_semantic_pipeline(
    english_query: str, schema: dict, aux_models: dict, qdrant_client
) -> dict:
    """
    Executes semantic search and returns a dictionary containing:
    semantic_success, semantic_data, semantic_error.
    """
    semantic_result = {
        "semantic_success": False,
        "semantic_data": None,
        "semantic_error": None,
    }
    semantic_data, error_msg = _perform_semantic_search(english_query, aux_models, qdrant_client, schema)
    if not error_msg:
        semantic_result["semantic_success"] = True
        semantic_result["semantic_data"] = semantic_data
    else:
        semantic_result["semantic_error"] = error_msg
    return semantic_result


def run_secondary_route(
    english_query: str,
    primary_route: str,
    sql_result: dict,
    semantic_result: dict,
    aux_models,
    conn,
    schema: dict,
    llm_wrapper,
    qdrant_client,
    sample_data_str: str
) -> tuple[dict, dict]:
    """
    If the primary route did not yield both SQL and semantic results,
    this function attempts to run the missing branch once.
    Returns updated sql_result and semantic_result dictionaries.
    """
    run_secondary = True  # Change to False if you want to disable secondary route

    if not run_secondary:
        return sql_result, semantic_result

    if primary_route == "SQL" and not semantic_result.get("semantic_success", False):
        sec_semantic, sec_err = _perform_semantic_search(english_query, aux_models, qdrant_client, schema)
        if not sec_err:
            semantic_result["semantic_success"] = True
            semantic_result["semantic_data"] = sec_semantic
        else:
            semantic_result["semantic_error"] = sec_err
    elif primary_route == "SEMANTIC" and not sql_result.get("sql_success", False):
        # Run only one attempt for secondary SQL; include sample data for context
        sec_sql = _generate_sql_query(english_query, schema, sample_data_str, aux_models)
        sql_result["generated_sql"] = sec_sql
        if "-- Error" not in sec_sql:
            sec_sql_df, sec_sql_err = _execute_sql_query(conn, sec_sql)
            if not sec_sql_err:
                sql_result["sql_success"] = True
                sql_result["sql_data"] = sec_sql_df
            else:
                sql_result["sql_error"] = sec_sql_err
        else:
            sql_result["sql_error"] = sec_sql

    return sql_result, semantic_result



def generate_final_summary(
    original_query: str,
    english_query: str,
    was_translated: bool,
    sql_data,
    semantic_data,
    llm_wrapper
) -> str:
    """
    Aggregates SQL and semantic results into a final natural language summary using the LLM.
    """
    summary_context = f'User Query: "{original_query}"\n'
    if was_translated:
        summary_context += f'Query Interpreted as (English): "{english_query}"\n\n'

    if sql_data is not None:
        if not sql_data.empty:
            max_rows = 5
            summary_context += "SQL Query Results (Sample):\n"
            summary_context += sql_data.head(max_rows).to_markdown(index=False)
            if len(sql_data) > max_rows:
                summary_context += f"\n... ({len(sql_data) - max_rows} more rows)\n\n"
        else:
            summary_context += "The SQL query returned no matching rows.\n\n"

    if semantic_data and "No relevant matches found" not in semantic_data[0]:
        max_snippets = 5
        summary_context += "Semantic Search Results (Snippets):\n"
        summary_context += "\n".join(semantic_data[:max_snippets])
        if len(semantic_data) > max_snippets:
            summary_context += f"\n... ({len(semantic_data) - max_snippets} more snippets)\n\n"

    if summary_context.strip():
        summary_prompt = f"""Based only on the following retrieved data (which may include SQL results, semantic search snippets, or both), provide a concise and helpful natural language answer.

Retrieved Data:
---
{summary_context.strip()}
---

Answer:"""
        logging.info("Requesting final LLM summary based on aggregated results...")
        nl_summary = llm_wrapper.generate_response(summary_prompt, max_tokens=300)
        return nl_summary
    return "I searched the data, but couldn't find specific information matching your query."


def process_natural_language_query(
    original_query: str, conn, schema: dict, llm_wrapper, aux_models, qdrant_client, max_retries: int = 1
) -> dict:
    """
    Main pipeline:
    1. Validate inputs and ensure query is in English.
    2. Retrieve sample data for context.
    3. Route the query using the schema and sample data.
    4. Execute primary pipeline (SQL or semantic).
    5. Optionally run secondary route for aggregated context.
    6. Generate a final natural language summary.
    Returns a result dictionary with relevant data.
    """
    
    result = validate_inputs(original_query, conn, schema, llm_wrapper, qdrant_client)
    if result.get("status") == "error":
        return result

    # Use the conversation history from session state
    conversation_history = st.session_state.get("messages", [])
    # Combine refinement, table selection, and routing in one LLM call:
    refinement = refine_and_select(original_query, conversation_history, schema, llm_wrapper)
    refined_query = refinement.get("refined_query", original_query)
    selected_table = refinement.get("selected_table", next(iter(schema)) if schema else "")
    recommended_route = refinement.get("route", "SQL")
    result["refined_query"] = refined_query
    result["selected_table"] = selected_table
    result["determined_route"] = recommended_route
    logging.info(f"Refined query: {refined_query}, Selected Table: {selected_table}, Recommended Route: {recommended_route}")

    # Get sample data only for the selected table
    sample_data_str = get_table_sample_data(conn, selected_table, limit=3)

    # Depending on the recommended route, execute the corresponding pipeline:
    sql_result = {}
    semantic_result = {}
    if recommended_route.upper() == "SQL":
        sql_result = execute_sql_pipeline(refined_query, {selected_table: schema[selected_table]}, sample_data_str, aux_models, conn, llm_wrapper, max_retries)
    elif recommended_route.upper() == "SEMANTIC":
        semantic_result = execute_semantic_pipeline(refined_query, {selected_table: schema[selected_table]}, aux_models, qdrant_client)
    else:
        return {"status": "error", "message": f"Unknown route: {recommended_route}"}

    # Combine primary results and optionally run a secondary route for aggregation
    sql_result, semantic_result = run_secondary_route(refined_query, recommended_route, sql_result, semantic_result, aux_models, conn, schema, llm_wrapper, qdrant_client, sample_data_str)
    result["generated_sql"] = sql_result.get("generated_sql")
    result["sql_data"] = sql_result.get("sql_data")
    result["sql_error"] = sql_result.get("sql_error")
    result["semantic_data"] = semantic_result.get("semantic_data")
    result["semantic_error"] = semantic_result.get("semantic_error")

    # Generate a final natural language summary
    nl_summary = generate_final_summary(original_query, refined_query, False, result.get("sql_data"), result.get("semantic_data"), llm_wrapper)
    result["natural_language_summary"] = nl_summary
    result["message"] = nl_summary
    result["status"] = "success"
    result["raw_display_data"] = {"sql": result.get("sql_data"), "semantic": result.get("semantic_data")}
    return result

# --- NEW Validation Helper ---
def _validate_sql_results(user_query: str, executed_sql: str, result_df: pd.DataFrame, llm_wrapper: LLMWrapper) -> tuple[bool, str | None]:
    """
    Uses the LLM wrapper to validate if the SQL results satisfy the user query.

    Returns:
        tuple[bool, str | None]: (is_satisfactory, feedback_or_None)
            - True, None: If results are deemed satisfactory or validation fails safely.
            - False, str: If results are unsatisfactory, returns feedback string.
    """
    logging.info("Validating SQL results using LLM...")

    # Check if wrapper is ready and supports structured output (preferred)
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for validation. Assuming results are satisfactory.")
        return True, None # Fail safely - don't retry if validator isn't working

    # Format context for the LLM
    max_rows_for_context = 3 # Keep context small for validation prompt
    if result_df.empty:
        context_str = "The query returned zero rows."
    else:
        context_str = result_df.head(max_rows_for_context).to_markdown(index=False)
        if len(result_df) > max_rows_for_context:
             context_str += f"\n\n...(and {len(result_df) - max_rows_for_context} more rows)"

    json_structure_description = """Respond ONLY with a valid JSON object with two keys:
1. "satisfactory": boolean (true if the data answers the query, false otherwise).
2. "reason": string (if false, a BRIEF explanation of why, e.g., "missing filter for category", "wrong ordering", "needs different columns". If true, this can be an empty string or null).
Example if unsatisfactory: {"satisfactory": false, "reason": "The query returned the highest price, but the user asked for the cheapest."}
Example if satisfactory: {"satisfactory": true, "reason": ""}"""

    prompt = f"""You are validating SQL query results.
User's original query: "{user_query}"
Executed SQL query: `{executed_sql}`
Data returned (sample):
{context_str}

Does the returned data accurately and completely answer the user's original query? Consider if the filtering, ordering, and selected columns are appropriate.

{json_structure_description}"""

    logging.debug(f"Sending validation prompt to LLM:\n{prompt}")

    try:
        # Prioritize structured output if available
        if llm_wrapper.mode == 'openrouter':
            structured_result = llm_wrapper.generate_structured_response(prompt)
            if structured_result and isinstance(structured_result, dict):
                is_satisfactory = structured_result.get("satisfactory")
                reason = structured_result.get("reason")

                if isinstance(is_satisfactory, bool):
                    feedback = reason if not is_satisfactory and isinstance(reason, str) and reason else None
                    logging.info(f"LLM validation result: Satisfactory={is_satisfactory}, Feedback='{feedback}'")
                    return is_satisfactory, feedback
                else:
                     logging.warning(f"LLM validation JSON missing/invalid 'satisfactory' boolean: {structured_result}. Assuming satisfactory.")
            else:
                 logging.warning(f"LLM structured validation failed or returned non-dict: {structured_result}. Assuming satisfactory.")
        else: # Fallback to text generation for local GGUF or if structured failed
             logging.info("Using text generation for validation feedback.")
             # Try to guide the text model towards the JSON format anyway
             response_text = llm_wrapper.generate_response(prompt + ' {"satisfactory": true, "reason": ""}', max_tokens=100, temperature=0.1)
             logging.debug(f"Raw text validation response: {response_text}")
             try:
                 # Attempt to parse JSON even from text response
                 parsed_json = json.loads(response_text)
                 if isinstance(parsed_json, dict):
                     is_satisfactory = parsed_json.get("satisfactory")
                     reason = parsed_json.get("reason")
                     if isinstance(is_satisfactory, bool):
                         feedback = reason if not is_satisfactory and isinstance(reason, str) and reason else None
                         logging.info(f"LLM validation result (from text): Satisfactory={is_satisfactory}, Feedback='{feedback}'")
                         return is_satisfactory, feedback
                     else: logging.warning("Parsed validation JSON missing/invalid 'satisfactory'. Assuming satisfactory.")
                 else: logging.warning("Parsed validation response was not a dictionary. Assuming satisfactory.")
             except json.JSONDecodeError:
                  logging.warning(f"Could not parse LLM text validation response as JSON: {response_text}. Assuming satisfactory.")

    except Exception as e:
        logging.error(f"Error during LLM validation call: {e}", exc_info=True)
        # Fail safely on unexpected errors

    # Default return if anything goes wrong with validation
    return True, None

# --- Helper to get data sample ---
def _get_table_sample_df(conn: sqlite3.Connection, table_name: str, limit: int = 3) -> pd.DataFrame | None:
    """Fetches the first few rows of a table as a DataFrame sample."""
    if not conn or not table_exists(conn, table_name):
        return None
    try:
        # Use LIMIT to fetch only a small sample
        query = f'SELECT * FROM "{table_name}" LIMIT {limit}' # Quote table name just in case
        df_sample = pd.read_sql_query(query, conn)
        return df_sample
    except Exception as e:
        logging.error(f"Error fetching sample data for table '{table_name}': {e}")
        return None
    
def _ensure_english_query(user_query: str, llm_wrapper: LLMWrapper) -> tuple[str, bool]:
    """
    Checks if query is English. If not, attempts translation using LLM.
    Returns (potentially_translated_query, was_translation_needed)
    """
    logging.info("Checking query language...")
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for language check/translation. Assuming English.")
        return user_query, False # Assume English if LLM unavailable

    try:
        # Simple check first
        lang_prompt = f"Is the following query primarily written in English? Respond only with YES or NO.\n\nQuery: \"{user_query}\""
        lang_response = llm_wrapper.generate_response(lang_prompt, max_tokens=5, temperature=0.1).strip().upper()

        if lang_response.startswith("YES"):
            logging.info("Query identified as English.")
            return user_query, False
        elif lang_response.startswith("NO"):
            logging.info("Query identified as non-English. Attempting translation...")
            trans_prompt = f"Translate the following query accurately into English. Respond ONLY with the English translation.\n\nQuery: \"{user_query}\"\n\nEnglish Translation:"
            translated_query = llm_wrapper.generate_response(trans_prompt, max_tokens=len(user_query) + 50, temperature=0.5).strip() # Allow more tokens for translation

            # Basic validation of translation
            if translated_query and translated_query != user_query and len(translated_query) > 5:
                logging.info(f"Translated query to English: \"{translated_query}\"")
                return translated_query, True
            else:
                logging.warning(f"Translation attempt failed or returned unusable result: '{translated_query}'. Using original query.")
                return user_query, False # Use original if translation fails
        else:
            logging.warning(f"Language check returned ambiguous response: '{lang_response}'. Assuming English.")
            return user_query, False # Assume English on ambiguous response

    except Exception as e:
        logging.error(f"Error during language check/translation: {e}", exc_info=True)
        return user_query, False # Fail safe - use original query
    
def get_sqlite_table_row_count(conn: sqlite3.Connection, table_name: str) -> int | None:
    """Gets the row count for a specific SQLite table."""
    if not conn or not table_exists(conn, table_name):
        return None
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}") # Safe as table_name checked by table_exists
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    except Exception as e:
        logging.error(f"Error getting row count for table '{table_name}': {e}")
        return None
    
def get_qdrant_collection_info(qdrant_client: QdrantClient, collection_name: str) -> dict | None:
    """Gets point count and vector dimension for a Qdrant collection."""
    if not qdrant_client:
        return None
    try:
        info = qdrant_client.get_collection(collection_name=collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count, # Might differ if vectors are named
            "vector_size": info.vectors_config.params.size if isinstance(info.vectors_config, models.VectorParams) else "N/A", # Handle named vectors if needed
            "status": str(info.status)
        }
    except Exception as e:
        # Handle case where collection doesn't exist gracefully
        logging.warning(f"Could not get Qdrant collection info for '{collection_name}': {e}")
        return None
    

def reindex_table(conn: sqlite3.Connection, table_name: str, llm_wrapper: LLMWrapper, aux_models: dict, qdrant_client: QdrantClient) -> tuple[bool, str]:
    """
    Reads data from SQLite table, re-suggests columns (optional),
    re-embeds, and re-indexes in Qdrant (replaces existing collection).
    """
    logging.info(f"Attempting to re-index table '{table_name}'...")
    # --- Readiness Checks ---
    if not conn: return False, "Database connection unavailable."
    if not table_exists(conn, table_name): return False, f"SQLite table '{table_name}' not found."
    if not qdrant_client: return False, "Qdrant client unavailable."
    if not aux_models or aux_models.get('status') != 'loaded': return False, "Aux models not loaded."
    # LLM wrapper needed for suggesting columns
    if not llm_wrapper or not llm_wrapper.is_ready: logging.warning("LLM wrapper not ready, cannot re-suggest columns.")

    try:
        # --- Read Data ---
        logging.info(f"Reading data from SQLite table '{table_name}' for re-indexing...")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn) # Safe as table name exists
        if df.empty: return False, f"Table '{table_name}' is empty, nothing to re-index."
        logging.info(f"Read {len(df)} rows from '{table_name}'.")

        # --- Suggest Semantic Columns (Optional - could reuse previous logic or let user choose later) ---
        # For now, let's re-run the suggestion
        current_schema = {table_name: df.columns.tolist()}
        semantic_cols_to_embed = _suggest_semantic_columns(
            df.head(), current_schema, table_name, llm_wrapper
        )
        if not semantic_cols_to_embed:
             return False, "No semantic columns identified or suggested for re-embedding."

        # --- Embed and Index ---
        logging.info(f"Calling embed and index for re-indexing '{table_name}'...")
        embed_success, embed_msg = _embed_and_index_data(
            df, table_name, semantic_cols_to_embed, aux_models, qdrant_client
        )

        if embed_success:
            logging.info(f"Successfully re-indexed table '{table_name}'.")
            return True, f"Successfully re-indexed table '{table_name}'. {embed_msg}"
        else:
            logging.error(f"Re-indexing failed for table '{table_name}'. Reason: {embed_msg}")
            return False, f"Re-indexing failed: {embed_msg}"

    except Exception as e:
        logging.error(f"Unexpected error during re-indexing of '{table_name}': {e}", exc_info=True)
        return False, f"Unexpected error during re-indexing: {e}"


def delete_table_data(conn: sqlite3.Connection, table_name: str, qdrant_client: QdrantClient) -> tuple[bool, str]:
    """Deletes table from SQLite and corresponding collection from Qdrant."""
    logging.info(f"Attempting to delete data for table '{table_name}'...")
    if not conn: return False, "Database connection unavailable."
    if not qdrant_client: return False, "Qdrant client unavailable."

    sqlite_success = False
    qdrant_success = False
    sqlite_msg = "Not Attempted"
    qdrant_msg = "Not Attempted"

    # Delete from SQLite
    try:
        with conn: # Use transaction
            conn.execute(f"DROP TABLE IF EXISTS {table_name}") # Safe syntax
        sqlite_success = True
        sqlite_msg = f"SQLite table '{table_name}' dropped successfully."
        logging.info(sqlite_msg)
    except Exception as e:
        sqlite_msg = f"Error dropping SQLite table '{table_name}': {e}"
        logging.error(sqlite_msg, exc_info=True)

    # Delete from Qdrant
    collection_name = f"{QDRANT_COLLECTION_PREFIX}{table_name}"
    try:
        # Check if collection exists before attempting delete
        # This avoids errors if only SQLite part existed
        try:
             qdrant_client.get_collection(collection_name=collection_name)
             collection_exists = True
        except Exception:
             collection_exists = False
             qdrant_msg = f"Qdrant collection '{collection_name}' did not exist."
             qdrant_success = True # Treat as success if it's already gone

        if collection_exists:
            qdrant_client.delete_collection(collection_name=collection_name, timeout=10)
            qdrant_success = True
            qdrant_msg = f"Qdrant collection '{collection_name}' deleted successfully."
            logging.info(qdrant_msg)

    except Exception as e:
        qdrant_msg = f"Error deleting Qdrant collection '{collection_name}': {e}"
        logging.error(qdrant_msg, exc_info=True)

    # Combine results
    overall_success = sqlite_success and qdrant_success
    final_message = f"SQLite: {sqlite_msg} | Qdrant: {qdrant_msg}"

    return overall_success, final_message

def select_relevant_table(user_query: str, schema: dict, llm_wrapper: LLMWrapper) -> tuple[str, dict]:
    """
    Determines the most relevant table for the user query.
    Returns a tuple containing:
    - the selected table name, and
    - a filtered schema dictionary for that table.
    """
    # Simple heuristic: check if any table name appears in the query (case-insensitive)
    for table in schema:
        if table.lower() in user_query.lower():
            return table, {table: schema[table]}
    # Fallback: if only one table exists, use it
    if len(schema) == 1:
        table = next(iter(schema))
        return table, {table: schema[table]}
    # Advanced approach: ask the LLM to decide
    prompt = f"""Based on the following database tables: {list(schema.keys())}
    and the user query: "{user_query}",
    which table is most relevant? Respond ONLY with the table name."""
    selected_table = llm_wrapper.generate_response(prompt, max_tokens=20, temperature=0.2).strip()
    if selected_table in schema:
        return selected_table, {selected_table: schema[selected_table]}
    # Fallback: return first table
    table = next(iter(schema))
    return table, {table: schema[table]}

def get_table_sample_data(conn, table_name: str, limit: int = 3) -> str:
    sample_df = _get_table_sample_df(conn, table_name, limit=limit)
    if sample_df is None:
        return f"Could not retrieve sample for table '{table_name}'."
    elif sample_df.empty:
        return f"Table '{table_name}' exists but is empty."
    return sample_df.to_markdown(index=False)


def refine_query(user_query: str, filtered_schema: dict, sample_data_str: str, llm_wrapper: LLMWrapper) -> str:
    """
    Refines a broad user query by combining it with the filtered schema and sample data context.
    Returns a more detailed, refined query that includes additional instructions for SQL generation.
    """
    prompt = f"""You are an expert data analyst.
        The user has asked the following broad question:
        "{user_query}"

        Here is the context:
        Database Schema:
        {filtered_schema}

        Sample Data:
        {sample_data_str}

        Please rewrite the user question into a more detailed query that includes specific instructions on which columns, filters, and aggregations might be relevant. Provide a refined question in clear English that can guide a SQL generation model effectively.
        Refined Query:"""
    refined = llm_wrapper.generate_response(prompt, max_tokens=150, temperature=0.3)
    return refined.strip()

def refine_query_with_feedback(user_query: str, feedback: str, filtered_schema: dict, sample_data_str: str, llm_wrapper: LLMWrapper) -> str:
    """
    Refines a broad user query by combining it with additional user feedback, the filtered schema, and sample data.
    Returns a more detailed query that provides clear instructions for SQL generation.
    """
    prompt = f"""You are an expert data analyst.
The user initially asked:
"{user_query}"

Additional feedback provided:
"{feedback}"

Context:
Database Schema:
{filtered_schema}

Sample Data:
{sample_data_str}

Please rewrite the user's query into a more detailed, precise question that indicates which columns, filters, and aggregations might be necessary.
Refined Query:"""
    refined = llm_wrapper.generate_response(prompt, max_tokens=150, temperature=0.3)
    return refined.strip()


def derive_requirements_from_history(conversation_history: list[dict], llm_wrapper: LLMWrapper, max_turns: int = 10) -> str:
    """
    Takes the last 'max_turns' messages from the conversation history and asks the LLM
    to summarize the user's overall requirements and intention.
    Returns a refined requirements string.
    """
    # Extract the last max_turns messages (both user and assistant)
    recent_turns = conversation_history[-max_turns:]
    # Format the conversation into a readable context
    context_lines = []
    for msg in recent_turns:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        context_lines.append(f"{role.capitalize()}: {content}")
    context = "\n".join(context_lines)
    
    prompt = f"""You are an expert data assistant. Based on the following conversation history between a user and an assistant, derive a concise summary of the user's real requirements for searching the database. Focus on any specific filters, desired aggregations, or conditions mentioned by the user.

Conversation History:
{context}

Please provide a refined summary of the user's requirements that can be used to generate the best search query.
Refined Requirements:"""
    
    refined_requirements = llm_wrapper.generate_response(prompt, max_tokens=150, temperature=0.3)
    return refined_requirements.strip()


def refine_and_select(user_query: str, conversation_history: list[dict], schema: dict, llm_wrapper: LLMWrapper) -> dict:
    """
    Consolidates query refinement, table selection, and routing into a single LLM call.
    It uses the latest conversation history, the user query, and the database schema.
    Returns a dictionary with keys:
      - "refined_query": a refined and detailed version of the user's query,
      - "selected_table": the most relevant table from the schema,
      - "route": either "SQL" or "SEMANTIC" indicating the recommended search method.
    """
    recent_context = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in conversation_history[-10:]
    )
    prompt = f"""You are an expert data analyst.
Based on the following conversation history:
{recent_context}

And the database schema:
{schema}

The user originally asked:
"{user_query}"

Please produce a JSON object with the following keys:
- "refined_query": a more detailed version of the user's query that includes explicit instructions (e.g., desired columns, filters, aggregations) to help generate a precise SQL query.
- "selected_table": the name of the table that is most relevant for answering the query.
- "route": either "SQL" if the refined query should be answered via SQL query generation, or "SEMANTIC" if a semantic search is more appropriate.

Respond ONLY with the JSON object.
Example:
{{
  "refined_query": "Find the total sales for Electronics in the last quarter, grouped by region.",
  "selected_table": "sales_data",
  "route": "SQL"
}}"""
    response = llm_wrapper.generate_response(prompt, max_tokens=150, temperature=0.3)
    try:
        data = json.loads(response)
    except Exception as e:
        logging.error(f"Error parsing LLM response: {e}")
        # Fallback defaults if parsing fails
        data = {
            "refined_query": user_query,
            "selected_table": next(iter(schema)) if schema else "",
            "route": "SQL"
        }
    return data
