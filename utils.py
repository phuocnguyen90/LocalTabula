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
# --- NEW: Load environment variables from .env file ---
from dotenv import load_dotenv
import yaml # <-- NEW: Import YAML
import logging
import uuid
import traceback # For detailed error logging in query processing

# --- Imports for Google Auth (If needed, keep commented otherwise) ---
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
# import pickle
# import gspread

# Import the new LLM wrapper and aux model loader
from llm_interface import LLMWrapper
from model import load_aux_models # Renamed import


dotenv.load_dotenv('.env', override=True) # Searches for .env file and loads variables

SQLITE_TIMEOUT_SECONDS = 15
# SHEETS_SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly'] # Keep if using private sheets
# DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
# ALL_SCOPES = SHEETS_SCOPES + DRIVE_SCOPES
QDRANT_COLLECTION_PREFIX = "table_data_"
PROMPT_FILE = "prompts.yaml" # <-- NEW: Define prompt file path

# --- NEW: Load Prompts ---
@st.cache_data # Cache the loaded prompts
def load_prompts(filepath=PROMPT_FILE):
    """Loads prompts from a YAML file."""
    try:
        with open(filepath, 'r') as f:
            prompts = yaml.safe_load(f)
            logging.info(f"Prompts loaded successfully from {filepath}")
            return prompts
    except FileNotFoundError:
        logging.error(f"FATAL: Prompts file not found at {filepath}")
        st.error(f"Error: Prompts file '{filepath}' not found. Cannot continue.")
        return None # Indicate failure
    except yaml.YAMLError as e:
        logging.error(f"FATAL: Error parsing YAML file {filepath}: {e}")
        st.error(f"Error: Could not parse prompts file '{filepath}'. Check its format.")
        return None # Indicate failure
    except Exception as e:
        logging.error(f"FATAL: Unexpected error loading prompts from {filepath}: {e}")
        st.error(f"An unexpected error occurred loading prompts: {e}")
        return None

PROMPTS = load_prompts() # Load prompts globally for the module

# Ensure prompts loaded before proceeding
if PROMPTS is None:
    st.stop() # Halt execution if prompts failed to load

# --- NEW: Environment Setup Function ---
def setup_environment():
    """
    Determines DB path based on DEVELOPMENT_MODE env variable.
    Creates the necessary directories.
    Returns the absolute database path or None on failure.
    """
    dev_mode_str = os.getenv('DEVELOPMENT_MODE', 'false').lower()
    is_dev_mode = dev_mode_str in ('true', '1', 't', 'yes', 'y')
    app_data_dir = None
    db_filename = "chat_database.db"

    try:
        if is_dev_mode:
            home_dir = os.path.expanduser("~")
            app_data_dir = os.path.join(home_dir, ".streamlit_chat_dev_data")
            print("INFO: Running in DEVELOPMENT mode.")
        else:
            current_working_dir = os.getcwd()
            app_data_dir = os.path.join(current_working_dir, "app_data")
            print("INFO: Running in PRODUCTION mode.")

        print(f"INFO: App data directory set to: {app_data_dir}")
        os.makedirs(app_data_dir, exist_ok=True)
        print(f"INFO: Ensured app data directory exists.")
        db_path = os.path.join(app_data_dir, db_filename)
        print(f"INFO: Database path set to: {db_path}")
        return db_path

    except OSError as e:
        print(f"ERROR: Failed to create data directory '{app_data_dir}': {e}. Check permissions.")
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
    if not sheet_url.startswith("https://docs.google.com/spreadsheets/d/"):
        raise ValueError("Invalid Google Sheet URL. Ensure it's a published link.")
    sheet_id = sheet_url.split("/d/")[1].split("/")[0]
    url= f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise ValueError(f"Error reading Google Sheet: {e}")
    if df.empty:
        raise ValueError("The Google Sheet is empty or not accessible.")
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

# --- NEW: Cached LLM Wrapper Initialization ---
@st.cache_resource
def get_llm_wrapper():
    """Initializes and caches the LLMWrapper."""
    logging.info("Attempting to get/initialize LLMWrapper...")
    wrapper = LLMWrapper()
    if not wrapper.is_ready:
        logging.error("LLMWrapper initialization failed.")
    return wrapper # app.py should check wrapper.is_ready

# --- Modified: Cached Auxiliary Model Loading ---
@st.cache_resource
def get_cached_aux_models():
    """Loads and caches auxiliary models (SQL, Embedder)."""
    logging.info("Cache miss or first run: Calling load_aux_models...")
    models_dict_from_loader = load_aux_models()

    if not isinstance(models_dict_from_loader, dict):
        logging.error("CRITICAL [utils.py]: load_aux_models did NOT return a dictionary!")
        return {"status": "error", "error_message": f"Internal Error: Model loader returned type {type(models_dict_from_loader)} instead of dict."}

    if models_dict_from_loader.get("status") != "loaded":
         logging.warning(f"Auxiliary model loading status: {models_dict_from_loader.get('status')}. Error: {models_dict_from_loader.get('error_message', 'N/A')}")
    else:
        logging.info("Aux models loaded and cached successfully.")
    return models_dict_from_loader


def _suggest_semantic_columns(df_head: pd.DataFrame, schema: dict, table_name: str, llm_wrapper: LLMWrapper) -> list[str]:
    """
    Uses the LLM to suggest columns suitable for semantic search embedding.
    Loads prompt from PROMPTS and cleans LLM response markdown.
    """
    logging.info(f"Requesting LLM suggestion for semantic columns in table '{table_name}'...")
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for semantic column suggestion. Falling back to simple text column selection.")
        return df_head.select_dtypes(include=['object', 'string']).columns.tolist()

    schema_str = json.dumps(schema.get(table_name, "Schema unavailable"), indent=2)
    df_head_str = df_head.to_string(index=False, max_rows=5)

    prompt = PROMPTS['suggest_semantic_columns'].format(
        table_name=table_name,
        schema_str=schema_str,
        df_head_str=df_head_str
    )
    logging.debug(f"Sending semantic column suggestion prompt to LLM:\n{prompt}")

    suggested_columns = []
    raw_llm_output = None # Store raw output for debugging
    try:
        if llm_wrapper.mode == 'openrouter':
             # Use generate_response for consistency and handle markdown cleaning below
             raw_llm_output = llm_wrapper.generate_response(prompt + " []", max_tokens=150) # Add default if empty
             logging.debug(f"Raw OpenRouter text suggestion response: {raw_llm_output}")
             # --- FIX: Clean Markdown before parsing JSON ---
             cleaned_text = raw_llm_output.strip()
             if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
             if cleaned_text.startswith("```"): cleaned_text = cleaned_text[3:]
             if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
             cleaned_text = cleaned_text.strip()

             try:
                 parsed_list = json.loads(cleaned_text)
                 if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                     suggested_columns = parsed_list
                     logging.info(f"LLM suggested (parsed from OpenRouter text): {suggested_columns}")
                 else:
                     logging.warning(f"Parsed OpenRouter text response was not a valid list of strings: {parsed_list}")
             except json.JSONDecodeError as e:
                 logging.error(f"Failed to parse JSON response from OpenRouter text: {e}")
                 logging.error(f"Invalid JSON received: {cleaned_text}") # Log the cleaned text
                 # Optionally try a fallback or raise error
        else: # Local GGUF mode
             raw_llm_output = llm_wrapper.generate_response(prompt + ' []', max_tokens=150)
             logging.debug(f"Raw GGUF text suggestion response: {raw_llm_output}")
             # --- FIX: Clean Markdown before parsing JSON ---
             cleaned_text = raw_llm_output.strip()
             if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
             if cleaned_text.startswith("```"): cleaned_text = cleaned_text[3:]
             if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
             cleaned_text = cleaned_text.strip()

             try:
                 parsed_list = json.loads(cleaned_text)
                 if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                     suggested_columns = parsed_list
                     logging.info(f"LLM suggested (parsed from GGUF text): {suggested_columns}")
                 else:
                      logging.warning(f"Parsed GGUF text response was not a valid list of strings: {parsed_list}")
             except json.JSONDecodeError as e:
                  logging.error(f"Could not parse GGUF LLM text response as JSON: {e}")
                  logging.error(f"Invalid JSON (GGUF): '{cleaned_text}'")


    except Exception as e:
        logging.error(f"Error during LLM semantic column suggestion: {e}", exc_info=True)

    # Final fallback & validation
    if not suggested_columns:
        logging.warning("LLM suggestion failed or returned empty. Falling back to selecting all object/string columns.")
        suggested_columns = df_head.select_dtypes(include=['object', 'string']).columns.tolist()

    final_columns = [col for col in suggested_columns if col in df_head.columns]
    if len(final_columns) != len(suggested_columns):
         logging.warning(f"LLM suggested columns not present in DataFrame were filtered out. Original: {suggested_columns}, Final: {final_columns}")

    logging.info(f"Final semantic columns selected for embedding: {final_columns}")
    return final_columns

def _embed_and_index_data(df: pd.DataFrame, table_name: str, semantic_columns: list, aux_models: dict, qdrant_client: QdrantClient):
    """
    Embeds specified semantic columns and indexes them in Qdrant.
    Uses UUIDs for Qdrant point IDs universally.
    Stores original meaningful ID in payload.
    Includes enhanced logging.
    """
    logging.info(f"Starting embedding and indexing process for table '{table_name}'...")

    # --- Readiness Checks ---
    if not qdrant_client:
        logging.error(f"[{table_name}] Qdrant client not available. Aborting indexing.")
        return False, "Qdrant client not initialized."
    if not aux_models or aux_models.get('status') != 'loaded' or not aux_models.get('embedding_model'):
        logging.error(f"[{table_name}] Embedding model not loaded or aux models failed. Aborting indexing.")
        return False, "Embedding model not loaded or aux models failed."

    embedding_model = aux_models.get('embedding_model')
    if df.empty:
        logging.info(f"[{table_name}] DataFrame is empty. Skipping embedding and collection creation.")
        return True, "DataFrame is empty."
    if not semantic_columns:
        logging.warning(f"[{table_name}] No semantic columns specified. Skipping embedding and collection creation.")
        return True, "No semantic columns specified."

    valid_semantic_cols = [col for col in semantic_columns if col in df.columns]
    logging.info(f"[{table_name}] Valid semantic columns found in DataFrame: {valid_semantic_cols}")
    if not valid_semantic_cols:
        logging.warning(f"[{table_name}] Valid semantic columns list is empty. Skipping embedding.")
        return True, "Valid semantic columns list is empty."

    logging.info(f"[{table_name}] Proceeding with embedding for columns: {valid_semantic_cols}")
    collection_name = f"{QDRANT_COLLECTION_PREFIX}{table_name}"
    status_messages = []

    # --- Universal UUID Strategy for Qdrant Point IDs ---
    logging.info(f"[{table_name}] Using UUIDs for Qdrant point IDs.")

    try:
        documents_to_embed = []
        payloads = []
        point_ids = [] # These will ALWAYS be UUIDs now

        potential_pk = None
        if not df.columns.empty:
             potential_pk = next((col for col in df.columns if col.lower() == 'id'),
                                 df.columns[0] if df.columns[0].lower() in ['id', 'pk', 'key'] else None)
        logging.info(f"[{table_name}] Potential primary key column identified: {potential_pk}")

        logging.info(f"[{table_name}] Preparing documents for embedding from {len(df)} rows...")
        rows_with_text = 0
        for index, row in df.iterrows():
            text_parts = [f"{col}: {row[col]}" for col in valid_semantic_cols if pd.notna(row[col]) and str(row[col]).strip()]
            if not text_parts: continue
            rows_with_text += 1
            doc_text = " | ".join(text_parts)

            payload = row.to_dict()
            for k, v in payload.items():
                 if pd.isna(v): payload[k] = None
            payload["_table_name"] = table_name
            payload["_source_text"] = doc_text

            # --- Determine Original Meaningful ID (for payload) ---
            original_id_value = f"{table_name}_idx_{index}" # Default
            if potential_pk and potential_pk in row and pd.notna(row[potential_pk]):
                try:
                     pk_val_str = str(row[potential_pk])
                     # Basic sanitization for common problematic chars if needed for display ID
                     # pk_val_str_sanitized = pk_val_str.replace(" ", "_").replace("/", "-")
                     original_id_value = f"{table_name}_{pk_val_str}"
                except Exception as pk_str_err:
                     logging.warning(f"[{table_name}] Could not convert PK '{row[potential_pk]}' to string for payload ID (Row {index}): {pk_str_err}. Using index fallback.")
                     original_id_value = f"{table_name}_idx_{index}"
            # --- Store Original ID in Payload ---
            payload["_original_id"] = original_id_value
            # --- End Original ID Logic ---

            # --- Generate UUID for Qdrant Point ID ---
            qdrant_point_id = str(uuid.uuid4())
            # --- End Qdrant Point ID Logic ---

            point_ids.append(qdrant_point_id) # Add the UUID to the list for Qdrant
            payloads.append(payload)
            documents_to_embed.append(doc_text)

        logging.info(f"[{table_name}] Prepared {len(documents_to_embed)} documents from {rows_with_text} rows with text.")
        if not documents_to_embed:
            logging.warning(f"[{table_name}] No non-empty documents found to embed. Skipping embedding.")
            return True, "No non-empty documents to embed."

        # --- Get Embeddings ---
        logging.info(f"[{table_name}] Generating {len(documents_to_embed)} embeddings...")
        try:
            embeddings_result = embedding_model.embed(documents_to_embed)
            if not isinstance(embeddings_result, (list, np.ndarray)):
                 embeddings = list(embeddings_result)
                 logging.info(f"[{table_name}] Converted embedding result from {type(embeddings_result)} to list.")
            else:
                 embeddings = embeddings_result

            embedding_count = len(embeddings) if embeddings is not None else 0
            logging.info(f"[{table_name}] Embedding generation completed. Received {embedding_count} embeddings.")
            if not embeddings or embedding_count != len(documents_to_embed):
                 logging.error(f"[{table_name}] Embedding failed or returned incorrect count ({embedding_count} vs {len(documents_to_embed)}). Aborting.")
                 return False, "Embedding failed or returned incorrect number of vectors."
        except Exception as embed_e:
            logging.error(f"[{table_name}] Error during embedding call: {embed_e}", exc_info=True)
            return False, f"Embedding generation failed: {embed_e}"

        # --- Get Vector Size ---
        try:
            vector_size = len(embeddings[0])
            logging.info(f"[{table_name}] Determined vector size: {vector_size}")
        except IndexError:
             logging.error(f"[{table_name}] Failed to get vector size from empty embeddings list. Aborting.")
             return False, "Failed to get vector size from embeddings."
        except Exception as size_e:
             logging.error(f"[{table_name}] Error determining vector size: {size_e}", exc_info=True)
             return False, f"Error determining vector size: {size_e}"

        # --- Collection Handling ---
        logging.info(f"[{table_name}] Checking/Creating Qdrant collection '{collection_name}'...")
        try:
            qdrant_client.get_collection(collection_name=collection_name)
            logging.warning(f"[{table_name}] Collection '{collection_name}' exists. Recreating.")
            qdrant_client.delete_collection(collection_name=collection_name, timeout=10)
            status_messages.append(f"Recreated collection '{collection_name}'.")
            time.sleep(0.5)
        except Exception:
             logging.info(f"[{table_name}] Collection '{collection_name}' not found. Will create new.")

        try:
            logging.info(f"[{table_name}] Attempting creation of '{collection_name}' size {vector_size}...")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            logging.info(f"[{table_name}] Successfully created/ensured collection '{collection_name}'.")
            status_messages.append(f"Created collection '{collection_name}'.")
        except Exception as create_e:
             logging.error(f"[{table_name}] Failed to create Qdrant collection '{collection_name}': {create_e}", exc_info=True)
             return False, f"Failed to create Qdrant collection: {create_e}"

        # --- Upsert data ---
        batch_size = 100
        num_batches = (len(point_ids) + batch_size - 1) // batch_size
        logging.info(f"[{table_name}] Upserting {len(point_ids)} points to Qdrant in {num_batches} batches...")
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(point_ids))
            batch_ids = point_ids[start_idx:end_idx] # Use UUIDs here
            batch_vectors = [list(map(float, vec)) for vec in embeddings[start_idx:end_idx]]
            batch_payloads = payloads[start_idx:end_idx]

            logging.debug(f"[{table_name}] Upserting batch {i+1}/{num_batches} (size {len(batch_ids)})")
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=models.Batch(
                        ids=batch_ids, # These are now UUIDs
                        vectors=batch_vectors,
                        payloads=batch_payloads
                    ),
                    wait=True
                )
            except Exception as upsert_e:
                 logging.error(f"[{table_name}] Error upserting batch {i+1} to Qdrant: {upsert_e}", exc_info=True)
                 return False, f"Error during Qdrant upsert: {upsert_e}"

        logging.info(f"[{table_name}] Qdrant upsert complete.")
        status_messages.append(f"Indexed {len(point_ids)} points.")
        return True, "\n".join(status_messages)

    except Exception as e:
        logging.error(f"[{table_name}] Unexpected error during embedding/indexing: {e}", exc_info=True)
        return False, f"Unexpected error during indexing: {e}"


# --- utils.py ---
# utils.py

from qdrant_client import QdrantClient, models # Ensure models is imported

# ... other code ...

def get_qdrant_collection_info(qdrant_client: QdrantClient, collection_name: str) -> dict | None:
    """Gets point count and vector dimension for a Qdrant collection."""
    if not qdrant_client: return None
    try:
        info = qdrant_client.get_collection(collection_name=collection_name)

        # --- Extract basic info safely ---
        points_count = getattr(info, 'points_count', 0)
        vectors_count = getattr(info, 'vectors_count', 0)
        status = getattr(info, 'status', 'Unknown')
        logging.debug(f"[{collection_name}] Raw CollectionInfo attributes: {dir(info)}") # Log all attributes

        # --- Attempt to find vector size in different potential locations ---
        vec_size = "N/A"
        vectors_config_obj = getattr(info, 'vectors_config', None)
        config_obj = getattr(info, 'config', None)

        # 1. Standard location (info.vectors_config)
        if isinstance(vectors_config_obj, models.VectorParams):
             vec_size = getattr(vectors_config_obj, 'size', 'N/A')
             logging.debug(f"[{collection_name}] Vector size found in info.vectors_config")
        elif isinstance(vectors_config_obj, dict): # Named vectors
            first_vec_name = next(iter(vectors_config_obj), None)
            if first_vec_name and isinstance(vectors_config_obj[first_vec_name], models.VectorParams):
                vec_size = getattr(vectors_config_obj[first_vec_name], 'size', 'N/A')
                logging.debug(f"[{collection_name}] Vector size found in info.vectors_config (named)")

        # 2. Nested location (info.config.params.vectors - check structure carefully)
        # This structure might vary greatly depending on client version and type
        elif config_obj and hasattr(config_obj, 'params') and hasattr(config_obj.params, 'vectors'):
             vectors_param = getattr(config_obj.params, 'vectors', None)
             if isinstance(vectors_param, models.VectorParams):
                  vec_size = getattr(vectors_param, 'size', 'N/A')
                  logging.debug(f"[{collection_name}] Vector size found in info.config.params.vectors")
             elif isinstance(vectors_param, dict): # Named vectors under config.params
                  first_vec_name = next(iter(vectors_param), None)
                  if first_vec_name and isinstance(vectors_param[first_vec_name], models.VectorParams):
                       vec_size = getattr(vectors_param[first_vec_name], 'size', 'N/A')
                       logging.debug(f"[{collection_name}] Vector size found in info.config.params.vectors (named)")

        # 3. Log warning if still not found
        if vec_size == "N/A":
             logging.warning(f"[{collection_name}] Could not determine vector size from CollectionInfo. Checked info.vectors_config and info.config.params.vectors.")
             # Log the structure for debugging if size wasn't found
             if config_obj and hasattr(config_obj, 'params'):
                  logging.debug(f"[{collection_name}] info.config.params structure: {getattr(config_obj.params, '__dict__', 'N/A')}")
             elif vectors_config_obj:
                  logging.debug(f"[{collection_name}] info.vectors_config structure: {getattr(vectors_config_obj, '__dict__', 'N/A')}")
             else:
                 logging.debug(f"[{collection_name}] Neither info.vectors_config nor info.config found or structured as expected.")


        return {
            "points_qdrantcount": points_count,
            "vectors_count": vectors_count,
            "vector_size": vec_size,
            "status": str(status)
        }
    except Exception as e:
        # Handle common errors gracefully
        if "not found" in str(e).lower() or "status_code=404" in str(e):
             logging.info(f"Qdrant collection '{collection_name}' not found.")
        elif isinstance(e, AttributeError):
             logging.warning(f"Attribute error getting Qdrant info for '{collection_name}': {e}. Structure might have changed.")
        else:
             logging.warning(f"Could not get Qdrant info for '{collection_name}': {type(e).__name__}: {e}")
        return None

# --- process_uploaded_data ---
def process_uploaded_data(uploaded_file, gsheet_published_url, table_name, conn, llm_model, aux_models, qdrant_client, replace_confirmed=False):
    """
    Reads data, writes to SQLite (with confirmation), analyzes, embeds, indexes.
    Includes detailed debugging for Length mismatch errors.
    """
    # --- Initial checks ---
    if not conn: return False, "Database connection lost."
    if not aux_models or aux_models.get('status') != 'loaded': return False, "Models not loaded."
    if not qdrant_client: return False, "Qdrant client not available."
    if not table_name or not table_name.isidentifier(): return False, f"Invalid table name: '{table_name}'."
    if not uploaded_file and not gsheet_published_url: return False, "No data source provided."
    if uploaded_file and gsheet_published_url: return False, "Provide only one data source."

    status_messages = []
    df = None
    error_msg = None
    expected_columns = 0 # To track expected column count

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
                error_msg = f"Error reading Excel file: {e}"; st.error(error_msg)
        elif gsheet_published_url:
            status_messages.append(f"Attempting to read Published Google Sheet CSV URL...")
            try:
                df = read_google_sheet(gsheet_published_url)
                status_messages.append(f"Read {len(df)} rows from URL. Shape: {df.shape}")
                read_success = True
            except ValueError as ve:
                error_msg = f"Error reading Google Sheet: {ve}"; st.error(error_msg)

        if not read_success or df is None:
            final_error = error_msg if error_msg else "Failed to read data source or data source is empty."
            return False, "\n".join(status_messages + [final_error])

        initial_shape = df.shape
        original_columns = df.columns.tolist()
        expected_columns = len(original_columns)
        status_messages.append(f"Initial DataFrame shape: {initial_shape}. Columns: {expected_columns}")
        logging.debug(f"DEBUG: Initial columns read ({expected_columns}): {original_columns}")

        # --- Column Sanitization (Robust Implementation) ---
        status_messages.append("Sanitizing column names for SQL compatibility...")
        sanitized_columns = []
        try:
            for i, col in enumerate(original_columns):
                col_str = str(col).strip()
                if not col_str:
                    sanitized = f'col_{i}_blank'
                else:
                    sanitized = ''.join(c for c in col_str.replace(' ', '_').replace('%', 'perc').replace('/', '_').replace('.', '').replace(':', '').replace('-', '_').replace('(', '').replace(')', '') if c.isalnum() or c == '_')
                    if not sanitized or (not sanitized[0].isalpha() and sanitized[0] != '_'):
                        sanitized = '_' + sanitized
                reserved_sql = ['select', 'from', 'where', 'index', 'table', 'insert', 'update', 'delete', 'order', 'group']
                if not sanitized.isidentifier() or sanitized.lower() in reserved_sql:
                     safe_original = ''.join(c for c in col_str if c.isalnum() or c == '_')[:20]
                     sanitized = f'col_{i}_' + safe_original
                     if not sanitized.isidentifier(): sanitized = f'col_{i}' # Absolute fallback
                sanitized_columns.append(sanitized)

            if len(sanitized_columns) != expected_columns:
                error_msg = f"CRITICAL SANITIZE ERROR: Length mismatch AFTER sanitizing. Expected {expected_columns}, Got {len(sanitized_columns)}."
                logging.error(error_msg); logging.error(f"Original: {original_columns}, Sanitized: {sanitized_columns}")
                st.error(error_msg)
                from collections import Counter; dupes = [item for item, count in Counter(sanitized_columns).items() if count > 1]
                if dupes: st.warning(f"Potential cause: Duplicate column names after sanitization: {dupes}")
                return False, "\n".join(status_messages + [error_msg])
            else:
                df.columns = sanitized_columns
                renamed_columns = df.columns.tolist()
                status_messages.append("Column names sanitized.")
                if original_columns != renamed_columns:
                    st.info("Note: Some column names were adjusted for compatibility.")

        except Exception as sanitize_e:
            error_msg = f"Error during column name sanitization: {sanitize_e}"
            logging.error(error_msg, exc_info=True); st.error(error_msg)
            return False, "\n".join(status_messages + [error_msg])

        # --- Check DataFrame state BEFORE writing ---
        current_shape = df.shape
        current_columns_count = len(df.columns)
        status_messages.append(f"DataFrame shape before write: {current_shape}. Columns: {current_columns_count}")
        logging.debug(f"DEBUG: Columns before write ({current_columns_count}): {df.columns.tolist()}")
        if current_columns_count != expected_columns:
             error_msg = f"CRITICAL ERROR: Column count changed unexpectedly before write. Expected {expected_columns}, Found {current_columns_count}."
             logging.error(error_msg); st.error(error_msg)
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
        except sqlite3.OperationalError as e: error_msg = f"SQLite error: {'Database locked' if 'locked' in str(e) else e}"; logging.error(error_msg); st.error(error_msg)
        except ValueError as e: error_msg = f"ValueError writing data (check types/lengths): {e}"; logging.error(error_msg); st.error(error_msg)
        except Exception as e: error_msg = f"Unexpected error writing data: {e}"; logging.error(error_msg, exc_info=True); st.error(error_msg)
        if "Length mismatch" in str(error_msg): error_msg += f". DF Shape: {df.shape}. Expected cols: {expected_columns}."

        if write_successful:
            # --- Analyze, Embed, Index ---
            status_messages.append("Analyzing columns for embedding (using LLM)...")
            current_schema = {table_name: df.columns.tolist()}
            semantic_cols_to_embed = _suggest_semantic_columns(df.head(), current_schema, table_name, llm_model)
            status_messages.append(f"LLM suggested for embedding: {semantic_cols_to_embed}")

            status_messages.append("Starting embedding & indexing...")
            embed_success, embed_msg = _embed_and_index_data(df, table_name, semantic_cols_to_embed, aux_models, qdrant_client)
            status_messages.append(f"Embedding/Indexing result: {embed_msg}")

            if embed_success:
                status_messages.append(f"Processing complete for `{table_name}`.")
                return True, "\n".join(status_messages)
            else:
                 status_messages.append(f"SQL write OK, but embedding/indexing failed.")
                 return False, "\n".join(status_messages)
        else: # Write failed
             return False, "\n".join(status_messages + [f"Failed during SQLite write: {error_msg}"])

    except Exception as e:
        critical_error_msg = f"Critical error during data processing: {e} (Expected Columns: {expected_columns}, DF State: {'Exists' if df is not None else 'None'})"
        st.error(critical_error_msg); logging.error(critical_error_msg, exc_info=True)
        return False, critical_error_msg

# --- Database Connection ---
@st.cache_resource
def get_db_connection(db_path):
    """Establishes and caches SQLite connection using the provided path."""
    print(f"Attempting DB connection to: {db_path}")
    if not db_path:
         logging.error("Invalid database path provided to get_db_connection.")
         return None
    try:
        conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_SECONDS, check_same_thread=False)
        try: conn.execute("PRAGMA journal_mode=WAL;") ; print("INFO: WAL mode enabled.")
        except Exception as wal_e: print(f"WARNING: Could not enable WAL mode: {wal_e}")
        print("INFO: DB Connection Successful.")
        return conn
    except Exception as e:
        print(f"ERROR: Failed to connect to DB at {db_path}: {e}")
        return None

# --- Qdrant Client Initialization ---
@st.cache_resource
def init_qdrant_client():
    """Initializes and caches an in-memory Qdrant client."""
    print("Initializing in-memory Qdrant client...")
    try:
        client = QdrantClient(":memory:")
        print("Qdrant client initialized.")
        return client
    except Exception as e:
        print(f"FATAL QDRANT ERROR: {e}")
        return None

# --- REMOVED load_models (replaced by get_cached_aux_models) ---

# --- Data Processing Functions ---

def get_schema_info(conn):
    """Fetches table names and columns from the SQLite DB."""
    if not conn: return {}
    schema = {}
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            if table_name.startswith("sqlite_"): continue
            cursor.execute(f"PRAGMA table_info('{table_name}');") # Quote table name
            columns = [info[1] for info in cursor.fetchall()]
            schema[table_name] = columns
        return schema
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return {}

# --- REMOVED _analyze_columns_for_embedding placeholder ---


# --- Query Processing Functions ---

def _route_query(user_query: str, schema: dict, llm_wrapper: LLMWrapper, sample_data_str: str | None = None) -> str:
    """
    Determines query route ('SQL' or 'SEMANTIC') using LLM structured output if available,
    otherwise falls back to keywords. Uses prompt from PROMPTS.
    """
    default_route = "SQL"
    logging.info("Routing query...")

    if not llm_wrapper or not llm_wrapper.is_ready or llm_wrapper.mode != 'openrouter':
        logging.warning(f"LLM wrapper not ready or not in OpenRouter mode ({getattr(llm_wrapper, 'mode', 'N/A')}). Falling back to keyword routing.")
        query_lower = user_query.lower()
        sql_kws = ["how many", "list", "total", "average", "maximum", "minimum", "count", "sum", "max", "min", "avg", "group by", "order by", " where ", " id ", " date ", " year ", " month ", " price", " number of ", " value of "]
        sem_kws = ["describe", "description", "details about", "similar to", "meaning", "related to", "find products like", "tell me about", "explain"]
        if any(kw in query_lower for kw in sem_kws): return "SEMANTIC"
        if any(kw in query_lower for kw in sql_kws): return "SQL"
        logging.info(f"Keyword fallback ambiguous. Defaulting to: {default_route}")
        return default_route

    logging.info("Attempting routing via OpenRouter structured output...")
    # --- Use loaded prompt ---
    sample_context_for_prompt = f"\nSample Data:\n{sample_data_str}\n" if sample_data_str else ""
    prompt = PROMPTS['route_query_structured'].format(
        schema=schema,
        user_query=user_query,
        sample_data_context=sample_context_for_prompt # Pass sample data here
    )
    # --- End Use loaded prompt ---
    logging.debug(f"Sending structured routing prompt to LLM:\n{prompt}")

    structured_result = llm_wrapper.generate_structured_response(prompt)

    if structured_result and isinstance(structured_result, dict):
        query_type = structured_result.get("query_type")
        if query_type in ["SQL", "SEMANTIC"]:
            logging.info(f"Structured routing successful. Determined type: {query_type}")
            return query_type
        else:
            logging.warning(f"LLM returned valid JSON but with unexpected 'query_type' value: '{query_type}'. Falling back.")
    else:
        logging.warning(f"LLM structured response failed or returned invalid result: {structured_result}. Falling back.")

    logging.info(f"Structured routing failed. Defaulting route to: {default_route}")
    return default_route


def _generate_sql_query(user_query: str, schema: dict, sample_data_str: str | None, aux_models, previous_sql: str | None = None, feedback: str | None = None) -> str:
    """
    Generates SQL query, incorporating feedback from previous attempts if provided.
    Loads prompt components from PROMPTS.
    """
    sql_llm: Llama = aux_models.get('sql_gguf_model')
    if not sql_llm:
        return "-- Error: SQL GGUF model object missing."

    sample_context = f"Data Sample (representative table):\n{sample_data_str}\n" if sample_data_str else "Sample data not available.\n"

    # --- Construct Prompt using loaded components ---
    prompt_parts = [
        PROMPTS['generate_sql_base'].format(
            schema=schema,
            sample_context=sample_context,
            user_query=user_query
        )
    ]

    if previous_sql and feedback:
        if "syntax error" in feedback.lower():
            prompt_parts.append(
                PROMPTS['generate_sql_feedback_syntax'].format(
                    previous_sql=previous_sql,
                    feedback=feedback
                )
            )
        else:
            prompt_parts.append(
                PROMPTS['generate_sql_feedback_other'].format(
                    previous_sql=previous_sql,
                    feedback=feedback
                )
            )

    prompt_parts.append(PROMPTS['generate_sql_response_format'])
    prompt = "\n".join(prompt_parts)
    # --- End Prompt Construction ---

    logging.info(f"Sending prompt to SQL GGUF model (Retry Attempt? {'Yes' if previous_sql else 'No'}):\n{prompt}")
    try:
        output = sql_llm(
            prompt, max_tokens=500, temperature=0.1, top_p=0.9,
            stop=[";", "\n\n", "```"], echo=False
        )
        if output and 'choices' in output and len(output['choices']) > 0:
            generated_sql = output['choices'][0]['text'].strip()
            logging.info(f"Raw SQL GGUF output: {generated_sql}")
            # Clean the output (same logic as before)
            if "```sql" in generated_sql: generated_sql = generated_sql.split("```sql")[1]
            if "```" in generated_sql: generated_sql = generated_sql.split("```")[0]
            sql_keywords = ["SELECT", "WITH"]; first_kw_pos = -1
            for keyword in sql_keywords:
                pos = generated_sql.upper().find(keyword)
                if pos != -1 and (first_kw_pos == -1 or pos < first_kw_pos): first_kw_pos = pos
            cleaned_sql = generated_sql[first_kw_pos:] if first_kw_pos != -1 else generated_sql
            if ';' in cleaned_sql: cleaned_sql = cleaned_sql.split(';')[0] + ';'
            elif cleaned_sql.strip(): cleaned_sql += ';'; logging.warning("Added missing semicolon.")
            cleaned_sql = cleaned_sql.strip()
            logging.info(f"Cleaned SQL query: {cleaned_sql}")
            if not cleaned_sql or not any(kw in cleaned_sql.upper() for kw in ["SELECT", "WITH"]):
                return "-- Error: Model did not generate a valid SQL query structure."
            return cleaned_sql
        else:
            logging.error(f"SQL GGUF model returned empty/invalid output: {output}")
            return "-- Error: SQL model returned empty or invalid output."
    except Exception as e:
        logging.error(f"Error during SQL GGUF generation: {e}", exc_info=True)
        return f"-- Error generating SQL: {e}"


def _execute_sql_query(conn, sql_query):
    """Executes the SQL query and returns results as a DataFrame."""
    if not conn:
        return pd.DataFrame(), "Database connection is not available."
    try:
        logging.info(f"Executing SQL: {sql_query}")
        result_df = pd.read_sql_query(sql_query, conn)
        logging.info("SQL execution successful.")
        return result_df, None # Data, No error message
    except pd.io.sql.DatabaseError as e:
         logging.error(f"SQL Execution Error: {e}")
         err_msg = f"SQL Error: {e}"
         if "no such table" in str(e): err_msg = f"Error: Table in query might not exist. Check schema. ({e})"
         elif "no such column" in str(e): err_msg = f"Error: Column in query might not exist. Check schema. ({e})"
         return pd.DataFrame(), err_msg
    except Exception as e:
        logging.error(f"Unexpected SQL Execution Error: {e}", exc_info=True)
        return pd.DataFrame(), f"An unexpected error occurred during SQL execution: {e}"


# --- Semantic Search Function ---
def _perform_semantic_search(user_query: str, aux_models: dict, qdrant_client: QdrantClient, schema: dict) -> tuple[list[str], str | None]:
    """
    Performs semantic search using the embedding model and Qdrant.
    (No changes needed here related to prompts)
    """
    logging.info("Performing semantic search...")
    if not qdrant_client: return [], "Qdrant client not initialized."
    if not aux_models or aux_models.get('status') != 'loaded' or not aux_models.get('embedding_model'):
        return [], "Embedding model not loaded or aux models failed."

    embedding_model = aux_models.get('embedding_model')

    try:
        logging.info(f"Embedding user query: '{user_query[:100]}...'")
        query_embedding_result = list(embedding_model.embed([user_query]))
        if not query_embedding_result: return [], "Failed to generate query embedding."
        query_vector = query_embedding_result[0]
        logging.info("Query embedded successfully.")

        all_collections = qdrant_client.get_collections()
        target_collections = [c.name for c in all_collections.collections if c.name.startswith(QDRANT_COLLECTION_PREFIX)]

        if not target_collections:
            return ["No vector data found for searching. Please process data first."], None
        logging.info(f"Searching in Qdrant collections: {target_collections}")

        all_hits = []
        search_limit_per_collection = 5

        for collection_name in target_collections:
            try:
                search_result = qdrant_client.search(collection_name=collection_name, query_vector=query_vector, limit=search_limit_per_collection)
                logging.info(f"Found {len(search_result)} hits in '{collection_name}'.")
                all_hits.extend(search_result)
            except Exception as search_e:
                 logging.error(f"Error searching collection '{collection_name}': {search_e}")

        if not all_hits:
            return ["No relevant matches found in the vector data."], None

        all_hits.sort(key=lambda x: x.score, reverse=True)
        formatted_results = []
        max_total_results = 10

        logging.info(f"Processing top {min(len(all_hits), max_total_results)} hits...")
        for hit in all_hits[:max_total_results]:
            payload = hit.payload
            if not payload: continue

            table_name = payload.get("_table_name", "Unknown")
            display_text = payload.get("_source_text", str(payload))[:200] + ("..." if len(payload.get("_source_text", "")) > 200 else "")
            display_id = payload.get("_original_id", str(hit.id)) # Prefer original ID if stored
            # Clean up display ID if it has table prefix
            if isinstance(display_id, str) and display_id.startswith(f"{table_name}_"):
                 display_id = display_id.split('_', 1)[1]

            result_str = f"**Table:** `{table_name}` | **ID:** `{display_id}` | **Score:** {hit.score:.3f}\n> {display_text}"
            formatted_results.append(result_str)

        if not formatted_results:
             return ["No relevant matches found after processing hits."], None

        return formatted_results, None

    except Exception as e:
        logging.error(f"Unexpected error during semantic search: {e}", exc_info=True)
        return [], f"Error during semantic search: {e}"

def validate_inputs(query: str, conn, schema: dict, llm_wrapper, qdrant_client) -> dict:
    """Validates inputs and returns an initial result dict or an error dict."""
    if not query: return {"status": "error", "message": "Query empty."}
    if not conn: return {"status": "error", "message": "DB connection unavailable."}
    if not schema: return {"status": "error", "message": "No DB schema found. Load data first."}
    if not llm_wrapper: return {"status": "error", "message": "LLM wrapper not available."}
    if not qdrant_client: return {"status": "error", "message": "Qdrant client not available."}
    return {"status": "ok"}

def get_table_sample_data(conn, table_name: str, limit: int = 3) -> str:
    """Gets sample data as a markdown string."""
    sample_df = _get_table_sample_df(conn, table_name, limit=limit)
    if sample_df is None: return f"Could not retrieve sample for table '{table_name}'."
    if sample_df.empty: return f"Table '{table_name}' exists but is empty."
    return sample_df.to_markdown(index=False)


def _ensure_english_query(user_query: str, llm_wrapper: LLMWrapper) -> tuple[str, bool]:
    """
    Checks if query is English. If not, attempts translation using LLM.
    Uses prompts from PROMPTS.
    Returns (potentially_translated_query, was_translation_needed)
    """
    logging.info("Checking query language...")
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for language check/translation. Assuming English.")
        return user_query, False

    try:
        # --- Use loaded prompts ---
        lang_prompt = PROMPTS['check_language'].format(user_query=user_query)
        lang_response = llm_wrapper.generate_response(lang_prompt, max_tokens=5, temperature=0.1).strip().upper()

        if lang_response.startswith("YES"):
            logging.info("Query identified as English.")
            return user_query, False
        elif lang_response.startswith("NO"):
            logging.info("Query identified as non-English. Attempting translation...")
            trans_prompt = PROMPTS['translate_to_english'].format(user_query=user_query)
            # --- End Use loaded prompts ---
            translated_query = llm_wrapper.generate_response(trans_prompt, max_tokens=len(user_query) + 70, temperature=0.5).strip()

            if translated_query and translated_query != user_query and len(translated_query) > 5:
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

# --- REMOVED route_query_wrapper - logic integrated into _route_query ---

def execute_sql_pipeline(
    english_query: str,
    schema: dict, # Should be filtered schema for the selected table
    sample_data_str:str,
    aux_models,
    conn,
    llm_wrapper,
    max_retries: int = 1
) -> dict:
    """
    Attempts to generate and execute an SQL query with retries.
    Uses the filtered schema and sample data for the selected table.
    Returns a dictionary containing keys: sql_success, generated_sql, sql_data, sql_error.
    """
    sql_result = { "sql_success": False, "generated_sql": None, "sql_data": None, "sql_error": None }
    current_sql = None
    previous_feedback = None

    if not isinstance(aux_models, dict):
        logging.error("ERROR: aux_models is not a dict in execute_sql_pipeline!")
        sql_result["sql_error"] = "Internal Error: Model configuration invalid."
        return sql_result

    for attempt in range(max_retries + 1):
        logging.info(f"SQL Generation/Execution Attempt #{attempt + 1}")
        # Ensure schema and sample are for the relevant table only
        current_sql = _generate_sql_query(
            user_query=english_query, # Use the refined query
            schema=schema,           # Pass the filtered schema
            sample_data_str=sample_data_str, # Pass relevant sample
            aux_models=aux_models,
            previous_sql=current_sql,
            feedback=previous_feedback
        )

        sql_result["generated_sql"] = current_sql
        previous_feedback = None # Reset feedback

        if current_sql.startswith("-- Error"):
            feedback = f"SQL generation failed: {current_sql}"
            if attempt < max_retries: previous_feedback = feedback; continue
            else: sql_result["sql_error"] = feedback; break

        sql_df, exec_error = _execute_sql_query(conn, current_sql)
        if exec_error:
            feedback = f"SQL execution error: {exec_error}"
            if attempt < max_retries: previous_feedback = feedback; continue
            else: sql_result["sql_error"] = feedback; break

        # Validate results using the original query intent (or refined query?)
        # Using refined query for validation might be more precise
        is_valid, validation_feedback = _validate_sql_results(english_query, current_sql, sql_df, llm_wrapper)
        if is_valid:
            sql_result["sql_success"] = True; sql_result["sql_data"] = sql_df
            logging.info("SQL attempt successful and validated.")
            break
        else:
            feedback = f"Result validation failed: {validation_feedback}"
            if attempt < max_retries: previous_feedback = validation_feedback; continue
            else: sql_result["sql_error"] = feedback; break

    return sql_result


def execute_semantic_pipeline(
    english_query: str, # Use refined query
    schema: dict, # Pass filtered schema
    aux_models: dict,
    qdrant_client
) -> dict:
    """
    Executes semantic search and returns a dictionary containing:
    semantic_success, semantic_data, semantic_error.
    """
    semantic_result = { "semantic_success": False, "semantic_data": None, "semantic_error": None }
    semantic_data, error_msg = _perform_semantic_search(english_query, aux_models, qdrant_client, schema)
    if not error_msg:
        semantic_result["semantic_success"] = True
        semantic_result["semantic_data"] = semantic_data
    else:
        semantic_result["semantic_error"] = error_msg
    return semantic_result


def run_secondary_route(
    english_query: str, # Use refined query
    primary_route: str,
    sql_result: dict,
    semantic_result: dict,
    aux_models,
    conn,
    schema: dict, # Filtered schema
    llm_wrapper,
    qdrant_client,
    sample_data_str: str # Relevant sample
) -> tuple[dict, dict]:
    """
    If the primary route did not yield results, run the other route once.
    Returns updated sql_result and semantic_result dictionaries.
    """
    # Check if secondary run is needed (e.g., primary failed or yielded nothing)
    run_sql_secondary = (primary_route == "SEMANTIC" and not sql_result.get("sql_success"))
    run_semantic_secondary = (primary_route == "SQL" and not semantic_result.get("semantic_success"))

    if run_semantic_secondary:
        logging.info("Primary SQL failed or empty, running secondary Semantic search...")
        sec_semantic, sec_err = _perform_semantic_search(english_query, aux_models, qdrant_client, schema)
        if not sec_err: semantic_result.update({"semantic_success": True, "semantic_data": sec_semantic})
        else: semantic_result["semantic_error"] = sec_err # Keep primary SQL error info
    elif run_sql_secondary:
        logging.info("Primary Semantic failed or empty, running secondary SQL generation...")
        # Run only one attempt for secondary SQL
        sec_sql = _generate_sql_query(english_query, schema, sample_data_str, aux_models)
        sql_result["generated_sql"] = sec_sql # Update with secondary attempt
        if sec_sql.startswith("-- Error"):
             sql_result["sql_error"] = sec_sql # Keep primary semantic error info
        else:
            sec_sql_df, sec_sql_err = _execute_sql_query(conn, sec_sql)
            if not sec_sql_err:
                # Basic validation for secondary SQL result (optional, could skip LLM call)
                 is_valid, _ = _validate_sql_results(english_query, sec_sql, sec_sql_df, llm_wrapper)
                 if is_valid: sql_result.update({"sql_success": True, "sql_data": sec_sql_df, "sql_error": None})
                 else: sql_result["sql_error"] = "Secondary SQL result failed validation." # Keep primary error?
            else:
                sql_result["sql_error"] = sec_sql_err # Keep primary error?

    return sql_result, semantic_result


def generate_final_summary(
    original_query: str,
    english_query: str, # Refined query
    was_translated: bool,
    sql_data,
    semantic_data,
    llm_wrapper
) -> str:
    """
    Aggregates SQL and semantic results into a final natural language summary.
    Uses prompt from PROMPTS.
    """
    summary_parts = []
    if sql_data is not None:
        if not sql_data.empty:
            max_rows = 5
            summary_parts.append("SQL Query Results (Sample):")
            summary_parts.append(sql_data.head(max_rows).to_markdown(index=False))
            if len(sql_data) > max_rows:
                summary_parts.append(f"... ({len(sql_data) - max_rows} more rows)")
        else:
            summary_parts.append("The SQL query returned no matching rows.")

    if semantic_data and not ("No relevant matches found" in semantic_data[0] if semantic_data else True):
        max_snippets = 5
        summary_parts.append("\nSemantic Search Results (Snippets):")
        summary_parts.append("\n".join(semantic_data[:max_snippets]))
        if len(semantic_data) > max_snippets:
            summary_parts.append(f"... ({len(semantic_data) - max_snippets} more snippets)")

    summary_context = "\n".join(summary_parts)
    translated_context = f'\nQuery Interpreted as (English): "{english_query}"' if was_translated else ""

    if summary_context.strip():
        # --- Use loaded prompt ---
        summary_prompt = PROMPTS['generate_final_summary'].format(
            original_query=original_query,
            translated_context=translated_context,
            summary_context=summary_context.strip()
        )
        # --- End Use loaded prompt ---
        logging.info("Requesting final LLM summary based on aggregated results...")
        logging.debug(f"Summary prompt:\n{summary_prompt}")
        nl_summary = llm_wrapper.generate_response(summary_prompt, max_tokens=350) # Increased tokens slightly
        return nl_summary.strip()
    else:
        # If both SQL and Semantic yielded nothing useful
        return "I searched the data based on your query, but couldn't find specific information matching your request."


def process_natural_language_query(
    original_query: str, conn, schema: dict, llm_wrapper, aux_models, qdrant_client, max_retries: int = 1
) -> dict:
    """
    Main pipeline Refactored:
    1. Validate inputs. Check English.
    2. Consolidate refinement, table selection, routing via LLM (refine_and_select).
    3. Retrieve sample data for the selected table.
    4. Execute primary pipeline (SQL or semantic) using refined query and filtered context.
    5. Optionally run secondary route if primary failed.
    6. Generate a final natural language summary.
    Returns a result dictionary with relevant data.
    """
    # Initial validation
    result = validate_inputs(original_query, conn, schema, llm_wrapper, qdrant_client)
    if result.get("status") == "error": return result

    # Check/Translate Query Language (do this early)
    processed_query, was_translated = _ensure_english_query(original_query, llm_wrapper)
    result["query_was_translated"] = was_translated
    result["processed_query"] = processed_query # Store the query used for processing

    # Use conversation history
    conversation_history = st.session_state.get("messages", [])

    # --- Consolidate refinement, table selection, routing ---
    refinement_data = refine_and_select(processed_query, conversation_history, schema, llm_wrapper)
    refined_query = refinement_data.get("refined_query", processed_query) # Fallback to processed query
    selected_table = refinement_data.get("selected_table")
    recommended_route = refinement_data.get("route", "SQL").upper() # Default to SQL

    # Validate selected table
    if not selected_table or selected_table not in schema:
        logging.warning(f"LLM selected invalid table '{selected_table}'. Falling back.")
        # Fallback logic (e.g., first table or simple keyword match)
        selected_table = next(iter(schema), None) # Just take the first table
        if not selected_table: return {"status": "error", "message": "No tables found after LLM fallback."}
        # Optionally, re-route based on fallback table? For now, keep original route.

    result["refined_query"] = refined_query
    result["selected_table"] = selected_table
    result["determined_route"] = recommended_route
    logging.info(f"Refined Query: {refined_query}")
    logging.info(f"Selected Table: {selected_table}")
    logging.info(f"Recommended Route: {recommended_route}")

    # --- Prepare context for the selected table ---
    filtered_schema = {selected_table: schema[selected_table]}
    sample_data_str = get_table_sample_data(conn, selected_table, limit=3)

    # --- Execute Primary Pipeline ---
    sql_result = {"sql_success": False} # Initialize placeholders
    semantic_result = {"semantic_success": False}

    if recommended_route == "SQL":
        sql_result = execute_sql_pipeline(
            refined_query, filtered_schema, sample_data_str, aux_models, conn, llm_wrapper, max_retries
        )
    elif recommended_route == "SEMANTIC":
        semantic_result = execute_semantic_pipeline(
            refined_query, filtered_schema, aux_models, qdrant_client
        )
    else:
        return {"status": "error", "message": f"Internal Error: Unknown route '{recommended_route}' determined."}

    # --- Optionally Run Secondary Route ---
    # Pass the filtered context (schema, sample) for the *selected table* to the secondary route as well
    sql_result, semantic_result = run_secondary_route(
        refined_query, recommended_route, sql_result, semantic_result, aux_models, conn,
        filtered_schema, llm_wrapper, qdrant_client, sample_data_str
    )

    # --- Aggregate Results ---
    result["generated_sql"] = sql_result.get("generated_sql")
    result["sql_data"] = sql_result.get("sql_data")
    result["sql_error"] = sql_result.get("sql_error")
    result["semantic_data"] = semantic_result.get("semantic_data")
    result["semantic_error"] = semantic_result.get("semantic_error")

    # --- Generate Final Summary ---
    nl_summary = generate_final_summary(
        original_query,
        refined_query, # Use the refined query for context in the summary prompt
        was_translated,
        result.get("sql_data"),
        result.get("semantic_data"),
        llm_wrapper
    )
    result["natural_language_summary"] = nl_summary
    result["message"] = nl_summary # Main message for display
    result["status"] = "success" # Assuming summary generation is always 'successful' in structure
    # Data for potential detailed display (DataFrames, semantic strings)
    result["raw_display_data"] = {"sql": result.get("sql_data"), "semantic": result.get("semantic_data")}

    return result

# --- NEW Validation Helper ---
def _validate_sql_results(user_query: str, executed_sql: str, result_df: pd.DataFrame, llm_wrapper: LLMWrapper) -> tuple[bool, str | None]:
    """
    Uses the LLM wrapper to validate if the SQL results satisfy the user query.
    Loads prompt from PROMPTS.
    Returns: tuple[bool, str | None]: (is_satisfactory, feedback_or_None)
    """
    logging.info("Validating SQL results using LLM...")
    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for validation. Assuming results are satisfactory.")
        return True, None # Fail safely

    max_rows_for_context = 3
    context_str = "The query returned zero rows." if result_df.empty else result_df.head(max_rows_for_context).to_markdown(index=False)
    if not result_df.empty and len(result_df) > max_rows_for_context:
         context_str += f"\n\n...(and {len(result_df) - max_rows_for_context} more rows)"

    # --- Use loaded prompt ---
    prompt = PROMPTS['validate_sql_results'].format(
        user_query=user_query, # Use original or refined query here? Let's use the one passed in.
        executed_sql=executed_sql,
        context_str=context_str
    )
    # --- End Use loaded prompt ---

    logging.debug(f"Sending validation prompt to LLM:\n{prompt}")

    try:
        if llm_wrapper.mode == 'openrouter':
            structured_result = llm_wrapper.generate_structured_response(prompt)
            if structured_result and isinstance(structured_result, dict):
                is_satisfactory = structured_result.get("satisfactory")
                reason = structured_result.get("reason")
                if isinstance(is_satisfactory, bool):
                    feedback = reason if not is_satisfactory and isinstance(reason, str) and reason else None
                    logging.info(f"LLM validation (structured): Satisfactory={is_satisfactory}, Feedback='{feedback}'")
                    return is_satisfactory, feedback
                else: logging.warning(f"LLM validation JSON invalid 'satisfactory': {structured_result}. Assuming satisfactory.")
            else: logging.warning(f"LLM structured validation failed: {structured_result}. Assuming satisfactory.")
        else: # Fallback to text generation
             response_text = llm_wrapper.generate_response(prompt + ' {"satisfactory": true, "reason": ""}', max_tokens=100, temperature=0.1)
             logging.debug(f"Raw text validation response: {response_text}")
             try:
                 # Attempt to parse JSON even from text response
                 cleaned_text = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                 parsed_json = json.loads(cleaned_text)
                 if isinstance(parsed_json, dict):
                     is_satisfactory = parsed_json.get("satisfactory")
                     reason = parsed_json.get("reason")
                     if isinstance(is_satisfactory, bool):
                         feedback = reason if not is_satisfactory and isinstance(reason, str) and reason else None
                         logging.info(f"LLM validation (parsed text): Satisfactory={is_satisfactory}, Feedback='{feedback}'")
                         return is_satisfactory, feedback
                     else: logging.warning("Parsed validation JSON invalid 'satisfactory'. Assuming satisfactory.")
                 else: logging.warning("Parsed validation response not dict. Assuming satisfactory.")
             except json.JSONDecodeError:
                  logging.warning(f"Could not parse LLM text validation as JSON: {response_text}. Assuming satisfactory.")

    except Exception as e:
        logging.error(f"Error during LLM validation call: {e}", exc_info=True)

    return True, None # Default return if validation fails


# --- Helper to get data sample ---
def _get_table_sample_df(conn: sqlite3.Connection, table_name: str, limit: int = 3) -> pd.DataFrame | None:
    """Fetches the first few rows of a table as a DataFrame sample."""
    if not conn or not table_name or not table_exists(conn, table_name): # Added table_name check
        logging.warning(f"Cannot get sample: Connection invalid or table '{table_name}' does not exist.")
        return None
    try:
        query = f'SELECT * FROM "{table_name}" LIMIT {limit}' # Quote table name
        df_sample = pd.read_sql_query(query, conn)
        return df_sample
    except Exception as e:
        logging.error(f"Error fetching sample data for table '{table_name}': {e}")
        return None

# --- REMOVED redundant ensure_english_query (using _ensure_english_query) ---

def get_sqlite_table_row_count(conn: sqlite3.Connection, table_name: str) -> int | None:
    """Gets the row count for a specific SQLite table."""
    if not conn or not table_exists(conn, table_name): return None
    try:
        cursor = conn.cursor(); cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        count = cursor.fetchone()[0]; cursor.close(); return count
    except Exception as e:
        logging.error(f"Error getting row count for table '{table_name}': {e}")
        return None


def reindex_table(conn: sqlite3.Connection, table_name: str, llm_wrapper: LLMWrapper, aux_models: dict, qdrant_client: QdrantClient) -> tuple[bool, str]:
    """
    Reads data from SQLite table, re-suggests columns, re-embeds, and re-indexes in Qdrant.
    """
    logging.info(f"Attempting to re-index table '{table_name}'...")
    if not conn: return False, "DB connection unavailable."
    if not table_exists(conn, table_name): return False, f"SQLite table '{table_name}' not found."
    if not qdrant_client: return False, "Qdrant client unavailable."
    if not aux_models or aux_models.get('status') != 'loaded': return False, "Aux models not loaded."
    if not llm_wrapper or not llm_wrapper.is_ready: logging.warning("LLM wrapper not ready for re-suggesting columns.")

    try:
        logging.info(f"Reading data from SQLite table '{table_name}' for re-indexing...")
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        if df.empty: return False, f"Table '{table_name}' is empty, nothing to re-index."
        logging.info(f"Read {len(df)} rows from '{table_name}'.")

        current_schema = {table_name: df.columns.tolist()}
        semantic_cols_to_embed = _suggest_semantic_columns(df.head(), current_schema, table_name, llm_wrapper)
        if not semantic_cols_to_embed:
             return False, "No semantic columns identified or suggested for re-embedding."
        logging.info(f"Re-embedding with columns: {semantic_cols_to_embed}")

        logging.info(f"Calling embed and index for re-indexing '{table_name}'...")
        embed_success, embed_msg = _embed_and_index_data(df, table_name, semantic_cols_to_embed, aux_models, qdrant_client)

        if embed_success:
            msg = f"Successfully re-indexed table '{table_name}'. {embed_msg}"
            logging.info(msg); return True, msg
        else:
            msg = f"Re-indexing failed for table '{table_name}'. Reason: {embed_msg}"
            logging.error(msg); return False, msg

    except Exception as e:
        logging.error(f"Unexpected error during re-indexing of '{table_name}': {e}", exc_info=True)
        return False, f"Unexpected error during re-indexing: {e}"


def delete_table_data(conn: sqlite3.Connection, table_name: str, qdrant_client: QdrantClient) -> tuple[bool, str]:
    """Deletes table from SQLite and corresponding collection from Qdrant."""
    logging.info(f"Attempting to delete data for table '{table_name}'...")
    if not conn: return False, "DB connection unavailable."
    if not qdrant_client: return False, "Qdrant client unavailable."

    sqlite_success, qdrant_success = False, False
    sqlite_msg, qdrant_msg = "Not Attempted", "Not Attempted"

    try:
        with conn: conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        sqlite_success = True; sqlite_msg = f"SQLite table '{table_name}' dropped."
        logging.info(sqlite_msg)
    except Exception as e:
        sqlite_msg = f"Error dropping SQLite table '{table_name}': {e}"
        logging.error(sqlite_msg, exc_info=True)

    collection_name = f"{QDRANT_COLLECTION_PREFIX}{table_name}"
    try:
        try: qdrant_client.get_collection(collection_name=collection_name); collection_exists = True
        except Exception: collection_exists = False; qdrant_msg = f"Qdrant collection '{collection_name}' did not exist."; qdrant_success = True

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


def refine_and_select(user_query: str, conversation_history: list[dict], schema: dict, llm_wrapper: LLMWrapper) -> dict:
    """
    Consolidates query refinement, table selection, and routing into a single LLM call.
    Uses prompt from PROMPTS.
    Returns a dictionary with keys: "refined_query", "selected_table", "route".
    Provides fallbacks if LLM fails.
    """
    # Default fallbacks
    default_result = {
        "refined_query": user_query,
        "selected_table": next(iter(schema)) if schema else "", # First table if exists
        "route": "SQL"
    }

    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for refine/select/route. Using defaults.")
        return default_result

    recent_context = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in conversation_history[-10:] # Limit context size
    ) if conversation_history else "No previous conversation history."

    # --- Use loaded prompt ---
    prompt = PROMPTS['refine_and_select'].format(
        recent_context=recent_context,
        schema=schema, # Pass the full schema here for table selection
        user_query=user_query
    )
    # --- End Use loaded prompt ---
    logging.debug(f"Sending refine_and_select prompt to LLM:\n{prompt}")

    try:
        # Prefer structured output if available
        response_data = None
        if llm_wrapper.mode == 'openrouter':
            response_data = llm_wrapper.generate_structured_response(prompt)
            if not isinstance(response_data, dict):
                logging.warning(f"Structured refine_and_select failed ({type(response_data)}). Trying text generation.")
                response_data = None # Force text fallback

        if response_data is None: # If not openrouter or structured failed
            response_text = llm_wrapper.generate_response(prompt, max_tokens=250, temperature=0.2) # Increased tokens slightly
            logging.debug(f"Raw text response for refine_and_select: {response_text}")
            # Try parsing the text response as JSON
            cleaned_text = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                response_data = json.loads(cleaned_text)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON from refine_and_select text response: {cleaned_text}")
                return default_result # Use defaults if parsing fails

        # Validate the structure of the response_data dict
        if isinstance(response_data, dict) and \
           "refined_query" in response_data and \
           "selected_table" in response_data and \
           "route" in response_data:
            # Basic validation of route value
            if response_data["route"].upper() not in ["SQL", "SEMANTIC"]:
                 logging.warning(f"LLM returned invalid route '{response_data['route']}'. Defaulting to SQL.")
                 response_data["route"] = "SQL"
            # Ensure table exists (though process_natural_language_query does fallback too)
            if response_data["selected_table"] not in schema:
                 logging.warning(f"LLM selected non-existent table '{response_data['selected_table']}'. Will use fallback later.")
                 # Keep the LLM's choice for now, let downstream handle fallback
            logging.info(f"refine_and_select successful: {response_data}")
            return response_data
        else:
            logging.error(f"LLM response for refine_and_select lacked required keys or wasn't a dict: {response_data}")
            return default_result # Use defaults if structure is wrong

    except Exception as e:
        logging.error(f"Error during refine_and_select LLM call: {e}", exc_info=True)
        return default_result # Use defaults on error