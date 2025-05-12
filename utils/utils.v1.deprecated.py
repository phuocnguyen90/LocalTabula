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
from utils.aux_model import load_aux_models # Renamed import


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
PROMPT_PATH = CONFIG_DIR / "prompts.yaml"
PROMPT_FILE = "prompts.yaml"
PROMPT_PATH = CONFIG_DIR / PROMPT_FILE
ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(dotenv_path=str(ENV_PATH))  

SQLITE_TIMEOUT_SECONDS = 15
# SHEETS_SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly'] # Keep if using private sheets
# DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
# ALL_SCOPES = SHEETS_SCOPES + DRIVE_SCOPES
QDRANT_COLLECTION_PREFIX = "table_data_"


# --- Load Prompts ---
def load_prompts():
    """
    Returns a dict (or list) from your prompt YAML file.
    """
    prompts = yaml.safe_load(PROMPT_PATH.open())
    logging.info("Loaded prompts keys:", list(prompts.keys()))
    return prompts
    

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

def prepare_collection(q_client: QdrantClient, collection_name: str, vector_size: int, table_name: str) -> tuple[bool, str, list[str]]:
    """
    Checks if a Qdrant collection with the given name exists and deletes it if so,
    then creates a new collection with the specified vector size.
    
    Returns:
        (success: bool, error_message: str, status_messages: list[str])
    """
    status_messages = []
    try:
        q_client.get_collection(collection_name=collection_name)
        logging.warning(f"[{table_name}] Collection '{collection_name}' exists. Recreating it.")
        q_client.delete_collection(collection_name=collection_name, timeout=10)
        status_messages.append(f"Recreated collection '{collection_name}'.")
        time.sleep(0.5)
    except Exception:
        logging.info(f"[{table_name}] Collection '{collection_name}' not found. Will create new.")
    
    try:
        logging.info(f"[{table_name}] Creating collection '{collection_name}' with vector size {vector_size}...")
        q_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        logging.info(f"[{table_name}] Collection '{collection_name}' created successfully.")
        status_messages.append(f"Created collection '{collection_name}'.")
    except Exception as create_e:
        logging.error(f"[{table_name}] Failed to create collection '{collection_name}': {create_e}", exc_info=True)
        return False, f"Failed to create Qdrant collection: {create_e}", status_messages

    return True, "", status_messages

def prepare_documents(df: pd.DataFrame, valid_cols: list[str], table_name: str) -> tuple[list[str], list[dict], list[str]]:
    """
    Prepares a list of text documents, payloads, and unique point IDs for each row in the DataFrame.
    Attempts to detect a primary key for better payload linkage.
    
    Returns:
        documents_to_embed (list[str]), payloads (list[dict]), point_ids (list[str])
    """
    documents_to_embed = []
    payloads = []
    point_ids = []

    # Try to detect a primary key column (if it exists)
    potential_pk = next((col for col in df.columns if col.lower() == 'id'), None)
    if potential_pk is None and len(df.columns) > 0 and df.columns[0].lower() in ['id', 'pk', 'key']:
        potential_pk = df.columns[0]
    logging.info(f"[{table_name}] Potential primary key: {potential_pk}")

    rows_with_text = 0
    for index, row in df.iterrows():
        # Concatenate the content of valid semantic columns
        text_parts = [f"{col}: {row[col]}" for col in valid_cols 
                      if pd.notna(row[col]) and str(row[col]).strip()]
        if not text_parts:
            continue
        rows_with_text += 1
        doc_text = " | ".join(text_parts)
        
        # Build payload from row data
        payload = row.to_dict()
        for k, v in payload.items():
            if pd.isna(v):
                payload[k] = None
        payload["_table_name"] = table_name
        payload["_source_text"] = doc_text
        
        # Determine an original ID value (using primary key if available)
        original_id = f"{table_name}_idx_{index}"
        if potential_pk and potential_pk in row and pd.notna(row[potential_pk]):
            try:
                original_id = f"{table_name}_{str(row[potential_pk])}"
            except Exception as pk_err:
                logging.warning(f"[{table_name}] Could not convert PK '{row[potential_pk]}' for row {index}: {pk_err}")
        payload["_original_id"] = original_id

        # Always generate a unique UUID for Qdrant point ID
        unique_point_id = str(uuid.uuid4())
        point_ids.append(unique_point_id)
        payloads.append(payload)
        documents_to_embed.append(doc_text)
    
    logging.info(f"[{table_name}] Prepared {len(documents_to_embed)} documents from {rows_with_text} rows with text.")
    return documents_to_embed, payloads, point_ids

def compute_vector_size(embeddings: list, table_name: str) -> tuple[int, str]:
    """
    Determines the vector size from the embeddings.
    Uses the environment variable EMBEDDING_VECTOR_SIZE as a reference and logs a warning if they differ.
    
    Returns:
        (vector_size: int, error_message: str)
    """
    try:
        computed_size = len(embeddings[0])
        expected_size = int(os.getenv("EMBEDDING_VECTOR_SIZE", "768"))
        if computed_size != expected_size:
            logging.warning(f"[{table_name}] Computed vector size {computed_size} differs from EMBEDDING_VECTOR_SIZE {expected_size}. Using computed value.")
            return computed_size, ""
        return expected_size, ""
    except Exception as e:
        logging.error(f"[{table_name}] Error determining vector size: {e}", exc_info=True)
        return 0, f"Error determining vector size: {e}"

def create_qdrant_collection_with_retry(q_client: QdrantClient, collection_name: str, vector_size: int, table_name: str,
                                          retries: int = 3, delay: int = 2) -> tuple[bool, str, list[str]]:
    """
    Attempts to create (or recreate) a Qdrant collection.
    Retries up to 'retries' times in case of transient network/DNS errors.
    
    Returns: (success, error_message, status_messages)
    """
    status_messages = []
    for attempt in range(1, retries + 1):
        try:
            # If the collection exists, delete it first.
            try:
                q_client.get_collection(collection_name=collection_name)
                logging.warning(f"[{table_name}] Collection '{collection_name}' exists. Recreating.")
                q_client.delete_collection(collection_name=collection_name, timeout=10)
                status_messages.append(f"Recreated collection '{collection_name}'.")
                time.sleep(0.5)
            except Exception:
                logging.info(f"[{table_name}] Collection '{collection_name}' not found. Proceeding to create a new one.")

            logging.info(f"[{table_name}] Attempt {attempt}: Creating collection '{collection_name}' with vector size {vector_size}...")
            q_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            logging.info(f"[{table_name}] Collection '{collection_name}' created successfully.")
            status_messages.append(f"Created collection '{collection_name}'.")
            return True, "", status_messages
        except Exception as e:
            logging.error(f"[{table_name}] Attempt {attempt} failed to create collection '{collection_name}': {e}", exc_info=True)
            if attempt < retries:
                time.sleep(delay)
            else:
                return False, f"Failed to create Qdrant collection after {retries} attempts: {e}", status_messages
    return False, "Unexpected error in collection creation", status_messages

def compute_vector_size(embeddings: list, table_name: str) -> tuple[int, str]:
    try:
        computed_size = len(embeddings[0])
        expected_size = int(os.getenv("EMBEDDING_VECTOR_SIZE", "768"))
        if computed_size != expected_size:
            logging.warning(f"[{table_name}] Computed vector size {computed_size} differs from EMBEDDING_VECTOR_SIZE {expected_size}. Using computed value.")
            return computed_size, ""
        return expected_size, ""
    except Exception as e:
        logging.error(f"[{table_name}] Error determining vector size: {e}", exc_info=True)
        return 0, f"Error determining vector size: {e}"

def prepare_documents(df: pd.DataFrame, valid_cols: list[str], table_name: str) -> tuple[list[str], list[dict], list[str]]:
    """Prepare text documents, payloads, and UUID point IDs from the DataFrame using the valid semantic columns."""
    documents, payloads, point_ids = [], [], []

    # Attempt to find a primary key column for more meaningful payload IDs
    potential_pk = next((col for col in df.columns if col.lower() == 'id'), None)
    if potential_pk is None and len(df.columns) > 0 and df.columns[0].lower() in ['id', 'pk', 'key']:
        potential_pk = df.columns[0]
    logging.info(f"[{table_name}] Potential primary key: {potential_pk}")

    for idx, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in valid_cols if pd.notna(row[col]) and str(row[col]).strip()]
        if not parts:
            continue
        doc_text = " | ".join(parts)
        documents.append(doc_text)
        payload = row.to_dict()
        for k, v in payload.items():
            if pd.isna(v):
                payload[k] = None
        payload["_table_name"] = table_name

        # Create an original ID using the primary key if possible
        original_id = f"{table_name}_idx_{idx}"
        if potential_pk and potential_pk in row and pd.notna(row[potential_pk]):
            try:
                original_id = f"{table_name}_{str(row[potential_pk])}"
            except Exception as ex:
                logging.warning(f"[{table_name}] Error converting PK for row {idx}: {ex}")
        payload["_original_id"] = original_id
        payload["_source_text"] = doc_text

        # Use a UUID for the Qdrant point ID
        point_ids.append(str(uuid.uuid4()))
        payloads.append(payload)

    logging.info(f"[{table_name}] Prepared {len(documents)} documents.")
    return documents, payloads, point_ids

def create_or_update_collection(q_client: QdrantClient, collection_name: str, vector_size: int, table_name: str) -> tuple[bool, str, list[str]]:
    """
    Uses a fixed collection name. If the collection exists, delete it and re-create it.
    Returns (success, error_message, status_messages).
    """
    status_messages = []
    try:
        # Check if collection exists
        q_client.get_collection(collection_name=collection_name)
        logging.info(f"[{table_name}] Collection '{collection_name}' exists. Deleting to update.")
        q_client.delete_collection(collection_name=collection_name, timeout=10)
        status_messages.append(f"Deleted existing collection '{collection_name}'.")
        time.sleep(0.5)
    except Exception:
        logging.info(f"[{table_name}] No existing collection '{collection_name}'.")
    
    try:
        logging.info(f"[{table_name}] Creating collection '{collection_name}' with vector size {vector_size}...")
        q_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        logging.info(f"[{table_name}] Collection '{collection_name}' created successfully.")
        status_messages.append(f"Created collection '{collection_name}'.")
    except Exception as e:
        logging.error(f"[{table_name}] Failed to create collection '{collection_name}': {e}", exc_info=True)
        return False, f"Failed to create Qdrant collection: {e}", status_messages
    return True, "", status_messages

def _embed_and_index_data(df: pd.DataFrame, table_name: str, semantic_columns: list, aux_models: dict,
                          qdrant_client: QdrantClient) -> tuple[bool, str]:
    """
    Embeds specified semantic columns and indexes them in the Qdrant collection associated
    with the SQLite table. Uses a fixed collection name (table_data_{table_name}).
    """
    logging.info(f"Starting embedding and indexing for table '{table_name}'...")
    
    # --- Readiness Checks ---
    if not qdrant_client:
        logging.error(f"[{table_name}] Qdrant client not available.")
        return False, "Qdrant client not initialized."
    if not aux_models or aux_models.get('status') != 'loaded' or not aux_models.get('embedding_model'):
        logging.error(f"[{table_name}] Embedding model or aux models not loaded.")
        return False, "Embedding model not loaded or aux models failed."
    if df.empty:
        logging.info(f"[{table_name}] DataFrame is empty; skipping embedding.")
        return True, "DataFrame is empty."
    if not semantic_columns:
        logging.warning(f"[{table_name}] No semantic columns specified.")
        return True, "No semantic columns specified."
    
    valid_cols = [col for col in semantic_columns if col in df.columns]
    logging.info(f"[{table_name}] Valid semantic columns: {valid_cols}")
    if not valid_cols:
        logging.warning(f"[{table_name}] No valid semantic columns found.")
        return True, "Valid semantic columns list is empty."
    
    # --- Prepare Documents ---
    documents, payloads, point_ids = prepare_documents(df, valid_cols, table_name)
    if not documents:
        logging.warning(f"[{table_name}] No non-empty documents prepared for embedding.")
        return True, "No non-empty documents to embed."
    
    # --- Generate Embeddings ---
    embedding_model = aux_models.get("embedding_model")
    logging.info(f"[{table_name}] Generating embeddings for {len(documents)} documents...")
    try:
        embeddings_result = embedding_model.embed(documents)
        embeddings = list(embeddings_result) if not isinstance(embeddings_result, (list, np.ndarray)) else embeddings_result
        if len(embeddings) != len(documents):
            logging.error(f"[{table_name}] Embedding count ({len(embeddings)}) does not match document count ({len(documents)}).")
            return False, "Embedding count mismatch."
    except Exception as embed_e:
        logging.error(f"[{table_name}] Error generating embeddings: {embed_e}", exc_info=True)
        return False, f"Embedding generation failed: {embed_e}"
    
    # --- Compute Vector Size ---
    vector_size, size_error = compute_vector_size(embeddings, table_name)
    if vector_size == 0:
        return False, size_error
    logging.info(f"[{table_name}] Using vector size: {vector_size}")
    
    # --- Fixed Collection Name ---
    collection_name = f"{QDRANT_COLLECTION_PREFIX}{table_name}"
    logging.info(f"[{table_name}] Using fixed collection name: '{collection_name}'")
    
    # --- Create or Update Collection ---
    coll_ok, coll_error, coll_status = create_or_update_collection(qdrant_client, collection_name, vector_size, table_name)
    if not coll_ok:
        return False, coll_error
    
    # --- Upsert Data in Batches ---
    batch_size = 100
    num_batches = (len(point_ids) + batch_size - 1) // batch_size
    logging.info(f"[{table_name}] Upserting {len(point_ids)} points into '{collection_name}' in {num_batches} batches...")
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(point_ids))
        batch_ids = point_ids[start_idx:end_idx]
        batch_vectors = [list(map(float, vec)) for vec in embeddings[start_idx:end_idx]]
        batch_payloads = payloads[start_idx:end_idx]
        
        logging.debug(f"[{table_name}] Upserting batch {i+1}/{num_batches} with {len(batch_ids)} points.")
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    payloads=batch_payloads
                ),
                wait=True
            )
        except Exception as upsert_e:
            logging.error(f"[{table_name}] Error upserting batch {i+1}: {upsert_e}", exc_info=True)
            return False, f"Error during Qdrant upsert: {upsert_e}"
    
    logging.info(f"[{table_name}] Successfully indexed {len(point_ids)} points into collection '{collection_name}'.")
    return True, f"Indexed {len(point_ids)} points into collection '{collection_name}'."

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

@st.cache_resource
# utils.py

@st.cache_resource # Keep caching
def init_qdrant_client():
    """
    Initializes and caches a Qdrant client.
    Tries persistent local file, falls back to in-memory.
    """
    persistent_path_dir = None
    qdrant_filename = "qdrant_storage" # Changed filename slightly

    # Use the app_data directory determined by setup_environment
    # We need setup_environment to run first, which happens in app.py
    # Let's try to determine the path similarly here, or ideally pass it
    dev_mode_str = os.getenv('DEVELOPMENT_MODE', 'false').lower()
    is_dev_mode = dev_mode_str in ('true', '1', 't', 'yes', 'y')
    if is_dev_mode:
        home_dir = os.path.expanduser("~")
        app_data_dir = os.path.join(home_dir, ".streamlit_chat_dev_data")
    else:
        current_working_dir = os.getcwd()
        app_data_dir = os.path.join(current_working_dir, "app_data")

    # Ensure directory exists
    try:
        os.makedirs(app_data_dir, exist_ok=True)
        persistent_path_dir = os.path.join(app_data_dir, qdrant_filename)
        logging.info(f"Checking for persistent Qdrant storage at: {persistent_path_dir}")
    except OSError as e:
        logging.warning(f"Could not create/access Qdrant storage directory '{app_data_dir}': {e}. Will use in-memory.")
        persistent_path_dir = None

    client = None
    # Try persistent path only if directory seems accessible
    if persistent_path_dir:
        try:
            logging.info(f"Attempting persistent Qdrant client initialization at: {persistent_path_dir}")
            # Explicitly use path, avoid default URL connection attempts
            client = QdrantClient(path=persistent_path_dir)
            # Perform a basic operation to confirm connectivity/usability
            client.get_collections() # Check if basic interaction works
            logging.info(f"Persistent Qdrant client initialized successfully at {persistent_path_dir}")
            return client
        except Exception as e:
            # Catching broader exceptions as specific http errors might not cover all init issues
            logging.error(f"Persistent Qdrant client initialization failed at {persistent_path_dir}: {type(e).__name__} - {e}")
            logging.warning("Falling back to in-memory Qdrant client.")
            client = None # Ensure client is None for fallback

    # Fallback to in-memory if persistent path wasn't attempted or failed
    if client is None:
        try:
            logging.info("Initializing in-memory Qdrant client...")
            client = QdrantClient(":memory:")
            # Basic check for in-memory
            client.get_collections()
            logging.info("In-memory Qdrant client initialized successfully.")
            return client
        except Exception as e:
            logging.fatal(f"FATAL: Failed to initialize even in-memory Qdrant client: {e}", exc_info=True)
            return None # Indicate total failure


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


def _generate_sql_query(
    refined_query_input: str, # Renamed for clarity - this IS the refined query
    schema: dict,
    sample_data_str: Optional[str],
    aux_models,
    augmented_keywords: Optional[list[str]] = None, # Accept augmented keywords
    previous_sql: str | None = None,
    feedback: str | None = None
) -> str:
    """
    Generates SQL query using the refined query and schema context.
    Optionally incorporates augmented keywords if the prompt template supports it.
    Loads prompt components from PROMPTS.
    """
    sql_llm: Llama = aux_models.get('sql_gguf_model')
    if not sql_llm:
        return "-- Error: SQL GGUF model object missing."

    sample_context = f"Data Sample (representative table):\n{sample_data_str}\n" if sample_data_str else "Sample data not available.\n"
    prompt_context = {
        "schema": json.dumps(schema, indent=2), # Format schema as JSON string
        "sample_data": sample_context,
        "user_query": refined_query_input, # Use the refined query received as input
        "augmented_keywords": augmented_keywords if augmented_keywords else []
    }
    # --- Construct Prompt using loaded components ---
    prompt_parts = [
        PROMPTS['generate_sql_base'].format(**prompt_context) # Use prepared context
    ]


    if previous_sql and feedback:
        if "syntax error" in feedback.lower():
            prompt_parts.append(PROMPTS['generate_sql_feedback_syntax'].format(previous_sql=previous_sql, feedback=feedback))
        else:
            prompt_parts.append(PROMPTS['generate_sql_feedback_other'].format(previous_sql=previous_sql, feedback=feedback))
            
    prompt_parts.append(PROMPTS['generate_sql_response_format'])
    prompt = "\n".join(prompt_parts)
    # --- End Prompt Construction ---

    logging.debug(f"Sending prompt to SQL GGUF model (Retry Attempt? {'Yes' if previous_sql else 'No'}):\n{prompt}")
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
        start_time = 0.0
        end_time = 0.0
        duration = 0.0
        start_time = time.perf_counter()
        result_df = pd.read_sql_query(sql_query, conn)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.info(f"SQL execution time: {duration:.2f} seconds")
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
    refined_query: str,
    schema: dict,
    sample_data_str:str,
    aux_models,
    conn,
    llm_wrapper,
    augmented_keywords: Optional[list[str]] = None, # Add parameter
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
            refined_query_input=refined_query, 
            schema=schema,         
            sample_data_str=sample_data_str, 
            aux_models=aux_models,
            augmented_keywords=augmented_keywords, 
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
        is_valid, validation_feedback = _validate_sql_results(refined_query, current_sql, sql_df, llm_wrapper)
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
    original_query: str, 
    conn: Optional[sqlite3.Connection], 
    schema: dict, # This is now expected to be the FULL schema dict {db_id: {table: [cols]}}
    llm_wrapper, aux_models, qdrant_client, max_retries: int = 1,
    conversation_history: Optional[list] = None,
    pre_selected_db_id: Optional[str] = None # From previous refactor
) -> dict:
    """
    Main pipeline (Two-Stage Analysis & Generation, Optional Execution):
    1. Validate inputs. Check language.
    2. Stage 1: Select relevant db_id.
    3. Stage 2: Refine query, get keywords, determine route.
    4. If route is SQL, generate SQL string.
    5. If conn is provided, proceed with execution, secondary route, and summary.
    6. Returns analysis results (incl. generated SQL) and optionally execution results.
    """
    # Initial validation
    if not llm_wrapper or not llm_wrapper.is_ready: return {"status": "error", "message": "LLM wrapper not ready."}
    if not aux_models or aux_models.get('status') != 'loaded': return {"status": "error", "message": "Auxiliary models not ready."}
    if not qdrant_client: logging.warning("Qdrant client not ready, semantic search may fail."); # Non-fatal for SQL focus
    if not schema: return {"status": "error", "message": "Full schema dictionary not provided."}

    result = { # Initialize result dict
        "status": "processing", "generated_sql": None, "selected_db_id": None,
        "determined_route": None, "refined_query": None, "augmented_keywords": [],
        "sql_data": None, "sql_error": None, "semantic_data": None,
        "semantic_error": None, "natural_language_summary": None, "raw_display_data": None
    }


    # Check/Translate Query Language early on
    processed_query, was_translated = _ensure_english_query(original_query, llm_wrapper)
    result["query_was_translated"] = was_translated
    result["processed_query"] = processed_query  # Store the query used for processing
    
    # Ensure that schema is up-to-date: if it's empty, refresh it from the DB.
    if not schema:
        schema = get_schema_info(conn)
        if not schema:
            return {"status": "error", "message": "No tables found in the database."}

    current_conversation_history = conversation_history if conversation_history is not None else [] # Assuming streamlit context
    selected_db_id = None
    formatted_context_for_selection = "{}" # Default empty
    schemas_truncated = False
    # --- Stage 0: Select Database ID ---
    logging.info("Stage 1: Selecting Database ID...")
    if pre_selected_db_id and pre_selected_db_id in schema:
        selected_db_id = pre_selected_db_id
    else:        
        temp_schemas_for_prompt, schemas_truncated = format_schemas_for_prompt(schema) # Use schema-only formatter

        selected_db_id = select_database_id(
            processed_query,
            temp_schemas_for_prompt, # Pass formatted schemas string
            schemas_truncated,
            list(schema.keys()), # Pass original keys for validation
            llm_wrapper,
            current_conversation_history
        )
        logging.info(f"Stage 1: Selected DB ID = {selected_db_id}")

    
    if not selected_db_id or selected_db_id not in schema:
        msg = f"Could not determine relevant database ID (Selected: {selected_db_id})."
        logging.error(msg + " Cannot proceed.")
        result.update({"status": "error", "message": msg})
        return result

    logging.info(f"Stage 1 Result: Selected DB ID = {selected_db_id}")
    result["selected_db_id"] = selected_db_id
    selected_db_schema_tables = schema[selected_db_id] 
    # --- Stage 1: Get Sample Data ---
    if selected_db_id:
        sample_data_str_selected = get_table_sample_data(conn, selected_db_id, limit=3)
        if not sample_data_str_selected:
            msg = f"Could not retrieve sample data for selected DB ID '{selected_db_id}'."
            logging.error(msg + " Cannot proceed.")
            result.update({"status": "error", "message": msg})
            return result
    else:
        msg = f"Selected DB ID '{selected_db_id}' not found in schema."
        logging.error(msg + " Cannot proceed.")
        result.update({"status": "error", "message": msg})
        return result


    # --- Stage 2: Refine Query and Determine Route ---
    logging.info("Stage 2: Refining Query and Determining Route...")
    refinement_data = refine_and_select(
        processed_query, 
        current_conversation_history,
        {selected_db_id: selected_db_schema_tables},        
        sample_data=sample_data_str_selected,        
        llm_wrapper=llm_wrapper,
    )

    refined_query = refinement_data.get("refined_query", processed_query)
    recommended_route = refinement_data.get("route", "SQL").upper()
    augmented_keywords = refinement_data.get("augmented_keywords", [])

    result["refined_query"] = refined_query
    result["determined_route"] = recommended_route
    result["augmented_keywords"] = augmented_keywords
    logging.info(f"Stage 2 Result: Refined Query='{refined_query}', Route='{recommended_route}', Keywords={augmented_keywords}")

    # --- Stage 3: Generate SQL (if route is SQL) ---
    if recommended_route == "SQL":
        logging.info("Stage 3: Generating SQL Query...")
        # Pass None for sample_data_str initially, execution part will get it if needed
        generated_sql_string = _generate_sql_query(
            refined_query_input=refined_query,
            schema=selected_db_schema_tables,
            sample_data_str=sample_data_str_selected, 
            aux_models=aux_models,
            augmented_keywords=augmented_keywords
        )
        result["generated_sql"] = generated_sql_string # Store generated SQL or error string

        if generated_sql_string.startswith("-- Error"):
             result["status"] = "error_sql_generation"
             result["message"] = f"SQL Generation Failed: {generated_sql_string}"
             logging.error(f"SQL Generation Failed: {generated_sql_string}")
             # Return now, cannot execute
             return result
        else:
             result["status"] = "analysis_complete_sql_generated"
             result["message"] = "Analysis & SQL Generation complete. Execution pending."
             logging.info("SQL Generation Successful.")

    elif recommended_route == "SEMANTIC":
        result["status"] = "analysis_complete_semantic_route"
        result["message"] = "Analysis complete, SEMANTIC route determined."
        logging.info("Semantic route determined.")
        # Return now, benchmark doesn't focus on semantic execution (for now)
        return result
    else:
        result["status"] = "error_unknown_route"
        result["message"] = f"Analysis failed: Unknown route '{recommended_route}'."
        logging.error(f"Unknown route '{recommended_route}'.")
        return result

    # --- Stage 4: Execute SQL, Run Secondary, Summarize (Requires Connection) ---
    if conn is not None:
        logging.info("Stage 4: Executing SQL and Summarizing (Connection provided)...")
        sql_result = {"sql_success": False}
        semantic_result = {"semantic_success": False} # Needed for secondary route logic

        if recommended_route == "SQL" and not result["generated_sql"].startswith("-- Error"):
            
            sql_result = execute_sql_pipeline(
                refined_query=refined_query,
                schema=selected_db_schema_tables,
                sample_data_str=sample_data_str_selected, # Now we have sample data
                aux_models=aux_models,
                conn=conn, # Pass the actual connection
                llm_wrapper=llm_wrapper,
                augmented_keywords=augmented_keywords,
                max_retries=max_retries
            )
            # Update result dict with execution outcome
            result["generated_sql"] = sql_result.get("generated_sql", result["generated_sql"]) # Keep generated SQL
            result["sql_data"] = sql_result.get("sql_data")
            result["sql_error"] = sql_result.get("sql_error")
            if not sql_result.get("sql_success"):
                result["status"] = "error_sql_execution"
                result["message"] = f"SQL Execution Failed: {sql_result.get('sql_error')}"
            # else: Status remains related to match/summary below

        # Run secondary route (optional, might need adjustment)
        # Pass the results from the primary execution attempt
        sql_result_sec, semantic_result_sec = run_secondary_route(
            refined_query, recommended_route, sql_result, semantic_result,
            aux_models, conn, selected_db_schema_tables, llm_wrapper, qdrant_client, sample_data_str_selected
            # Pass keywords?
        )
        # Update results if secondary route provided something new/better
        # (Careful not to overwrite primary errors if secondary also fails)
        if not result.get("sql_data") and sql_result_sec.get("sql_data") is not None:
             result["sql_data"] = sql_result_sec.get("sql_data")
             result["sql_error"] = sql_result_sec.get("sql_error") # Update error too
             result["generated_sql"] = sql_result_sec.get("generated_sql", result["generated_sql"])
        if not result.get("semantic_data") and semantic_result_sec.get("semantic_data"):
             result["semantic_data"] = semantic_result_sec.get("semantic_data")
             result["semantic_error"] = semantic_result_sec.get("semantic_error")


        # Generate final summary using available data
        nl_summary = generate_final_summary(
            original_query, refined_query, was_translated,
            result.get("sql_data"), result.get("semantic_data"), llm_wrapper
        )
        result["natural_language_summary"] = nl_summary
        result["message"] = nl_summary # Final user message is the summary

        # Determine final status based on execution outcomes
        sql_ok = sql_result.get("sql_success") or sql_result_sec.get("sql_success")
        sem_ok = semantic_result.get("semantic_success") or semantic_result_sec.get("semantic_success") # Assuming semantic benchmarked elsewhere
        if sql_ok or sem_ok:
             result["status"] = "success"
        elif result["sql_error"] or result["semantic_error"]:
              result["status"] = "error_execution" # Keep specific error if execution failed
        else:
              result["status"] = "success_no_results" # Ran ok but found nothing

        result["raw_display_data"] = {"sql": result.get("sql_data"), "semantic": result.get("semantic_data")}

    # If conn was None, the status will be analysis_complete_*
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


def refine_and_select(
    user_query: str,
    conversation_history: Optional[list[dict]],
    selected_schema: dict, # Schema for the single selected DB
    sample_data: Optional[str], # <-- NEW: Sample data string
    llm_wrapper: LLMWrapper
) -> dict:
    """ Refines query, gets keywords, determines route using schema AND sample data. """
    # Default fallback result
    default_result = {
        "refined_query": user_query,
        "route": "SQL",
        "augmented_keywords": []
    }
    recent_context = "\n".join(...) if conversation_history else "No history."

    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for refine/select/route. Using defaults.")
        return default_result
    if not selected_schema:
         logging.warning("No schema provided to refine_and_select. Using defaults.")
         return default_result

    # Prepare recent conversation context
    recent_context = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in conversation_history[-10:]
    ) if conversation_history else "No previous conversation history."

    # Format schema as JSON (to help LLM read it cleanly)
    formatted_schema = json.dumps(selected_schema, indent=2)
    sample_data_str = sample_data if sample_data is not None else "N/A" # Handle None case

    try:
        prompt = PROMPTS['refine_and_select'].format( 
            recent_context=recent_context,
            user_query=user_query,
            schema=formatted_schema,
            sample_data=sample_data_str 
        )
        logging.debug(f"Sending refine_and_select prompt:\n{prompt}")

        # --- Generation and Parsing Logic (similar to previous refine_and_select) ---
        raw_response_content = llm_wrapper.generate_response(prompt, max_tokens=300, temperature=0.2)
        logging.debug(f"Raw LLM response for refine_and_select: {raw_response_content}")
        # Clean
        cleaned_text = raw_response_content.strip()
        if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"): cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        # Parse
        try:
            response_data = json.loads(cleaned_text)
            if not isinstance(response_data, dict):
                 logging.error(f"Parsed JSON not dict: {cleaned_text} -> {type(response_data)}")
                 return default_result
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse failed: {e}. Content: '{cleaned_text}'")
            return default_result

        # --- Robust Key Access & Validation ---
        final_result = {}
        # Note: 'selected_table' is NOT expected here anymore
        expected_keys = ["refined_query", "augmented_keywords", "route"]
        found_keys = {key.strip().strip('"'): value for key, value in response_data.items()}

        for expected_key in expected_keys:
            found_value = found_keys.get(expected_key)
            if found_value is not None: final_result[expected_key] = found_value
            else:
                logging.warning(f"LLM missing '{expected_key}' in refine/route response. Using fallback.")
                final_result[expected_key] = default_result.get(expected_key)
                if expected_key == "augmented_keywords" and final_result[expected_key] is None: final_result[expected_key] = []

        # --- Post-processing Validation (similar logic) ---
        if not isinstance(final_result.get("route"), str) or final_result["route"].upper() not in ["SQL", "SEMANTIC"]:
            final_result["route"] = "SQL" # Default if invalid
        else: final_result["route"] = final_result["route"].upper()

        if not isinstance(final_result.get("augmented_keywords"), list):
             final_result["augmented_keywords"] = []
        else: final_result["augmented_keywords"] = [str(item) for item in final_result["augmented_keywords"] if isinstance(item, (str, int, float))]

        if not isinstance(final_result.get("refined_query"), str): final_result["refined_query"] = default_result["refined_query"]

        logging.info(f"refine_and_select processed result: {final_result}")
        return final_result

    except KeyError:
        logging.error("Prompt key 'refine_and_select' not found in prompts.yaml!")
        return default_result
    except Exception as e:
        logging.error(f"Unexpected error during refine_and_select LLM call: {e}", exc_info=True)
        return default_result
    

MAX_SCHEMA_CHARS_ESTIMATE = 20000 # Rough estimate for ~6k-7k tokens

def format_schemas_for_prompt(available_schemas: dict, max_chars: int = MAX_SCHEMA_CHARS_ESTIMATE) -> Tuple[str, bool]:
    """Formats schemas into a JSON string for the prompt, handling potential truncation."""
    # Format: { "db_id1": {"table1": ["col1", "col2"], ...}, "db_id2": {...} }
    full_schemas_str = json.dumps(available_schemas, indent=2)
    truncated = False

    if len(full_schemas_str) > max_chars:
        logging.warning(f"Full schema string length ({len(full_schemas_str)} chars) exceeds estimated limit ({max_chars}). Attempting truncation.")
        truncated = True
        # Simple truncation strategy: Keep removing schemas until under limit
        # More sophisticated: Prioritize based on keyword match first (Two-Pass approach needed)
        schemas_to_include = dict(available_schemas) # Start with a copy
        while len(json.dumps(schemas_to_include, indent=2)) > max_chars and schemas_to_include:
             # Remove the 'last' item (arbitrary for dict) - could be improved
             schemas_to_include.popitem()
        formatted_str = json.dumps(schemas_to_include, indent=2)
        if not schemas_to_include:
             logging.error("Schema string too large even after attempting truncation. Cannot proceed.")
             return "{}", True # Return empty dict string, signal truncation happened

        logging.warning(f"Truncated schema string length: {len(formatted_str)} chars. Kept {len(schemas_to_include)} schemas.")
        return formatted_str, truncated
    else:
        # Fits within limits
        return full_schemas_str, truncated


def select_database_id(
    user_query: str,
    # Pass the formatted string directly
    formatted_schemas_and_samples_str: str,
    schemas_were_truncated: bool, # Know if truncation happened
    available_schema_keys: list, # Pass original keys for validation
    llm_wrapper: LLMWrapper,
    conversation_history: Optional[list[dict]] = None # Keep history if prompt uses it
) -> Optional[str]:
    """ Uses LLM to select DB ID using formatted schema+sample context. """
    template = PROMPTS.get('select_database_id')
    if template is None:
        logging.error("PROMPTS has no 'select_database_id' key—aborting.")
        return None

    if not llm_wrapper or not llm_wrapper.is_ready:
        logging.warning("LLM wrapper not ready for DB ID selection.")
        return None # Or fallback

    try:
        prompt = template.format(
            formatted_schemas_and_samples_str=formatted_schemas_and_samples_str,
            user_query=user_query
            )
        logging.debug(f"Sending DB ID selection prompt (schemas truncated: {schemas_were_truncated})...")
        # Log only part of the potentially massive prompt
        prompt_preview = prompt[:500] + ("..." if len(prompt) > 500 else "")
        logging.debug(f"Prompt Preview:\n{prompt_preview}")


        # Use simple text generation - response should just be the ID string
        selected_id_raw = llm_wrapper.generate_response(prompt, max_tokens=50, temperature=0.1) # Short response needed
        # Clean the response robustly
        selected_id = selected_id_raw.strip().strip('`"\'') # Remove common delimiters

        logging.info(f"LLM suggested DB ID: '{selected_id}' (Raw: '{selected_id_raw}')")

        # --- Validation ---
        # Check against the ORIGINAL, full list of schemas
        if selected_id in available_schema_keys:
            return selected_id
        else:
            # LLM failed or hallucinated
            logging.warning(f"LLM selection '{selected_id}' invalid or not found in available schemas ({list(available_schema_keys.keys())[:10]}...). Attempting fallback.")

            # --- Fallback Strategy (Simple Keyword Match) ---
            query_lower = user_query.lower()
            # Prioritize matching db_id first
            for db_id in available_schema_keys:
                 if db_id.lower() in query_lower:
                      logging.info(f"Fallback: Matched DB ID '{db_id}' in query.")
                      return db_id       

            logging.error("LLM selection failed and fallback keyword match also failed.")
            return None # Indicate failure

    except KeyError:
        logging.error("Prompt key 'select_database_id' not found in prompts.yaml!")
        return None
    except Exception as e:
        logging.error(f"Error during DB ID selection LLM call: {e}", exc_info=True)
        return None
    
def get_table_sample_data(conn: sqlite3.Connection, table_name: str, limit: int = 3) -> Optional[str]:
    """Fetches the first few rows of a table as a DataFrame sample string."""
    if not conn or not table_name: # Basic check
        return None
    try:
        # Check if table exists first
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        if cursor.fetchone() is None:
             logging.warning(f"Table '{table_name}' not found in DB during sample fetching.")
             return None # Table doesn't exist

        # Use LIMIT to fetch only a small sample
        # Quote table name just in case
        query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
        df_sample = pd.read_sql_query(query, conn)
        if df_sample.empty:
             return f"Table '{table_name}' exists but is empty (sample query returned no rows)."
        # Convert to markdown, handle potential errors
        return df_sample.to_markdown(index=False)
    except pd.io.sql.DatabaseError as e:
        # Handle cases where query fails even if table exists (e.g., permissions, malformed table)
        logging.error(f"Pandas SQL Error fetching sample data for table '{table_name}': {e}")
        return f"Error fetching sample data for table '{table_name}': {e}"
    except Exception as e:
        logging.error(f"Unexpected error fetching sample data for table '{table_name}': {e}", exc_info=True)
        return f"Unexpected error fetching sample for table '{table_name}'."
    
def format_context_for_db_selection(
    available_schemas: dict,
    db_samples: dict, # NEW: {db_id: sample_string or None}
    max_chars: int = MAX_SCHEMA_CHARS_ESTIMATE
) -> Tuple[str, bool]:
    """Formats schemas AND sample data into a JSON string for the DB selection prompt."""

    context_data = {}
    for db_id, schema_tables in available_schemas.items():
        context_data[db_id] = {
            "schema": schema_tables,
            "sample_data_snippet": db_samples.get(db_id, "N/A") # Get sample or use placeholder
        }

    full_context_str = json.dumps(context_data, indent=2)
    truncated = False

    if len(full_context_str) > max_chars:
        logging.warning(f"Full schema+sample context length ({len(full_context_str)} chars) exceeds limit ({max_chars}). Truncating.")
        truncated = True
        # Simple truncation - remove DBs until it fits. Better pre-filtering needed for prod.
        context_to_include = dict(context_data)
        while len(json.dumps(context_to_include, indent=2)) > max_chars and context_to_include:
             context_to_include.popitem() # Remove arbitrary item
        formatted_str = json.dumps(context_to_include, indent=2)
        if not context_to_include:
             logging.error("Schema+sample context too large even after truncation.")
             return "{}", True
        logging.warning(f"Truncated context string length: {len(formatted_str)}. Kept {len(context_to_include)} DB contexts.")
        return formatted_str, truncated
    else:
        return full_context_str, truncated