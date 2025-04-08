# model.py
import os
import dotenv
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fast_embed import TextEmbedding
from huggingface_hub import hf_hub_download

try:
    from llama_cpp import Llama
    llama_cpp_available_for_aux = True
except ImportError:
    llama_cpp_available_for_aux = False
    logging.warning("llama_cpp library not found. Cannot load GGUF SQL model.")


# Set up a succinct logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Load environment variables
dotenv.load_dotenv('.env', override=True)
HF_TOKEN = os.getenv('HF_TOKEN')
logging.info("Loading auxiliary models...")

# --- Helper function for succinct logging ---
def log_key(message):
    logging.info(message)

try:
    from llama_cpp import Llama
    llama_cpp_available_for_aux = True
except ImportError:
    llama_cpp_available_for_aux = False
    logging.warning("llama_cpp not installed; SQL GGUF model will not load.")


# --- Auxiliary Model Loading Function (Revised for GGUF SQL Model) ---
def load_aux_models():
    """
    Loads auxiliary models: SQL Generator (GGUF, downloads if needed) and Embedder.
    Returns a dictionary containing the loaded objects or status info.
    """
    models_dict = {}
    sql_model_loaded = False
    embed_model_loaded = False

    # --- Load SQL Model (GGUF with Auto-Download) ---
    if not llama_cpp_available_for_aux:
        logging.error("Cannot load SQL GGUF model: llama_cpp library is not installed.")
        models_dict['sql_load_error'] = "llama_cpp not installed"
    else:
        # Get config from environment variables
        repo_id = os.getenv('SQL_GGUF_REPO_ID')
        print("SQL_GGUF_REPO_ID:", repo_id)

        filename = os.getenv('SQL_GGUF_FILENAME')
        # Use specified cache dir or let huggingface_hub use its default
        cache_dir = os.getenv('MODEL_CACHE_DIR', None) # None means use default HF cache

        if not repo_id or not filename:
            logging.error("SQL_GGUF_REPO_ID or SQL_GGUF_FILENAME env vars not set.")
            models_dict['sql_load_error'] = "GGUF Repo ID/Filename not configured"
        else:
            local_sql_model_path = None
            try:
                logging.info(f"Checking cache/downloading SQL GGUF: repo='{repo_id}', file='{filename}'")
                if cache_dir:
                    logging.info(f"Using local cache directory: {cache_dir}")
                    # Ensure cache dir exists if specified
                    os.makedirs(cache_dir, exist_ok=True)

                # hf_hub_download handles caching and downloading automatically
                local_sql_model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=cache_dir,
                    token=HF_TOKEN or None # Use token if provided (for private models)
                    # Add resume_download=True if desired
                )
                logging.info(f"SQL GGUF model path resolved to: {local_sql_model_path}")

            
            except Exception as e:
                logging.error(f"Error resolving/downloading SQL GGUF model: {e}", exc_info=True)
                models_dict['sql_load_error'] = f"Download/Cache Error: {e}"

            # Proceed to load if download/cache check was successful
            if local_sql_model_path and os.path.exists(local_sql_model_path):
                try:
                    # Get GGUF parameters (same logic as before)
                    sql_n_gpu_layers_str = os.getenv('SQL_GGUF_N_GPU_LAYERS', os.getenv('GGUF_N_GPU_LAYERS', '-1'))
                    sql_n_gpu_layers = int(sql_n_gpu_layers_str) if sql_n_gpu_layers_str.lstrip('-').isdigit() else -1
                    sql_n_ctx_str = os.getenv('SQL_GGUF_N_CTX', os.getenv('GGUF_N_CTX', '1024'))
                    sql_n_ctx = int(sql_n_ctx_str) if sql_n_ctx_str.isdigit() else 1024

                    logging.info(f"Loading SQL GGUF model using llama_cpp from: {local_sql_model_path}")
                    # logging.info(f"SQL GGUF Params: n_gpu_layers={sql_n_gpu_layers}, n_ctx={sql_n_ctx}")

                    sql_gguf_model = Llama(
                        model_path=local_sql_model_path, # Use the resolved path
                        n_gpu_layers=sql_n_gpu_layers,
                        n_ctx=sql_n_ctx,
                        verbose=True
                    )
                    models_dict['sql_gguf_model'] = sql_gguf_model
                    sql_model_loaded = True
                    logging.info("SQL GGUF model loaded successfully.")

                except Exception as e:
                    logging.error(f"Failed to load SQL GGUF model with llama_cpp: {e}", exc_info=True)
                    models_dict['sql_load_error'] = f"llama_cpp Load Error: {e}"
            else:
                 # This case is hit if hf_hub_download failed and returned None/invalid path
                 logging.error("SQL GGUF model path not available after download/cache check.")
                 if 'sql_load_error' not in models_dict: # Add error if not already set by download exception
                      models_dict['sql_load_error'] = "Model path resolution failed"


    # --- Load Embedding Model ---
    try:
        logging.info("Loading Embedding model (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)...")
        embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # FastEmbed likely uses huggingface_hub internally too, respecting cache
        embedding_model = TextEmbedding.add_custom_model(model=embedding_model_name, model_file=cache_dir)
        models_dict['embedding_model'] = embedding_model
        embed_model_loaded = True
        logging.info("Embedding model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Embedding model: {e}", exc_info=True)
        models_dict['embed_load_error'] = f"Embedder Load Error: {e}"


    # Determine overall status
    if sql_model_loaded and embed_model_loaded:
        models_dict['status'] = 'loaded'
        logging.info("All auxiliary models loaded successfully.")
    elif sql_model_loaded or embed_model_loaded:
        models_dict['status'] = 'partial'
        logging.warning("Some auxiliary models failed to load.")
    else:
        models_dict['status'] = 'error'
        # Consolidate error messages if available
        errors = []
        if 'sql_load_error' in models_dict: errors.append(f"SQL: {models_dict['sql_load_error']}")
        if 'embed_load_error' in models_dict: errors.append(f"Embed: {models_dict['embed_load_error']}")
        models_dict['error_message'] = "; ".join(errors) if errors else "Aux model loading failed (unknown reason)."
        logging.error(f"Auxiliary models loading failed: {models_dict['error_message']}")

    return models_dict

# --- Optional: Test Aux Models ---
if __name__ == "__main__":
    print("Testing auxiliary model loading directly...")
    aux_models = load_aux_models()
    print(f"\nAux models loading status: {aux_models.get('status')}")

    if aux_models.get('sql_gguf_model'):
        print("SQL GGUF model appears loaded.")
        # Add a simple generation test if desired
        try:
            sql_llm = aux_models.get('sql_gguf_model')
            test_sql_prompt = "SELECT COUNT(*) FROM employees;" # Simple prompt
            print(f"\nTesting SQL GGUF with prompt: '{test_sql_prompt}'")
            output = sql_llm(test_sql_prompt, max_tokens=50, temperature=0.1)
            print("SQL GGUF Response:", output)
        except Exception as e:
            print(f"Error during SQL GGUF test generation: {e}")

    if aux_models.get('embedding_model'):
        print("Embedding model appears loaded.")
        # Add embedding test if desired

    if aux_models.get('status') != 'loaded':
        print("\nError:", aux_models.get('error_message', 'Unknown loading error'))