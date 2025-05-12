# aux_model.py
import os
import dotenv
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastembed import TextEmbedding
from huggingface_hub import hf_hub_download
from pathlib import Path # <--- ADD
from dotenv import load_dotenv


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
# Use Path for robustness, assuming .env is in project_root
_PROJECT_ROOT_MODEL_PY = Path(__file__).resolve().parent.parent # Adjust if model.py is nested deeper
dotenv.load_dotenv(_PROJECT_ROOT_MODEL_PY / '.env', override=True)
HF_TOKEN = os.getenv('HF_TOKEN')
logging.info("Loading auxiliary models...")

# --- Define MODELS_BASE_DIR (could also be imported from utils if preferred) ---
# For simplicity here, we define it relative to this file's project structure
# More robust: from utils import MODELS_BASE_DIR
try:
    # This assumes utils.py is in a 'utils' subdirectory from where app.py (and thus BASE_DIR) is defined
    # Or that model.py and utils.py share a common understanding of BASE_DIR
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent # Adjust if model.py is nested deeper
    MODELS_BASE_DIR = _PROJECT_ROOT / "models"
except NameError: # Fallback if utils.MODELS_BASE_DIR is not easily importable here
    _PROJECT_ROOT = Path(os.getcwd()) # Or Path(__file__).resolve().parent
    MODELS_BASE_DIR = _PROJECT_ROOT / "models"
    logging.warning(f"Could not import MODELS_BASE_DIR, defaulting to relative path in model.py: {MODELS_BASE_DIR}")


# --- Helper function for succinct logging ---


def log_key(message):
    logging.info(message)

try:
    from llama_cpp import Llama
    llama_cpp_available_for_aux = True
except ImportError:
    llama_cpp_available_for_aux = False
    logging.warning("llama_cpp not installed; SQL GGUF model will not load.")


import pynvml
def get_free_gpu_memory_mb(device_index: int = 0) -> int:
    """Returns free GPU memory in MiB, or -1 on failure."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free // (1024**2)
    except Exception as e:
        logging.warning(f"Could not query GPU memory: {e}")
        return -1


# --- Auxiliary Model Loading Function (Revised for GGUF SQL Model) ---
def load_aux_models() -> dict:
    """
    Loads auxiliary models: SQL Generator (GGUF) and Embedder.
    - Downloads/caches the SQL GGUF file via hf_hub_download.
    - Decides whether to load SQL model on GPU based on remaining VRAM.
    - Always returns a dict with keys: status, sql_gguf_model (optional),
      embedding_model (optional), plus any error fields.
    """
    models: dict = {}
    sql_model_loaded = False
    embed_model_loaded = False
    env_with_local_paths = _PROJECT_ROOT / "config" / ".env" # Or "config/paths.env"
    load_dotenv(env_with_local_paths, override=True) # Ensure it re-reads potentially updated .env
    # --- Define Auxiliary Model Cache Directory ---
    default_aux_cache_dir = MODELS_BASE_DIR / "auxiliary" # <--- Default Cache Subdirectory
    # Get from env var, or use our project-local default
    raw_cache_dir_env = os.getenv("MODEL_CACHE_DIR")
    if raw_cache_dir_env:
        cache_dir = Path(raw_cache_dir_env)
        logging.info(f"Using MODEL_CACHE_DIR from environment: {cache_dir}")
    else:
        cache_dir = default_aux_cache_dir
        logging.info(f"MODEL_CACHE_DIR not set, defaulting to: {cache_dir}")
    
    os.makedirs(cache_dir, exist_ok=True) # Ensure it exists

    # 1. SQL GGUF Model
    sql_model_path_str = os.getenv("SQL_GGUF_LOCAL_MODEL_PATH")
    if not sql_model_path_str:
        logging.error("SQL_GGUF_LOCAL_MODEL_PATH not set. Run download_models.py.")
        models["sql_load_error"] = "SQL model path not configured."
    else:
        sql_model_path = Path(sql_model_path_str)
        if not sql_model_path.exists():
            logging.error(f"SQL GGUF file not found at {sql_model_path}. Run download_models.py.")
            models["sql_load_error"] = f"SQL file not found at {sql_model_path}."
        else:
            sql_n_gpu_layers=os.getenv("SQL_GPU_LAYERS", "0")
            # ...
            try:
                sql_n_ctx = int(os.getenv("SQL_GGUF_N_CTX", "1024")) # Keep n_ctx setting
                # Determine sql_n_gpu_layers as before (based on SQL_GPU_LAYERS and VRAM)
                # ... (existing n_gpu_layers logic for SQL model) ...
                logging.info(f"Loading SQL model from: {sql_model_path} (n_gpu_layers={sql_n_gpu_layers}, n_ctx={sql_n_ctx})")
                models["sql_gguf_model"] = Llama(
                    model_path=str(sql_model_path),
                    n_gpu_layers=sql_n_gpu_layers,
                    n_ctx=sql_n_ctx,
                    verbose=False
                )
                sql_model_loaded = True
                logging.info("SQL GGUF model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading SQL GGUF model: {e}", exc_info=True)
                models["sql_load_error"] = f"Load error: {e}"


    # 2. Embedding Model
    embed_name = os.getenv("EMBEDDING_MODEL_NAME")
    # Use the cache directory specified by the downloader
    embedding_cache_dir_for_loader = os.getenv("EMBEDDING_MODEL_CACHE_DIR_FOR_LOADER")

    if not embed_name:
        logging.error("EMBEDDING_MODEL_NAME not set.")
        models["embed_load_error"] = "Embedding model name not configured."
    elif not embedding_cache_dir_for_loader:
        logging.warning("EMBEDDING_MODEL_CACHE_DIR_FOR_LOADER not set. TextEmbedding might use default cache or fail if model not pre-cached by downloader.")
        # Fallback or error, depending on strictness
        # For now, let TextEmbedding try its default if this isn't set.
        # Ideally, download_models.py ensures this is set and the dir is primed.
        try:
             logging.info(f"Loading embedding model: {embed_name} (cache dir not specified by downloader, using TextEmbedding default behavior)")
             models["embedding_model"] = TextEmbedding(model_name=embed_name) # cache_dir will be TextEmbedding's default
             embed_model_loaded = True
             logging.info("Embedding model loaded successfully (using default cache path).")
        except Exception as e:
             logging.error(f"Error loading embedding model (default cache): {e}", exc_info=True)
             models["embed_load_error"] = f"Embed load error (default cache): {e}"
    else:
        try:
            logging.info(f"Loading embedding model: {embed_name} from cache: {embedding_cache_dir_for_loader}")
            models["embedding_model"] = TextEmbedding(model_name=embed_name, cache_dir=str(embedding_cache_dir_for_loader))
            embed_model_loaded = True
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}", exc_info=True)
            models["embed_load_error"] = f"Embed load error: {e}"


    # 6) Finalize status
    if sql_model_loaded and embed_model_loaded:
        models["status"] = "loaded"
        logging.info("All auxiliary models loaded.")
    elif sql_model_loaded or embed_model_loaded:
        models["status"] = "partial"
        logging.warning("Partial auxiliary model load.")
    else:
        models["status"] = "error"
        errs = []
        if "sql_load_error" in models: errs.append(f"SQL: {models['sql_load_error']}")
        if "embed_load_error" in models: errs.append(f"Embed: {models['embed_load_error']}")
        models["error_message"] = "; ".join(errs) or "Unknown error"
        logging.error(f"Aux model loading failed: {models['error_message']}")

    return models

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