# model.py
import os
import dotenv
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastembed import TextEmbedding
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

    # 1) Environment / flags
    sql_use_cpu = os.getenv("SQL_USE_CPU", "false").lower() in ("true", "1", "t", "y", "yes")
    gpu_available = torch.cuda.is_available()

    # 2) Download / resolve the SQL GGUF model file
    repo_id  = os.getenv("SQL_GGUF_REPO_ID")
    filename = os.getenv("SQL_GGUF_FILENAME")
    cache_dir = os.getenv("MODEL_CACHE_DIR", None)

    if not repo_id or not filename:
        logging.error("SQL_GGUF_REPO_ID or SQL_GGUF_FILENAME not set")
        models["sql_load_error"] = "Config missing"
    else:
        try:
            logging.info(f"Downloading/caching SQL GGUF: repo={repo_id}, file={filename}")
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                logging.info(f"Using cache dir: {cache_dir}")
            local_sql_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                token=os.getenv("HF_TOKEN", None)
            )
            logging.info(f"Resolved SQL model path: {local_sql_path}")
        except Exception as e:
            logging.error(f"Failed download/caching of SQL model: {e}", exc_info=True)
            models["sql_load_error"] = f"Download error: {e}"
            local_sql_path = None

    # 3) Decide GPU layers for SQL model
    sql_n_gpu_layers = 0
    if local_sql_path and gpu_available and not sql_use_cpu:
        free_vram = get_free_gpu_memory_mb()
        logging.info(f"Free VRAM after main LLM: {free_vram} MiB")

        # Estimate footprint (1.2× file size)
        size_mb = os.path.getsize(local_sql_path) // (1024**2)
        required_mb = int(size_mb * 1.2)

        if free_vram >= required_mb:
            try:
                sql_n_gpu_layers = int(os.getenv("SQL_GPU_LAYERS", "-1"))
            except ValueError:
                sql_n_gpu_layers = -1
            logging.info(f"Allocating {sql_n_gpu_layers} GPU layers to SQL model (need ~{required_mb} MiB)")
        else:
            logging.warning(f"Insufficient VRAM ({free_vram} MiB < required {required_mb} MiB), loading SQL on CPU")

    # 4) Load the SQL GGUF model
    if local_sql_path:
        try:
            sql_n_ctx = int(os.getenv("SQL_GGUF_N_CTX", os.getenv("GGUF_N_CTX", "1024")))
            logging.info(f"Loading SQL model (n_gpu_layers={sql_n_gpu_layers}, n_ctx={sql_n_ctx})")
            models["sql_gguf_model"] = Llama(
                model_path=local_sql_path,
                n_gpu_layers=sql_n_gpu_layers,
                n_ctx=sql_n_ctx,
                verbose=False
            )
            sql_model_loaded = True
            logging.info("SQL GGUF model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading SQL GGUF model: {e}", exc_info=True)
            models["sql_load_error"] = f"Load error: {e}"

    # 5) Load the embedding model
    try:
        embed_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        logging.info(f"Loading embedding model: {embed_name}")
        models["embedding_model"] = TextEmbedding(model=embed_name, model_file=cache_dir)
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