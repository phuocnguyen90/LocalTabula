# utils/aux_model.py
import os
import logging
from fastembed import TextEmbedding
from pathlib import Path 
from dotenv import load_dotenv
# import torch
# Import ENV_PATH from .utils (sibling module in the same package)
from utils.utils import ENV_PATH


try:
    from llama_cpp import Llama
    llama_cpp_available_for_aux = True
except ImportError:
    llama_cpp_available_for_aux = False
    # logging.warning is configured below, so it's fine
    # print("Warning: llama_cpp library not found. Cannot load GGUF SQL model.")
# --- pynvml for VRAM check and GPU availability ---

try:
    import pynvml
    pynvml_available_aux = True
    pynvml.nvmlInit() # Initialize NVML once per module import (or globally)
except ImportError:
    pynvml_available_aux = False
    logging.info("pynvml not available in aux_model, GPU detection and VRAM checks will be limited.")
except pynvml.NVMLError as e:
    pynvml_available_aux = False # Treat init error as unavailable
    logging.warning(f"pynvml initialization error in aux_model: {e}. GPU features might be impacted.")

# --- Load Environment Variables ---
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True) # Override to get latest paths from download script
else:
    logging.warning(f".env file not found at {ENV_PATH}. Auxiliary models might not load correctly.")


# Set up a succinct logging format (can be defined once in app.py or utils.py and imported)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
if not llama_cpp_available_for_aux: # Log after basicConfig
    logging.warning("llama_cpp library not found. Cannot load GGUF SQL model.")

def is_gpu_available_pynvml() -> bool:
    """Checks if any NVIDIA GPU is accessible via pynvml."""
    if not pynvml_available_aux:
        return False
    try:
        # pynvml.nvmlInit() # Already initialized at module level
        device_count = pynvml.nvmlDeviceGetCount()
        return device_count > 0
    except pynvml.NVMLError:
        return False
    # finally:
        # pynvml.nvmlShutdown() # Shutdown if you init/shutdown per call

def get_free_gpu_memory_mb_aux(device_index: int = 0) -> int:
    """Returns free GPU memory in MiB, or -1 on failure."""
    if not pynvml_available_aux or not is_gpu_available_pynvml(): # Check if GPU even present
        return -1
    try:
        # pynvml.nvmlInit() # Already initialized
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = info.free // (1024**2)
        # total_mb = info.total // (1024**2) # If you need total for logging
        # logging.debug(f"GPU-{device_index} (aux_model): Free VRAM: {free_mb} MiB / Total: {total_mb} MiB")
        return free_mb
    except pynvml.NVMLError: # Catch specific pynvml errors
        return -1
    # finally:
        # pynvml.nvmlShutdown()


def load_aux_models(available_vram_mb: int = -1) -> dict:
    """
    Loads auxiliary models: SQL Generator (GGUF) and Embedder.
    Assumes models are pre-downloaded by scripts/download_models.py.
    """
    logging.info("Attempting to load auxiliary models...")
    models: dict = {}
    sql_model_loaded = False
    embed_model_loaded = False

    # 1. SQL LLM Model
    if not llama_cpp_available_for_aux:
        models["sql_load_error"] = "llama_cpp library not available."
    else:
        sql_model_path_str = os.getenv("LOCAL_SQL_LLM_PATH")
        if not sql_model_path_str:
            logging.error("LOCAL_SQL_LLM_PATH not set in .env. Run scripts/download_models.py.")
            models["sql_load_error"] = "SQL LLM path not configured."
        else:
            sql_model_path = Path(sql_model_path_str)
            sql_model_size_mb = sql_model_path.stat().st_size / (1024**2)
            logging.info(f"SQL LLM GGUF file size: {sql_model_size_mb:.1f} MiB")
            if not sql_model_path.exists():
                logging.error(f"SQL LLM GGUF file not found at: {sql_model_path}. Run scripts/download_models.py.")
                models["sql_load_error"] = f"SQL LLM file not found at {sql_model_path}."
            else:
                use_cpu_str = os.getenv("SQL_LLM_USE_CPU", "false").lower()
                use_cpu = use_cpu_str in ("true", "1", "t", "y", "yes")

                sql_n_gpu_layers = 0
                gpu_system_available = is_gpu_available_pynvml() 
                if use_cpu:
                    logging.info("SQL_LLM_USE_CPU is true; loading SQL LLM on CPU.")
                    sql_n_gpu_layers = 0
                elif not gpu_system_available:
                    logging.info("No GPU detected by pynvml; loading SQL LLM on CPU.")
                    sql_n_gpu_layers = 0
                else:
                    logging.info("GPU detected by pynvml; loading SQL LLM on GPU.")
                    try:
                        sql_n_gpu_layers_str = os.getenv("SQL_LLM_N_GPU_LAYERS", "-1")
                        sql_n_gpu_layers = int(sql_n_gpu_layers_str)
                    except ValueError:
                        logging.warning(f"Invalid value for SQL_LLM_N_GPU_LAYERS: '{sql_n_gpu_layers_str}'. Defaulting to -1.")
                        sql_n_gpu_layers = -1
                
                logging.info(f"SQL LLM GGUF n_gpu_layers={sql_n_gpu_layers}")

                try:
                    sql_n_ctx_str = os.getenv("SQL_LLM_N_CTX", "2048")
                    sql_n_ctx = int(sql_n_ctx_str)
                except ValueError:
                    logging.warning(f"Invalid value for SQL_LLM_N_CTX: '{sql_n_ctx_str}'. Defaulting to 2048.")
                    sql_n_ctx = 2048
                logging.info(f"SQL LLM GGUF n_ctx={sql_n_ctx}")
                
                try:
                    vram_before_sql_load_aux_mb = get_free_gpu_memory_mb_aux() if sql_n_gpu_layers != 0 and gpu_system_available else -1
                    logging.info(f"Loading SQL LLM GGUF model from: {sql_model_path}")
                    models["sql_gguf_model"] = Llama(
                        model_path=str(sql_model_path),
                        n_gpu_layers=sql_n_gpu_layers,
                        n_ctx=sql_n_ctx,
                        verbose=False
                    )
                    sql_model_loaded = True
                    logging.info("SQL LLM (local GGUF) loaded successfully.")
                    
                    vram_after = get_free_gpu_memory_mb_aux()
                    delta = (vram_after - vram_before_sql_load_aux_mb) / (1024**2)
                    logging.info(f"VRAM after SQL LLM load: {vram_after/1024**2:.1f} MiB  (Δ {delta:.1f} MiB)")
                except Exception as e:
                    logging.error(f"Error loading SQL LLM GGUF model: {e}", exc_info=True)
                    models["sql_load_error"] = f"Load error: {e}"

    # 2. Embedding Model
    embed_repo_name = os.getenv("EMBEDDING_MODEL_REPO_NAME")
    embedding_cache_dir_str = os.getenv("LOCAL_EMBEDDING_CACHE_DIR")

    if not embed_repo_name:
        logging.error("EMBEDDING_MODEL_REPO_NAME not set in .env.")
        models["embed_load_error"] = "Embedding model name not configured."
    elif not embedding_cache_dir_str:
        logging.error("LOCAL_EMBEDDING_CACHE_DIR not set in .env. Run scripts/download_models.py.")
        models["embed_load_error"] = "Embedding model cache directory not configured."
    else:
        embedding_cache_path = Path(embedding_cache_dir_str)
        if not embedding_cache_path.exists(): # The download script should create it.
             logging.error(f"Embedding cache directory {embedding_cache_path} does not exist. Run scripts/download_models.py.")
             models["embed_load_error"] = f"Embedding cache dir {embedding_cache_path} missing."
        else:
            try:
                logging.info(f"Loading embedding model: {embed_repo_name} using cache: {embedding_cache_path}")
                # FastEmbed expects the model_name (repo ID) and the cache_dir where it can find/store it
                models["embedding_model"] = TextEmbedding(model_name=embed_repo_name, cache_dir=str(embedding_cache_path))
                embed_model_loaded = True
                logging.info("Embedding model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading embedding model: {e}", exc_info=True)
                models["embed_load_error"] = f"Embed load error: {e}"

    # Finalize status
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
        models["error_message"] = "; ".join(errs) or "Unknown error loading auxiliary models."
        logging.error(f"Aux model loading failed: {models['error_message']}")

    return models
