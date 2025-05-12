import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# These imports are now from your actual project structure
from utils.llm_interface import LLMWrapper 
from utils.aux_model import load_aux_models     

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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import from the 'utils' package
from utils.utils import ENV_PATH
from utils.llm_interface import LLMWrapper
from utils.aux_model import load_aux_models

def initialize_all_models():
    """
    Loads all required models using their respective loaders.
    Assumes scripts/download_models.py has been run and .env contains local paths.
    """
    print("Initializing all models...")

    if not ENV_PATH.exists():
        print(f"ERROR: Environment file not found at {ENV_PATH}. "
              "Run scripts/download_models.py script first.")
        return None, None
    
    # Load environment variables from the central .env file
    # This ensures that LLMWrapper and load_aux_models get the correct paths
    # set by the download script.
    load_dotenv(ENV_PATH, override=True)

    main_llm_wrapper = LLMWrapper() # LLMWrapper will load .env internally using ENV_PATH
    if not main_llm_wrapper.is_ready:
        print("ERROR: Main LLM Wrapper failed to initialize.")
        # Decide on error handling, e.g., return None for main_llm_wrapper
    
    auxiliary_models = load_aux_models() # load_aux_models will also load .env internally
    if auxiliary_models.get("status") != "loaded":
        # Logged within load_aux_models, but can add more here
        print(f"Warning/Error: Auxiliary models did not load completely. Status: {auxiliary_models.get('status')}")

    print("Model initialization process finished.")
    return main_llm_wrapper, auxiliary_models

if __name__ == "__main__":
    wrapper, aux_models_dict = initialize_all_models()

    if wrapper and wrapper.is_ready:
        print(f"Main LLM ready in mode: {wrapper.mode}")
    else:
        print("Main LLM failed to load or not initialized.")

    if aux_models_dict:
        print(f"Aux models status: {aux_models_dict.get('status')}")
        if aux_models_dict.get('sql_gguf_model'):
            print("SQL GGUF model appears loaded.")
        if aux_models_dict.get('embedding_model'):
            print("Embedding model appears loaded.")
        if aux_models_dict.get('status') != 'loaded':
            print(f"Aux models error: {aux_models_dict.get('error_message')}")
    else:
        print("Auxiliary models dictionary not returned or load failed.")