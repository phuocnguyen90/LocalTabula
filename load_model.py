import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# These imports are now from your actual project structure
from llm_interface import LLMWrapper # Assuming llm_interface.py is at project_root or in a module
from aux_model import load_aux_models     # Assuming model.py is at project_root or in a module

def initialize_all_models():
    """
    Loads all required models using their respective loaders.
    Assumes download_models.py has been run and .env contains local paths.
    """
    print("Initializing all models...")

    # Load the .env file that contains the local model paths
    # (set by download_models.py)
    env_with_local_paths = PROJECT_ROOT / "config" / ".env" # Or "config/paths.env"
    if not env_with_local_paths.exists():
        print(f"ERROR: Environment file with model paths not found at {env_with_local_paths}. Run download_models.py script first.")
        return None, None # Or raise an error
    load_dotenv(env_with_local_paths, override=True) # Override to pick up latest paths

    main_llm_wrapper = LLMWrapper()
    if not main_llm_wrapper.is_ready:
        print("ERROR: Main LLM Wrapper failed to initialize.")
        # Potentially return or handle this error
    
    auxiliary_models = load_aux_models()
    if auxiliary_models.get("status") != "loaded":
        print(f"ERROR: Auxiliary models failed to load fully. Status: {auxiliary_models.get('status')}, Error: {auxiliary_models.get('error_message')}")
        # Potentially return or handle

    print("Model initialization process finished.")
    return main_llm_wrapper, auxiliary_models

if __name__ == "__main__":
    # This is for testing the load_all_models.py script directly
    # In your app, you'd import and call initialize_all_models()
    wrapper, aux_models_dict = initialize_all_models()

    if wrapper and wrapper.is_ready:
        print(f"Main LLM ready in mode: {wrapper.mode}")
    else:
        print("Main LLM failed to load.")

    if aux_models_dict:
        print(f"Aux models status: {aux_models_dict.get('status')}")
        if aux_models_dict.get('sql_gguf_model'):
            print("SQL GGUF model appears loaded.")
        if aux_models_dict.get('embedding_model'):
            print("Embedding model appears loaded.")
    else:
        print("Auxiliary models dictionary not returned.")