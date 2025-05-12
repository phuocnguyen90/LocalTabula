import os
import sys
from pathlib import Path
from dotenv import load_dotenv, set_key
from huggingface_hub import hf_hub_download
from fastembed import TextEmbedding # To trigger its download

# Ensure the project root is in sys.path to import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# from utils.utils import MODELS_BASE_DIR # Assuming MODELS_BASE_DIR is defined in utils

# Define MODELS_BASE_DIR here if not importing from utils
MODELS_BASE_DIR = PROJECT_ROOT / "models"

# Define specific cache subdirectories
MAIN_LLM_CACHE_DIR = MODELS_BASE_DIR / "main_llm"
SQL_GGUF_CACHE_DIR = MODELS_BASE_DIR / "auxiliary" / "sql_gguf"
EMBEDDING_CACHE_DIR = MODELS_BASE_DIR / "auxiliary" / "embedding_cache"

# Path to the .env file that stores final model paths
# This could be config/.env or a separate config/paths.env
ENV_FILE_PATH = PROJECT_ROOT / "config" / ".env" # Or "config/paths.env"

def update_env_file(key, value):
    """Updates or adds a key-value pair in the .env file."""
    print(f"Updating .env: {key} = {value}")
    # Create .env if it doesn't exist or if set_key requires it
    ENV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE_PATH.touch(exist_ok=True)
    set_key(str(ENV_FILE_PATH), key, str(value), quote_mode="always")


def download_all_models():
    print(f"Using base directory for models: {MODELS_BASE_DIR}")
    MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SQL_GGUF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load repo/filename configs (could be from a primary .env or defaults)
    primary_env_path = PROJECT_ROOT / "config" / ".env" # The one with REPO_IDs
    load_dotenv(primary_env_path)

    hf_token = os.getenv("HF_TOKEN")

    # 1. Main LLM
    main_repo = os.getenv("MAIN_LLM_REPO_ID")
    main_file = os.getenv("MAIN_LLM_FILENAME")
    if main_repo and main_file:
        print(f"Downloading Main LLM: {main_repo}/{main_file}...")
        try:
            # Download directly into the target subfolder, not using HF's general cache_dir structure here for predictability
            downloaded_main_llm_path = hf_hub_download(
                repo_id=main_repo,
                filename=main_file,
                local_dir=MAIN_LLM_CACHE_DIR,
                local_dir_use_symlinks=False, # Store actual file
                token=hf_token
            )
            print(f"Main LLM downloaded to: {downloaded_main_llm_path}")
            # Update .env with the exact local path for LLMWrapper to use
            update_env_file("LOCAL_LLM_GGUF_MODEL_PATH", downloaded_main_llm_path)
            # Also set os.environ for current session if download is followed by load in same script/notebook
            os.environ["LOCAL_LLM_GGUF_MODEL_PATH"] = str(downloaded_main_llm_path)
        except Exception as e:
            print(f"ERROR downloading Main LLM: {e}")
    else:
        print("Skipping Main LLM download: MAIN_LLM_REPO_ID or MAIN_LLM_FILENAME not set.")

    # 2. SQL GGUF
    sql_repo = os.getenv("SQL_GGUF_REPO_ID")
    sql_file = os.getenv("SQL_GGUF_FILENAME")
    if sql_repo and sql_file:
        print(f"Downloading SQL GGUF: {sql_repo}/{sql_file}...")
        try:
            downloaded_sql_gguf_path = hf_hub_download(
                repo_id=sql_repo,
                filename=sql_file,
                local_dir=SQL_GGUF_CACHE_DIR,
                local_dir_use_symlinks=False,
                token=hf_token
            )
            print(f"SQL GGUF downloaded to: {downloaded_sql_gguf_path}")
            update_env_file("SQL_GGUF_LOCAL_MODEL_PATH", downloaded_sql_gguf_path)
            os.environ["SQL_GGUF_LOCAL_MODEL_PATH"] = str(downloaded_sql_gguf_path)
        except Exception as e:
            print(f"ERROR downloading SQL GGUF: {e}")
    else:
        print("Skipping SQL GGUF download: SQL_GGUF_REPO_ID or SQL_GGUF_FILENAME not set.")

    # 3. Embedding Model
    embed_name = os.getenv("EMBEDDING_MODEL_NAME")
    if embed_name:
        print(f"Ensuring Embedding Model ({embed_name}) is cached...")
        try:
            # TextEmbedding will download to its own structure within EMBEDDING_CACHE_DIR
            TextEmbedding(model_name=embed_name, cache_dir=str(EMBEDDING_CACHE_DIR))
            print(f"Embedding model '{embed_name}' cached in '{EMBEDDING_CACHE_DIR}'.")
            # The loader (load_aux_models) will need to know this cache_dir
            update_env_file("EMBEDDING_MODEL_CACHE_DIR_FOR_LOADER", str(EMBEDDING_CACHE_DIR))
            os.environ["EMBEDDING_MODEL_CACHE_DIR_FOR_LOADER"] = str(EMBEDDING_CACHE_DIR)
        except Exception as e:
            print(f"ERROR caching Embedding Model: {e}")
    else:
        print("Skipping Embedding Model download: EMBEDDING_MODEL_NAME not set.")

    print("Model download process finished.")

if __name__ == "__main__":
    download_all_models()