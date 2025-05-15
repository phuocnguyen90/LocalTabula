import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging


from pathlib import Path
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from qdrant_client import QdrantClient, models
import sqlite3
# These imports are now from your actual project structure
from .llm_interface import load_gguf_model, LLMWrapper, _get_free_gpu_memory_mb_internal, Llama
from utils.aux_model import load_aux_models     

import pynvml
SQLITE_TIMEOUT_SECONDS = 15
QDRANT_COLLECTION_PREFIX = "table_data_" # Using prefix convention
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODELS_BASE_DIR = PROJECT_ROOT  / "models"
CONFIG_DIR = PROJECT_ROOT  / "config"
PROMPT_PATH = CONFIG_DIR / "prompts.yaml"

MODELS_BASE_DIR = PROJECT_ROOT / "models"
MAIN_LLM_SUBDIR = MODELS_BASE_DIR / "main_llm"
AUX_MODELS_SUBDIR = MODELS_BASE_DIR / "auxiliary"
SQL_LLM_SUBDIR_IN_AUX = AUX_MODELS_SUBDIR / "sql_llm"
EMBEDDING_SUBDIR_IN_AUX = AUX_MODELS_SUBDIR / "embedding_cache"

ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(dotenv_path=str(ENV_PATH))  
MAX_SCHEMA_CHARS_ESTIMATE = 20000 # Rough estimate for prompt length control


# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)
# FastEmbed for embedding model
try:
    from fastembed import TextEmbedding
    fastembed_available = True
except ImportError:
    fastembed_available = False
    logging.warning("FastEmbed library not found. Embedding model cannot be loaded.")



class ModelResources:
    """
    Holds all loaded model resources.
    Initializes them sequentially, considering VRAM for GGUF models.
    """
    def __init__(self, force_verbose_llms: bool = False):
        logger.info("--- Initializing All Model Resources ---")
        self.main_llm_wrapper: Optional[LLMWrapper] = None
        self.sql_gguf_model: Optional[Llama] = None # llama_cpp.Llama type
        self.embedding_model: Optional[TextEmbedding] = None
        self.status: str = "pending"
        self.error_messages: Dict[str, str] = {}
        self.force_verbose_llms = force_verbose_llms # For debugging specific loads

        self._load_all()

    def _load_all(self):
        # --- 1. Main LLM ---
        logger.info("--- Loading Main LLM ---")
        # LLMWrapper handles its own .env loading for paths and GGUF params
        # It uses the load_gguf_model internally now.
        self.main_llm_wrapper = LLMWrapper(force_verbose_main_llm=self.force_verbose_llms)
        if not self.main_llm_wrapper or not self.main_llm_wrapper.is_ready:
            self.error_messages["main_llm"] = "Main LLM (LLMWrapper) failed to initialize."
            logger.error(self.error_messages["main_llm"])
            # Decide if we should stop or try to load other models
            # For now, let's continue to load aux models

        # --- Get VRAM after Main LLM for Aux models ---
        vram_after_main_llm_mb = -1
        if self.main_llm_wrapper and self.main_llm_wrapper.is_ready and self.main_llm_wrapper.mode == "local_gguf":
            # Access free VRAM via the method in llm_interface
            vram_after_main_llm_mb = _get_free_gpu_memory_mb_internal()
            logger.info(f"VRAM available after Main LLM load (for Aux models): {vram_after_main_llm_mb} MiB")
        else:
            # If main LLM is OpenRouter or failed, get current free VRAM
            vram_after_main_llm_mb = _get_free_gpu_memory_mb_internal()
            logger.info(f"Main LLM not local GGUF or not ready. Current free VRAM (for Aux models): {vram_after_main_llm_mb} MiB")


        # --- 2. SQL GGUF Model ---
        logger.info("--- Loading SQL GGUF Model ---")
        self.sql_gguf_model = load_gguf_model(
            model_name_log_prefix="SQL LLM",
            model_path_str=os.getenv("LOCAL_SQL_LLM_PATH"),
            n_gpu_layers_env_var="SQL_LLM_N_GPU_LAYERS",
            n_ctx_env_var="SQL_LLM_N_CTX",
            use_cpu_env_var="SQL_LLM_USE_CPU",
            default_n_gpu_layers=-1, # Or 0 if you prefer CPU default for aux
            default_n_ctx=2048,
            available_vram_mb=vram_after_main_llm_mb, # Pass VRAM after main LLM
            model_vram_buffer_mb=int(os.getenv("SQL_LLM_VRAM_BUFFER_MB", "300")), # Smaller buffer for aux
            force_verbose_llama_cpp=self.force_verbose_llms
        )
        if not self.sql_gguf_model:
            self.error_messages["sql_llm"] = "SQL GGUF model failed to load."
            logger.error(self.error_messages["sql_llm"])

        # --- 3. Embedding Model ---
        if not fastembed_available:
            self.error_messages["embedding"] = "FastEmbed library not available."
            logger.error(self.error_messages["embedding"])
        else:
            logger.info("--- Loading Embedding Model ---")
            embed_repo_name = os.getenv("EMBEDDING_MODEL_REPO_NAME")
            embedding_cache_dir_str = os.getenv("LOCAL_EMBEDDING_CACHE_DIR")

            if not embed_repo_name:
                self.error_messages["embedding"] = "EMBEDDING_MODEL_REPO_NAME not set."
                logger.error(self.error_messages["embedding"])
            elif not embedding_cache_dir_str:
                self.error_messages["embedding"] = "LOCAL_EMBEDDING_CACHE_DIR not set."
                logger.error(self.error_messages["embedding"])
            else:
                embedding_cache_path = Path(embedding_cache_dir_str)
                if not embedding_cache_path.exists():
                    self.error_messages["embedding"] = f"Embedding cache dir {embedding_cache_path} missing."
                    logger.error(self.error_messages["embedding"])
                else:
                    try:
                        logger.info(f"Loading embedding model: {embed_repo_name} from cache: {embedding_cache_path}")
                        self.embedding_model = TextEmbedding(
                            model_name=embed_repo_name,
                            cache_dir=str(embedding_cache_path)
                            # providers = ['CPUExecutionProvider'] # Example: Force CPU for FastEmbed if needed
                        )
                        logger.info("Embedding model loaded successfully.")
                    except Exception as e:
                        self.error_messages["embedding"] = f"Embedding model load error: {e}"
                        logger.error(self.error_messages["embedding"], exc_info=True)
        
        # --- Final Status ---
        if self.main_llm_wrapper and self.main_llm_wrapper.is_ready and \
           self.sql_gguf_model and self.embedding_model:
            self.status = "loaded_all"
            logger.info("All model resources loaded successfully.")
        elif self.error_messages:
            self.status = "error_partial"
            logger.error(f"Partial or failed model resource loading. Errors: {self.error_messages}")
        else: # Should not happen if logic is correct, but as a fallback
            self.status = "unknown_state_no_errors_no_success" # Should be caught by specific checks
            logger.warning("Model resources in an unknown state after loading attempts (no errors, but not all loaded).")


    def get_models_dict(self) -> Dict[str, Any]:
        """Returns a dictionary similar to what load_aux_models used to return, plus main LLM."""
        return {
            "main_llm_wrapper": self.main_llm_wrapper,
            "sql_gguf_model": self.sql_gguf_model,
            "embedding_model": self.embedding_model,
            "status": self.status, # "loaded_all", "error_partial", etc.
            "load_errors": self.error_messages if self.error_messages else None
        }
