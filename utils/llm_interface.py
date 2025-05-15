
# utils/llm_interface.py
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from llama_cpp import Llama
    llama_cpp_available = True
except ImportError:
    llama_cpp_available = False
    # Logging will be handled by the module that calls this

# pynvml for GPU checks (can be initialized once globally if preferred)
try:
    import pynvml
    pynvml_available_interface = True
except ImportError:
    pynvml_available_interface = False

if pynvml_available_interface:
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        pynvml_available_interface = False # Mark as unavailable if init fails
        logging.error(f"llm_interface: pynvml.nvmlInit() error: {e}. GPU features might be impaired.")


def _is_gpu_available_pynvml_internal() -> bool:
    if not pynvml_available_interface: return False
    try:
        return pynvml.nvmlDeviceGetCount() > 0
    except pynvml.NVMLError: return False

def _get_free_gpu_memory_mb_internal(device_index: int = 0) -> int:
    if not pynvml_available_interface or not _is_gpu_available_pynvml_internal(): return -1
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free // (1024**2)
    except pynvml.NVMLError: return -1


def load_gguf_model(
    model_name_log_prefix: str, # For logging clarity (e.g., "Main LLM", "SQL LLM")
    model_path_str: Optional[str],
    n_gpu_layers_env_var: str, # e.g., "MAIN_LLM_N_GPU_LAYERS"
    n_ctx_env_var: str,        # e.g., "MAIN_LLM_N_CTX"
    use_cpu_env_var: str,      # e.g., "MAIN_LLM_USE_CPU"
    default_n_gpu_layers: int = -1,
    default_n_ctx: int = 4096,
    available_vram_mb: int = -1, # Optional: pass current VRAM for smart decisions
    model_vram_buffer_mb: int = 500, # Buffer for VRAM check
    force_verbose_llama_cpp: bool = False # For debugging specific model loads
) -> Optional[Llama]:
    """
    Reusable function to load a GGUF model using llama.cpp.
    Handles .env variable reading, GPU availability, and VRAM checks.
    """
    if not llama_cpp_available:
        logging.error(f"[{model_name_log_prefix}] llama_cpp library not found. Cannot load GGUF model.")
        return None

    if not model_path_str:
        logging.error(f"[{model_name_log_prefix}] Model path environment variable not set or empty.")
        return None

    model_path = Path(model_path_str)
    if not model_path.exists():
        logging.error(f"[{model_name_log_prefix}] GGUF file not found at: {model_path}.")
        return None

    model_size_mb = model_path.stat().st_size / (1024**2)
    logging.info(f"[{model_name_log_prefix}] File path: {model_path}, Size: {model_size_mb:.1f} MiB")

    # Determine if GPU should be used
    n_gpu_layers_to_use = 0
    force_cpu_str = os.getenv(use_cpu_env_var, "false").lower()
    force_cpu_flag = force_cpu_str in ("true", "1", "t", "y", "yes")
    
    gpu_system_available = _is_gpu_available_pynvml_internal()

    if force_cpu_flag:
        logging.info(f"[{model_name_log_prefix}] {use_cpu_env_var} is true; loading on CPU.")
        n_gpu_layers_to_use = 0
    elif not gpu_system_available:
        logging.info(f"[{model_name_log_prefix}] No GPU detected by pynvml; loading on CPU.")
        n_gpu_layers_to_use = 0
    else:
        # GPU available and not forced to CPU, consider VRAM and .env config
        if available_vram_mb != -1 : # Only check if valid VRAM is passed
            required_vram_for_model_mb = model_size_mb + model_vram_buffer_mb
            if available_vram_mb < required_vram_for_model_mb:
                logging.warning(
                    f"[{model_name_log_prefix}] Low available VRAM ({available_vram_mb} MiB) "
                    f"(requires approx. {required_vram_for_model_mb:.1f} MiB with buffer). Forcing to CPU."
                )
                n_gpu_layers_to_use = 0
            else: # Sufficient VRAM, use .env setting
                try:
                    n_gpu_layers_val_str = os.getenv(n_gpu_layers_env_var, str(default_n_gpu_layers))
                    n_gpu_layers_to_use = int(n_gpu_layers_val_str)
                    logging.info(f"[{model_name_log_prefix}] Sufficient VRAM ({available_vram_mb} MiB); "
                                 f"Using {n_gpu_layers_env_var} = {n_gpu_layers_to_use}")
                except ValueError:
                    logging.warning(f"[{model_name_log_prefix}] Invalid value for {n_gpu_layers_env_var}: "
                                    f"'{n_gpu_layers_val_str}'. Defaulting to {default_n_gpu_layers}.")
                    n_gpu_layers_to_use = default_n_gpu_layers
        else: # VRAM not passed or invalid, rely solely on .env
            logging.info(f"[{model_name_log_prefix}] Available VRAM not specified for check; "
                         f"relying on {n_gpu_layers_env_var} from .env.")
            try:
                n_gpu_layers_val_str = os.getenv(n_gpu_layers_env_var, str(default_n_gpu_layers))
                n_gpu_layers_to_use = int(n_gpu_layers_val_str)
            except ValueError:
                logging.warning(f"[{model_name_log_prefix}] Invalid value for {n_gpu_layers_env_var}: "
                                f"'{n_gpu_layers_val_str}'. Defaulting to {default_n_gpu_layers}.")
                n_gpu_layers_to_use = default_n_gpu_layers

    logging.info(f"[{model_name_log_prefix}] Determined n_gpu_layers = {n_gpu_layers_to_use}")

    try:
        n_ctx_val_str = os.getenv(n_ctx_env_var, str(default_n_ctx))
        n_ctx_to_use = int(n_ctx_val_str)
    except ValueError:
        logging.warning(f"[{model_name_log_prefix}] Invalid value for {n_ctx_env_var}: "
                        f"'{n_ctx_val_str}'. Defaulting to {default_n_ctx}.")
        n_ctx_to_use = default_n_ctx
    logging.info(f"[{model_name_log_prefix}] n_ctx = {n_ctx_to_use}")

    # Log crucial environment variables for llama.cpp GPU behavior
    logging.info(f"[{model_name_log_prefix}] For Llama.cpp - GGML_CUDA_NO_PINNED_MEMORY: {os.getenv('GGML_CUDA_NO_PINNED_MEMORY')}")
    logging.info(f"[{model_name_log_prefix}] For Llama.cpp - LD_LIBRARY_PATH (as seen by Python): {os.getenv('LD_LIBRARY_PATH')}")

    # Record VRAM before this specific model load (if using GPU)
    vram_before_this_model_load_mb = -1
    if n_gpu_layers_to_use != 0 and gpu_system_available:
        vram_before_this_model_load_mb = _get_free_gpu_memory_mb_internal()

    try:
        # verbose=True is critical for debugging llama.cpp GPU issues
        # force_verbose_llama_cpp allows overriding the default False for specific debug scenarios
        effective_verbose = bool(os.getenv("LLAMA_CPP_VERBOSE", "False").lower() == "true") or force_verbose_llama_cpp

        logging.info(f"[{model_name_log_prefix}] Instantiating Llama model (verbose={effective_verbose})...")
        loaded_model = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers_to_use,
            n_ctx=n_ctx_to_use,
            verbose=effective_verbose
        )
        logging.info(f"[{model_name_log_prefix}] Llama GGUF model instantiated.")

        if n_gpu_layers_to_use != 0 and gpu_system_available:
            vram_after_this_model_load_mb = _get_free_gpu_memory_mb_internal()
            if vram_before_this_model_load_mb != -1 and vram_after_this_model_load_mb != -1:
                delta_vram = vram_before_this_model_load_mb - vram_after_this_model_load_mb
                logging.info(f"[{model_name_log_prefix}] VRAM delta after load (pynvml): {delta_vram:.1f} MiB. "
                             f"Free now: {vram_after_this_model_load_mb} MiB.")
            # Report actual layers offloaded
            actual_layers_on_gpu = getattr(loaded_model.model_params, 'n_gpu_layers', 0)
            if actual_layers_on_gpu == 2147483647: # INT_MAX
                logging.info(f"[{model_name_log_prefix}] Llama.cpp reports ALL layers offloaded to GPU.")
            elif actual_layers_on_gpu > 0:
                logging.info(f"[{model_name_log_prefix}] Llama.cpp reports {actual_layers_on_gpu} layers offloaded to GPU.")
            else:
                logging.info(f"[{model_name_log_prefix}] Llama.cpp reports 0 layers offloaded (CPU).")
                if n_gpu_layers_to_use != 0 : # Requested GPU but got CPU
                    logging.warning(f"[{model_name_log_prefix}] GPU offload was requested (n_gpu_layers={n_gpu_layers_to_use}) but model loaded on CPU. Check llama.cpp verbose logs.")
        
        return loaded_model

    except Exception as e:
        logging.error(f"[{model_name_log_prefix}] Error loading GGUF model with Llama(): {e}", exc_info=True)
        return None


class LLMWrapper:
    """
    A wrapper to interact with LLMs via OpenRouter (dev) or a local GGUF model (prod).
    """
    def __init__(self, force_verbose_main_llm: bool = False):
        self.mode: Optional[str] = None
        self.openrouter_client: Optional[Any] = None # OpenAI client type
        self.openrouter_model_name: Optional[str] = None
        self.local_model: Optional[Llama] = None # llama_cpp.Llama type
        self.is_ready: bool = False

        # Basic logging setup if not already done by importing module
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s]: %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__) # Specific logger

        is_dev = os.getenv("DEVELOPMENT_MODE", "false").lower() in ("true", "1", "t", "y", "yes")
        if is_dev:
            self._init_openrouter()
        else:
            self.mode = "local_gguf"
            self.logger.info("Initializing in PRODUCTION mode (Local GGUF - Main LLM)...")
            self.local_model = load_gguf_model(
                model_name_log_prefix="Main LLM",
                model_path_str=os.getenv("LOCAL_MAIN_LLM_PATH"),
                n_gpu_layers_env_var="MAIN_LLM_N_GPU_LAYERS",
                n_ctx_env_var="MAIN_LLM_N_CTX",
                use_cpu_env_var="MAIN_LLM_USE_CPU",
                default_n_gpu_layers=-1,
                default_n_ctx=4096,
                # For main LLM, we don't pass available_vram_mb, let it try to use as much as it needs
                # based on its own config, assuming it's the primary model.
                force_verbose_llama_cpp=force_verbose_main_llm
            )
            if self.local_model:
                self.is_ready = True
                self.logger.info("Main LLM (LLMWrapper) initialized successfully.")
            else:
                self.logger.error("Main LLM (LLMWrapper) FAILED to initialize.")

        if not self.is_ready:
            self.logger.warning("LLMWrapper initialization failed; generate_response will not work.")

    def _init_openrouter(self):
        self.logger.info("Initializing in DEVELOPMENT mode (OpenRouter)...")
        self.mode = "openrouter"
        # Assuming openai_available is checked or handled within OpenRouter specific code
        global openai_available # Check if needed or pass as arg
        if not openai_available:
            self.logger.error("OpenAI library not found. OpenRouter mode disabled.")
            return

        from openai import OpenAI, APIError, AuthenticationError, RateLimitError # Localize import
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = os.getenv("OPENROUTER_MODEL_NAME")

        if not api_key: self.logger.error("OPENROUTER_API_KEY not set."); return
        if not model_name: self.logger.error("OPENROUTER_MODEL_NAME not set."); return

        try:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            self.openrouter_model_name = model_name
            self.is_ready = True
            self.logger.info(f"OpenRouter client ready, model: {self.openrouter_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenRouter client: {e}", exc_info=True)

    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self.is_ready:
            self.logger.error("LLM is not ready. Cannot generate response.")
            return "Error: LLM Wrapper is not initialized."

        self.logger.info(f"Generating response using mode: {self.mode}")
        try:
            if self.mode == 'openrouter':
                # ... (OpenRouter chat completion logic as before) ...
                # Make sure to import OpenAI, APIError etc. if not global
                from openai import APIError, AuthenticationError, RateLimitError # Localize import
                if not self.openrouter_client or not self.openrouter_model_name:
                     return "Error: OpenRouter client not configured correctly."
                messages = [{"role": "user", "content": prompt}]
                self.logger.debug(f"Sending to OpenRouter: {self.openrouter_model_name}")
                response = self.openrouter_client.chat.completions.create( # type: ignore
                    model=self.openrouter_model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    return content.strip() if content else "Error: Empty content from OpenRouter."
                return "Error: No choices from OpenRouter."

            elif self.mode == 'local_gguf':
                if not self.local_model:
                    self.logger.error("Local GGUF model not loaded in LLMWrapper.")
                    return "Error: Local GGUF model not loaded."
                
                self.logger.debug(f"Generating with local GGUF model...")
                # Using create_chat_completion for consistency if model supports it well
                response = self.local_model.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if response and 'choices' in response and response['choices']: # type: ignore
                    content = response['choices'][0]['message']['content'] # type: ignore
                    return content.strip() if content else "Error: Empty content from local model chat."
                self.logger.warning(f"Unexpected local GGUF chat response: {response}")
                return "Error: Unexpected output from local model chat."

        except AuthenticationError as e: # Specific to OpenAI lib
            self.logger.error(f"OpenRouter Authentication Error: {e}")
            return "Error: OpenRouter Authentication Failed."
        except RateLimitError as e: # Specific to OpenAI lib
             self.logger.error(f"OpenRouter Rate Limit Error: {e}")
             return "Error: OpenRouter rate limit exceeded."
        except APIError as e: # Specific to OpenAI lib
            self.logger.error(f"OpenRouter API Error: {e}")
            return f"Error: OpenRouter API error ({e.status_code})."
        except Exception as e:
            self.logger.error(f"Error during LLM generation ({self.mode}): {e}", exc_info=True)
            return f"Error: Exception during text generation ({type(e).__name__})."
        return "Error: Unknown state in generate_response."

    # generate_structured_response (for OpenRouter) can remain similar
    # ...
    
    # --- NEW METHOD for Structured Output ---
    def generate_structured_response(self, prompt: str, response_schema: dict = None) -> dict | None:
        """
        Generates a structured JSON response using OpenRouter API.
        Returns the parsed JSON dictionary or None on failure.
        Note: This currently only works in 'openrouter' mode.
        """
        if not self.is_ready:
            logging.error("Cannot generate structured response: LLM Wrapper not initialized.")
            return None
        if self.mode != 'openrouter':
            logging.error(f"Cannot generate structured response: Not in OpenRouter mode (current: {self.mode}).")
            return None
        if not self.openrouter_client:
             logging.error("Cannot generate structured response: OpenRouter client not configured.")
             return None

        logging.info(f"Generating STRUCTURED JSON response using OpenRouter model: {self.openrouter_model_name}")
        try:
            messages = [{"role": "user", "content": prompt}]
            # Add system prompt guiding JSON output if helpful for the model
            # messages.insert(0, {"role": "system", "content": "You are an assistant that only responds in valid JSON."})

            completion_params = {
                "model": self.openrouter_model_name,
                "messages": messages,
                "response_format": {"type": "json_object"}, # Specify JSON mode
                "temperature": 0.2, # Lower temp often better for structured tasks
                # Max tokens should be sufficient for the JSON structure
                "max_tokens": 200,
            }
            # If a specific JSON schema is provided (for more complex validation by model)
            # Note: Check OpenRouter/model compatibility for json_schema parameter
            # if response_schema:
            #    completion_params["response_format"]["json_schema"] = response_schema

            response = self.openrouter_client.chat.completions.create(**completion_params)

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    logging.debug(f"Raw JSON response content: {content}")
                    # Parse the JSON string into a Python dictionary
                    try:
                        parsed_json = json.loads(content)
                        logging.info("Successfully parsed structured JSON response.")
                        return parsed_json
                    except json.JSONDecodeError as json_e:
                        logging.error(f"Failed to parse JSON response from API: {json_e}")
                        logging.error(f"Invalid JSON received: {content}")
                        return None # Indicate parsing failure
                else:
                    logging.warning("OpenRouter response choice content is empty for structured request.")
                    return None # Indicate empty content failure
            else:
                logging.warning(f"OpenRouter returned no choices for structured request: {response}")
                return None # Indicate API failure

        # Handle specific API errors
        except AuthenticationError as e: logging.error(f"OpenRouter Auth Error: {e}"); return None
        except RateLimitError as e: logging.error(f"OpenRouter Rate Limit Error: {e}"); return None
        except APIError as e: logging.error(f"OpenRouter API Error: {e}"); return None
        except Exception as e:
            logging.error(f"Error during structured generation: {e}", exc_info=True)
            return None # Indicate general failure

    # Helper methods within LLMWrapper for its own pynvml use
    def _is_gpu_available_pynvml_internal(self) -> bool:
        if not pynvml_available_interface: return False
        try:
            # nvmlInit is at module level
            return pynvml.nvmlDeviceGetCount() > 0
        except pynvml.NVMLError: return False

    def _get_free_gpu_memory_mb_internal(self, device_index: int = 0) -> int:
        if not pynvml_available_interface or not self._is_gpu_available_pynvml_internal(): return -1
        try:
            # nvmlInit is at module level
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free // (1024**2)
        except pynvml.NVMLError: return -1