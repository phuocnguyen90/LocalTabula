# llm_interface.py
import os
import logging
from dotenv import load_dotenv
import json
import torch
from pathlib import Path

# Conditional Imports
try:
    # Using the openai library to interact with OpenRouter's compatible API
    from openai import OpenAI, APIError, AuthenticationError, RateLimitError
    openai_available = True
except ImportError:
    openai_available = False
    logging.warning("openai library not found. OpenRouter API mode disabled.")

try:
    from llama_cpp import Llama
    llama_cpp_available = True
except ImportError:
    llama_cpp_available = False
    logging.warning("llama_cpp library not found. Local GGUF mode disabled.")

try:
    from huggingface_hub import hf_hub_download # <--- ADD
    hf_hub_available = True
except ImportError:
    hf_hub_available = False
    logging.warning("huggingface_hub library not found. Automatic GGUF download disabled for main LLM.")

# Import ENV_PATH from .utils (sibling module in the same package)
from .utils import ENV_PATH # Use relative import for sibling modules

# --- Load Environment Variables ---
# Load from the central .env file specified by utils.py
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)
else:
    logging.warning(f".env file not found at {ENV_PATH}. LLMWrapper might not function correctly.")

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMWrapper:
    """
    A wrapper to interact with LLMs via OpenRouter (dev) or a local GGUF model (prod).
    In prod, will load the GGUF into GPU memory if available.
    """
    def __init__(self):
        self.mode = None           # 'openrouter' or 'local_gguf'
        self.openrouter_client = None
        self.openrouter_model_name = None
        self.local_model = None
        self.is_ready = False

        # 1) Dev vs Prod
        is_dev = os.getenv("DEVELOPMENT_MODE", "false").lower() in ("true","1","t","y","yes")
        if is_dev:
            self._init_openrouter()
        else:
            self._init_local_gguf()

        if not self.is_ready:
            logging.warning("LLMWrapper initialization failed; generate_response will not work.")

    def _init_openrouter(self):
        logging.info("Initializing in DEVELOPMENT mode (OpenRouter)...")
        self.mode = "openrouter"
        # … your existing OpenRouter logic …
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = os.getenv("OPENROUTER_MODEL_NAME")
        if not api_key or not model_name:
            logging.error("Missing OpenRouter credentials/model name.")
            return
        try:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            self.openrouter_model_name = model_name
            self.is_ready = True
            logging.info(f"OpenRouter ready: {model_name}")
        except Exception as e:
            logging.error(f"OpenRouter init error: {e}", exc_info=True)

    def _init_local_gguf(self):
        logging.info("Initializing in PRODUCTION mode (Local GGUF - Main LLM)...")
        self.mode = "local_gguf"

        # Read the local path set by download_models.py
        model_path_str = os.getenv("LOCAL_MAIN_LLM_PATH")

        if not model_path_str:
            logging.error("LOCAL_MAIN_LLM_PATH not set in .env. Run scripts/download_models.py first.")
            return
        
        model_path = Path(model_path_str)
        if not model_path.exists():
            logging.error(f"Main LLM GGUF file not found at specified path: {model_path}. "
                          "Ensure it was downloaded correctly by scripts/download_models.py.")
            return

        vram_before = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        logging.info(f"VRAM before main LLM load: {vram_before/1024**2:.1f} MiB")

        use_cpu_str = os.getenv("MAIN_LLM_USE_CPU", "false").lower()
        use_cpu = use_cpu_str in ("true", "1", "t", "y", "yes")

        if not torch.cuda.is_available():
            logging.info("No GPU detected; loading Main LLM on CPU.")
            use_cpu = True # Force CPU if no CUDA
        elif use_cpu:
            logging.info("MAIN_LLM_USE_CPU=true; loading Main LLM on CPU.")
        
        try:
            n_gpu_layers_str = os.getenv("MAIN_LLM_N_GPU_LAYERS", "-1")
            n_gpu_layers = int(n_gpu_layers_str)
        except ValueError:
            logging.warning(f"Invalid value for MAIN_LLM_N_GPU_LAYERS: '{n_gpu_layers_str}'. Defaulting to -1.")
            n_gpu_layers = -1

        if use_cpu:
            n_gpu_layers = 0
        
        logging.info(f"Main LLM GGUF n_gpu_layers={n_gpu_layers}")

        try:
            n_ctx_str = os.getenv("MAIN_LLM_N_CTX", "4096")
            n_ctx = int(n_ctx_str)
        except ValueError:
            logging.warning(f"Invalid value for MAIN_LLM_N_CTX: '{n_ctx_str}'. Defaulting to 4096.")
            n_ctx = 4096
        logging.info(f"Main LLM GGUF n_ctx={n_ctx}")

        try:
            logging.info(f"Loading Main LLM GGUF model from: {model_path}")
            self.local_model = Llama(
                model_path=str(model_path), # Llama cpp expects string
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False
            )
            
            self.is_ready = True
            logging.info("Main LLM (local GGUF) loaded successfully.")
            vram_after = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
            delta = (vram_after - vram_before) / (1024**2)
            logging.info(f"VRAM after main LLM load: {vram_after/1024**2:.1f} MiB  (Δ {delta:.1f} MiB)")
        except Exception as e:
            logging.error(f"Failed to load Main LLM GGUF: {e}", exc_info=True)


    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generates a text response using the configured LLM (OpenRouter or Local GGUF).
        """
        if not self.is_ready:
            return "Error: LLM Wrapper is not initialized."

        logging.info(f"Generating response using mode: {self.mode}")
        try:
            if self.mode == 'openrouter':
                if not self.openrouter_client or not self.openrouter_model_name:
                     return "Error: OpenRouter client not configured correctly."

                # Construct messages payload for Chat Completions API
                messages = [{"role": "user", "content": prompt}]
                # Optional: Add system prompt if needed
                # messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

                logging.info(f"Sending request to OpenRouter model: {self.openrouter_model_name}")
                response = self.openrouter_client.chat.completions.create(
                    model=self.openrouter_model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    # Add other parameters like top_p, stream=False etc. if needed
                )

                # Extract response text
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content:
                         return content.strip()
                    else:
                         logging.warning("OpenRouter response choice content is empty.")
                         # Check finish reason
                         finish_reason = response.choices[0].finish_reason
                         if finish_reason == 'length': return "Error: Response truncated by max_tokens limit."
                         if finish_reason == 'content_filter': return "Error: Response blocked by OpenRouter content filter."
                         return "Error: Received empty content from API."
                else:
                    logging.warning(f"OpenRouter returned no choices in response: {response}")
                    return "Error: Received no response choices from API."

            elif self.mode == 'local_gguf':
                if not self.local_model:
                    return "Error: Local GGUF model not loaded."

                logging.info(f"Generating with local GGUF model...")
                try:
                    response = self.local_model.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        # stop=["<|end_of_turn|>", "<|user|>"] # Adjust stop tokens for Gemma 3 if needed
                    )
                    if response and 'choices' in response and len(response['choices']) > 0:
                        content = response['choices'][0]['message']['content']
                        return content.strip() if content else "Error: Received empty content from local model."
                    else:
                         logging.warning(f"Local GGUF model chat completion returned unexpected structure: {response}")
                         return "Error: Unexpected output from local model chat completion."

                except Exception as e:
                    # Fallback to simple invocation if create_chat_completion fails (e.g., older llama-cpp)
                    # Or if templating isn't working as expected
                    logging.warning(f"create_chat_completion failed ({e}), trying direct invocation...")
                    try:
                        output = self.local_model(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            # stop=["<|end_of_turn|>", "<|user|>"] # Stop tokens might be needed here too
                        )
                        if output and 'choices' in output and len(output['choices']) > 0:
                            return output['choices'][0]['text'].strip()
                        else:
                            logging.warning(f"Local GGUF model direct invocation returned unexpected output: {output}")
                            return "Error: Received unexpected output from local model direct invocation."
                    except Exception as e2:
                        logging.error(f"Both chat completion and direct invocation failed for local GGUF: {e2}", exc_info=True)
                        return f"Error: Failed to generate response from local model ({type(e2).__name__})."



        # Handle specific API errors if using openai library
        except AuthenticationError as e:
            logging.error(f"OpenRouter Authentication Error: {e}")
            return "Error: OpenRouter Authentication Failed. Check your API key."
        except RateLimitError as e:
             logging.error(f"OpenRouter Rate Limit Error: {e}")
             return "Error: OpenRouter rate limit exceeded. Please wait and try again."
        except APIError as e: # Catch other API related errors
            logging.error(f"OpenRouter API Error: {e}")
            return f"Error: OpenRouter API error ({e.status_code})."
        except Exception as e:
            logging.error(f"Error during LLM generation ({self.mode}): {e}", exc_info=True)
            return f"Error: An exception occurred during text generation ({type(e).__name__})."

        return "Error: Unknown state in generate_response."
    
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
