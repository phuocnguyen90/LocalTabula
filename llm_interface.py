# llm_interface.py
import os
import logging
from dotenv import load_dotenv
import json

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

# --- Load Environment Variables ---
load_dotenv('.env')

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMWrapper:
    """
    A wrapper to interact with LLMs either via OpenRouter API (Dev Mode)
    or a local GGUF model (Production/Local Mode).
    """
    def __init__(self):
        self.mode = None # 'openrouter' or 'local_gguf'
        self.openrouter_client = None
        self.openrouter_model_name = None
        self.local_model = None
        self.is_ready = False

        # Determine mode from environment variable
        dev_mode_str = os.getenv('DEVELOPMENT_MODE', 'false').lower()
        is_dev_mode = dev_mode_str in ('true', '1', 't', 'yes', 'y')

        if is_dev_mode:
            logging.info("Attempting to initialize in DEVELOPMENT mode (OpenRouter API)...")
            self.mode = 'openrouter'
            if not openai_available:
                logging.error("DEVELOPMENT_MODE is true, but openai library is not installed.")
                return # Cannot initialize

            api_key = os.getenv('OPENROUTER_API_KEY')
            model_name = os.getenv('OPENROUTER_MODEL_NAME')

            if not api_key:
                logging.error("DEVELOPMENT_MODE is true, but OPENROUTER_API_KEY env var is not set.")
                return
            if not model_name:
                logging.error("DEVELOPMENT_MODE is true, but OPENROUTER_MODEL_NAME env var is not set.")
                return

            try:
                # Configure the OpenAI client to point to OpenRouter
                self.openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                self.openrouter_model_name = model_name
                # Optional: Add a simple check like listing models? Be careful with rate limits.
                # Or just assume it's ready if config is present.
                self.is_ready = True
                logging.info(f"OpenRouter client initialized successfully for model: {self.openrouter_model_name}")

            except Exception as e:
                logging.error(f"Failed to initialize OpenRouter client: {e}", exc_info=True)
                # self.is_ready remains False

        else: # Production / Local GGUF Mode
            logging.info("Attempting to initialize in PRODUCTION mode (Local GGUF)...")
            self.mode = 'local_gguf'
            if not llama_cpp_available:
                logging.error("DEVELOPMENT_MODE is false, but llama_cpp is not installed.")
                return

            model_path = os.getenv('GGUF_MODEL_PATH')
            if not model_path or not os.path.exists(model_path):
                logging.error(f"GGUF_MODEL_PATH env var not set or path invalid: '{model_path}'")
                return

            try:
                n_gpu_layers_str = os.getenv('GGUF_N_GPU_LAYERS', '-1')
                n_gpu_layers = int(n_gpu_layers_str) if n_gpu_layers_str.lstrip('-').isdigit() else -1
                n_ctx_str = os.getenv('GGUF_N_CTX', '2048')
                n_ctx = int(n_ctx_str) if n_ctx_str.isdigit() else 2048

                logging.info(f"Loading GGUF model from: {model_path}")
                logging.info(f"GGUF Params: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")

                self.local_model = Llama(
                    model_path=model_path, n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx, verbose=True
                )
                self.is_ready = True
                logging.info("Local GGUF model loaded successfully.")

            except Exception as e:
                logging.error(f"Failed to load GGUF model from '{model_path}': {e}", exc_info=True)

        if not self.is_ready:
             logging.warning("LLMWrapper initialization failed. generate_response will not work.")


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

# --- Optional: Test ---
if __name__ == "__main__":
     print("Testing LLMWrapper...")
     wrapper = LLMWrapper()
     if wrapper.is_ready:
         print(f"Wrapper ready in mode: {wrapper.mode}")
         test_prompt = "Who are you and what model are you based on?"
         print(f"\nTest Prompt: {test_prompt}")
         response = wrapper.generate_response(test_prompt, max_tokens=100)
         print(f"\nTest Response:\n{response}")
     else:
         print("\nWrapper failed to initialize. Check env vars and logs.")