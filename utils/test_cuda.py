from llama_cpp import Llama
import llama_cpp.llama_cpp as llama_cpp_internal

import os
LOCAL_MAIN_LLM_PATH='/home/phuoc/project/AI_agent_gemma3/models/main_llm/google_gemma-3-4b-it-Q5_K_M.gguf'


try:
     print("\nAttempting to load model with 1 GPU layer...")
     llm = Llama(model_path=LOCAL_MAIN_LLM_PATH, n_gpu_layers=-1, verbose=True) # verbose=True is helpful
     print("Model loaded successfully with n_gpu_layers=1.")
     print(f"Actual n_gpu_layers used: {llm.model_params.n_gpu_layers}")
     if llm.model_params.n_gpu_layers > 0:
          print("✅ GPU offloading successful!")
     else:
          print("⚠️ Model loaded, but no layers were offloaded to GPU despite requesting. Check llama.cpp verbose output.")
except Exception as e:
     print(f"Error loading model with GPU layers: {e}")