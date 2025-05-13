# WSL GPU Setup Note

> **Note to Future Self:**
>
> For WSL When using a GPU, you need to rebuild **llama-cpp-python** with CUDA support.
> Replace `cuda-xx.x` with the version you have installed on your system.

1. **Set CMake arguments** (adjust paths to your CUDA installation):

   ```bash
   export CMAKE_ARGS="-DGGML_CUDA=on \
   -DCUDA_PATH=/usr/local/cuda-12.5 \
   -DCUDAToolkit_ROOT=/usr/local/cuda-12.5 \
   -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include \
   -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.5/lib64"
   ```

2. **Point to the NVCC compiler**:

   ```bash
   export CUDACXX=/usr/local/cuda-12.5/bin/nvcc
   ```

3. **Reinstall llama-cpp-python from source** (force-reinstall to pick up the new build flags):

   ```bash
   pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
   ```

# Running on Google colab T4

```bash
!CMAKE_ARGS="-DGGML_CUDA=on"
!pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp311-cp311-linux_x86_64.whl
```

**Warning:** This works and uses the GPU but doesnt suppport newer architectures like Gemma 3
https://github.com/abetlen/llama-cpp-python/issues/1780 

Change MAIN_LLM_REPO_ID to 
MAIN_LLM_FILENAME="google_gemma-3-4b-it-Q5_K_M.gguf" 