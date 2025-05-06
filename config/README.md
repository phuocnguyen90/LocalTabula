# GPU Setup Note

> **Note to Future Self:**
>
> When using a GPU, you need to rebuild **llama-cpp-python** with CUDA support.
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
