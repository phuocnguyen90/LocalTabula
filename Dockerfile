FROM public.ecr.aws/lambda/python:3.12
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y build-essential \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && apt-get clean

RUN pip install uv
RUN uv init .
RUN export CC=/usr/bin/gcc CXX=/usr/bin/g++
RUN export LD_LIBRARY_PATH=/usr/lib/gcc/$(gcc -dumpmachine)/$(gcc -dumpversion):$LD_LIBRARY_PATH
RUN CMAKE_ARGS="-DGGML_CUDA=on \
            -DCMAKE_CUDA_ARCHITECTURES=75 \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TESTS=OFF" FORCE_CMAKE=1 \
uv pip install --system --upgrade --force-reinstall llama-cpp-python==0.3.8 \
--index-url https://pypi.org/simple \
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 \
--index-strategy unsafe-best-match