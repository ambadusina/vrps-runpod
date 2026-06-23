# syntax=docker/dockerfile:1.7
#
# VRPS-RunPod — offline 2D->3D/VR + passthrough matting on RunPod GPU pods.
#
# Base: CUDA 12.8 runtime + cuDNN. 12.8 (not 12.6) is REQUIRED because CuPy must
# emit native sm_120 cubins via NVRTC >= 12.8 for Blackwell GPUs (RTX PRO 4500 /
# RTX 50xx). On 12.6 CuPy falls back to the slow PTX->driver-JIT path or fails
# with CUDA_ERROR_NO_BINARY_FOR_GPU.
#
# Host requirement on RunPod: NVIDIA driver >= 570 (CUDA 12.8 userland).
# RunPod pod filter: select a CUDA 12.8 (or "12.8+") template / GPU.
#
# Models are HYBRID:
#   - Baked into the image at build time: DA3 base + base_hd, NVDS, RVM  (~1 GB)
#   - Downloaded on first use at runtime:  MatAnyone2, SAM3, DA3 large_hd (cached
#     on the pod's container disk under /workspace/models)
#
# AGPL-3.0: this image bundles AGPL source (upstream VRPS). If you expose it over
# a network to users, you must publish the complete corresponding source,
# including this Dockerfile, start.sh, and any modifications. Keep the GitHub
# repo public and referenced from the running service.

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8:replace \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# ---- System deps: python3.12 (default on ubuntu24.04), ffmpeg w/ NVENC, git ----
# ffmpeg from the Ubuntu repo is built with NVENC/NVDEC support enabled and uses
# the driver's libnvidia-encode at runtime (provided by the RunPod host), so no
# CUDA SDK is needed at build time for encoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python venv (avoids clobbering the distro python) ----
ENV VENV=/opt/venv
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"
RUN python -m pip install --upgrade pip wheel

WORKDIR /app

# ---- Python deps ----
# We DO NOT use the upstream pyproject wholesale: it pulls PySide6, pyinstaller,
# osam, pymp4, av etc. that the headless offline pipeline does not need. The
# trimmed requirements below cover the 2D->3D/VR + matting offline paths.
#
# Pinned for sm_120 (Blackwell):
#   - cupy-cuda12x[ctk]  : ships pip CUDA 12.x toolkit (nvrtc 12.9 emits sm_120)
#   - onnxruntime-gpu    : CUDA + TensorRT execution providers
#   - tensorrt-cu12      : TRT engine build/cache
#   - nvidia-cudnn-cu12  : ORT CUDA EP needs cuDNN 9 in the venv
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
# osam separately, WITHOUT deps, so it cannot drag in plain onnxruntime and
# shadow the GPU build. Only its CLIP tokenizer (numpy/pillow) is used offline.
RUN pip install --no-deps osam==0.4.0

# ---- Application source (full upstream tree; AGPL) ----
# Copy the whole repo: the offline paths use lazy cross-module imports, so
# cherry-picking files risks breaking MatAnyone2/SAM3 at runtime. It's only
# ~22 MB of Python.
COPY src/ /app/

# ---- Runtime layout & env ----
# config.py uses PT_* env vars, but config.ROOT itself is hardcoded to the
# source dir (/app) and is NOT env-overridable. start.sh therefore symlinks
# /app/models and /app/runtime_cache onto the pod's writable /workspace disk so
# that (a) runtime-downloaded heavy models and (b) GPU-specific TRT engines
# persist on the container disk rather than bloating the image or vanishing.
ENV PT_ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider \
    PT_ONNX_TRT_ENGINE_CACHE_ENABLE=1 \
    HF_HOME=/workspace/hf

# Expose pip-installed NVIDIA libs to the dynamic linker so onnxruntime's
# TensorRT EP can load libnvinfer.so.10 (and cuDNN/cuBLAS). These wheels drop
# their .so files in per-package site-packages dirs that are NOT on the default
# linker search path, which otherwise forces a silent fallback to the CUDA EP.
# Must be set before build-models.sh (that step imports onnxruntime).
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/tensorrt_libs:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}

# ---- Bake the lightweight models into the image (hybrid: heavy ones at runtime) ----
# build-models.sh downloads DA3 base + base_hd, NVDS, and RVM into /app/models.
# Heavy models (MatAnyone2, SAM3, large_hd) are intentionally skipped here and
# fetched on first use by start.sh / the job runner.
COPY build-models.sh /app/build-models.sh
RUN bash /app/build-models.sh --baked-only

# ---- Entry ----
COPY start.sh /app/start.sh
COPY job_runner.py /app/job_runner.py
RUN chmod +x /app/start.sh

# Pod working area (RunPod mounts /workspace as the container disk)
VOLUME ["/workspace"]
WORKDIR /workspace

ENTRYPOINT ["/app/start.sh"]
