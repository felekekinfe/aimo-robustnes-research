# syntax=docker/dockerfile:1
# AIMO Interpretability Challenge (NeurIPS 2026) - Production Codabench Container
# Built on PyTorch 2.12.1 + CUDA 13.2 + cuDNN 9 (NVIDIA Blackwell sm_120 & Hopper sm_90 ready)

FROM pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime@sha256:f48553a21cd5944e8093a74801e403283311b460a5b602ec02d15fe98f04ff5f

LABEL maintainer="AIMO Interpretability Challenge Participant"
LABEL description="Unified 3-Tier Calibrated Robustness Solution (SCSP + FLA Triton + Meta-Ensemble)"

# Set working directory
WORKDIR /app

# Ensure offline operation per Codabench evaluation requirements
ENV HF_HUB_OFFLINE=1 \
    HF_HOME=/app/data/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy dependency specifications
COPY requirements.txt /app/requirements.txt

# Install pinned packages into base container
RUN python -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt \
    && python -c "import torch, transformers, sklearn, lightgbm; print('Container runtime verified:', torch.__version__, transformers.__version__, sklearn.__version__)"

# Copy solution source files and artifacts
COPY solution.py /app/solution.py
COPY scsp_core.py /app/scsp_core.py
COPY triton_kernels.py /app/triton_kernels.py
COPY perceptors.py /app/perceptors.py
COPY components /app/components
COPY scripts /app/scripts

# Default entrypoint for Codabench ingestion or standalone execution
ENTRYPOINT ["python", "solution.py"]
