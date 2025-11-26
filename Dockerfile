# Multi-stage Dockerfile for VieNeu-TTS
# Supports both CPU and GPU execution

ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=11.8
ARG UBUNTU_VERSION=22.04

# Base stage with common dependencies
FROM python:${PYTHON_VERSION}-slim-bookworm as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng1 \
    libsndfile1 \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Install uv for faster dependency management
RUN pip install uv

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/models /app/config /app/output_audio /app/cache

# Set permissions
RUN chmod -R 755 /app

# Expose port for Gradio app
EXPOSE 7860

# CPU-only stage
FROM base as cpu

ENV DEVICE=cpu
CMD ["python", "gradio_app.py"]

# GPU stage with CUDA support
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as gpu-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEVICE=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng1 \
    libsndfile1 \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install uv for faster dependency management
RUN pip install uv

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/models /app/config /app/output_audio /app/cache

# Set permissions
RUN chmod -R 755 /app

# Expose port for Gradio app
EXPOSE 7860

# GPU stage
FROM gpu-base as gpu

ENV DEVICE=cuda
CMD ["python", "gradio_app.py"]

# Default stage (CPU)
FROM cpu as default
