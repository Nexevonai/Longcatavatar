# Dockerfile for LongCat Avatar on RunPod Serverless
# Base image: RunPod PyTorch with CUDA 12.4

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements_avatar.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_avatar.txt

# Install RunPod SDK and R2 client dependencies
RUN pip install --no-cache-dir runpod boto3

# Copy application code
COPY longcat_video /app/longcat_video
COPY handler.py /app/handler.py
COPY r2_client.py /app/r2_client.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Entry point - runs the serverless handler
CMD ["python", "-u", "handler.py"]
