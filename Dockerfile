# CosmicML-Biodetect Production Docker Image
#
# Usage:
#   docker build -t cosmicml-biodetect:latest .
#   docker run -v $(pwd)/models:/app/models cosmicml-biodetect:latest
#
# References:
#   PyTorch official images: pytorch/pytorch:latest
#   Python 3.10+

FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

# Create directories for models and data
RUN mkdir -p /app/models/checkpoints && \
    mkdir -p /app/data/simulated

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default command: Run training
CMD ["python", "scripts/train_pinn_v3.py", "--epochs", "100", "--batch-size", "32"]

# Labels
LABEL maintainer="CosmicML-Biodetect"
LABEL version="1.0"
LABEL description="Physics-informed deep learning for exoplanet biosignature detection"
