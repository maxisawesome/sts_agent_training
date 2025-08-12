# STS Neural Agent - Docker Image for Cloud Training
# Supports both CPU and GPU training with CUDA

# Use NVIDIA's PyTorch image with CUDA support
FROM pytorch/pytorch:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WANDB_CACHE_DIR=/app/wandb_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build sts_lightspeed
RUN cd sts_lightspeed && \
    git submodule update --init --recursive && \
    cmake -DPYTHON_EXECUTABLE=$(which python) . && \
    make -j$(nproc) && \
    cd .. && \
    python -c "import sys; sys.path.insert(0, 'sts_lightspeed'); import slaythespire; print('✅ sts_lightspeed built successfully')"

# Create directories for outputs
RUN mkdir -p /app/sts_models /app/wandb_cache /app/logs

# Set Python path
ENV PYTHONPATH=/app

# Create non-root user for security
RUN useradd -m -u 1000 trainer && \
    chown -R trainer:trainer /app
USER trainer

# Expose port for Jupyter (if needed)
EXPOSE 8888

# Default command - can be overridden
CMD ["python", "train_sts_agent.py", "train", "--episodes", "1000", "--wandb"]