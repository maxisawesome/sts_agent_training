# STS Neural Agent - Docker Image for Cloud Training
# Supports both CPU and GPU training with CUDA

# Use NVIDIA's PyTorch image with CUDA support
FROM pytorch/pytorch:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WANDB_CACHE_DIR=/app/wandb_cache

# Install system dependencies (rarely changes - good for caching)
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

# Copy and install Python requirements first (changes infrequently)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only sts_lightspeed C++ source code and build files (stable structure)
COPY sts_lightspeed/include/ sts_lightspeed/include/
COPY sts_lightspeed/src/ sts_lightspeed/src/
COPY sts_lightspeed/apps/ sts_lightspeed/apps/
COPY sts_lightspeed/bindings/ sts_lightspeed/bindings/
COPY sts_lightspeed/json/ sts_lightspeed/json/
COPY sts_lightspeed/pybind11/ sts_lightspeed/pybind11/
COPY sts_lightspeed/CMakeLists.txt sts_lightspeed/CMakeLists.txt

# Build sts_lightspeed (this layer will be cached unless C++ code changes)
RUN cd sts_lightspeed && \
    cmake -DPYTHON_EXECUTABLE=$(which python) . && \
    make -j$(nproc) && \
    echo "✅ sts_lightspeed built successfully"

# Copy setup.py and install package structure (changes less frequently than Python scripts)
COPY setup.py .
RUN pip install -e . || echo "Package install attempted"

# Copy Python source code (changes most frequently - put at end)
COPY sts_*.py ./
COPY train_sts_agent.py ./
COPY setup_wandb.py ./
COPY test*.py ./
COPY analyze_rewards.py ./
COPY simple_reward_analysis.py ./

# Test that everything works
RUN python -c "import sys; sys.path.insert(0, 'sts_lightspeed'); import slaythespire; print('✅ sts_lightspeed import test successful')"

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
EXPOSE 1234

# Default command - can be overridden
CMD ["python", "train_sts_agent.py", "train", "--episodes", "1000", "--wandb"]