# Docker Troubleshooting Guide

## Common Issues and Solutions

### 1. Docker Build Takes Too Long / Times Out

**Problem**: `./docker-scripts/test-docker.sh` fails because Docker build times out.

**Cause**: The PyTorch Docker image is very large (3.6GB+) and takes 10-15 minutes to download and build on first run.

**Solutions**:

#### Quick Fix - Use Local Testing
```bash
# Test the training script locally without Docker first
python train_sts_agent.py train --episodes 1 --no-wandb

# If that works, the issue is just Docker build time
```

#### Extended Build Time
```bash
# Allow more time for Docker build (30 minutes)
timeout 1800 docker build -t sts-neural-agent .

# Or build without timeout (be patient!)
docker build -t sts-neural-agent .
```

#### Use Pre-built Image (Future)
```bash
# Pull from Docker Hub instead of building locally
docker pull your-username/sts-neural-agent:latest
docker tag your-username/sts-neural-agent:latest sts-neural-agent
```

### 2. Base Image Not Found

**Problem**: `pytorch/pytorch:2.1.0-cuda11.8-devel-ubuntu20.04: not found`

**Solution**: The Dockerfile has been updated to use `pytorch/pytorch:latest` which is always available.

### 3. Out of Disk Space

**Problem**: Docker build fails with "no space left on device"

**Solutions**:
```bash
# Clean up Docker cache
docker system prune -f

# Remove unused images
docker image prune -a

# Check disk space
df -h
```

### 4. Memory Issues During Build

**Problem**: Build fails during PyTorch installation or sts_lightspeed compilation

**Solutions**:
```bash
# Limit Docker memory usage
docker build --memory=4g -t sts-neural-agent .

# Standard build (includes GPU support)
docker build -t sts-neural-agent .
```

### 5. GPU Support Not Working

**Problem**: GPU not detected in Docker container

**Solutions**:
```bash
# Install NVIDIA Docker support
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## Debugging Steps

### 1. Run Debug Script
```bash
./docker-scripts/debug-docker.sh
```

### 2. Check System Requirements
- **Docker**: Version 20.10+
- **Disk Space**: At least 10GB free
- **Memory**: At least 4GB available
- **Network**: Stable internet for downloads

### 3. Test Minimal Build
```bash
# Create minimal test image
cat > Dockerfile.minimal << 'EOF'
FROM python:3.8-slim
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
CMD python -c "import torch; print('✅ PyTorch works')"
EOF

docker build -f Dockerfile.minimal -t test-minimal .
docker run --rm test-minimal
```

### 4. Check Build Logs
```bash
# Build with detailed output
docker build --progress=plain -t sts-neural-agent .

# Check specific build stage
docker build --target=build-stage -t sts-neural-agent .
```

## Performance Tips

### 1. Use Docker Layer Caching
```bash
# Requirements change less frequently than code
# Dockerfile already optimized for this
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .  # This comes after pip install
```

### 2. Multi-stage Builds (Future Enhancement)
```dockerfile
FROM pytorch/pytorch:latest as builder
# Build dependencies
RUN compile-everything

FROM pytorch/pytorch:latest as runtime
COPY --from=builder /compiled /app
# Much smaller final image
```

### 3. Use .dockerignore
```bash
# Already configured to exclude:
.git
*.log
__pycache__
*.pyc
wandb/
sts_models/
```

## Alternative Approaches

### 1. Use Local Virtual Environment
```bash
# Instead of Docker, use conda/venv
conda create -n sts python=3.8
conda activate sts
pip install -r requirements.txt
python train_sts_agent.py train --episodes 10
```

### 2. Use Development Containers
```bash
# For VS Code users
# Use .devcontainer/devcontainer.json
# Provides consistent development environment
```

### 3. Cloud Development
```bash
# Use GitHub Codespaces or similar
# Pre-configured cloud development environment
# Faster than local Docker builds
```

## Docker Best Practices

1. **Build Time**: First build takes 10-15 minutes. Subsequent builds are faster due to layer caching.

2. **Resource Usage**: Monitor with `docker stats` during training.

3. **Data Persistence**: Always use volume mounts for models and logs:
   ```bash
   -v $(pwd)/sts_models:/app/sts_models
   -v $(pwd)/logs:/app/logs
   ```

4. **Security**: Container runs as non-root user 'trainer'.

5. **Cleanup**: Regular cleanup prevents disk space issues:
   ```bash
   docker system prune -f  # Weekly
   ```

---

**Next Steps**: If you're still having issues, the training works perfectly without Docker. You can proceed with local development and use Docker later for cloud deployment.