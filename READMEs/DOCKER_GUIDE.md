# Docker Deployment Guide for STS Neural Agent

Complete guide for deploying and running STS neural network training on cloud instances using Docker.

## 🚀 Quick Start

### For New Cloud Instances

1. **Run the deployment script** (as root on fresh instances):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/your-username/sts-neural-agent/main/docker-scripts/deploy.sh | bash
   ```

2. **Set your Wandb API key**:
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

3. **Start training**:
   ```bash
   cd /home/trainer/sts-neural-agent
   ./docker-scripts/train.sh --episodes 2000 --wandb-name my-experiment
   ```

### For Existing Setups

```bash
# Clone repository
git clone https://github.com/your-username/sts-neural-agent.git
cd sts-neural-agent

# Build Docker image
docker build -t sts-neural-agent .

# Start training
./docker-scripts/train.sh --episodes 1000
```

## 🐳 Docker Setup

### File Overview

- **`Dockerfile`** - Main container definition with CUDA support
- **`docker-compose.yml`** - Multi-service orchestration
- **`requirements.txt`** - Python dependencies
- **`.dockerignore`** - Optimize build context
- **`docker-scripts/`** - Deployment and training scripts

### Building the Image

```bash
# Build with default settings
docker build -t sts-neural-agent .

# Build with specific tag
docker build -t sts-neural-agent:v1.0 .

# Build with build args (if needed)
docker build --build-arg CUDA_VERSION=11.8 -t sts-neural-agent .
```

## 🎯 Training Options

### Using the Training Script

```bash
# Basic training
./docker-scripts/train.sh

# Custom configuration
./docker-scripts/train.sh \
    --episodes 5000 \
    --reward-function comprehensive \
    --wandb-name gpu-large-scale \
    --wandb-tags gpu comprehensive production

# CPU-only training
./docker-scripts/train.sh --cpu --episodes 1000

# Background training
./docker-scripts/train.sh --detached --episodes 10000

# Help
./docker-scripts/train.sh --help
```

### Using Docker Compose

```bash
# GPU training (default)
docker-compose up sts-trainer

# CPU training
docker-compose --profile cpu up sts-trainer-cpu

# Jupyter development environment
docker-compose --profile jupyter up sts-jupyter

# Background training
docker-compose up -d sts-trainer
```

### Direct Docker Commands

```bash
# GPU training
docker run --gpus all --rm -it \
    -v $(pwd)/sts_models:/app/sts_models \
    -v $(pwd)/logs:/app/logs \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    sts-neural-agent \
    python train_sts_agent.py train --episodes 2000

# CPU training
docker run --rm -it \
    -v $(pwd)/sts_models:/app/sts_models \
    -v $(pwd)/logs:/app/logs \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    sts-neural-agent \
    python train_sts_agent.py train --episodes 1000 --wandb-name cpu-training

# Interactive shell
docker run --gpus all --rm -it \
    -v $(pwd)/sts_models:/app/sts_models \
    sts-neural-agent bash
```

## ☁️ Cloud Platform Setup

### AWS EC2

1. **Launch Instance**:
   - Use Ubuntu 20.04 LTS AMI
   - Instance type: `g4dn.xlarge` (GPU) or `c5.2xlarge` (CPU)
   - Storage: 50GB+ SSD

2. **Setup**:
   ```bash
   # Connect to instance
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Run deployment script
   sudo bash <(curl -fsSL https://raw.githubusercontent.com/your-username/sts-neural-agent/main/docker-scripts/deploy.sh)
   
   # Switch to trainer user
   sudo su - trainer
   cd sts-neural-agent
   ```

3. **Training**:
   ```bash
   export WANDB_API_KEY=your_key
   ./docker-scripts/train.sh --episodes 5000 --detached
   ```

### Google Cloud Platform

1. **Create VM**:
   ```bash
   gcloud compute instances create sts-trainer \
       --zone=us-central1-a \
       --machine-type=n1-standard-4 \
       --accelerator=type=nvidia-tesla-t4,count=1 \
       --image-family=ubuntu-2004-lts \
       --image-project=ubuntu-os-cloud \
       --boot-disk-size=50GB \
       --maintenance-policy=TERMINATE
   ```

2. **Setup and Run**:
   ```bash
   gcloud compute ssh sts-trainer
   sudo bash <(curl -fsSL https://raw.githubusercontent.com/your-username/sts-neural-agent/main/docker-scripts/deploy.sh)
   ```

### Azure

1. **Create VM**:
   ```bash
   az vm create \
       --resource-group myResourceGroup \
       --name sts-trainer \
       --image UbuntuLTS \
       --size Standard_NC6 \
       --admin-username azureuser \
       --generate-ssh-keys
   ```

2. **Setup**:
   ```bash
   ssh azureuser@your-vm-ip
   sudo bash <(curl -fsSL https://raw.githubusercontent.com/your-username/sts-neural-agent/main/docker-scripts/deploy.sh)
   ```

### Vast.ai / RunPod (Budget GPU Options)

1. **Select Instance** with PyTorch template
2. **Upload Code**:
   ```bash
   git clone https://github.com/your-username/sts-neural-agent.git
   cd sts-neural-agent
   ```
3. **Train**:
   ```bash
   docker build -t sts-neural-agent .
   ./docker-scripts/train.sh --episodes 2000
   ```

## 📊 Monitoring and Management

### Container Management

```bash
# List running containers
docker ps

# View logs
docker logs -f container_id

# Stop training
docker stop container_id

# Monitor GPU usage
nvidia-smi

# Monitor resources
docker stats
```

### Persistent Data

```bash
# Check model outputs
ls -la sts_models/

# View training logs
ls -la logs/

# Wandb cache
ls -la wandb_cache/
```

### Debugging

```bash
# Interactive shell in running container
docker exec -it container_id bash

# Test GPU access
docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check build logs
docker build --progress=plain -t sts-neural-agent .
```

## 🔧 Configuration

### Environment Variables

```bash
# Wandb authentication
export WANDB_API_KEY=your_api_key

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0

# Training configuration
export STS_EPISODES=2000
export STS_REWARD_FUNCTION=comprehensive
```

### Volume Mounts

- **`./sts_models:/app/sts_models`** - Trained models (persistent)
- **`./logs:/app/logs`** - Training logs (persistent)
- **`./wandb_cache:/app/wandb_cache`** - Wandb cache (performance)
- **`./configs:/app/configs`** - Custom configurations (optional)

### Resource Limits

```bash
# Limit memory usage
docker run --memory=8g --gpus all sts-neural-agent

# Limit CPU usage
docker run --cpus=4 sts-neural-agent

# Set restart policy
docker run --restart=unless-stopped --detach sts-neural-agent
```

## 💰 Cost Optimization

### Instance Selection

- **Development**: `g4dn.xlarge` ($0.50/hr) or `t3.large` ($0.08/hr CPU)
- **Production**: `g4dn.2xlarge` ($1.00/hr) or `p3.2xlarge` ($3.06/hr)
- **Budget**: Vast.ai starting at $0.15/hr for RTX 3080

### Training Strategies

```bash
# Short validation runs
./docker-scripts/train.sh --episodes 100 --wandb-name validation

# Checkpoint-based training
./docker-scripts/train.sh --episodes 1000 --save-interval 50

# Spot instances (AWS/GCP)
# Use --detached and save frequently
```

### Auto-shutdown

```bash
# Add to training script
./docker-scripts/train.sh --episodes 5000 && sudo shutdown -h now

# Or use cloud instance scheduling
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU not detected**:
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Check Docker GPU support
   docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Out of memory**:
   ```bash
   # Reduce batch size
   docker run ... sts-neural-agent python train_sts_agent.py train --batch-size 32
   ```

3. **Build failures**:
   ```bash
   # Clean rebuild
   docker build --no-cache -t sts-neural-agent .
   
   # Check logs
   docker build --progress=plain -t sts-neural-agent .
   ```

4. **Wandb authentication**:
   ```bash
   # Login interactively
   docker run -it sts-neural-agent wandb login
   
   # Or set API key
   export WANDB_API_KEY=your_key
   ```

### Performance Issues

```bash
# Monitor resource usage
htop
nvidia-smi -l 1

# Check I/O performance
iostat -x 1

# Profile training
docker run --gpus all -it sts-neural-agent python -m cProfile train_sts_agent.py train
```

## 📚 Advanced Usage

### Multi-GPU Training

```bash
# Use all GPUs
docker run --gpus all sts-neural-agent

# Specific GPUs
docker run --gpus '"device=0,1"' sts-neural-agent
```

### Custom Images

```dockerfile
# Custom Dockerfile extending base
FROM sts-neural-agent:latest

# Add custom dependencies
RUN pip install additional-package

# Custom training script
COPY my_custom_trainer.py .
CMD ["python", "my_custom_trainer.py"]
```

### Hyperparameter Sweeps

```bash
# Parallel training runs
for lr in 1e-4 3e-4 1e-3; do
    docker run -d --gpus all \
        -e WANDB_API_KEY=$WANDB_API_KEY \
        sts-neural-agent \
        python train_sts_agent.py train --lr $lr --wandb-name lr-$lr
done
```

---

🎯 **Your STS neural agent is now ready for scalable cloud training!** Start with a small instance for testing, then scale up for production training runs.