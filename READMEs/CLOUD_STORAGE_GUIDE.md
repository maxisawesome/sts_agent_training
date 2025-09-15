# Cloud Storage Guide for STS Neural Agent

This guide shows how to persist training data, models, and logs when running on cloud platforms.

## 🎯 What Needs to Be Persisted

### Critical Data
- **`sts_models/`** - Trained model files (.pt files)
- **`wandb/`** - Weights & Biases logs and cache
- **`logs/`** - Training logs and debug output
- **`training_config_*.json`** - Training configurations

### Optional Data
- **`wandb_cache/`** - Wandb offline cache (can be recreated)
- **Build artifacts** - Usually regenerated each time

## 🌩️ Cloud Platform Solutions

### AWS EC2 + EBS Volumes

**1. Create and attach EBS volume:**
```bash
# Create 100GB volume (adjust size as needed)
aws ec2 create-volume --size 100 --volume-type gp3 --availability-zone us-west-2a

# Attach to your instance
aws ec2 attach-volume --volume-id vol-xxxxx --instance-id i-xxxxx --device /dev/sdf
```

**2. Mount and setup:**
```bash
# On your EC2 instance
sudo mkfs.ext4 /dev/xvdf
sudo mkdir -p /mnt/sts-data
sudo mount /dev/xvdf /mnt/sts-data
sudo chown -R $USER:$USER /mnt/sts-data

# Add to /etc/fstab for auto-mount
echo '/dev/xvdf /mnt/sts-data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
```

**3. Run with volume mounting:**
```bash
docker run --gpus all --rm \
    -v /mnt/sts-data/models:/app/sts_models \
    -v /mnt/sts-data/logs:/app/logs \
    -v /mnt/sts-data/wandb:/app/wandb \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    sts-neural-agent \
    python train_sts_agent.py train --episodes 2000
```

### Google Cloud Platform + Persistent Disks

**1. Create persistent disk:**
```bash
gcloud compute disks create sts-training-data \
    --size=100GB \
    --zone=us-central1-a \
    --type=pd-ssd
```

**2. Attach to VM:**
```bash
gcloud compute instances attach-disk sts-trainer \
    --disk=sts-training-data \
    --zone=us-central1-a
```

**3. Mount disk:**
```bash
# Format and mount
sudo mkfs.ext4 -F /dev/sdb
sudo mkdir -p /mnt/training-data
sudo mount /dev/sdb /mnt/training-data
sudo chown -R $USER:$USER /mnt/training-data
```

### Azure + Azure Disk

**1. Create managed disk:**
```bash
az disk create \
    --resource-group myResourceGroup \
    --name sts-training-disk \
    --size-gb 100 \
    --sku Premium_LRS
```

**2. Attach to VM:**
```bash
az vm disk attach \
    --resource-group myResourceGroup \
    --vm-name sts-trainer \
    --name sts-training-disk
```

### Generic Cloud / VPS

**1. Create data directory:**
```bash
sudo mkdir -p /opt/sts-data/{models,logs,wandb,configs}
sudo chown -R $USER:$USER /opt/sts-data
```

## 🐳 Docker Compose for Cloud

### Basic Cloud Compose File

```yaml
# docker-compose.cloud.yml
version: '3.8'

services:
  sts-trainer:
    build: .
    volumes:
      # Mount persistent storage
      - ${DATA_PATH}/models:/app/sts_models
      - ${DATA_PATH}/logs:/app/logs
      - ${DATA_PATH}/wandb:/app/wandb
      - ${DATA_PATH}/configs:/app/configs
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - WANDB_CACHE_DIR=/app/wandb
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: python train_sts_agent.py train --episodes 2000 --wandb
    
  # Optional: Jupyter for analysis
  sts-jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ${DATA_PATH}/models:/app/sts_models
      - ${DATA_PATH}/logs:/app/logs
      - ${DATA_PATH}/wandb:/app/wandb
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Usage with environment file:

```bash
# Create .env file
echo "DATA_PATH=/mnt/sts-data" > .env
echo "WANDB_API_KEY=your_key_here" >> .env

# Run training
docker-compose -f docker-compose.cloud.yml up sts-trainer

# Run Jupyter for analysis
docker-compose -f docker-compose.cloud.yml up sts-jupyter
```

## 💾 Backup and Sync Strategies

### 1. Periodic S3 Sync (AWS)

```bash
#!/bin/bash
# backup-to-s3.sh
aws s3 sync /mnt/sts-data/models s3://your-bucket/sts-models/ --delete
aws s3 sync /mnt/sts-data/logs s3://your-bucket/sts-logs/ 
aws s3 sync /mnt/sts-data/configs s3://your-bucket/sts-configs/

# Run this via cron every hour
# 0 * * * * /path/to/backup-to-s3.sh
```

### 2. Google Cloud Storage Sync

```bash
#!/bin/bash
# backup-to-gcs.sh
gsutil -m rsync -r -d /mnt/training-data/models gs://your-bucket/sts-models/
gsutil -m rsync -r /mnt/training-data/logs gs://your-bucket/sts-logs/
```

### 3. Real-time Sync with rsync

```bash
#!/bin/bash
# sync-to-backup-server.sh
rsync -avz --delete /mnt/sts-data/ backup-server:/backup/sts-data/
```

## 🚀 Production Training Script

Create a production training script that handles cloud storage:

```bash
#!/bin/bash
# cloud-train.sh

set -e

# Configuration
DATA_DIR="${DATA_DIR:-/mnt/sts-data}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
EPISODES="${EPISODES:-2000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cloud-training-$(date +%Y%m%d-%H%M%S)}"

# Ensure directories exist
mkdir -p "$DATA_DIR"/{models,logs,wandb,configs}

# Pre-training backup
if [ "$BACKUP_ENABLED" = "true" ]; then
    echo "📦 Creating pre-training backup..."
    ./scripts/backup-data.sh
fi

# Run training with proper volume mounts
echo "🚀 Starting training: $EXPERIMENT_NAME"
docker run --gpus all --rm \
    -v "$DATA_DIR/models":/app/sts_models \
    -v "$DATA_DIR/logs":/app/logs \
    -v "$DATA_DIR/wandb":/app/wandb \
    -v "$DATA_DIR/configs":/app/configs \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e WANDB_CACHE_DIR=/app/wandb \
    sts-neural-agent \
    python train_sts_agent.py train \
        --episodes "$EPISODES" \
        --wandb-name "$EXPERIMENT_NAME" \
        --wandb-tags cloud production

# Post-training backup
if [ "$BACKUP_ENABLED" = "true" ]; then
    echo "💾 Creating post-training backup..."
    ./scripts/backup-data.sh
fi

echo "✅ Training completed: $EXPERIMENT_NAME"
```

## 📊 Monitoring and Management

### Check Storage Usage

```bash
#!/bin/bash
# check-storage.sh

echo "=== STS Training Data Storage Usage ==="
echo "Models: $(du -sh /mnt/sts-data/models 2>/dev/null || echo 'N/A')"
echo "Logs: $(du -sh /mnt/sts-data/logs 2>/dev/null || echo 'N/A')"
echo "Wandb: $(du -sh /mnt/sts-data/wandb 2>/dev/null || echo 'N/A')"
echo "Total: $(du -sh /mnt/sts-data 2>/dev/null || echo 'N/A')"
echo ""
echo "=== Disk Usage ==="
df -h /mnt/sts-data
```

### Clean Old Data

```bash
#!/bin/bash
# cleanup-old-data.sh

# Remove logs older than 30 days
find /mnt/sts-data/logs -name "*.log" -mtime +30 -delete

# Remove old model checkpoints (keep latest 10)
cd /mnt/sts-data/models
ls -t *.pt | tail -n +11 | xargs rm -f

# Clean wandb cache
wandb artifact cache cleanup 10GB
```

## 🔧 Quick Setup Commands

### For AWS EC2
```bash
# One-line setup
curl -fsSL https://raw.githubusercontent.com/your-repo/sts/main/scripts/setup-aws-storage.sh | bash
```

### For Google Cloud
```bash
# One-line setup  
curl -fsSL https://raw.githubusercontent.com/your-repo/sts/main/scripts/setup-gcp-storage.sh | bash
```

### For Generic Cloud
```bash
# Manual setup
sudo mkdir -p /opt/sts-data/{models,logs,wandb,configs}
sudo chown -R $USER:$USER /opt/sts-data
export DATA_PATH=/opt/sts-data
```

## 💡 Best Practices

1. **Size Planning**: Start with 100GB, monitor usage, expand as needed
2. **Backup Strategy**: Sync critical data (models) frequently, logs less often
3. **Monitoring**: Set up alerts for disk usage > 80%
4. **Cost Optimization**: Use cheaper storage tiers for old logs/models
5. **Recovery Plan**: Test restore procedures before you need them

This setup ensures your training data persists across container restarts and instance reboots, with proper backup strategies for long-running cloud training jobs.