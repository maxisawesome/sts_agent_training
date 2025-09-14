#!/bin/bash
# STS Neural Agent - Cloud Training Script
# Handles persistent storage and backup for cloud training

set -e

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo "⚠️  No .env file found. Copy .env.example to .env and configure it."
    echo "   cp .env.example .env"
    exit 1
fi

# Configuration with defaults
DATA_DIR="${DATA_PATH:-/mnt/sts-data}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
EPISODES="${EPISODES:-2000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cloud-training-$(date +%Y%m%d-%H%M%S)}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.cloud.yml}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 STS Neural Agent - Cloud Training${NC}"
echo "============================================"
echo "📊 Experiment: $EXPERIMENT_NAME"
echo "📂 Data Path: $DATA_DIR"
echo "🎯 Episodes: $EPISODES"
echo ""

# Check dependencies
command -v docker >/dev/null 2>&1 || { echo -e "${RED}❌ Docker not found${NC}"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo -e "${RED}❌ Docker Compose not found${NC}"; exit 1; }

# Setup storage directories
echo -e "${YELLOW}📁 Setting up storage directories...${NC}"
mkdir -p "$DATA_DIR"/{models,logs,wandb,configs,backups}

# Verify storage is writable
if ! touch "$DATA_DIR/test-write" 2>/dev/null; then
    echo -e "${RED}❌ Cannot write to $DATA_DIR. Check permissions or mount status.${NC}"
    exit 1
fi
rm -f "$DATA_DIR/test-write"

# Check available space
AVAILABLE_SPACE=$(df -BG "$DATA_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    echo -e "${YELLOW}⚠️  Warning: Less than 10GB available space ($AVAILABLE_SPACE GB)${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Pre-training backup
if [ "$BACKUP_ENABLED" = "true" ] && [ -f "scripts/backup-data.sh" ]; then
    echo -e "${YELLOW}📦 Creating pre-training backup...${NC}"
    ./scripts/backup-data.sh
fi

# Save training configuration
CONFIG_FILE="$DATA_DIR/configs/training_config_$(date +%Y%m%d_%H%M%S).json"
cat > "$CONFIG_FILE" << EOF
{
    "experiment_name": "$EXPERIMENT_NAME",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "episodes": $EPISODES,
    "reward_function": "$REWARD_FUNCTION",
    "hidden_size": $HIDDEN_SIZE,
    "batch_size": $BATCH_SIZE,
    "learning_rate": $LEARNING_RATE,
    "data_path": "$DATA_DIR",
    "wandb_project": "$WANDB_PROJECT",
    "extra_tags": "$EXTRA_TAGS"
}
EOF

echo -e "${GREEN}💾 Configuration saved: $CONFIG_FILE${NC}"

# Start training
echo -e "${BLUE}🏋️  Starting training...${NC}"
export EXPERIMENT_NAME
export DATA_PATH="$DATA_DIR"

# Use docker-compose for orchestration
docker-compose -f "$COMPOSE_FILE" up --build sts-trainer

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    
    # Post-training backup
    if [ "$BACKUP_ENABLED" = "true" ] && [ -f "scripts/backup-data.sh" ]; then
        echo -e "${YELLOW}📦 Creating post-training backup...${NC}"
        ./scripts/backup-data.sh
    fi
    
    # Show results summary
    echo ""
    echo -e "${BLUE}📊 Training Summary${NC}"
    echo "===================="
    echo "🎯 Experiment: $EXPERIMENT_NAME"
    echo "📈 Models saved: $(ls -1 "$DATA_DIR/models" | wc -l) files"
    echo "📋 Logs: $(du -sh "$DATA_DIR/logs" | cut -f1)"
    echo "🔗 Wandb: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
    echo ""
    echo "🎉 Training complete! Check wandb for detailed results."
    
else
    echo -e "${RED}❌ Training failed. Check logs in $DATA_DIR/logs${NC}"
    exit 1
fi