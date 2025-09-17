#!/bin/bash
# STS Neural Agent - Docker Training Script
# Usage: ./docker-scripts/train.sh [OPTIONS]

set -e

# Default values
EPISODES=2000
REWARD_FUNCTION="comprehensive"
WANDB_NAME=""
WANDB_TAGS=""
GPU=true
DETACHED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --reward-function)
            REWARD_FUNCTION="$2"
            shift 2
            ;;
        --wandb-name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --wandb-tags)
            WANDB_TAGS="$2"
            shift 2
            ;;
        --cpu)
            GPU=false
            shift
            ;;
        --detached|-d)
            DETACHED=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --episodes N           Number of training episodes (default: 2000)"
            echo "  --reward-function F    Reward function type (default: comprehensive)"
            echo "  --wandb-name NAME      Wandb run name"
            echo "  --wandb-tags TAGS      Wandb tags (space-separated)"
            echo "  --cpu                  Use CPU-only training"
            echo "  --detached, -d         Run in detached mode"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --episodes 5000 --wandb-name my-experiment"
            echo "  $0 --cpu --episodes 1000"
            echo "  $0 --detached --wandb-name background-training"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if wandb API key is set
if [ -z "${WANDB_API_KEY}" ]; then
    echo "⚠️  Warning: WANDB_API_KEY not set. You may need to run:"
    echo "   export WANDB_API_KEY=your_api_key_here"
    echo ""
fi

# Build command
TRAIN_CMD="python train_sts_agent.py train --episodes $EPISODES --reward-function $REWARD_FUNCTION --wandb"

if [ -n "$WANDB_NAME" ]; then
    TRAIN_CMD="$TRAIN_CMD --wandb-name $WANDB_NAME"
fi

if [ -n "$WANDB_TAGS" ]; then
    TRAIN_CMD="$TRAIN_CMD --wandb-tags $WANDB_TAGS"
fi

# Docker run options
DOCKER_OPTS="--rm"
if [ "$DETACHED" = true ]; then
    DOCKER_OPTS="$DOCKER_OPTS -d"
else
    DOCKER_OPTS="$DOCKER_OPTS -it"
fi

# GPU or CPU
if [ "$GPU" = true ]; then
    DOCKER_OPTS="$DOCKER_OPTS --gpus all"
    echo "🚀 Starting GPU training..."
else
    echo "🚀 Starting CPU training..."
fi

# Environment variables
ENV_OPTS=""
if [ -n "${WANDB_API_KEY}" ]; then
    ENV_OPTS="-e WANDB_API_KEY=${WANDB_API_KEY}"
fi

# Volume mounts
VOLUME_OPTS="-v $(pwd)/sts_models:/app/sts_models"
VOLUME_OPTS="$VOLUME_OPTS -v $(pwd)/logs:/app/logs"
VOLUME_OPTS="$VOLUME_OPTS -v $(pwd)/wandb_cache:/app/wandb_cache"

# Create directories if they don't exist
mkdir -p sts_models logs wandb_cache

echo "Training Configuration:"
echo "  Episodes: $EPISODES"
echo "  Reward Function: $REWARD_FUNCTION"
echo "  GPU: $GPU"
echo "  Detached: $DETACHED"
if [ -n "$WANDB_NAME" ]; then
    echo "  Wandb Name: $WANDB_NAME"
fi
if [ -n "$WANDB_TAGS" ]; then
    echo "  Wandb Tags: $WANDB_TAGS"
fi
echo ""

# Build image if it doesn't exist
if [ -z "$(docker images -q sts-neural-agent 2> /dev/null)" ]; then
    echo "📦 Building Docker image..."
    docker build -t sts-neural-agent .
fi

# Run training
echo "🎯 Command: $TRAIN_CMD"
echo ""

docker run $DOCKER_OPTS $ENV_OPTS $VOLUME_OPTS sts-neural-agent bash -c "$TRAIN_CMD"

if [ "$DETACHED" = true ]; then
    echo ""
    echo "✅ Training started in background!"
    echo "📊 Monitor progress with: docker logs -f \$(docker ps -q --filter ancestor=sts-neural-agent)"
    echo "🛑 Stop training with: docker stop \$(docker ps -q --filter ancestor=sts-neural-agent)"
fi