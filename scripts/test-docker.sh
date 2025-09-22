#!/bin/bash
# STS Neural Agent - Docker Test Script
# Quick validation that Docker setup works correctly

set -e

# Parse command line arguments
USE_CACHE=true
DOCKERFILE="Dockerfile"
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-cache       Build without using Docker cache"
            echo "  --force-rebuild  Force a complete rebuild"
            echo "  --build-only    Only build, skip functionality tests"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🧪 Testing STS Neural Agent Docker Setup"
echo "======================================="
echo "📄 Using: $DOCKERFILE"
echo "🗃️  Cache enabled: $USE_CACHE"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

echo "✅ Docker found: $(docker --version)"

# Check if GPU support is available
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers found: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
    
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        echo "✅ Docker GPU support working"
        GPU_AVAILABLE=true
    else
        echo "⚠️  Docker GPU support not working"
    fi
else
    echo "ℹ️  No NVIDIA GPU detected (CPU-only mode)"
fi

# Build the image
echo ""
echo "📦 Building STS Neural Agent image..."

# Prepare build command
BUILD_CMD="docker build -f $DOCKERFILE -t sts-neural-agent-test"

if [ "$FORCE_REBUILD" = true ]; then
    echo "🔄 Force rebuilding (removing existing image)..."
    docker rmi sts-neural-agent-test 2>/dev/null || true
fi

if [ "$USE_CACHE" = false ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
    echo "ℹ️  Building without cache (will take 10-15 minutes)..."
else
    echo "ℹ️  Building with cache (much faster for subsequent builds)..."
fi

BUILD_CMD="$BUILD_CMD ."

echo "🔨 Running: $BUILD_CMD"
if timeout 1800 $BUILD_CMD > build.log 2>&1; then
    echo "✅ Image built successfully"
    
    # Show cache efficiency info
    if [ "$USE_CACHE" = true ] && [ "$FORCE_REBUILD" = false ]; then
        CACHED_LAYERS=$(grep -c "CACHED" build.log || echo "0")
        if [ "$CACHED_LAYERS" -gt 0 ]; then
            echo "🚀 Cache efficiency: $CACHED_LAYERS layers cached"
        fi
    fi
else
    echo "❌ Image build failed. Check build.log for details:"
    tail -20 build.log
    exit 1
fi

# Test basic functionality (skip if build-only mode)
if [ "$BUILD_ONLY" = true ]; then
    echo ""
    echo "✅ Build completed successfully (skipping tests as requested)"
    echo ""
    echo "💡 To run full tests: ./docker-scripts/test-docker.sh"
    exit 0
fi

echo ""
echo "🧪 Testing basic functionality..."

# Create test directories
mkdir -p test_output test_logs

# Test CPU training (very short)
echo "Testing CPU training..."
# Max: this was messing up so I changed it to tee 
docker run --rm \
    -v $(pwd)/test_output:/app/sts_models \
    -v $(pwd)/test_logs:/app/logs \
    sts-neural-agent-test \
    python train_sts_agent.py train --episodes 1 --no-wandb | tee cpu_test.log # 2>&1; then

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ CPU training test passed"
else
    echo "❌ CPU training test failed. Check cpu_test.log for details:"
    tail -10 cpu_test.log
    exit 1
fi

# Test GPU training if available
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Testing GPU training..."
    if docker run --rm --gpus all \
        -v $(pwd)/test_output:/app/sts_models \
        -v $(pwd)/test_logs:/app/logs \
        sts-neural-agent-test \
        python train_sts_agent.py train --episodes 1 --no-wandb > gpu_test.log 2>&1; then
        echo "✅ GPU training test passed"
    else
        echo "⚠️  GPU training test failed. Check gpu_test.log for details:"
        tail -10 gpu_test.log
    fi
fi

# Test model evaluation
echo "Testing model evaluation..."
if docker run --rm \
    -v $(pwd)/test_output:/app/sts_models \
    sts-neural-agent-test \
    python train_sts_agent.py list > list_test.log 2>&1; then
    echo "✅ Model listing test passed"
else
    echo "⚠️  Model listing test failed. Check list_test.log"
fi

# Test interactive capabilities
echo "Testing interactive shell..."
if docker run --rm sts-neural-agent-test python -c '
import sys
sys.path.insert(0, "sts_lightspeed"); 
import slaythespire
import torch
print("Python imports: ✅")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 123456, 0)
print(f"Game context: HP={gc.cur_hp}/{gc.max_hp}")
print("All tests passed!")
' > interactive_test.log 2>&1; then
    echo "✅ Interactive test passed"
else
    echo "❌ Interactive test failed. Check interactive_test.log:"
    cat interactive_test.log
    exit 1
fi

# Clean up test files
rm -f build.log cpu_test.log gpu_test.log list_test.log interactive_test.log
rm -rf test_output test_logs

# Summary
echo ""
echo "🎉 Docker Setup Test Complete!"
echo "=============================="
echo "✅ Docker image builds successfully"
echo "✅ CPU training works"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "✅ GPU support available"
else
    echo "ℹ️  GPU support not available (CPU-only)"
fi
echo "✅ All core functionality working"
echo ""
echo "🚀 Ready for production training!"
echo ""
echo "Next steps:"
echo "1. Set WANDB_API_KEY: export WANDB_API_KEY=your_key"
echo "2. Start training: ./docker-scripts/train.sh --episodes 100"
echo "3. For help: ./docker-scripts/train.sh --help"
echo ""
echo "💡 Cache optimization tips:"
echo "• Subsequent builds will be much faster (cached layers)"
echo "• Use --dev for development (faster Python code iteration)"
echo "• Use --optimized for production (multi-stage build)"
echo "• Use --force-rebuild only when needed (e.g., base image updates)"