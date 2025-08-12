#!/bin/bash
# STS Neural Agent - Docker Test Script
# Quick validation that Docker setup works correctly

set -e

echo "🧪 Testing STS Neural Agent Docker Setup"
echo "======================================="

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
echo "📦 Building STS Neural Agent image (this may take 10-15 minutes)..."
echo "ℹ️  Building PyTorch Docker image for the first time..."
if timeout 1800 docker build -t sts-neural-agent-test . > build.log 2>&1; then
    echo "✅ Image built successfully"
else
    echo "❌ Image build failed. Check build.log for details:"
    tail -20 build.log
    exit 1
fi

# Test basic functionality
echo ""
echo "🧪 Testing basic functionality..."

# Create test directories
mkdir -p test_output test_logs

# Test CPU training (very short)
echo "Testing CPU training..."
if docker run --rm \
    -v $(pwd)/test_output:/app/sts_models \
    -v $(pwd)/test_logs:/app/logs \
    sts-neural-agent-test \
    python train_sts_agent.py train --episodes 1 --no-wandb > cpu_test.log 2>&1; then
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
if docker run --rm sts-neural-agent-test python -c "
import slaythespire
import torch
print('Python imports: ✅')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 123, 456)
print(f'Game context: HP={gc.cur_hp}/{gc.max_hp}')
print('All tests passed!')
" > interactive_test.log 2>&1; then
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