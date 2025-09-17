#!/bin/bash
# STS Neural Agent - Docker Debug Script
# Step-by-step debugging for Docker issues

set -e

echo "🔍 STS Neural Agent Docker Debug"
echo "==============================="

# Function to check command existence
check_command() {
    if command -v $1 &> /dev/null; then
        echo "✅ $1 is available"
        return 0
    else
        echo "❌ $1 is not available"
        return 1
    fi
}

# Function to test Docker functionality
test_docker_basic() {
    echo "🐳 Testing basic Docker functionality..."
    
    if docker run --rm hello-world > /dev/null 2>&1; then
        echo "✅ Docker basic functionality works"
        return 0
    else
        echo "❌ Docker basic functionality failed"
        echo "Try running: docker run --rm hello-world"
        return 1
    fi
}

# Function to check file structure
check_files() {
    echo "📁 Checking required files..."
    
    local files=(
        "Dockerfile"
        "requirements.txt"
        "sts_lightspeed"
        "sts_neural_network.py"
        "sts_training.py"
        "train_sts_agent.py"
    )
    
    for file in "${files[@]}"; do
        if [ -e "$file" ]; then
            echo "✅ $file exists"
        else
            echo "❌ $file missing"
        fi
    done
}

# Function to check sts_lightspeed build
check_sts_lightspeed() {
    echo "🎮 Checking sts_lightspeed..."
    
    if [ -d "sts_lightspeed" ]; then
        echo "✅ sts_lightspeed directory exists"
        
        if [ -f "sts_lightspeed/slaythespire.cpython-38-darwin.so" ] || [ -f "sts_lightspeed/slaythespire.cpython-*.so" ]; then
            echo "✅ sts_lightspeed appears to be built"
        else
            echo "⚠️  sts_lightspeed may not be built"
            echo "Build files in sts_lightspeed:"
            ls -la sts_lightspeed/ | grep -E "\.(so|dylib)$" || echo "No shared libraries found"
        fi
        
        # Check if submodules are initialized
        if [ -d "sts_lightspeed/.git" ] || [ -f "sts_lightspeed/.git" ]; then
            echo "✅ sts_lightspeed appears to be a git submodule"
        else
            echo "⚠️  sts_lightspeed may not be a proper git submodule"
        fi
    else
        echo "❌ sts_lightspeed directory missing"
    fi
}

# Function to test Python imports locally
test_python_imports() {
    echo "🐍 Testing Python imports locally..."
    
    # Test basic imports
    if python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
        echo "✅ PyTorch import works"
    else
        echo "❌ PyTorch import failed"
    fi
    
    if python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null; then
        echo "✅ NumPy import works"
    else
        echo "❌ NumPy import failed"
    fi
    
    # Test sts_lightspeed import
    if python3 -c "import sys; sys.path.insert(0, 'sts_lightspeed'); import slaythespire; print('sts_lightspeed import: ✅')" 2>/dev/null; then
        echo "✅ sts_lightspeed import works locally"
    else
        echo "❌ sts_lightspeed import failed locally"
        echo "This suggests the module needs to be rebuilt"
    fi
}

# Function to test minimal Docker build
test_minimal_build() {
    echo "🔨 Testing minimal Docker build..."
    
    # Create a minimal Dockerfile for testing
    cat > Dockerfile.test << 'EOF'
FROM pytorch/pytorch:2.1.0-cuda11.8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Test basic functionality
RUN python -c "import torch; print('PyTorch works')"
RUN python -c "import numpy; print('NumPy works')"

CMD ["echo", "Minimal build successful"]
EOF
    
    echo "Building minimal test image..."
    if docker build -f Dockerfile.test -t sts-test-minimal . > minimal_build.log 2>&1; then
        echo "✅ Minimal Docker build successful"
        rm -f Dockerfile.test
        return 0
    else
        echo "❌ Minimal Docker build failed"
        echo "Last 10 lines of build log:"
        tail -10 minimal_build.log
        return 1
    fi
}

# Main debug function
main() {
    echo "Starting comprehensive Docker debug..."
    echo ""
    
    # Basic checks
    echo "=== System Checks ==="
    check_command docker
    check_command python3
    check_command git
    echo ""
    
    # Docker functionality
    echo "=== Docker Checks ==="
    test_docker_basic
    echo ""
    
    # File structure
    echo "=== File Structure ==="
    check_files
    echo ""
    
    # sts_lightspeed specific
    echo "=== STS Lightspeed ==="
    check_sts_lightspeed
    echo ""
    
    # Python imports
    echo "=== Python Environment ==="
    test_python_imports
    echo ""
    
    # Minimal build test
    echo "=== Docker Build Test ==="
    test_minimal_build
    echo ""
    
    # Summary and recommendations
    echo "=== Debug Summary ==="
    echo "Debug complete. Check the output above for any ❌ failures."
    echo ""
    echo "Common fixes:"
    echo "1. If Docker basic test failed:"
    echo "   sudo systemctl start docker"
    echo "   sudo usermod -aG docker \$USER  # then logout/login"
    echo ""
    echo "2. If sts_lightspeed missing or not built:"
    echo "   git submodule update --init --recursive"
    echo "   cd sts_lightspeed && cmake . && make -j4"
    echo ""
    echo "3. If Python imports fail:"
    echo "   pip install torch numpy wandb matplotlib"
    echo ""
    echo "4. If Docker build fails:"
    echo "   Check minimal_build.log for detailed error messages"
    echo "   docker system prune -f  # Clean up Docker cache"
    echo ""
    echo "After fixing issues, try running the test again:"
    echo "./docker-scripts/test-docker.sh"
}

# Run main function
main "$@"