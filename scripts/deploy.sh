#!/bin/bash
# STS Neural Agent - Cloud Deployment Script
# Quick setup for cloud instances (AWS, GCP, Azure, etc.)

set -e

echo "🌥️  STS Neural Agent - Cloud Instance Setup"
echo "=========================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Running as root. This script will create a 'trainer' user."
    CREATE_USER=true
else
    CREATE_USER=false
fi

# Function to detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        DISTRO=$ID
    else
        echo "❌ Cannot detect OS. This script supports Ubuntu/Debian."
        exit 1
    fi
    echo "🔍 Detected OS: $OS"
}

# Function to install Docker
install_docker() {
    echo "🐳 Installing Docker..."
    
    if command -v docker &> /dev/null; then
        echo "✅ Docker already installed"
        return
    fi
    
    case $DISTRO in
        ubuntu|debian)
            apt-get update
            apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
            curl -fsSL https://download.docker.com/linux/$DISTRO/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
            echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/$DISTRO $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
            apt-get update
            apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            ;;
        *)
            echo "❌ Unsupported distribution: $DISTRO"
            exit 1
            ;;
    esac
    
    systemctl enable docker
    systemctl start docker
    echo "✅ Docker installed successfully"
}

# Function to install NVIDIA Docker (for GPU support)
install_nvidia_docker() {
    echo "🎮 Installing NVIDIA Docker support..."
    
    # Check if NVIDIA drivers are installed
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️  NVIDIA drivers not found. Installing..."
        case $DISTRO in
            ubuntu|debian)
                apt-get update
                apt-get install -y nvidia-driver-470
                ;;
        esac
        echo "🔄 Please reboot after driver installation and re-run this script"
        exit 0
    fi
    
    # Install NVIDIA Container Toolkit
    if ! command -v nvidia-container-runtime &> /dev/null; then
        case $DISTRO in
            ubuntu|debian)
                distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
                curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
                curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
                apt-get update
                apt-get install -y nvidia-container-toolkit
                systemctl restart docker
                ;;
        esac
        echo "✅ NVIDIA Docker support installed"
    else
        echo "✅ NVIDIA Docker support already installed"
    fi
}

# Function to create user
create_trainer_user() {
    if [ "$CREATE_USER" = true ]; then
        echo "👤 Creating trainer user..."
        if ! id "trainer" &>/dev/null; then
            useradd -m -s /bin/bash trainer
            usermod -aG docker trainer
            echo "✅ Created trainer user"
        else
            echo "✅ Trainer user already exists"
        fi
    fi
}

# Function to setup project
setup_project() {
    echo "📁 Setting up project directory..."
    
    PROJECT_DIR="/home/trainer/sts-neural-agent"
    if [ "$CREATE_USER" = false ]; then
        PROJECT_DIR="$HOME/sts-neural-agent"
    fi
    
    # Clone repository (you'll need to replace with your actual repo URL)
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "📥 Cloning repository..."
        if [ "$CREATE_USER" = true ]; then
            sudo -u trainer git clone https://github.com/your-username/sts-neural-agent.git $PROJECT_DIR
        else
            git clone https://github.com/your-username/sts-neural-agent.git $PROJECT_DIR
        fi
    else
        echo "✅ Project directory already exists"
    fi
    
    # Set ownership
    if [ "$CREATE_USER" = true ]; then
        chown -R trainer:trainer $PROJECT_DIR
    fi
    
    echo "✅ Project setup complete at: $PROJECT_DIR"
}

# Function to test installation
test_installation() {
    echo "🧪 Testing installation..."
    
    # Test Docker
    docker --version
    echo "✅ Docker working"
    
    # Test NVIDIA Docker (if available)
    if command -v nvidia-smi &> /dev/null; then
        if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            echo "✅ NVIDIA Docker working"
        else
            echo "⚠️  NVIDIA Docker test failed"
        fi
    fi
    
    echo "✅ Installation test complete"
}

# Function to show usage examples
show_usage() {
    PROJECT_DIR="/home/trainer/sts-neural-agent"
    if [ "$CREATE_USER" = false ]; then
        PROJECT_DIR="$HOME/sts-neural-agent"
    fi
    
    echo ""
    echo "🚀 Setup Complete! Next Steps:"
    echo "============================="
    echo ""
    echo "1. Set your Wandb API key:"
    echo "   export WANDB_API_KEY=your_api_key_here"
    echo ""
    echo "2. Navigate to project directory:"
    echo "   cd $PROJECT_DIR"
    echo ""
    echo "3. Quick test run:"
    echo "   ./docker-scripts/train.sh --episodes 5 --wandb-name test-run"
    echo ""
    echo "4. Full training run:"
    echo "   ./docker-scripts/train.sh --episodes 2000 --wandb-name production-run"
    echo ""
    echo "5. Background training:"
    echo "   ./docker-scripts/train.sh --detached --episodes 5000"
    echo ""
    echo "6. Monitor training:"
    echo "   docker logs -f \$(docker ps -q --filter ancestor=sts-neural-agent)"
    echo ""
    echo "💡 For more options: ./docker-scripts/train.sh --help"
}

# Main execution
main() {
    detect_os
    install_docker
    
    # Install NVIDIA Docker if GPU available
    if lspci | grep -i nvidia &> /dev/null; then
        echo "🎮 NVIDIA GPU detected"
        install_nvidia_docker
    else
        echo "💻 No NVIDIA GPU detected, CPU-only setup"
    fi
    
    create_trainer_user
    setup_project
    test_installation
    show_usage
}

# Run main function
main "$@"