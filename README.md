# STS Neural Agent - Reinforcement Learning for Slay the Spire

A complete reinforcement learning system for training neural network agents to play Slay the Spire using the [sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) simulation engine.

## 🎯 Features

- **Fast C++ Simulation**: Uses sts_lightspeed for high-performance game simulation
- **Modern RL**: Proximal Policy Optimization (PPO) with actor-critic architecture
- **Flexible Rewards**: Multiple reward functions (simple, comprehensive, sparse, shaped)
- **Experiment Tracking**: Full Weights & Biases integration for monitoring training
- **GPU Support**: CUDA acceleration for faster training
- **Professional Tools**: Model management, analysis tools, and interactive evaluation

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **CMake 3.10+**
- **C++ compiler** with C++17 support (GCC/Clang)
- **CUDA** (optional, for GPU acceleration)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd sts-neural-agent
git submodule update --init --recursive

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# OR for CPU only:
# pip install torch torchvision torchaudio

pip install numpy matplotlib wandb
```

### 2. Build sts_lightspeed

```bash
cd sts_lightspeed

# Clean any previous builds
rm -rf CMakeCache.txt CMakeFiles/

# Configure with your Python installation
cmake -DPYTHON_EXECUTABLE=$(which python3) .

# Build (use -j4 for 4 parallel jobs, adjust based on your CPU)
make -j4

# Verify the build worked
python3 -c "import slaythespire; print('✅ sts_lightspeed installed successfully!')"
```

### 3. Set up Weights & Biases (Optional but Recommended)

```bash
# Install wandb if not already installed
pip install wandb

# Set up authentication
python3 setup_wandb.py
```

### 4. Start Training!

```bash
# Quick test (5 episodes, ~1 minute)
python3 train_sts_agent.py train --episodes 5 --no-wandb

# Full training with tracking (recommended)
python3 train_sts_agent.py train \
    --episodes 2000 \
    --reward-function comprehensive \
    --wandb \
    --wandb-name my-first-experiment

# GPU training (if CUDA available)
python3 train_sts_agent.py train \
    --episodes 5000 \
    --hidden-size 1024 \
    --batch-size 128 \
    --wandb-name gpu-large-model
```

## 🛠️ Detailed Installation

### System Requirements

**Minimum:**
- 4 GB RAM
- 2 CPU cores
- 2 GB disk space

**Recommended:**
- 16+ GB RAM
- 8+ CPU cores
- GPU with 8+ GB VRAM (RTX 3070/4060 or better)
- 10 GB disk space

### Ubuntu/Debian Installation

```bash
# Install system dependencies
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip python3-venv

# Install CUDA (optional, for GPU support)
# Follow: https://developer.nvidia.com/cuda-downloads

# Clone and setup
git clone <your-repo-url>
cd sts-neural-agent
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (choose your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy matplotlib wandb

# Build sts_lightspeed
cd sts_lightspeed
cmake -DPYTHON_EXECUTABLE=$(which python3) .
make -j$(nproc)
cd ..

# Test installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import slaythespire; print('sts_lightspeed: ✅')"
```

### macOS Installation

```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake python@3.11

# Clone and setup
git clone <your-repo-url>
cd sts-neural-agent
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU version for macOS)
pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy matplotlib wandb

# Build sts_lightspeed
cd sts_lightspeed
cmake -DPYTHON_EXECUTABLE=$(which python3) .
make -j$(sysctl -n hw.ncpu)
cd ..

# Test installation
python3 test_nn_interface.py
```

### Docker Installation (Recommended for Cloud)

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install Python dependencies
RUN pip install numpy matplotlib wandb

# Build sts_lightspeed
RUN cd sts_lightspeed && \
    cmake -DPYTHON_EXECUTABLE=$(which python) . && \
    make -j$(nproc)

# Set default command
CMD ["python3", "train_sts_agent.py", "train", "--episodes", "1000"]
```

```bash
# Build and run
docker build -t sts-neural-agent .
docker run --gpus all -it sts-neural-agent
```

## 📊 Training Options

### Basic Usage

```bash
# List all available commands
python3 train_sts_agent.py --help

# Train with default settings
python3 train_sts_agent.py train

# Evaluate a trained model
python3 train_sts_agent.py eval sts_models/final_model.pt

# List available models
python3 train_sts_agent.py list

# Interactive play session
python3 train_sts_agent.py play --model sts_models/final_model.pt
```

### Training Parameters

```bash
python3 train_sts_agent.py train \
    --episodes 2000 \              # Number of episodes to train
    --lr 3e-4 \                    # Learning rate
    --batch-size 64 \              # Training batch size
    --hidden-size 512 \            # Neural network hidden layer size
    --reward-function comprehensive # Reward function type
```

### Reward Functions

- **`simple`**: Basic HP preservation (stable, limited learning)
- **`comprehensive`**: Multi-factor reward (recommended for most cases)
- **`sparse`**: Outcome-focused rewards (good for final optimization)
- **`shaped`**: Expert heuristics (domain knowledge guided)

```bash
# Compare different reward functions
python3 train_sts_agent.py train --reward-function simple --wandb-name simple-baseline
python3 train_sts_agent.py train --reward-function comprehensive --wandb-name comprehensive-baseline
python3 train_sts_agent.py train --reward-function sparse --wandb-name sparse-baseline
```

### Weights & Biases Integration

```bash
# Enable tracking (default)
python3 train_sts_agent.py train --wandb

# Custom project and naming
python3 train_sts_agent.py train \
    --wandb-project my-sts-experiments \
    --wandb-name baseline-experiment \
    --wandb-tags baseline comprehensive

# Disable tracking
python3 train_sts_agent.py train --no-wandb
```

## 🧠 Architecture

### Neural Network

- **Input**: 412-dimensional game state observation
  - Player HP (current/max), Gold, Floor number
  - Boss type (one-hot encoded, 10 bosses)
  - Deck composition (220 features for card counts)
  - Relics (178 binary features)

- **Architecture**: Actor-Critic with shared backbone
  - 3-layer fully connected network (512 hidden units default)
  - Separate policy and value heads
  - ~800K trainable parameters

- **Algorithm**: Proximal Policy Optimization (PPO)
  - Generalized Advantage Estimation (GAE)
  - Gradient clipping and entropy regularization
  - Configurable hyperparameters

### File Structure

```
sts-neural-agent/
├── README.md                    # This file
├── sts_lightspeed/             # C++ simulation engine
├── sts_neural_network.py       # Neural network architectures
├── sts_training.py             # PPO training implementation
├── sts_data_collection.py      # Environment and data collection
├── sts_reward_functions.py     # Reward function implementations
├── sts_neural_agent.py         # Trained agent wrapper
├── sts_model_manager.py        # Model saving/loading
├── train_sts_agent.py          # Main CLI interface
├── setup_wandb.py              # Wandb setup helper
├── WANDB_SETUP.md              # Detailed wandb guide
└── REWARD_SYSTEM.md            # Reward function documentation
```

## 🎮 Usage Examples

### Training a Baseline Model

```bash
# Train a comprehensive baseline (recommended first experiment)
python3 train_sts_agent.py train \
    --episodes 2000 \
    --reward-function comprehensive \
    --wandb-name baseline-comprehensive \
    --wandb-tags baseline comprehensive

# Monitor progress at: https://wandb.ai/your-username/sts-neural-agent
```

### Hyperparameter Sweeps

```bash
# Test different learning rates
for lr in 1e-4 3e-4 1e-3; do
    python3 train_sts_agent.py train \
        --episodes 1000 \
        --lr $lr \
        --wandb-name lr-sweep-$lr \
        --wandb-tags sweep learning-rate
done

# Test different architectures
for hidden in 256 512 1024; do
    python3 train_sts_agent.py train \
        --episodes 1000 \
        --hidden-size $hidden \
        --wandb-name arch-sweep-$hidden \
        --wandb-tags sweep architecture
done
```

### Model Evaluation

```bash
# Evaluate best model
python3 train_sts_agent.py eval sts_models/final_model.pt --games 100

# Interactive session
python3 train_sts_agent.py play --model sts_models/final_model.pt

# Compare models
python3 -c "
from sts_model_manager import STSModelManager
manager = STSModelManager()
models = manager.list_models()
for model in models[:5]:
    print(f'{model[\"filename\"]}: {model.get(\"final_avg_reward\", \"N/A\")} avg reward')
"
```

## 🐛 Troubleshooting

### Common Issues

1. **Build fails with CMake errors**
   ```bash
   # Clean and reconfigure
   cd sts_lightspeed
   rm -rf CMakeCache.txt CMakeFiles/
   cmake -DPYTHON_EXECUTABLE=$(which python3) .
   make clean && make -j4
   ```

2. **Import error: "No module named 'slaythespire'"**
   ```bash
   # Check if the .so file exists
   ls sts_lightspeed/*.so
   
   # Test import directly
   cd sts_lightspeed && python3 -c "import slaythespire"
   ```

3. **CUDA out of memory**
   ```bash
   # Reduce batch size and model size
   python3 train_sts_agent.py train \
       --batch-size 32 \
       --hidden-size 256
   ```

4. **Wandb authentication issues**
   ```bash
   # Reset wandb login
   wandb login --relogin
   
   # Or run setup script
   python3 setup_wandb.py
   ```

### Performance Optimization

**For CPU training:**
- Use smaller models (`--hidden-size 256`)
- Reduce batch size (`--batch-size 32`)
- Lower episode count for testing

**For GPU training:**
- Increase batch size (`--batch-size 128`)
- Use larger models (`--hidden-size 1024`)
- Monitor GPU memory usage

**For cloud instances:**
- Use tmux/screen for long training runs
- Set up automatic model saving
- Monitor costs and usage

## 📚 Documentation

- **[WANDB_SETUP.md](WANDB_SETUP.md)** - Complete Weights & Biases setup guide
- **[REWARD_SYSTEM.md](REWARD_SYSTEM.md)** - Reward function documentation
- **[sts_lightspeed docs](https://github.com/gamerpuppy/sts_lightspeed)** - Simulation engine documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) - Fast STS simulation engine
- [Slay the Spire](https://www.megacrit.com/) - The amazing game this is based on
- [Weights & Biases](https://wandb.ai/) - Experiment tracking platform
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

**Happy training!** 🚀 If you run into issues, please open an issue or check the troubleshooting section above.