# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install package in editable mode (automatically builds sts_lightspeed)
pip install -e .
# Test the installation
python3 -c "import slaythespire; print('✅ sts_lightspeed installed successfully!')"
```

### Manual C++ Build (if needed)
```bash
cd sts_lightspeed
# Clean previous builds
rm -rf CMakeCache.txt CMakeFiles/
# Configure (adjust Python path as needed)
cmake -DPYTHON_EXECUTABLE=$(which python3) .
# Build with parallel jobs
make -j4
# Test the build
python3 -c "import slaythespire; print('✅ sts_lightspeed built successfully!')"
```

### Training and Evaluation
```bash
# Quick test run (5 episodes, ~1 minute)
python3 train_sts_agent.py train --episodes 5 --no-wandb

# Full training with Weights & Biases tracking
python3 train_sts_agent.py train --episodes 2000 --reward-function comprehensive --wandb

# Evaluate a trained model
python3 train_sts_agent.py eval sts_models/final_model.pt --games 100

# Two-network training (single environment)
python3 train_sts_agent.py train-two-net --episodes 1000

# Two-network training (parallel environments - RECOMMENDED)
python3 train_sts_agent.py train-parallel --batches 100 --num-envs 4

# Evaluate two-network model
python3 train_sts_agent.py eval-two-net sts_models/two_network_model.pt --episodes 100

# List available models
python3 train_sts_agent.py list

# Interactive play session
python3 train_sts_agent.py play --model sts_models/final_model.pt
```

### Testing and Development
```bash
# Test the neural network interface
python3 test_nn_interface.py

# Test basic functionality
python3 test.py

# Test Weights & Biases integration
python3 test_wandb.py

# Analyze reward functions
python3 simple_reward_analysis.py
```

### Docker Commands

**Quick Testing:**
```bash
# Test with cache (fast subsequent builds)
./docker-scripts/test-docker.sh

# Quick build-only test (faster, no functionality tests)
./docker-scripts/test-docker.sh --build-only

# Force clean rebuild
./docker-scripts/test-docker.sh --force-rebuild --no-cache
```

**Manual Building:**
```bash
# Build Docker image (with caching)
docker build -t sts-neural-agent .

# Run training in container
docker run --gpus all -it sts-neural-agent

# Use provided training script
./docker-scripts/train.sh --episodes 1000 --wandb-name experiment
```

**Docker Cache Optimization:**
- **First build**: Takes 10-15 minutes (downloads base image, builds C++)
- **Subsequent builds**: 30 seconds - 2 minutes (uses cached layers)
- **Python-only changes**: ~30 seconds (C++ layer cached)
- **C++ changes**: 2-5 minutes (only rebuilds C++ and later layers)
- **Dependency changes**: 5-10 minutes (rebuilds from dependency layer)

## Architecture Overview

### Core Components

**sts_lightspeed/** - High-performance C++ simulation engine
- Provides 100% RNG-accurate Slay the Spire simulation
- Speed: 1M random playouts in 5s with 16 threads
- Built with CMake, uses pybind11 for Python bindings
- Includes all enemies, relics, Ironclad cards, and colorless cards

**Neural Network Pipeline:**
- `sts_neural_network.py` - Actor-critic architectures with 412-dimensional state space
- `sts_training.py` - PPO (Proximal Policy Optimization) trainer implementation
- `sts_data_collection.py` - Environment wrapper and experience collection
- `sts_reward_functions.py` - Multiple reward function implementations
- `train_sts_agent.py` - Main CLI interface for training/evaluation

**Model Management:**
- `sts_model_manager.py` - Model saving, loading, and metadata tracking
- `sts_neural_agent.py` - Inference wrapper for trained agents
- `sts_models/` - Directory containing trained model files

### State Representation (412 dimensions)
- Player HP (current/max), Gold, Floor number
- Boss type (one-hot encoded, 10 bosses)  
- Deck composition (220 features for card counts)
- Relics (178 binary features)

### Reward Functions
- `simple` - Basic HP preservation (stable, limited learning)
- `comprehensive` - Multi-factor reward (recommended)
- `sparse` - Outcome-focused rewards
- `shaped` - Expert heuristics guided

### Dependencies
- PyTorch 2.0+ for neural networks
- Weights & Biases for experiment tracking
- NumPy, matplotlib for analysis
- CMake 3.10+ and C++17 compiler for building sts_lightspeed
- pybind11 for Python-C++ bindings (automatically installed)

## Key Files to Understand

**Configuration and Setup:**
- `requirements.txt` - Python dependencies
- `sts_lightspeed/CMakeLists.txt` - C++ build configuration
- `WANDB_SETUP.md` - Experiment tracking setup
- `REWARD_SYSTEM.md` - Detailed reward function documentation

**Training Artifacts:**
- `training_config_*.json` - Saved training configurations
- `wandb/` - Local experiment tracking data
- Model files are saved to `sts_models/` directory with timestamps

The system uses Docker for cloud deployment and includes comprehensive logging and model management for long-running experiments.

## Installation Notes

The project now uses proper Python packaging with `setup.py`. Simply run `pip install -e .` to:
- Automatically build the sts_lightspeed C++ extension
- Install all Python dependencies
- Make the `slaythespire` module available globally
- Enable console script commands like `sts-train`

This eliminates the need for `sys.path.insert()` workarounds throughout the codebase.

## Cloud Storage Setup

For cloud deployments, data persistence is critical. The system includes automated setup for major cloud providers:

**Quick Setup:**
```bash
# Auto-detect cloud provider and setup storage
./scripts/setup-cloud-storage.sh

# Provider-specific setup
./scripts/setup-cloud-storage.sh --aws --size 100    # AWS EBS
./scripts/setup-cloud-storage.sh --gcp --size 100    # GCP Persistent Disk
./scripts/setup-cloud-storage.sh --azure --size 100  # Azure Managed Disk
./scripts/setup-cloud-storage.sh --generic           # Generic directory
```

**Cloud Training:**
```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your WANDB_API_KEY and settings

# Start cloud training with persistent storage
./scripts/cloud-train.sh

# Using Docker Compose for cloud
docker-compose -f docker-compose.cloud.yml up sts-trainer
```

**Data Backup:**
```bash
# Manual backup to cloud storage
./scripts/backup-data.sh

# Automated backup (configure cloud credentials in .env)
# Supports AWS S3, Google Cloud Storage, Azure Blob Storage
```

**Important Cloud Files:**
- `docker-compose.cloud.yml` - Cloud-optimized container orchestration
- `.env.example` - Template for cloud environment configuration
- `CLOUD_STORAGE_GUIDE.md` - Detailed cloud setup documentation
- `scripts/` - Automated setup and backup scripts