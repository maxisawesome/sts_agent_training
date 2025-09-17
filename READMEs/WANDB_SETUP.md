# Weights & Biases Integration for STS Neural Agent

This guide explains how to use Weights & Biases (wandb) for experiment tracking with the STS neural network training system.

## 🚀 Quick Start

1. **Install wandb**:
   ```bash
   pip install wandb
   ```

2. **Set up authentication**:
   ```bash
   python3 setup_wandb.py
   ```

3. **Start training with tracking**:
   ```bash
   python3 train_sts_agent.py train --episodes 1000 --wandb
   ```

4. **View results**: Visit https://wandb.ai to see your experiments!

## 📊 What Gets Tracked

### Training Metrics
- **Rewards**: Average, min, max, std, distribution histograms
- **Episode Length**: Mean episode duration and variability
- **Loss Functions**: Policy loss, value loss, entropy
- **Performance**: Training FPS, collection time, update time
- **Learning Progress**: Improvement over time, reward trends

### Model Information
- **Architecture**: Total and trainable parameters
- **Gradients**: Gradient norms and distributions (via `wandb.watch`)
- **Checkpoints**: Automatic model saving to wandb

### Hyperparameters
- All training configuration automatically logged
- Easy comparison between different experiment settings

## 🎛️ Configuration Options

### Command Line Arguments

```bash
# Basic usage
python3 train_sts_agent.py train --episodes 1000 --wandb

# Custom project and naming
python3 train_sts_agent.py train \
  --episodes 2000 \
  --wandb-project my-sts-experiments \
  --wandb-name experiment-baseline \
  --wandb-entity your-username

# With tags for organization
python3 train_sts_agent.py train \
  --episodes 1000 \
  --reward-function comprehensive \
  --wandb-tags baseline comprehensive-reward first-attempt

# Disable wandb
python3 train_sts_agent.py train --episodes 1000 --no-wandb
```

### Configuration File

You can also set wandb options in your training configuration:

```python
config = TrainingConfig(
    # Training parameters
    num_episodes=1000,
    learning_rate=3e-4,
    reward_function='comprehensive',
    
    # Wandb configuration
    use_wandb=True,
    wandb_project='sts-neural-agent',
    wandb_entity='your-username',  # Optional
    wandb_run_name='experiment-1',  # Optional, auto-generated if None
    wandb_tags=['baseline', 'comprehensive-reward']  # Optional
)
```

## 📈 Experiment Organization

### Projects
Organize experiments by major themes:
- `sts-neural-agent` - Main development
- `sts-reward-functions` - Reward function experiments
- `sts-architecture` - Network architecture tests

### Tags
Use tags to categorize runs:
- `baseline` - Initial experiments
- `comprehensive-reward` - Using comprehensive reward function
- `sparse-reward` - Using sparse reward function
- `architecture-test` - Testing different network sizes
- `hyperparameter-sweep` - Grid search experiments

### Run Names
Use descriptive names:
- `baseline-simple-reward-20241224`
- `comprehensive-reward-512-hidden`
- `sparse-reward-long-training`

## 🔍 Monitoring Training

### Key Metrics to Watch

1. **`training/avg_reward`** - Primary success metric
2. **`training/policy_loss`** - Should decrease over time
3. **`training/value_loss`** - Should stabilize
4. **`episodes/reward_distribution`** - Should shift toward higher rewards
5. **`training/learning_progress`** - Improvement over time

### Dashboard Setup

Create custom dashboards in wandb to monitor:
- Reward trends across multiple runs
- Loss function comparisons
- Training efficiency metrics
- Model performance correlation with hyperparameters

## 🛠️ Advanced Features

### Model Versioning
```python
# Models are automatically saved to wandb
# Access them later via wandb API:
import wandb
api = wandb.Api()
run = api.run("your-entity/sts-neural-agent/run-id")
run.file("final_model.pt").download()
```

### Hyperparameter Sweeps
Create a sweep configuration:

```yaml
# sweep.yaml
program: train_sts_agent.py
method: grid
parameters:
  command:
    value: train
  learning_rate:
    values: [1e-4, 3e-4, 1e-3]
  reward_function:
    values: ['simple', 'comprehensive', 'sparse']
  hidden_size:
    values: [256, 512, 1024]
```

Run the sweep:
```bash
wandb sweep sweep.yaml
wandb agent your-entity/sts-neural-agent/sweep-id
```

### Artifact Tracking
Models and datasets are automatically tracked as artifacts:
- Model checkpoints at save intervals
- Final trained models
- Configuration files

## 🐛 Troubleshooting

### Common Issues

1. **"wandb not installed"**
   ```bash
   pip install wandb
   ```

2. **Authentication errors**
   ```bash
   wandb login
   # Or run: python3 setup_wandb.py
   ```

3. **Permission denied for project**
   - Check your wandb entity (username/team)
   - Ensure project name doesn't conflict with existing projects

4. **Runs not appearing**
   - Check internet connection
   - Verify wandb credentials: `wandb status`

### Debugging Mode
Enable debug logging:
```bash
export WANDB_MODE=debug
python3 train_sts_agent.py train --episodes 50
```

### Offline Mode
For development without internet:
```bash
export WANDB_MODE=offline
python3 train_sts_agent.py train --episodes 50
# Sync later with: wandb sync
```

## 📊 Analysis Examples

### Compare Reward Functions
```python
import wandb
api = wandb.Api()

# Get runs with different reward functions
runs = api.runs("your-entity/sts-neural-agent", 
               filters={"config.reward_function": {"$in": ["simple", "comprehensive"]}})

# Plot comparison
import matplotlib.pyplot as plt
for run in runs:
    history = run.scan_history(keys=["training/avg_reward"])
    rewards = [row["training/avg_reward"] for row in history if row["training/avg_reward"]]
    plt.plot(rewards, label=f"{run.name} ({run.config['reward_function']})")

plt.legend()
plt.xlabel("Training Updates")
plt.ylabel("Average Reward")
plt.title("Reward Function Comparison")
plt.show()
```

### Best Model Selection
```python
# Find best performing run
best_run = min(api.runs("your-entity/sts-neural-agent"), 
               key=lambda run: run.summary.get("training/final_avg_reward", float('-inf')))

print(f"Best run: {best_run.name}")
print(f"Final reward: {best_run.summary['training/final_avg_reward']}")
print(f"Config: {best_run.config}")
```

## 🎯 Best Practices

1. **Consistent Naming**: Use clear, descriptive run names
2. **Tag Everything**: Use tags liberally for easy filtering
3. **Monitor Early**: Check metrics after first few updates
4. **Save Configs**: Always save your configuration files
5. **Document Changes**: Use run notes to document what changed
6. **Regular Cleanup**: Archive or delete unsuccessful experiments

## 🔗 Useful Links

- [Wandb Documentation](https://docs.wandb.ai/)
- [PyTorch Integration](https://docs.wandb.ai/guides/integrations/pytorch)
- [Hyperparameter Sweeps](https://docs.wandb.ai/guides/sweeps)
- [Model Versioning](https://docs.wandb.ai/guides/artifacts)

---

**Happy Experimenting!** 🚀 Your STS neural agent training is now fully tracked and ready for serious experimentation.