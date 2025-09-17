# Slay the Spire Reward System

This document explains the reward function system implemented for training the STS neural network agent.

## Overview

The reward function is critical for reinforcement learning success. It defines what behavior we want the agent to learn by providing feedback signals during training. Our system supports multiple reward functions that can be easily swapped and compared.

## Location of Reward Functions

- **Main implementation**: `sts_reward_functions.py`
- **Integration**: `sts_data_collection.py` (lines 94-96)
- **Configuration**: `sts_training.py` (TrainingConfig.reward_function)
- **Analysis tools**: `simple_reward_analysis.py`, `analyze_rewards.py`

## Available Reward Functions

### 1. Simple HP Reward (`simple`)
**Current basic implementation**
```python
hp_ratio = game_context.cur_hp / max(game_context.max_hp, 1)
reward = hp_ratio * 0.1 + 0.01  # survival bonus
```

**Characteristics:**
- ✅ Stable and predictable
- ✅ Easy to understand
- ❌ Limited learning signal
- ❌ Doesn't encourage progression

**Use case:** Initial testing, baseline comparisons

### 2. Comprehensive Reward (`comprehensive`)
**Multi-factor reward considering all game aspects**

**Components:**
- **Health preservation**: `hp_ratio * 1.0`
- **Gold acquisition**: `min(gold/1000, 1.0) * 0.1`
- **Floor progression**: `floor_num * 0.5`
- **Survival bonus**: `0.01` per step
- **Delta rewards**: Bonuses for gaining HP, gold, advancing floors
- **Terminal rewards**: `-10.0` for death, `+100.0` for winning

**Characteristics:**
- ✅ Rich learning signal
- ✅ Encourages multiple good behaviors
- ✅ Balanced between survival and progression
- ❌ More complex, potentially noisy
- ❌ Requires tuning of weights

**Use case:** Main training, balanced agent development

### 3. Sparse Reward (`sparse`)
**Minimal feedback, only at key moments**

**Components:**
- **Death penalty**: `-50.0 + floor_progress * 2.0`
- **Win bonus**: `+200.0`
- **Floor milestone**: `+10.0` per floor advanced
- **No step-by-step feedback**

**Characteristics:**
- ✅ Focuses on outcomes
- ✅ Less noisy signal
- ✅ Closer to real game objectives
- ❌ Slower learning initially
- ❌ Requires longer episodes for signal

**Use case:** Advanced training, final optimization

### 4. Shaped Reward (`shaped`)
**Hand-crafted heuristics for strategic play**

**Components:**
- **Combat efficiency**: Rewards for winning with less HP loss
- **Health management**: Bonuses for staying healthy, penalties for low HP
- **Progressive rewards**: Increasing bonuses for advancement
- **Gold efficiency**: Rewards based on gold per floor

**Characteristics:**
- ✅ Incorporates domain expertise
- ✅ Encourages strategic play
- ✅ Good for specific behaviors
- ❌ Requires expert knowledge
- ❌ May limit exploration

**Use case:** Expert-guided training, specific strategy development

## How to Use Different Reward Functions

### In Training
```bash
# Use comprehensive reward (recommended for most cases)
python3 train_sts_agent.py train --reward-function comprehensive --episodes 1000

# Use sparse reward for outcome-focused training
python3 train_sts_agent.py train --reward-function sparse --episodes 2000

# Use shaped reward for strategic behavior
python3 train_sts_agent.py train --reward-function shaped --episodes 1000
```

### In Code
```python
# Create environment with specific reward function
env = STSEnvironmentWrapper(reward_function='comprehensive')

# Or change reward function in training config
config = TrainingConfig(reward_function='comprehensive')
```

### Analysis
```bash
# Quick analysis of all reward functions
python3 simple_reward_analysis.py

# Comprehensive analysis with episodes and plots
python3 analyze_rewards.py
```

## Reward Engineering Guidelines

### 1. **Reward Shaping Principles**
- **Alignment**: Rewards should align with actual game objectives
- **Density**: Provide enough signal for learning, but not too noisy
- **Scale**: Keep reward magnitudes reasonable (avoid extreme values)
- **Consistency**: Rewards should be consistent across similar states

### 2. **Common Pitfalls**
- **Reward hacking**: Agent finds unintended ways to maximize reward
- **Local optima**: Reward function creates suboptimal behaviors
- **Sparse rewards**: Too little feedback leads to slow learning
- **Dense rewards**: Too much feedback can be noisy and misleading

### 3. **Tuning Process**
1. Start with simple, interpretable rewards
2. Gradually add complexity based on observed behaviors
3. Monitor for unintended consequences
4. Compare different functions empirically
5. Use domain knowledge to guide reward design

## Implementing Custom Reward Functions

To create a new reward function:

1. **Inherit from BaseRewardFunction**:
```python
class MyCustomReward(BaseRewardFunction):
    def __init__(self):
        super().__init__("MyCustom")
    
    def calculate_reward(self, game_context, action, done):
        # Your reward logic here
        reward = 0.0
        # ... calculate reward based on game_context
        self.episode_rewards.append(reward)
        return reward
```

2. **Register in RewardFunctionManager**:
```python
# In sts_reward_functions.py
self.reward_functions['mycustom'] = MyCustomReward()
```

3. **Add to choices in training script**:
```python
# In train_sts_agent.py
choices=['simple', 'comprehensive', 'sparse', 'shaped', 'mycustom']
```

## Current Limitations & Future Work

### Limitations
1. **Action rewards**: Current system doesn't consider specific actions taken
2. **Game state detection**: Limited ability to detect combat vs non-combat states
3. **Deck evaluation**: No direct assessment of deck quality or synergies
4. **Long-term planning**: Rewards mostly focus on immediate state

### Future Improvements
1. **Context-aware rewards**: Different rewards for combat vs exploration
2. **Deck analysis**: Rewards based on deck synergy and power level
3. **Strategic rewards**: Bonuses for good long-term decisions
4. **Adaptive rewards**: Rewards that change based on training progress
5. **Curiosity rewards**: Bonuses for exploring new states or strategies

## Performance Analysis

Based on initial testing:

- **Simple**: Stable but limited learning
- **Comprehensive**: Best balance for most use cases (~926% higher rewards than simple)
- **Sparse**: Good for final optimization (0 reward until milestones)
- **Shaped**: Domain-specific, requires tuning

**Recommendation**: Start with `comprehensive` for initial training, then experiment with `sparse` for fine-tuning once the agent shows basic competency.

## Questions & Debugging

### "My agent isn't learning properly"
1. Check reward magnitudes - are they too large/small?
2. Analyze reward distribution - is there enough signal?
3. Look for reward hacking - is the agent exploiting the reward?
4. Try a different reward function

### "Rewards seem inconsistent"
1. Check if previous_state is being updated correctly
2. Verify game state is changing as expected
3. Look at reward function logic for edge cases

### "Want to understand what the agent is optimizing for"
```bash
python3 simple_reward_analysis.py  # Quick overview
python3 analyze_rewards.py         # Detailed analysis
```

The reward system is the heart of what your agent will learn - invest time in getting it right!