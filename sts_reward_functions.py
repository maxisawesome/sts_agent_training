#!/usr/bin/env python3
"""
Slay the Spire Reward Functions

This module contains different reward function implementations for training
the STS neural network agent. Reward engineering is crucial for RL success.
"""

import sys
import os
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

import slaythespire

class BaseRewardFunction(ABC):
    """Base class for reward functions."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset any episode-specific state."""
        self.previous_state = None
        self.episode_rewards = []
    
    @abstractmethod
    def calculate_reward(self, game_context: slaythespire.GameContext, action: int, done: bool) -> float:
        """Calculate reward for the current state and action."""
        pass
    
    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary statistics for the current episode."""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_reward': sum(self.episode_rewards),
            'avg_reward': sum(self.episode_rewards) / len(self.episode_rewards),
            'max_reward': max(self.episode_rewards),
            'min_reward': min(self.episode_rewards),
            'num_steps': len(self.episode_rewards)
        }

class SimpleHPReward(BaseRewardFunction):
    """Simple reward based on HP preservation (current implementation)."""
    
    def __init__(self):
        super().__init__("SimpleHP")
    
    def calculate_reward(self, game_context: slaythespire.GameContext, action: int, done: bool) -> float:
        # Simple reward based on current HP ratio
        hp_ratio = game_context.cur_hp / max(game_context.max_hp, 1)
        base_reward = hp_ratio * 0.1
        
        # Small positive reward for surviving each step
        survival_reward = 0.01
        
        reward = base_reward + survival_reward
        self.episode_rewards.append(reward)
        return reward

class ComprehensiveReward(BaseRewardFunction):
    """More sophisticated reward function considering multiple factors."""
    
    def __init__(self, 
                 hp_weight: float = 1.0,
                 gold_weight: float = 0.1,
                 floor_weight: float = 0.5,
                 survival_bonus: float = 0.01,
                 death_penalty: float = -10.0,
                 win_bonus: float = 100.0):
        super().__init__("Comprehensive")
        self.hp_weight = hp_weight
        self.gold_weight = gold_weight
        self.floor_weight = floor_weight
        self.survival_bonus = survival_bonus
        self.death_penalty = death_penalty
        self.win_bonus = win_bonus
    
    def calculate_reward(self, game_context: slaythespire.GameContext, action: int, done: bool) -> float:
        reward = 0.0
        
        # Health-based reward (encourage HP preservation)
        hp_ratio = game_context.cur_hp / max(game_context.max_hp, 1)
        hp_reward = hp_ratio * self.hp_weight
        
        # Gold acquisition reward (encourage resource gathering)
        gold_reward = min(game_context.gold / 1000.0, 1.0) * self.gold_weight
        
        # Floor progression reward (encourage advancing through the spire)
        floor_reward = game_context.floor_num * self.floor_weight
        
        # Base survival bonus
        survival_reward = self.survival_bonus
        
        # Calculate delta rewards if we have previous state
        if self.previous_state is not None:
            # Reward for gaining HP
            hp_delta = game_context.cur_hp - self.previous_state.cur_hp
            if hp_delta > 0:
                reward += hp_delta * 0.5  # Bonus for healing
            
            # Reward for gaining gold
            gold_delta = game_context.gold - self.previous_state.gold
            if gold_delta > 0:
                reward += gold_delta * 0.01  # Small bonus for gold
            
            # Reward for floor progression
            floor_delta = game_context.floor_num - self.previous_state.floor_num
            if floor_delta > 0:
                reward += floor_delta * 2.0  # Good bonus for advancing floors
        
        # Terminal rewards
        if done:
            if game_context.cur_hp <= 0:
                reward += self.death_penalty  # Large penalty for dying
            elif game_context.floor_num >= 50:  # Approximate win condition
                reward += self.win_bonus  # Large bonus for winning
        
        total_reward = hp_reward + gold_reward + floor_reward + survival_reward + reward
        
        # Store state for next calculation
        self.previous_state = game_context
        self.episode_rewards.append(total_reward)
        
        return total_reward

class SparseReward(BaseRewardFunction):
    """Sparse reward that only gives feedback at key moments."""
    
    def __init__(self):
        super().__init__("Sparse")
        self.initial_hp = None
        self.initial_floor = None
    
    def reset(self):
        super().reset()
        self.initial_hp = None
        self.initial_floor = None
    
    def calculate_reward(self, game_context: slaythespire.GameContext, action: int, done: bool) -> float:
        if self.initial_hp is None:
            self.initial_hp = game_context.cur_hp
            self.initial_floor = game_context.floor_num
        
        reward = 0.0
        
        # Only give rewards at terminal states or major milestones
        if done:
            if game_context.cur_hp <= 0:
                # Death penalty proportional to how far we got
                reward = -50.0 + (game_context.floor_num - self.initial_floor) * 2.0
            elif game_context.floor_num >= 50:  # Win condition
                reward = 200.0
            else:
                # Partial success reward
                reward = (game_context.floor_num - self.initial_floor) * 5.0
        
        # Small milestone rewards
        elif self.previous_state is not None:
            floor_delta = game_context.floor_num - self.previous_state.floor_num
            if floor_delta > 0:
                reward = 10.0 * floor_delta  # Reward for floor progression
        
        self.previous_state = game_context
        self.episode_rewards.append(reward)
        return reward

class ShapedReward(BaseRewardFunction):
    """Carefully shaped reward to guide learning toward good strategies."""
    
    def __init__(self):
        super().__init__("Shaped")
        self.combat_start_hp = None
        self.in_combat = False
    
    def calculate_reward(self, game_context: slaythespire.GameContext, action: int, done: bool) -> float:
        reward = 0.0
        
        # Detect combat start/end (would need proper game state detection)
        # This is a placeholder - real implementation would check screen_state
        current_in_combat = game_context.cur_hp < game_context.max_hp  # Rough heuristic
        
        if current_in_combat and not self.in_combat:
            # Combat started
            self.combat_start_hp = game_context.cur_hp
            self.in_combat = True
        elif not current_in_combat and self.in_combat:
            # Combat ended
            if self.combat_start_hp is not None:
                hp_lost = max(0, self.combat_start_hp - game_context.cur_hp)
                # Reward efficient combat (less HP lost)
                reward += max(0, 5.0 - hp_lost)
            self.in_combat = False
        
        # Progressive reward for getting further
        floor_bonus = game_context.floor_num * 0.1
        
        # Efficiency reward (more gold per floor)
        if game_context.floor_num > 0:
            gold_efficiency = game_context.gold / (game_context.floor_num + 1)
            reward += gold_efficiency * 0.01
        
        # Health management reward
        hp_ratio = game_context.cur_hp / max(game_context.max_hp, 1)
        if hp_ratio > 0.8:
            reward += 0.1  # Bonus for staying healthy
        elif hp_ratio < 0.3:
            reward -= 0.1  # Penalty for being low on health
        
        # Terminal rewards
        if done:
            if game_context.cur_hp <= 0:
                reward -= 20.0  # Death penalty
            else:
                # Success bonus based on how far we got
                reward += game_context.floor_num * 2.0
        
        self.episode_rewards.append(reward)
        return reward

class RewardFunctionManager:
    """Manager for different reward functions."""
    
    def __init__(self):
        self.reward_functions = {
            'simple': SimpleHPReward(),
            'comprehensive': ComprehensiveReward(),
            'sparse': SparseReward(),
            'shaped': ShapedReward()
        }
        self.current_function = self.reward_functions['simple']
    
    def set_reward_function(self, name: str):
        """Set the active reward function."""
        if name in self.reward_functions:
            self.current_function = self.reward_functions[name]
            print(f"Set reward function to: {name}")
        else:
            print(f"Unknown reward function: {name}")
            print(f"Available functions: {list(self.reward_functions.keys())}")
    
    def get_reward(self, game_context: slaythespire.GameContext, action: int, done: bool) -> float:
        """Get reward using the current reward function."""
        return self.current_function.calculate_reward(game_context, action, done)
    
    def reset_episode(self):
        """Reset the current reward function for a new episode."""
        self.current_function.reset()
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get episode summary from current reward function."""
        summary = self.current_function.get_episode_summary()
        summary['reward_function'] = self.current_function.name
        return summary
    
    def compare_reward_functions(self, game_context: slaythespire.GameContext, action: int, done: bool) -> Dict[str, float]:
        """Compare rewards from all reward functions for analysis."""
        rewards = {}
        for name, func in self.reward_functions.items():
            # Create a copy of the game context for each function
            rewards[name] = func.calculate_reward(game_context, action, done)
        return rewards

def test_reward_functions():
    """Test different reward functions with sample game states."""
    print("=== Testing STS Reward Functions ===\n")
    
    # Create sample game contexts
    game_states = [
        # Initial state
        slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 12345, 67890),
        # Later in game (simulate progression)
        slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 12346, 67891),
    ]
    
    # Modify the second state to simulate progression
    # Note: This is for testing - in real usage the game engine updates these values
    print("Game State 1 (Initial):")
    print(f"  HP: {game_states[0].cur_hp}/{game_states[0].max_hp}")
    print(f"  Gold: {game_states[0].gold}")
    print(f"  Floor: {game_states[0].floor_num}")
    
    print(f"\nGame State 2 (Simulated Later):")
    print(f"  HP: {game_states[1].cur_hp}/{game_states[1].max_hp}")
    print(f"  Gold: {game_states[1].gold}")
    print(f"  Floor: {game_states[1].floor_num}")
    
    # Test reward functions
    manager = RewardFunctionManager()
    
    print(f"\n--- Reward Function Comparison ---")
    for i, state in enumerate(game_states):
        print(f"\nState {i+1} Rewards:")
        rewards = manager.compare_reward_functions(state, action=42, done=False)
        for name, reward in rewards.items():
            print(f"  {name:12}: {reward:8.4f}")
    
    # Test episode tracking
    print(f"\n--- Episode Tracking Test ---")
    manager.set_reward_function('comprehensive')
    
    # Simulate a short episode
    for step in range(5):
        reward = manager.get_reward(game_states[0], action=step, done=False)
        print(f"Step {step}: Reward = {reward:.4f}")
    
    # Final step (episode end)
    final_reward = manager.get_reward(game_states[1], action=99, done=True)
    print(f"Final step: Reward = {final_reward:.4f}")
    
    # Get episode summary
    summary = manager.get_episode_summary()
    print(f"\nEpisode Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

if __name__ == "__main__":
    test_reward_functions()