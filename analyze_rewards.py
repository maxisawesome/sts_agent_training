#!/usr/bin/env python3
"""
Reward Function Analysis Tool

This script helps analyze and compare different reward functions
to understand their behavior and tune them for better training.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

import slaythespire

from sts_reward_functions import RewardFunctionManager
from sts_data_collection import STSEnvironmentWrapper, STSDataCollector, ExperienceBuffer

def simulate_game_progression():
    """Create a simulated game progression for reward analysis."""
    # Create multiple game states representing different scenarios
    scenarios = []
    
    # Scenario 1: Healthy start
    gc1 = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 100, 200)
    scenarios.append(("Healthy Start", gc1))
    
    # Scenario 2: Low health
    gc2 = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 101, 201)
    # Simulate damage (we can't actually modify the C++ object easily, but we can analyze current state)
    scenarios.append(("Low Health", gc2))
    
    # Scenario 3: Different character
    gc3 = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 102, 202)
    scenarios.append(("Different Seed", gc3))
    
    return scenarios

def analyze_reward_functions():
    """Analyze different reward functions across various game states."""
    print("=== Reward Function Analysis ===\n")
    
    # Get game scenarios
    scenarios = simulate_game_progression()
    
    # Create reward manager
    reward_manager = RewardFunctionManager()
    
    # Analyze each reward function
    functions = ['simple', 'comprehensive', 'sparse', 'shaped']
    results = {func: [] for func in functions}
    
    print("Scenario Analysis:")
    print("-" * 80)
    print(f"{'Scenario':<15} {'Simple':<10} {'Comprehensive':<15} {'Sparse':<10} {'Shaped':<10}")
    print("-" * 80)
    
    for scenario_name, game_context in scenarios:
        scenario_rewards = {}
        
        for func_name in functions:
            reward_manager.set_reward_function(func_name)
            reward = reward_manager.get_reward(game_context, action=42, done=False)
            scenario_rewards[func_name] = reward
            results[func_name].append(reward)
        
        print(f"{scenario_name:<15} {scenario_rewards['simple']:<10.4f} "
              f"{scenario_rewards['comprehensive']:<15.4f} {scenario_rewards['sparse']:<10.4f} "
              f"{scenario_rewards['shaped']:<10.4f}")
    
    return results

def analyze_episode_rewards():
    """Analyze rewards over a complete episode simulation."""
    print(f"\n=== Episode Reward Analysis ===\n")
    
    # Create environment with different reward functions
    functions = ['simple', 'comprehensive', 'sparse', 'shaped']
    episode_data = {}
    
    for func_name in functions:
        print(f"Analyzing {func_name} reward function...")
        
        env = STSEnvironmentWrapper(reward_function=func_name)
        buffer = ExperienceBuffer()
        collector = STSDataCollector(env, buffer)
        
        # Collect a few episodes
        episodes = collector.collect_random_episodes(3)
        
        episode_rewards = [ep.total_reward for ep in episodes]
        episode_lengths = [ep.episode_length for ep in episodes]
        
        episode_data[func_name] = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths)
        }
        
        print(f"  Avg Reward: {episode_data[func_name]['avg_reward']:.4f} ± {episode_data[func_name]['std_reward']:.4f}")
        print(f"  Avg Length: {episode_data[func_name]['avg_length']:.1f}")
    
    return episode_data

def plot_reward_comparison(episode_data: Dict[str, Any]):
    """Create plots comparing reward functions."""
    try:
        import matplotlib.pyplot as plt
        
        print(f"\n=== Creating Reward Comparison Plots ===")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Reward Function Comparison', fontsize=16)
        
        functions = list(episode_data.keys())
        colors = ['blue', 'red', 'green', 'orange']
        
        # Plot 1: Average Rewards
        avg_rewards = [episode_data[func]['avg_reward'] for func in functions]
        std_rewards = [episode_data[func]['std_reward'] for func in functions]
        
        ax1.bar(functions, avg_rewards, yerr=std_rewards, color=colors, alpha=0.7)
        ax1.set_title('Average Episode Rewards')
        ax1.set_ylabel('Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Reward Distribution
        all_rewards = []
        labels = []
        for func in functions:
            all_rewards.extend(episode_data[func]['rewards'])
            labels.extend([func] * len(episode_data[func]['rewards']))
        
        reward_by_func = [episode_data[func]['rewards'] for func in functions]
        ax2.boxplot(reward_by_func, labels=functions)
        ax2.set_title('Reward Distribution')
        ax2.set_ylabel('Reward')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Episode Lengths
        avg_lengths = [episode_data[func]['avg_length'] for func in functions]
        ax3.bar(functions, avg_lengths, color=colors, alpha=0.7)
        ax3.set_title('Average Episode Length')
        ax3.set_ylabel('Steps')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Reward per Step
        reward_per_step = [episode_data[func]['avg_reward'] / episode_data[func]['avg_length'] 
                          for func in functions]
        ax4.bar(functions, reward_per_step, color=colors, alpha=0.7)
        ax4.set_title('Average Reward per Step')
        ax4.set_ylabel('Reward/Step')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reward_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plots saved as 'reward_analysis.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping plots")
        print("Install with: pip install matplotlib")

def recommend_reward_function(episode_data: Dict[str, Any]) -> str:
    """Recommend the best reward function based on analysis."""
    print(f"\n=== Reward Function Recommendation ===")
    
    # Score each function based on different criteria
    scores = {}
    
    for func_name, data in episode_data.items():
        score = 0
        
        # Prefer higher average rewards
        normalized_reward = data['avg_reward'] / max([d['avg_reward'] for d in episode_data.values()])
        score += normalized_reward * 0.4
        
        # Prefer lower variance (more stable)
        if data['std_reward'] > 0:
            stability = 1 / (1 + data['std_reward'])
            score += stability * 0.3
        else:
            score += 0.3
        
        # Prefer reasonable episode lengths (not too short or too long)
        ideal_length = 500  # Arbitrary ideal
        length_score = 1 - abs(data['avg_length'] - ideal_length) / ideal_length
        score += max(0, length_score) * 0.3
        
        scores[func_name] = score
        
        print(f"{func_name}:")
        print(f"  Normalized Reward: {normalized_reward:.3f}")
        print(f"  Stability Score: {stability if data['std_reward'] > 0 else 1.0:.3f}")
        print(f"  Length Score: {max(0, length_score):.3f}")
        print(f"  Total Score: {score:.3f}")
        print()
    
    # Find best function
    best_function = max(scores.keys(), key=lambda k: scores[k])
    print(f"Recommended reward function: {best_function}")
    print(f"Score: {scores[best_function]:.3f}")
    
    return best_function

def main():
    """Main analysis function."""
    print("Starting comprehensive reward function analysis...\n")
    
    # Basic analysis
    scenario_results = analyze_reward_functions()
    
    # Episode analysis
    episode_data = analyze_episode_rewards()
    
    # Create plots
    plot_reward_comparison(episode_data)
    
    # Get recommendation
    recommended = recommend_reward_function(episode_data)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Recommended reward function: {recommended}")
    print(f"\nTo use in training:")
    print(f"python3 train_sts_agent.py train --reward-function {recommended}")

if __name__ == "__main__":
    main()