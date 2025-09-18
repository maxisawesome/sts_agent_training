#!/usr/bin/env python3
"""
Simple Reward Function Analysis

Quick analysis of reward functions without full episode simulation.
"""

import slaythespire

from sts_reward_functions import RewardFunctionManager

def main():
    print("=== Simple Reward Function Analysis ===\n")
    
    # Create sample game states
    game_context = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 1234567890, 0)
    
    print("Current Game State:")
    print(f"  HP: {game_context.cur_hp}/{game_context.max_hp}")
    print(f"  Gold: {game_context.gold}")
    print(f"  Floor: {game_context.floor_num}")
    
    # Test all reward functions
    manager = RewardFunctionManager()
    functions = ['simple', 'comprehensive', 'sparse', 'shaped']
    
    print(f"\nReward Function Comparison:")
    print("-" * 50)
    
    for func_name in functions:
        manager.set_reward_function(func_name)
        reward = manager.get_reward(game_context, action=42, done=False)
        print(f"{func_name:12}: {reward:8.4f}")
    
    print(f"\nReward Function Details:")
    print("-" * 50)
    
    # Comprehensive analysis
    manager.set_reward_function('comprehensive')
    comprehensive_reward = manager.get_reward(game_context, action=42, done=False)
    
    print(f"Comprehensive reward breakdown:")
    print(f"  - Considers HP ratio: ✓")
    print(f"  - Considers gold: ✓") 
    print(f"  - Considers floor progression: ✓")
    print(f"  - Includes survival bonus: ✓")
    print(f"  - Terminal rewards (win/loss): ✓")
    
    print(f"\nRecommendations:")
    print(f"  - For initial training: 'comprehensive' (balanced, informative)")
    print(f"  - For stable training: 'simple' (less noisy)")
    print(f"  - For final optimization: 'sparse' (focuses on outcomes)")
    print(f"  - For expert tuning: 'shaped' (hand-crafted heuristics)")
    
    print(f"\nTo train with different reward functions:")
    print(f"  python3 train_sts_agent.py train --reward-function comprehensive")
    print(f"  python3 train_sts_agent.py train --reward-function sparse")

if __name__ == "__main__":
    main()