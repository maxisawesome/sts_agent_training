#!/usr/bin/env python3
"""
Test script for the decision logging system.
Runs a short training session with logging enabled to verify the system works.
"""

import sys
import os
sys.path.insert(0, 'sts_lightspeed')

import torch
import json
from datetime import datetime

from sts_training import TrainingConfig, PPOTrainer

def test_decision_logging():
    """Test the decision logging system with a minimal training run."""
    print("=== Testing Decision Logging System ===\n")
    
    # Create a configuration with logging enabled
    config = TrainingConfig(
        # Minimal training setup
        num_episodes=4,
        collect_episodes_per_update=2,
        batch_size=16,
        update_epochs=1,
        log_interval=1,
        save_interval=10,
        
        # Enable decision logging
        enable_decision_logging=True,
        log_directory='test_logs',
        log_level='INFO',
        log_every_n_steps=1,  # Log every step for testing
        
        # Disable wandb for testing
        use_wandb=False
    )
    
    print(f"Configuration:")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Logging enabled: {config.enable_decision_logging}")
    print(f"  Log directory: {config.log_directory}")
    print(f"  Log every N steps: {config.log_every_n_steps}")
    print()
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = PPOTrainer(config)
    print(f"Trainer initialized successfully")
    print(f"Model parameters: {sum(p.numel() for p in trainer.actor_critic.parameters()):,}")
    print()
    
    # Run short training session
    print("Starting test training session...")
    trainer.train()
    
    # Check if log files were created
    log_dir = config.log_directory
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.startswith('decisions_') and f.endswith('.jsonl')]
        
        if log_files:
            print(f"\n✅ Log files created in {log_dir}:")
            for log_file in log_files:
                file_path = os.path.join(log_dir, log_file)
                file_size = os.path.getsize(file_path)
                print(f"  - {log_file} ({file_size} bytes)")
                
                # Read and analyze a few log entries
                print(f"\nAnalyzing {log_file}:")
                analyze_log_file(file_path)
        else:
            print(f"❌ No log files found in {log_dir}")
    else:
        print(f"❌ Log directory {log_dir} was not created")

def analyze_log_file(file_path: str, max_entries: int = 3):
    """Analyze the contents of a log file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"  Total log entries: {len(lines)}")
        
        # Analyze first few entries
        decision_count = 0
        episode_summary_count = 0
        
        for i, line in enumerate(lines[:max_entries]):
            try:
                entry = json.loads(line.strip())
                entry_type = entry.get('type', 'decision')
                
                if entry_type == 'episode_summary':
                    episode_summary_count += 1
                    print(f"  Entry {i+1}: Episode summary - Episode {entry.get('episode')}, "
                          f"Reward: {entry.get('total_reward', 0):.2f}, "
                          f"Length: {entry.get('episode_length', 0)}")
                else:
                    decision_count += 1
                    episode = entry.get('episode', 'N/A')
                    step = entry.get('step', 'N/A')
                    chosen_action = entry.get('model_output', {}).get('chosen_action', 'N/A')
                    state_value = entry.get('model_output', {}).get('state_value', 'N/A')
                    
                    print(f"  Entry {i+1}: Decision - Episode {episode}, Step {step}, "
                          f"Action: {chosen_action}, Value: {state_value}")
                    
                    # Show available choices if present
                    choices = entry.get('available_choices', [])
                    if choices:
                        print(f"    Available choices: {len(choices)} options")
                        for j, choice in enumerate(choices[:2]):  # Show first 2 choices
                            choice_types = choice.get('choice_types', {})
                            active_types = [k for k, v in choice_types.items() if v]
                            print(f"      Choice {j}: {', '.join(active_types) if active_types else 'No active types'}")
                    
            except json.JSONDecodeError as e:
                print(f"  Entry {i+1}: Invalid JSON - {str(e)}")
        
        # Count all entries by type
        total_decisions = 0
        total_summaries = 0
        
        for line in lines:
            try:
                entry = json.loads(line.strip())
                if entry.get('type') == 'episode_summary':
                    total_summaries += 1
                else:
                    total_decisions += 1
            except:
                pass
        
        print(f"  Total decisions logged: {total_decisions}")
        print(f"  Total episode summaries: {total_summaries}")
        
    except Exception as e:
        print(f"  Error analyzing log file: {str(e)}")

if __name__ == "__main__":
    test_decision_logging()