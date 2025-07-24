#!/usr/bin/env python3
"""
Main Training Script for Slay the Spire Neural Network Agent

This script provides a complete pipeline for training, evaluating, and playing
with neural network agents for Slay the Spire.
"""

import argparse
import sys
import os
from datetime import datetime

# Add the sts_lightspeed directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sts_lightspeed'))
import slaythespire

from sts_training import PPOTrainer, TrainingConfig
from sts_neural_agent import STSNeuralAgent, STSNeuralGameRunner
from sts_model_manager import STSModelManager

def train_agent(config_path: str = None, **kwargs):
    """Train a new STS agent."""
    print("=== Training STS Neural Network Agent ===\n")
    
    # Load or create configuration
    if config_path and os.path.exists(config_path):
        config = TrainingConfig.load(config_path)
        print(f"Loaded configuration from: {config_path}")
    else:
        config = TrainingConfig(**kwargs)
        print("Using default configuration")
    
    print(f"Training configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Save configuration
    config_save_path = f"training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    config.save(config_save_path)
    print(f"\nConfiguration saved to: {config_save_path}")
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Start training
    print(f"\nStarting training with {sum(p.numel() for p in trainer.actor_critic.parameters())} parameters...")
    trainer.train()
    
    print("Training completed!")

def evaluate_agent(model_path: str, num_games: int = 10, temperature: float = 1.0):
    """Evaluate a trained agent."""
    print(f"=== Evaluating STS Agent ===\n")
    print(f"Model: {model_path}")
    print(f"Games: {num_games}")
    print(f"Temperature: {temperature}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Create agent
    agent = STSNeuralAgent(model_path, temperature=temperature)
    game_runner = STSNeuralGameRunner(agent)
    
    # Run evaluation games
    print(f"\nRunning {num_games} evaluation games...")
    results = game_runner.play_multiple_games(num_games, verbose=True)
    
    # Print summary
    summary = game_runner.get_performance_summary()
    print(f"\n=== Evaluation Results ===")
    print(f"Total games: {summary['total_games']}")
    print(f"Average confidence: {summary['avg_confidence']:.4f}")
    print(f"Average state value: {summary['avg_state_value']:.4f}")
    print(f"Average actions per game: {summary['avg_actions_per_game']:.1f}")

def list_models():
    """List all available trained models."""
    print("=== Available Models ===\n")
    
    manager = STSModelManager()
    models = manager.list_models()
    
    if not models:
        print("No trained models found.")
        print("Run 'python train_sts_agent.py train' to train a new model.")
        return
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['filename']}")
        print(f"   Timestamp: {model['timestamp']}")
        print(f"   Parameters: {model['total_parameters']}")
        if 'final_avg_reward' in model:
            print(f"   Final Avg Reward: {model['final_avg_reward']:.3f}")
            print(f"   Total Episodes: {model['total_episodes']}")
        print()

def interactive_play(model_path: str = None):
    """Run an interactive play session."""
    print("=== Interactive STS Agent ===\n")
    
    if model_path and os.path.exists(model_path):
        agent = STSNeuralAgent(model_path, temperature=1.0)
        print(f"Loaded neural agent from: {model_path}")
    else:
        agent = STSNeuralAgent()  # Random agent
        print("Using random agent (no model provided)")
    
    print("\nAgent statistics:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create a game context for demonstration
    game_context = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 12345, 67890)
    
    print(f"\nGame state:")
    print(f"  HP: {game_context.cur_hp}/{game_context.max_hp}")
    print(f"  Gold: {game_context.gold}")
    print(f"  Floor: {game_context.floor_num}")
    
    # Get agent recommendations
    if agent.use_neural_network:
        top_actions = agent.get_top_actions(game_context, top_k=5)
        print(f"\nTop 5 recommended actions:")
        for i, (action, prob) in enumerate(top_actions, 1):
            print(f"  {i}. Action {action}: {prob:.4f} probability")
        
        action, confidence, value = agent.get_action_with_confidence(game_context)
        print(f"\nSelected action: {action}")
        print(f"Confidence: {confidence:.4f}")
        print(f"State value estimate: {value:.4f}")
    else:
        action = agent.get_action(game_context)
        print(f"\nRandom action: {action}")

def main():
    parser = argparse.ArgumentParser(description="STS Neural Network Agent Training and Evaluation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a new agent')
    train_parser.add_argument('--config', type=str, help='Path to configuration file')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    train_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--hidden-size', type=int, default=512, help='Hidden layer size')
    train_parser.add_argument('--reward-function', type=str, default='simple', 
                            choices=['simple', 'comprehensive', 'sparse', 'shaped'],
                            help='Reward function to use')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained agent')
    eval_parser.add_argument('model', type=str, help='Path to model file')
    eval_parser.add_argument('--games', type=int, default=10, help='Number of evaluation games')
    eval_parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for action selection')
    
    # List models command
    subparsers.add_parser('list', help='List available models')
    
    # Interactive play command
    play_parser = subparsers.add_parser('play', help='Interactive play session')
    play_parser.add_argument('--model', type=str, help='Path to model file (optional)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_kwargs = {
            'num_episodes': args.episodes,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'hidden_size': args.hidden_size,
            'reward_function': args.reward_function
        }
        train_agent(args.config, **train_kwargs)
    
    elif args.command == 'eval':
        evaluate_agent(args.model, args.games, args.temperature)
    
    elif args.command == 'list':
        list_models()
    
    elif args.command == 'play':
        interactive_play(args.model)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()