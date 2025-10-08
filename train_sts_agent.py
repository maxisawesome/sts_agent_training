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

# Temporary: will be fixed when proper packaging works
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sts_lightspeed'))
import slaythespire

from sts_training import PPOTrainer, TrainingConfig
from sts_neural_agent import STSNeuralAgent, STSNeuralGameRunner
from sts_model_manager import STSModelManager
from sts_two_network_trainer import TwoNetworkTrainer, TwoNetworkTrainingConfig
from sts_parallel_trainer import ParallelTwoNetworkTrainer, ParallelTwoNetworkTrainingConfig
from sts_data_collection import STSEnvironmentWrapper

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

def interactive_play(model_path: str = None, episodes: int = 1):
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
    game_context = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 1234567890, 0)
    
    step = 0
    while step < episodes and not (game_context.cur_hp <= 0 or game_context.outcome != slaythespire.GameOutcome.UNDECIDED):
        print(f"\n=== Step {step + 1} ===")
        print(f"HP: {game_context.cur_hp}/{game_context.max_hp}")
        print(f"Gold: {game_context.gold}")
        print(f"Floor: {game_context.floor_num}")
        print(f"Screen: {game_context.screen_state}")

        # Get agent action (now returns generic action)
        generic_action = agent.get_action(game_context)
        if generic_action == -1:
            print("Agent returned invalid action - stopping")
            break

        # Get action description
        action_desc = agent.get_action_description(game_context, generic_action)
        print(f"Agent selected {action_desc}")

        # Show agent recommendations if using neural network
        if agent.use_neural_network:
            action_idx_conf, confidence, value = agent.get_action_with_confidence(game_context)
            print(f"Confidence: {confidence:.4f}")
            print(f"State value estimate: {value:.4f}")

        # Execute the action
        success = agent.execute_action(game_context, generic_action)
        if success:
            print("✓ Action executed successfully")
        else:
            print("✗ Action execution failed")
            break

        step += 1

    print(f"\n=== Game Complete ===")
    print(f"Final HP: {game_context.cur_hp}/{game_context.max_hp}")
    print(f"Final Gold: {game_context.gold}")
    print(f"Final Floor: {game_context.floor_num}")
    print(f"Game Outcome: {game_context.outcome}")
    print(f"Steps taken: {step}")


def train_two_network_agent(episodes: int = 1000, save_path: str = None, **kwargs):
    """Train the two-network architecture agent."""
    print("=== Training Two-Network STS Agent ===\n")

    # Create configuration
    config = TwoNetworkTrainingConfig()

    # Override config with provided arguments
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    print(f"Training configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Memory size: {config.memory_size}")
    print(f"  Min experiences before training: {config.min_experiences_before_training}")

    # Initialize trainer
    trainer = TwoNetworkTrainer(config)

    print(f"\nNetwork Architecture:")
    print(f"  Events network parameters: {sum(p.numel() for p in trainer.events_network.parameters()):,}")
    print(f"  Combat network parameters: {sum(p.numel() for p in trainer.combat_network.parameters()):,}")
    print(f"  Shared embedding parameters: {sum(p.numel() for module in trainer.shared_embeddings.get_modules().values() for p in module.parameters()):,}")

    # Initialize environment
    env = STSEnvironmentWrapper()

    # Training loop
    print(f"\nStarting two-network training...")

    episode_rewards = []
    for episode in range(episodes):
        result = trainer.train_episode(env)
        episode_rewards.append(result['episode_reward'])

        # Print progress
        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            print(f"Episode {episode + 1:4d}/{episodes}")
            print(f"  Avg reward (last 50): {sum(recent_rewards)/len(recent_rewards):6.3f}")
            print(f"  Episode steps: {result['episode_steps']:3d}")
            print(f"  Events actions: {result['events_actions']:3d}")
            print(f"  Combat actions: {result['combat_actions']:3d}")
            print(f"  Buffer size: {result['buffer_size']:5d} (E:{result['events_buffer_size']}, C:{result['combat_buffer_size']})")
            print()

        # Save model periodically
        if save_path and (episode + 1) % config.save_frequency == 0:
            episode_save_path = save_path.replace('.pt', f'_episode_{episode + 1}.pt')
            trainer.save_model(episode_save_path)
            print(f"Model saved to: {episode_save_path}")

    # Final save
    if save_path:
        trainer.save_model(save_path)
        print(f"\nFinal model saved to: {save_path}")

    # Evaluation
    print(f"\nRunning final evaluation...")
    eval_results = trainer.evaluate(env, episodes=10)
    print(f"Evaluation results over 10 episodes:")
    print(f"  Average reward: {eval_results['avg_reward']:.3f} ± {eval_results['std_reward']:.3f}")
    print(f"  Average steps: {eval_results['avg_steps']:.1f}")
    print(f"  Events actions per episode: {eval_results['avg_events_actions']:.1f}")
    print(f"  Combat actions per episode: {eval_results['avg_combat_actions']:.1f}")

    print(f"\n✓ Two-network training complete!")


def train_two_network_parallel(batches: int = 100, save_path: str = None, **kwargs):
    """Train the two-network architecture using parallel environments."""
    print("=== Training Two-Network STS Agent (Parallel) ===\n")

    # Create configuration
    config = ParallelTwoNetworkTrainingConfig()

    # Override config with provided arguments
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    print(f"Parallel training configuration:")
    print(f"  Batches: {batches}")
    print(f"  Parallel environments: {config.num_parallel_envs}")
    print(f"  Batch collection steps: {config.batch_collection_steps}")
    print(f"  Total steps per batch: {config.num_parallel_envs * config.batch_collection_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Memory size: {config.memory_size}")

    # Initialize trainer
    trainer = ParallelTwoNetworkTrainer(config)

    print(f"\nNetwork Architecture:")
    print(f"  Events network parameters: {sum(p.numel() for p in trainer.events_network.parameters()):,}")
    print(f"  Combat network parameters: {sum(p.numel() for p in trainer.combat_network.parameters()):,}")
    print(f"  Shared embedding parameters: {sum(p.numel() for module in trainer.shared_embeddings.get_modules().values() for p in module.parameters()):,}")

    print(f"\nStarting parallel training...")

    # Training with parallel environments
    results = trainer.train_parallel_episodes(num_batches=batches)

    # Final save
    if save_path:
        trainer.save_model(save_path)
        print(f"\nFinal model saved to: {save_path}")

    # Final statistics
    print(f"\nParallel Training Results:")
    print(f"  Total batches: {results['total_batches']}")
    print(f"  Total experiences collected: {results['total_experiences']:,}")
    print(f"  Total episodes completed: {results['total_episodes']:,}")
    print(f"  Overall average reward: {results['overall_avg_reward']:.3f}")
    print(f"  Total training time: {results['total_time']:.1f}s")
    print(f"  Experiences per second: {results['experiences_per_second']:.1f}")
    print(f"  Episodes per hour: {results['total_episodes'] / (results['total_time'] / 3600):.1f}")

    # Cleanup
    trainer.close()

    print(f"\n✓ Parallel two-network training complete!")


def evaluate_two_network_agent(model_path: str, episodes: int = 100):
    """Evaluate a trained two-network agent."""
    print(f"=== Evaluating Two-Network Agent ===\n")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")

    # Initialize trainer and load model
    config = TwoNetworkTrainingConfig()
    trainer = TwoNetworkTrainer(config)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    trainer.load_model(model_path)
    print("Model loaded successfully")

    # Initialize environment
    env = STSEnvironmentWrapper()

    # Run evaluation
    print(f"\nRunning evaluation...")
    results = trainer.evaluate(env, episodes=episodes)

    print(f"\nEvaluation Results:")
    print(f"  Episodes: {results['episodes']}")
    print(f"  Average reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Average episode length: {results['avg_steps']:.1f}")
    print(f"  Average events actions: {results['avg_events_actions']:.1f}")
    print(f"  Average combat actions: {results['avg_combat_actions']:.1f}")

    # Show detailed results for first few episodes
    print(f"\nDetailed results (first 5 episodes):")
    for i, result in enumerate(results['detailed_results'][:5]):
        print(f"  Episode {i+1}: Reward={result['reward']:6.3f}, Steps={result['steps']:3d}, "
              f"HP={result['final_hp']:2d}, Floor={result['final_floor']:2d}")


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
    train_parser.add_argument('--ascension', type=int, default=0, help='Ascension level (0-20)')
    train_parser.add_argument('--wandb', action='store_true', default=True, help='Enable wandb tracking')
    train_parser.add_argument('--no-wandb', dest='wandb', action='store_false', help='Disable wandb tracking')
    train_parser.add_argument('--wandb-project', type=str, default='sts-neural-agent', help='Wandb project name')
    train_parser.add_argument('--wandb-entity', type=str, help='Wandb entity (username/team)')
    train_parser.add_argument('--wandb-name', type=str, help='Wandb run name')
    train_parser.add_argument('--wandb-tags', type=str, nargs='+', help='Wandb tags for organizing runs')
    
    # Decision logging arguments
    train_parser.add_argument('--enable-logging', action='store_true', 
                            help='Enable decision logging (logs available options and model choices)')
    train_parser.add_argument('--log-dir', type=str, default='logs', 
                            help='Directory to save decision logs')
    train_parser.add_argument('--log-level', type=str, default='INFO', 
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            help='Logging level')
    train_parser.add_argument('--log-every-n-steps', type=int, default=1,
                            help='Log decisions every N steps (1 = every step)')
    
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
    play_parser.add_argument('--episodes', type=int, default=1, help='Number of steps to take with the model')

    # Two-network training command
    two_net_train_parser = subparsers.add_parser('train-two-net', help='Train two-network architecture agent')
    two_net_train_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    two_net_train_parser.add_argument('--save', type=str, help='Path to save trained model')
    two_net_train_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    two_net_train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    two_net_train_parser.add_argument('--memory-size', type=int, default=50000, help='Experience replay buffer size')
    two_net_train_parser.add_argument('--min-experiences', type=int, default=1000, help='Min experiences before training starts')

    # Two-network evaluation command
    two_net_eval_parser = subparsers.add_parser('eval-two-net', help='Evaluate two-network architecture agent')
    two_net_eval_parser.add_argument('model', type=str, help='Path to two-network model file')
    two_net_eval_parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')

    # Parallel two-network training command
    parallel_train_parser = subparsers.add_parser('train-parallel', help='Train two-network agent with parallel environments')
    parallel_train_parser.add_argument('--batches', type=int, default=100, help='Number of training batches')
    parallel_train_parser.add_argument('--save', type=str, help='Path to save trained model')
    parallel_train_parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments')
    parallel_train_parser.add_argument('--batch-steps', type=int, default=256, help='Steps per environment per batch')
    parallel_train_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parallel_train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parallel_train_parser.add_argument('--memory-size', type=int, default=50000, help='Experience replay buffer size')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_kwargs = {
            'num_episodes': args.episodes,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'hidden_size': args.hidden_size,
            'reward_function': args.reward_function,
            'ascension': args.ascension,
            'use_wandb': args.wandb,
            'wandb_project': args.wandb_project,
            'wandb_entity': args.wandb_entity,
            'wandb_run_name': args.wandb_name,
            'wandb_tags': args.wandb_tags,
            'enable_decision_logging': args.enable_logging,
            'log_directory': args.log_dir,
            'log_level': args.log_level,
            'log_every_n_steps': args.log_every_n_steps
        }
        train_agent(args.config, **train_kwargs)
    
    elif args.command == 'eval':
        evaluate_agent(args.model, args.games, args.temperature)
    
    elif args.command == 'list':
        list_models()
    
    elif args.command == 'play':
        interactive_play(args.model, args.episodes)

    elif args.command == 'train-two-net':
        # Generate default save path if not provided
        save_path = args.save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"sts_models/two_network_model_{timestamp}.pt"

        train_kwargs = {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'memory_size': args.memory_size,
            'min_experiences_before_training': args.min_experiences
        }
        train_two_network_agent(args.episodes, save_path, **train_kwargs)

    elif args.command == 'eval-two-net':
        evaluate_two_network_agent(args.model, args.episodes)

    elif args.command == 'train-parallel':
        # Generate default save path if not provided
        save_path = args.save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"sts_models/parallel_model_{timestamp}.pt"

        train_kwargs = {
            'num_parallel_envs': args.num_envs,
            'batch_collection_steps': args.batch_steps,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'memory_size': args.memory_size
        }
        train_two_network_parallel(args.batches, save_path, **train_kwargs)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()