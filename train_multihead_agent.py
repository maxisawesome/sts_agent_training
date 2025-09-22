#!/usr/bin/env python3
"""
Multi-Head STS Agent Training Script

This script trains the multi-head value network using Double DQN with separate
heads for combat and meta-game decisions.
"""

import argparse
import os
import time
import json
from datetime import datetime
import numpy as np

import slaythespire
from sts_dqn_trainer import STSDQNTrainer, DQNConfig
from sts_data_collection import STSEnvironmentWrapper
from sts_action_classifier import STSActionContextClassifier
from sts_multihead_network import ActionContext
import torch


def train_multihead_agent(episodes: int = 1000,
                         model_save_path: str = None,
                         config: DQNConfig = None,
                         verbose: bool = True):
    """
    Train the multi-head value network.

    Args:
        episodes: Number of training episodes
        model_save_path: Path to save trained model
        config: DQN configuration
        verbose: Whether to print training progress
    """
    if config is None:
        config = DQNConfig()

    # Create trainer and environment
    trainer = STSDQNTrainer(config)
    env = STSEnvironmentWrapper()

    if verbose:
        print(f"=== Multi-Head STS Agent Training ===")
        print(f"Episodes: {episodes}")
        print(f"Device: {trainer.device}")
        print(f"Network parameters: {sum(p.numel() for p in trainer.q_network.parameters()):,}")
        print(f"Target update frequency: {config.target_update_freq}")
        print(f"Batch size: {config.batch_size}")
        print()

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    combat_action_counts = []
    meta_action_counts = []

    start_time = time.time()

    for episode in range(episodes):
        # Train one episode
        result = trainer.train_episode(env, max_steps=1000)

        episode_rewards.append(result['episode_reward'])
        episode_lengths.append(result['episode_steps'])

        # Count action types for this episode
        combat_count = 0
        meta_count = 0
        # This would require tracking during episode - simplified for now

        # Print progress
        if verbose and (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            recent_lengths = episode_lengths[-50:]

            elapsed = time.time() - start_time

            print(f"Episode {episode + 1:4d}/{episodes}")
            print(f"  Avg reward (last 50): {np.mean(recent_rewards):6.3f}")
            print(f"  Avg length (last 50): {np.mean(recent_lengths):5.1f}")
            print(f"  Epsilon: {result['epsilon']:5.3f}")
            print(f"  Buffer size: {result['buffer_size']:6d}")
            print(f"  Time elapsed: {elapsed/60:.1f}m")

            if result['losses']:
                avg_losses = {}
                for loss_dict in result['losses']:
                    for key, value in loss_dict.items():
                        if key not in avg_losses:
                            avg_losses[key] = []
                        avg_losses[key].append(value)

                for key, values in avg_losses.items():
                    print(f"  Avg {key}: {np.mean(values):.4f}")

            print()

    # Save model if path provided
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        trainer.save_model(model_save_path)
        if verbose:
            print(f"Model saved to: {model_save_path}")

    # Training summary
    if verbose:
        total_time = time.time() - start_time
        print(f"\n=== Training Complete ===")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average reward: {np.mean(episode_rewards):.3f}")
        print(f"Final epsilon: {trainer.get_epsilon():.3f}")
        print(f"Total steps: {trainer.steps_done}")

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_epsilon': trainer.get_epsilon(),
        'total_steps': trainer.steps_done,
        'training_time': time.time() - start_time
    }


def evaluate_agent(model_path: str, episodes: int = 100, verbose: bool = True):
    """
    Evaluate a trained multi-head agent.

    Args:
        model_path: Path to trained model
        episodes: Number of evaluation episodes
        verbose: Whether to print progress
    """
    config = DQNConfig()
    trainer = STSDQNTrainer(config)

    # Load trained model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    trainer.load_model(model_path)

    if verbose:
        print(f"=== Evaluating Multi-Head Agent ===")
        print(f"Model: {model_path}")
        print(f"Episodes: {episodes}")
        print()

    env = STSEnvironmentWrapper()
    results = []

    for episode in range(episodes):
        state = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)

        episode_reward = 0
        episode_steps = 0
        combat_actions = 0
        meta_actions = 0

        for step in range(1000):
            # Select action (no exploration)
            generic_action, action_data, action_context = trainer.select_action(
                state_tensor, env.game_context, explore=False
            )

            if generic_action == -1:
                break

            # Count action types
            if action_context == ActionContext.COMBAT:
                combat_actions += 1
            else:
                meta_actions += 1

            # Take step
            next_state, reward, done, info = env.step(generic_action)

            episode_reward += reward
            episode_steps += 1

            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)

            if done:
                break

        result = {
            'episode': episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'combat_actions': combat_actions,
            'meta_actions': meta_actions,
            'final_hp': env.game_context.cur_hp,
            'final_floor': env.game_context.floor_num,
            'outcome': str(env.game_context.outcome)
        }
        results.append(result)

        if verbose and (episode + 1) % 20 == 0:
            recent = results[-20:]
            print(f"Episode {episode + 1:3d}: "
                  f"Reward={episode_reward:6.3f}, "
                  f"Steps={episode_steps:3d}, "
                  f"HP={env.game_context.cur_hp:2d}, "
                  f"Floor={env.game_context.floor_num:2d}")

    # Summary statistics
    if verbose:
        rewards = [r['reward'] for r in results]
        steps = [r['steps'] for r in results]
        combat_actions = [r['combat_actions'] for r in results]
        meta_actions = [r['meta_actions'] for r in results]
        final_floors = [r['final_floor'] for r in results]

        print(f"\n=== Evaluation Summary ===")
        print(f"Average reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"Average steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        print(f"Average final floor: {np.mean(final_floors):.1f}")
        print(f"Combat actions per episode: {np.mean(combat_actions):.1f}")
        print(f"Meta actions per episode: {np.mean(meta_actions):.1f}")

    return results


def test_multihead_system():
    """Test the complete multi-head system end-to-end."""
    print("=== Testing Multi-Head System ===\n")

    # Quick training test
    config = DQNConfig(
        batch_size=8,
        memory_size=500,
        training_start=50,
        epsilon_decay=200
    )

    print("Testing training for 5 episodes...")
    results = train_multihead_agent(
        episodes=5,
        config=config,
        verbose=True
    )

    print("✓ Training test completed")
    print(f"  Average reward: {np.mean(results['episode_rewards']):.3f}")
    print(f"  Total steps: {results['total_steps']}")

    print("\n✓ Multi-head system test passed!")


def main():
    parser = argparse.ArgumentParser(description="Multi-Head STS Agent Training")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train a new multi-head agent')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    train_parser.add_argument('--save', type=str, help='Path to save trained model')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--no-verbose', action='store_true', help='Reduce output verbosity')

    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained agent')
    eval_parser.add_argument('model_path', help='Path to trained model')
    eval_parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    eval_parser.add_argument('--no-verbose', action='store_true', help='Reduce output verbosity')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test the multi-head system')

    args = parser.parse_args()

    if args.command == 'train':
        config = DQNConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

        # Generate default save path if not provided
        save_path = args.save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"sts_models/multihead_model_{timestamp}.pt"

        train_multihead_agent(
            episodes=args.episodes,
            model_save_path=save_path,
            config=config,
            verbose=not args.no_verbose
        )

    elif args.command == 'eval':
        evaluate_agent(
            model_path=args.model_path,
            episodes=args.episodes,
            verbose=not args.no_verbose
        )

    elif args.command == 'test':
        test_multihead_system()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()