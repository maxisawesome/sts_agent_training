#!/usr/bin/env python3
"""
Parallel Two-Network Trainer for STS Architecture

This module extends the two-network trainer to support parallel environments
for efficient batch training and experience collection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import deque
import time
import os
import json
from datetime import datetime

import slaythespire
from sts_two_network_trainer import (
    TwoNetworkTrainer, TwoNetworkTrainingConfig, Experience, ExperienceBuffer
)
from sts_parallel_environments import VectorizedStsEnvironment, ParallelExperience
from sts_input_processor import ScreenState


class ParallelTwoNetworkTrainingConfig(TwoNetworkTrainingConfig):
    """Extended configuration for parallel training."""

    def __init__(self):
        super().__init__()

        # Parallel training specific config
        self.num_parallel_envs = 4  # Number of parallel environments
        self.batch_collection_steps = 256  # Steps to collect before training
        self.parallel_timeout = 10.0  # Timeout for parallel operations

        # Adjust other parameters for parallel training
        self.min_experiences_before_training = max(500, self.batch_collection_steps * 2)


class ParallelTwoNetworkTrainer(TwoNetworkTrainer):
    """
    Two-network trainer with parallel environment support.

    Extends the base trainer to collect experiences from multiple environments
    in parallel for more efficient training.
    """

    def __init__(self, config: ParallelTwoNetworkTrainingConfig = None):
        if config is None:
            config = ParallelTwoNetworkTrainingConfig()

        super().__init__(config)
        self.parallel_config = config

        # Initialize parallel environments
        self.vec_env = VectorizedStsEnvironment(
            num_envs=config.num_parallel_envs,
            reward_function='simple',  # TODO: Make configurable
            timeout=config.parallel_timeout
        )

        # Parallel training state
        self.env_states = [None] * config.num_parallel_envs
        self.total_parallel_steps = 0

    def train_parallel_batch(self) -> Dict:
        """
        Collect a batch of experiences from parallel environments and train networks.

        Returns:
            Dictionary with training statistics
        """
        batch_start_time = time.time()

        # Reset any finished environments
        done_envs = [i for i, state in enumerate(self.env_states) if state is None]
        if done_envs or all(state is None for state in self.env_states):
            # Reset all environments if starting fresh
            if all(state is None for state in self.env_states):
                done_envs = list(range(self.parallel_config.num_parallel_envs))

            initial_states = self.vec_env.reset(done_envs)
            for i, env_id in enumerate(done_envs):
                self.env_states[env_id] = initial_states[i]

        # Collect batch of experiences
        experiences_collected = 0
        batch_experiences = []
        episode_rewards = [0.0] * self.parallel_config.num_parallel_envs
        episode_steps = [0] * self.parallel_config.num_parallel_envs
        episodes_completed = 0

        for step in range(self.parallel_config.batch_collection_steps):
            # Get current states (handle None values)
            current_states = []
            for state in self.env_states:
                if state is not None:
                    current_states.append(state)
                else:
                    # Use placeholder for None states
                    current_states.append(np.zeros(550))  # Default observation size
            current_states = np.array(current_states)

            # Get game contexts
            game_contexts = self.vec_env.get_game_contexts()

            # Select actions for all environments
            actions = []
            action_data = []

            for env_id in range(self.parallel_config.num_parallel_envs):
                if current_states[env_id] is not None:
                    try:
                        action, confidence, debug_info = self.input_processor.process_and_select_action(
                            current_states[env_id], self._mock_game_context(game_contexts[env_id])
                        )
                        actions.append(action)
                        action_data.append(debug_info)
                    except Exception as e:
                        # Handle action selection errors
                        actions.append(0)  # Default action
                        action_data.append({'network_used': 'events', 'error': str(e)})
                else:
                    actions.append(0)
                    action_data.append({'network_used': 'events'})

            # Take steps in all environments
            next_states, rewards, dones, infos = self.vec_env.step(actions)

            # Process experiences
            for env_id in range(self.parallel_config.num_parallel_envs):
                if current_states[env_id] is not None:
                    # Create experience
                    experience = Experience(
                        state=current_states[env_id],
                        action=actions[env_id],
                        reward=rewards[env_id],
                        next_state=next_states[env_id],
                        done=dones[env_id],
                        screen_state=ScreenState.UNKNOWN,  # Simplified for now
                        game_context=game_contexts[env_id],
                        network_specific_data=action_data[env_id]
                    )

                    # Add to experience buffer
                    self.experience_buffer.add(experience)
                    batch_experiences.append(experience)
                    experiences_collected += 1

                    # Update episode tracking
                    episode_rewards[env_id] += rewards[env_id]
                    episode_steps[env_id] += 1

                    # Handle episode completion
                    if dones[env_id]:
                        episodes_completed += 1
                        # Reset state (will be reset in next iteration)
                        self.env_states[env_id] = None
                    else:
                        # Update state for next step
                        self.env_states[env_id] = next_states[env_id]

            self.total_parallel_steps += self.parallel_config.num_parallel_envs

            # Train networks if we have enough experiences
            if (self.experience_buffer.size() >= self.parallel_config.min_experiences_before_training and
                self.total_parallel_steps % self.parallel_config.train_frequency == 0):

                self._update_networks()

        # Training statistics
        batch_time = time.time() - batch_start_time
        avg_episode_reward = np.mean([r for r in episode_rewards if r > 0]) if any(r > 0 for r in episode_rewards) else 0.0

        return {
            'experiences_collected': experiences_collected,
            'episodes_completed': episodes_completed,
            'avg_episode_reward': avg_episode_reward,
            'total_episode_steps': sum(episode_steps),
            'batch_time': batch_time,
            'buffer_size': self.experience_buffer.size(),
            'events_buffer_size': self.experience_buffer.size('events'),
            'combat_buffer_size': self.experience_buffer.size('combat'),
            'steps_per_second': self.parallel_config.batch_collection_steps * self.parallel_config.num_parallel_envs / batch_time
        }

    def train_parallel_episodes(self, num_batches: int) -> Dict:
        """
        Train for multiple batches using parallel environments.

        Args:
            num_batches: Number of parallel batches to collect and train on

        Returns:
            Training statistics
        """
        print(f"Starting parallel training with {self.parallel_config.num_parallel_envs} environments")
        print(f"Batch size: {self.parallel_config.batch_collection_steps} steps per environment")
        print(f"Total batches: {num_batches}")

        start_time = time.time()
        all_batch_stats = []

        for batch in range(num_batches):
            batch_stats = self.train_parallel_batch()
            all_batch_stats.append(batch_stats)

            # Print progress
            if (batch + 1) % 10 == 0 or batch == 0:
                recent_stats = all_batch_stats[-10:] if len(all_batch_stats) >= 10 else all_batch_stats

                avg_reward = np.mean([s['avg_episode_reward'] for s in recent_stats])
                avg_experiences = np.mean([s['experiences_collected'] for s in recent_stats])
                avg_episodes = np.mean([s['episodes_completed'] for s in recent_stats])
                avg_sps = np.mean([s['steps_per_second'] for s in recent_stats])

                print(f"\nBatch {batch + 1:4d}/{num_batches}")
                print(f"  Avg reward (last {len(recent_stats)}): {avg_reward:6.3f}")
                print(f"  Avg experiences/batch: {avg_experiences:5.1f}")
                print(f"  Avg episodes/batch: {avg_episodes:4.1f}")
                print(f"  Steps/second: {avg_sps:6.1f}")
                print(f"  Buffer size: {batch_stats['buffer_size']:6d} (E:{batch_stats['events_buffer_size']}, C:{batch_stats['combat_buffer_size']})")

        total_time = time.time() - start_time

        # Final statistics
        total_experiences = sum(s['experiences_collected'] for s in all_batch_stats)
        total_episodes = sum(s['episodes_completed'] for s in all_batch_stats)
        overall_avg_reward = np.mean([s['avg_episode_reward'] for s in all_batch_stats if s['avg_episode_reward'] > 0])

        return {
            'total_batches': num_batches,
            'total_experiences': total_experiences,
            'total_episodes': total_episodes,
            'overall_avg_reward': overall_avg_reward,
            'total_time': total_time,
            'experiences_per_second': total_experiences / total_time,
            'batch_stats': all_batch_stats
        }

    def _mock_game_context(self, context_data: Dict):
        """Create a mock game context from parallel environment data."""
        class MockContext:
            def __init__(self, data):
                self.cur_hp = data.get('hp', 80)
                self.max_hp = data.get('max_hp', 80)
                self.gold = data.get('gold', 99)
                self.floor_num = data.get('floor', 0)
                self.act = data.get('act', 1)

        return MockContext(context_data)

    def close(self):
        """Clean up parallel environments."""
        if hasattr(self, 'vec_env'):
            self.vec_env.close()


def test_parallel_trainer():
    """Test the parallel two-network trainer."""
    print("=== Testing Parallel Two-Network Trainer ===\n")

    # Create configuration for testing
    config = ParallelTwoNetworkTrainingConfig()
    config.num_parallel_envs = 3
    config.batch_collection_steps = 32
    config.min_experiences_before_training = 50

    print(f"Configuration:")
    print(f"  Parallel environments: {config.num_parallel_envs}")
    print(f"  Batch collection steps: {config.batch_collection_steps}")
    print(f"  Min experiences: {config.min_experiences_before_training}")

    try:
        # Create trainer
        trainer = ParallelTwoNetworkTrainer(config)

        print(f"\nTrainer initialized successfully")
        print(f"  Events network parameters: {sum(p.numel() for p in trainer.events_network.parameters()):,}")
        print(f"  Combat network parameters: {sum(p.numel() for p in trainer.combat_network.parameters()):,}")

        # Test parallel batch training
        print(f"\n--- Testing Parallel Batch Training ---")
        results = trainer.train_parallel_episodes(num_batches=3)

        print(f"\nParallel training results:")
        print(f"  Total experiences collected: {results['total_experiences']}")
        print(f"  Total episodes completed: {results['total_episodes']}")
        print(f"  Overall average reward: {results['overall_avg_reward']:.3f}")
        print(f"  Training time: {results['total_time']:.1f}s")
        print(f"  Experiences per second: {results['experiences_per_second']:.1f}")

        # Cleanup
        trainer.close()
        print(f"\n✓ Parallel trainer test completed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_parallel_trainer()