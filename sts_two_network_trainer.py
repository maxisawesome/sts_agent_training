#!/usr/bin/env python3
"""
Training Coordinator for Two-Network STS Architecture

This module coordinates training of both the Events/Planning Network and Combat Network
with shared embeddings and separate experience buffers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import deque
import random
import os
import json
from datetime import datetime

import slaythespire
from sts_shared_embeddings import SharedEmbeddingSystem
from sts_events_network import EventsPlanningNetwork, EventsNetworkTrainer
from sts_combat_network import CombatNetwork, CombatNetworkTrainer
from sts_input_processor import STSInputProcessor, ScreenState
from sts_data_collection import STSEnvironmentWrapper


class Experience(NamedTuple):
    """Single experience tuple for either network."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    screen_state: ScreenState
    game_context: Dict  # Simplified game context data
    network_specific_data: Dict  # Combat or event specific data


class ExperienceBuffer:
    """Experience replay buffer with network-specific filtering."""

    def __init__(self, max_size: int = 50000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self,
               batch_size: int,
               network_type: str = None) -> List[Experience]:
        """
        Sample experiences, optionally filtered by network type.

        Args:
            batch_size: Number of experiences to sample
            network_type: 'combat' or 'events' to filter by network

        Returns:
            List of sampled experiences
        """
        if network_type == 'combat':
            filtered = [exp for exp in self.buffer if exp.screen_state == ScreenState.COMBAT]
        elif network_type == 'events':
            filtered = [exp for exp in self.buffer
                       if exp.screen_state != ScreenState.COMBAT]
        else:
            filtered = list(self.buffer)

        if len(filtered) < batch_size:
            return filtered

        return random.sample(filtered, batch_size)

    def size(self, network_type: str = None) -> int:
        """Get buffer size, optionally filtered by network type."""
        if network_type == 'combat':
            return sum(1 for exp in self.buffer if exp.screen_state == ScreenState.COMBAT)
        elif network_type == 'events':
            return sum(1 for exp in self.buffer if exp.screen_state != ScreenState.COMBAT)
        else:
            return len(self.buffer)


class TwoNetworkTrainingConfig:
    """Configuration for two-network training."""

    def __init__(self):
        # General training config
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Soft update rate for target networks

        # Experience collection
        self.max_episode_steps = 1000
        self.memory_size = 50000
        self.min_experiences_before_training = 1000

        # Training frequency
        self.train_frequency = 4  # Steps between training updates
        self.target_update_frequency = 1000  # Steps between target network updates

        # Network-specific config
        self.events_update_frequency = 1  # Train events network every N updates
        self.combat_update_frequency = 1  # Train combat network every N updates

        # Evaluation
        self.eval_frequency = 100  # Episodes between evaluations
        self.eval_episodes = 10

        # Saving
        self.save_frequency = 500  # Episodes between model saves


class TwoNetworkTrainer:
    """
    Coordinates training of both Events/Planning and Combat networks.

    Manages separate experience buffers, training loops, and shared embeddings.
    """

    def __init__(self, config: TwoNetworkTrainingConfig = None):
        self.config = config or TwoNetworkTrainingConfig()

        # Initialize networks
        self.shared_embeddings = SharedEmbeddingSystem()
        self.events_network = EventsPlanningNetwork(self.shared_embeddings)
        self.combat_network = CombatNetwork(self.shared_embeddings)

        # Initialize input processor
        self.input_processor = STSInputProcessor(
            self.shared_embeddings,
            self.events_network,
            self.combat_network
        )

        # Initialize trainers for each network
        self.events_trainer = EventsNetworkTrainer(
            self.events_network,
            learning_rate=self.config.learning_rate
        )

        self.combat_trainer = CombatNetworkTrainer(
            self.combat_network,
            learning_rate=self.config.learning_rate
        )

        # Shared embedding optimizer (updates both networks)
        shared_params = []
        for module in self.shared_embeddings.get_modules().values():
            shared_params.extend(module.parameters())

        self.shared_optimizer = optim.Adam(shared_params, lr=self.config.learning_rate)

        # Experience buffer
        self.experience_buffer = ExperienceBuffer(self.config.memory_size)

        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.update_count = 0

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._move_to_device()

    def _move_to_device(self):
        """Move all networks to device."""
        for module in self.shared_embeddings.get_modules().values():
            module.to(self.device)
        self.events_network.to(self.device)
        self.combat_network.to(self.device)

    def train_episode(self, env: STSEnvironmentWrapper) -> Dict:
        """Train for one episode, collecting experiences and updating networks."""

        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        experiences_collected = 0
        events_actions = 0
        combat_actions = 0

        for step in range(self.config.max_episode_steps):
            # Select action using current policy
            action, confidence, debug_info = self.input_processor.process_and_select_action(
                state, env.game_context
            )

            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Create experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                screen_state=ScreenState(debug_info.get('screen_state', 'unknown')),
                game_context=self._extract_game_context_data(env.game_context),
                network_specific_data=debug_info
            )

            # Add to experience buffer
            self.experience_buffer.add(experience)
            experiences_collected += 1

            # Count action types
            if debug_info['network_used'] == 'combat':
                combat_actions += 1
            else:
                events_actions += 1

            # Update networks if we have enough experiences
            if (self.experience_buffer.size() >= self.config.min_experiences_before_training and
                self.total_steps % self.config.train_frequency == 0):

                self._update_networks()

            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            if done:
                break

        self.episode_count += 1

        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'experiences_collected': experiences_collected,
            'events_actions': events_actions,
            'combat_actions': combat_actions,
            'buffer_size': self.experience_buffer.size(),
            'events_buffer_size': self.experience_buffer.size('events'),
            'combat_buffer_size': self.experience_buffer.size('combat')
        }

    def _update_networks(self):
        """Update both networks using their respective training methods."""
        losses = {}

        # Update Events/Planning Network
        if (self.update_count % self.config.events_update_frequency == 0 and
            self.experience_buffer.size('events') >= self.config.batch_size):

            events_loss = self._update_events_network()
            losses['events_loss'] = events_loss

        # Update Combat Network
        if (self.update_count % self.config.combat_update_frequency == 0 and
            self.experience_buffer.size('combat') >= self.config.batch_size):

            combat_loss = self._update_combat_network()
            losses['combat_loss'] = combat_loss

        self.update_count += 1
        return losses

    def _update_events_network(self) -> float:
        """Update the Events/Planning Network using value-based learning."""

        # Sample events experiences
        experiences = self.experience_buffer.sample(self.config.batch_size, 'events')

        if not experiences:
            return 0.0

        total_loss = 0.0
        batch_size = len(experiences)

        for exp in experiences:
            # Process state through shared embeddings
            shared_data = self.shared_embeddings.process_full_game_state(
                exp.state, self._reconstruct_game_context(exp.game_context)
            )

            # Extract event data (simplified)
            event_data = exp.network_specific_data.get('event_data', {
                'event_type': torch.tensor(0),
                'choice_vectors': torch.randn(4, 20)
            })

            # Calculate target value (simplified TD target)
            target_value = exp.reward
            if not exp.done:
                # Would add discounted next state value
                target_value += self.config.gamma * 0.5  # Placeholder

            # Train on this experience
            loss = self.events_trainer.train_step(
                shared_data, event_data, exp.action, target_value
            )
            total_loss += loss

        return total_loss / batch_size

    def _update_combat_network(self) -> float:
        """Update the Combat Network using policy-based learning."""

        # Sample combat experiences
        experiences = self.experience_buffer.sample(self.config.batch_size, 'combat')

        if not experiences:
            return 0.0

        total_loss = 0.0
        batch_size = len(experiences)

        for exp in experiences:
            # Process state through shared embeddings
            shared_data = self.shared_embeddings.process_full_game_state(
                exp.state, self._reconstruct_game_context(exp.game_context)
            )

            # Extract combat data (simplified)
            combat_data = exp.network_specific_data.get('combat_data', {
                'enemy_types': torch.tensor([0]),
                'enemy_healths': torch.tensor([50]),
                'energy': torch.tensor(3),
                'hand_cards': torch.tensor([]),
                'powers': {}
            })

            # Calculate target value (simplified)
            target_value = exp.reward
            if not exp.done:
                target_value += self.config.gamma * 0.5  # Placeholder

            # Train on this experience
            loss = self.combat_trainer.train_step(
                shared_data, combat_data, exp.action, target_value
            )
            total_loss += loss

        return total_loss / batch_size

    def _extract_game_context_data(self, game_context: slaythespire.GameContext) -> Dict:
        """Extract relevant data from game context for storage."""
        return {
            'hp': game_context.cur_hp,
            'max_hp': game_context.max_hp,
            'gold': game_context.gold,
            'floor': game_context.floor_num,
            'act': game_context.act
        }

    def _reconstruct_game_context(self, context_data: Dict):
        """Reconstruct a simplified game context from stored data."""
        # This would create a mock object with the necessary attributes
        class MockContext:
            def __init__(self, data):
                # Map the stored names to the expected attribute names
                self.cur_hp = data.get('hp', 80)
                self.max_hp = data.get('max_hp', 80)
                self.gold = data.get('gold', 99)
                self.floor_num = data.get('floor', 0)
                self.act = data.get('act', 1)

                # Add placeholder deck and relics for shared embeddings
                self.deck = []  # Empty deck placeholder
                self.relics = []  # Empty relics placeholder

        return MockContext(context_data)

    def evaluate(self, env: STSEnvironmentWrapper, episodes: int = 10) -> Dict:
        """Evaluate both networks over multiple episodes."""

        results = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            events_actions = 0
            combat_actions = 0

            for step in range(self.config.max_episode_steps):
                # Select action (no exploration)
                with torch.no_grad():
                    action, confidence, debug_info = self.input_processor.process_and_select_action(
                        state, env.game_context
                    )

                # Count action types
                if debug_info['network_used'] == 'combat':
                    combat_actions += 1
                else:
                    events_actions += 1

                # Take step
                next_state, reward, done, info = env.step(action)

                state = next_state
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            results.append({
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'events_actions': events_actions,
                'combat_actions': combat_actions,
                'final_hp': env.game_context.cur_hp,
                'final_floor': env.game_context.floor_num
            })

        # Calculate summary statistics
        rewards = [r['reward'] for r in results]
        steps = [r['steps'] for r in results]
        events_actions = [r['events_actions'] for r in results]
        combat_actions = [r['combat_actions'] for r in results]

        return {
            'episodes': episodes,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(steps),
            'avg_events_actions': np.mean(events_actions),
            'avg_combat_actions': np.mean(combat_actions),
            'detailed_results': results
        }

    def save_model(self, filepath: str):
        """Save both networks and shared embeddings."""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            'shared_embeddings': {
                name: module.state_dict()
                for name, module in self.shared_embeddings.get_modules().items()
            },
            'events_network': self.events_network.state_dict(),
            'combat_network': self.combat_network.state_dict(),
            'config': self.config.__dict__,
            'training_state': {
                'total_steps': self.total_steps,
                'episode_count': self.episode_count,
                'update_count': self.update_count
            }
        }

        torch.save(save_dict, filepath)

        # Also save config as JSON for reference
        config_path = filepath.replace('.pt', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(save_dict['config'], f, indent=2)

    def load_model(self, filepath: str):
        """Load both networks and shared embeddings."""

        checkpoint = torch.load(filepath, map_location=self.device)

        # Load shared embeddings
        for name, module in self.shared_embeddings.get_modules().items():
            if name in checkpoint['shared_embeddings']:
                module.load_state_dict(checkpoint['shared_embeddings'][name])

        # Load networks
        self.events_network.load_state_dict(checkpoint['events_network'])
        self.combat_network.load_state_dict(checkpoint['combat_network'])

        # Load training state
        if 'training_state' in checkpoint:
            state = checkpoint['training_state']
            self.total_steps = state['total_steps']
            self.episode_count = state['episode_count']
            self.update_count = state['update_count']


def test_two_network_trainer():
    """Test the two-network training system."""
    print("=== Testing Two-Network Training System ===\n")

    # Create trainer
    config = TwoNetworkTrainingConfig()
    config.batch_size = 8
    config.min_experiences_before_training = 10

    trainer = TwoNetworkTrainer(config)

    print(f"Two-network trainer created successfully")
    print(f"Events network parameters: {sum(p.numel() for p in trainer.events_network.parameters()):,}")
    print(f"Combat network parameters: {sum(p.numel() for p in trainer.combat_network.parameters()):,}")
    print(f"Shared embedding parameters: {sum(p.numel() for module in trainer.shared_embeddings.get_modules().values() for p in module.parameters()):,}")

    # Test experience buffer
    print("\n--- Experience Buffer Test ---")
    buffer = ExperienceBuffer(max_size=100)

    # Add some mock experiences
    for i in range(20):
        exp = Experience(
            state=np.random.randn(550),
            action=i % 4,
            reward=0.1,
            next_state=np.random.randn(550),
            done=False,
            screen_state=ScreenState.COMBAT if i % 2 == 0 else ScreenState.EVENT,
            game_context={'hp': 80, 'gold': 99},
            network_specific_data={'network_used': 'combat' if i % 2 == 0 else 'events'}
        )
        buffer.add(exp)

    print(f"Total experiences: {buffer.size()}")
    print(f"Combat experiences: {buffer.size('combat')}")
    print(f"Events experiences: {buffer.size('events')}")

    # Test sampling
    combat_batch = buffer.sample(5, 'combat')
    events_batch = buffer.sample(5, 'events')
    print(f"Combat batch size: {len(combat_batch)}")
    print(f"Events batch size: {len(events_batch)}")

    print("\n✓ Two-network training system working!")


if __name__ == "__main__":
    test_two_network_trainer()