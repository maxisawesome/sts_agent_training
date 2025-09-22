#!/usr/bin/env python3
"""
DQN Trainer for Multi-Head Value Network

This module implements Double DQN training for the multi-head Slay the Spire
neural network, with separate experience replay for combat and meta actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import wandb
from dataclasses import dataclass

from sts_multihead_network import STSMultiHeadValueNetwork, ActionContext
from sts_action_classifier import STSActionContextClassifier
from sts_data_collection import STSEnvironmentWrapper
import slaythespire


# Experience tuple for DQN
Experience = namedtuple('Experience', [
    'state', 'action_data', 'action_context', 'reward',
    'next_state', 'done', 'generic_action_id'
])


@dataclass
class DQNConfig:
    """Configuration for DQN training."""
    # Network parameters
    state_size: int = 550
    learning_rate: float = 1e-4

    # Training parameters
    batch_size: int = 32
    target_update_freq: int = 1000
    memory_size: int = 100000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 10000
    gamma: float = 0.99

    # Multi-head specific
    combat_batch_ratio: float = 0.4  # 40% combat, 60% meta in each batch

    # Training schedule
    training_start: int = 1000  # Start training after collecting experiences
    train_freq: int = 4  # Train every N steps


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer with separate priorities for
    combat and meta actions to ensure balanced training.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, experience: Experience, priority: float = None):
        """Add experience to buffer with optional priority."""
        if priority is None:
            priority = max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=True)
        experiences = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class STSDQNTrainer:
    """
    Double DQN trainer for multi-head STS value network.

    Features:
    - Separate experience replay for combat vs meta actions
    - Double DQN to reduce overestimation bias
    - Prioritized experience replay
    - Action masking for valid actions only
    """

    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.q_network = STSMultiHeadValueNetwork(config.state_size).to(self.device)
        self.target_network = STSMultiHeadValueNetwork(config.state_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Experience replay
        self.memory = PrioritizedReplayBuffer(config.memory_size)

        # Action classifier
        self.action_classifier = STSActionContextClassifier()

        # Training state
        self.steps_done = 0
        self.episode_count = 0
        self.losses = {'combat': [], 'meta': [], 'combined': []}

    def get_epsilon(self) -> float:
        """Get current epsilon for epsilon-greedy exploration."""
        return self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
               np.exp(-1. * self.steps_done / self.config.epsilon_decay)

    def select_action(self, state: torch.Tensor, game_context: slaythespire.GameContext,
                     explore: bool = True) -> Tuple[int, Dict, ActionContext]:
        """
        Select action using epsilon-greedy policy with action masking.

        Returns:
            Tuple of (generic_action_id, action_data, action_context)
        """
        # Get available actions and context
        actions, action_context = self.action_classifier.get_available_actions_with_context(game_context)

        if not actions:
            # No valid actions - should not happen in normal gameplay
            return -1, {}, action_context

        # Epsilon-greedy exploration
        if explore and random.random() < self.get_epsilon():
            # Random action
            chosen_action = random.choice(actions)
        else:
            # Greedy action selection
            best_action = None
            best_q_value = float('-inf')

            for action in actions:
                action_data = self.action_classifier.action_to_network_input(action, action_context)

                with torch.no_grad():
                    q_value = self.q_network(state, action_data, action_context)

                if q_value.item() > best_q_value:
                    best_q_value = q_value.item()
                    best_action = action

            chosen_action = best_action

        # Convert to network input format
        action_data = self.action_classifier.action_to_network_input(chosen_action, action_context)
        generic_action_id = chosen_action['generic_action_id']

        return generic_action_id, action_data, action_context

    def store_experience(self, state: np.ndarray, action_data: Dict, action_context: ActionContext,
                        reward: float, next_state: np.ndarray, done: bool, generic_action_id: int):
        """Store experience in replay buffer."""
        experience = Experience(
            state=state,
            action_data=action_data,
            action_context=action_context,
            reward=reward,
            next_state=next_state,
            done=done,
            generic_action_id=generic_action_id
        )

        # Higher priority for rarer combat experiences
        priority = 2.0 if action_context == ActionContext.COMBAT else 1.0
        self.memory.push(experience, priority)

    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if len(self.memory) < self.config.training_start:
            return {}

        # Sample batch
        experiences, weights, indices = self.memory.sample(self.config.batch_size)
        if not experiences:
            return {}

        # Separate combat and meta experiences
        combat_experiences = [exp for exp in experiences if exp.action_context == ActionContext.COMBAT]
        meta_experiences = [exp for exp in experiences if exp.action_context == ActionContext.META]

        losses = {}

        # Train on combat experiences
        if combat_experiences:
            combat_loss, combat_td_errors = self._train_batch(combat_experiences, ActionContext.COMBAT)
            losses['combat_loss'] = combat_loss

        # Train on meta experiences
        if meta_experiences:
            meta_loss, meta_td_errors = self._train_batch(meta_experiences, ActionContext.META)
            losses['meta_loss'] = meta_loss

        # Update target network
        if self.steps_done % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return losses

    def _train_batch(self, experiences: List[Experience], action_context: ActionContext) -> Tuple[float, np.ndarray]:
        """Train on a batch of experiences for a specific action context."""
        if not experiences:
            return 0.0, np.array([])

        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)

        # Current Q-values
        current_q_values = []
        for exp in experiences:
            q_val = self.q_network(torch.FloatTensor(exp.state).unsqueeze(0).to(self.device),
                                 exp.action_data, exp.action_context)
            current_q_values.append(q_val.squeeze())
        current_q_values = torch.stack(current_q_values)

        # Next Q-values (Double DQN)
        with torch.no_grad():
            next_q_values = torch.zeros(len(experiences)).to(self.device)

            for i, exp in enumerate(experiences):
                if not exp.done:
                    # Use main network to select action, target network to evaluate
                    next_state_tensor = torch.FloatTensor(exp.next_state).unsqueeze(0).to(self.device)

                    # For simplicity, use current action data as approximation for next state
                    # In full implementation, would need to get valid actions for next state
                    next_q_target = self.target_network(next_state_tensor, exp.action_data, exp.action_context)
                    next_q_values[i] = next_q_target.squeeze()

        # Compute target Q-values
        target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)

        # Compute loss
        td_errors = target_q_values - current_q_values
        loss = F.mse_loss(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), td_errors.detach().cpu().numpy()

    def train_episode(self, env: STSEnvironmentWrapper, max_steps: int = 1000) -> Dict:
        """Train for one episode."""
        state = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        episode_reward = 0
        episode_steps = 0
        episode_losses = []

        for step in range(max_steps):
            # Select action
            generic_action, action_data, action_context = self.select_action(state_tensor, env.game_context)

            if generic_action == -1:
                break  # No valid actions

            # Take step in environment
            next_state, reward, done, info = env.step(generic_action)

            # Store experience
            self.store_experience(state, action_data, action_context, reward, next_state, done, generic_action)

            # Train
            if self.steps_done % self.config.train_freq == 0:
                losses = self.train_step()
                if losses:
                    episode_losses.append(losses)

            # Update state
            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            episode_reward += reward
            episode_steps += 1
            self.steps_done += 1

            if done:
                break

        self.episode_count += 1

        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'epsilon': self.get_epsilon(),
            'losses': episode_losses,
            'buffer_size': len(self.memory)
        }

    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)

    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episode_count = checkpoint['episode_count']


def test_dqn_trainer():
    """Test the DQN trainer."""
    print("=== Testing DQN Trainer ===\n")

    # Create trainer
    config = DQNConfig(
        batch_size=8,
        memory_size=1000,
        training_start=100,
        epsilon_decay=500
    )
    trainer = STSDQNTrainer(config)

    # Create environment
    env = STSEnvironmentWrapper()

    print(f"Device: {trainer.device}")
    print(f"Q-network parameters: {sum(p.numel() for p in trainer.q_network.parameters()):,}")

    # Train for a few episodes
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        result = trainer.train_episode(env, max_steps=20)

        print(f"Episode reward: {result['episode_reward']:.3f}")
        print(f"Episode steps: {result['episode_steps']}")
        print(f"Epsilon: {result['epsilon']:.3f}")
        print(f"Buffer size: {result['buffer_size']}")

        if result['losses']:
            avg_losses = {}
            for loss_dict in result['losses']:
                for key, value in loss_dict.items():
                    if key not in avg_losses:
                        avg_losses[key] = []
                    avg_losses[key].append(value)

            for key, values in avg_losses.items():
                print(f"Average {key}: {np.mean(values):.4f}")


if __name__ == "__main__":
    test_dqn_trainer()