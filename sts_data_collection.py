#!/usr/bin/env python3
"""
Slay the Spire Data Collection System

This module handles collecting training data from sts_lightspeed simulations
for reinforcement learning. It interfaces with the game engine to generate
episodes and collect state-action-reward trajectories.
"""

import torch
import numpy as np
import sys
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from collections import deque
import random

import slaythespire

from sts_reward_functions import RewardFunctionManager

@dataclass
class Experience:
    """Single experience tuple for reinforcement learning."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0

@dataclass
class Episode:
    """Complete episode data."""
    experiences: List[Experience]
    total_reward: float
    episode_length: int
    outcome: str

class STSEnvironmentWrapper:
    """
    Wrapper around sts_lightspeed to provide a standard RL environment interface.
    
    This class bridges the gap between the C++ simulation engine and Python RL training.
    """
    
    def __init__(self, character_class=None, ascension=0, reward_function='simple'):
        self.character_class = character_class or slaythespire.CharacterClass.IRONCLAD
        self.ascension = ascension
        self.nn_interface = slaythespire.getNNInterface()
        self.reward_manager = RewardFunctionManager()
        self.reward_manager.set_reward_function(reward_function)
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""
        # Generate random seeds for new game
        seed = random.randint(0, 2**31 - 1)
        
        self.game_context = slaythespire.GameContext(self.character_class, seed, self.ascension)
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.reward_manager.reset_episode()
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current game state as numpy array."""
        obs_array = self.nn_interface.getObservation(self.game_context)
        return np.array(obs_array, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment.
        
        Note: This is a placeholder implementation. The actual action execution
        would need to interface with the game's action system, which requires
        understanding the specific action encoding used by sts_lightspeed.
        """
        self.step_count += 1
        
        # Placeholder action execution
        # In a full implementation, this would:
        # 1. Convert action ID to game action
        # 2. Execute action in game_context
        # 3. Update game state
        # 4. Calculate reward based on game outcome
        
        # Calculate reward using the reward function manager
        reward = self.reward_manager.get_reward(self.game_context, action, self.done)
        self.total_reward += reward
        
        # Simple termination condition (placeholder)
        self.done = (self.step_count >= 1000) or self._is_game_over()
        
        next_obs = self._get_observation()
        info = {
            'step_count': self.step_count,
            'total_reward': self.total_reward
        }
        
        return next_obs, reward, self.done, info
    
    def _is_game_over(self) -> bool:
        """Check if the game is over (placeholder)."""
        # In actual implementation, this would check game state
        return self.game_context.cur_hp <= 0

class ExperienceBuffer:
    """Buffer for storing and managing training experiences."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.episodes = []
    
    def add_experience(self, experience: Experience):
        """Add a single experience to the buffer."""
        self.buffer.append(experience)
    
    def add_episode(self, episode: Episode):
        """Add a complete episode to the buffer."""
        self.episodes.append(episode)
        for exp in episode.experiences:
            self.add_experience(exp)
    
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """Sample a random batch of experiences."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def get_recent_episodes(self, num_episodes: int) -> List[Episode]:
        """Get the most recent episodes."""
        return self.episodes[-num_episodes:]
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.episodes.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

class STSDataCollector:
    """
    Main data collection class for STS reinforcement learning.
    
    Orchestrates environment interaction and experience collection.
    """
    
    def __init__(self, env_wrapper: STSEnvironmentWrapper, buffer: ExperienceBuffer):
        self.env = env_wrapper
        self.buffer = buffer
        self.episode_count = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def collect_random_episodes(self, num_episodes: int) -> List[Episode]:
        """Collect episodes using random actions (for initial data)."""
        episodes = []
        
        for episode_idx in range(num_episodes):
            episode = self._collect_random_episode()
            episodes.append(episode)
            self.buffer.add_episode(episode)
            
            if (episode_idx + 1) % 10 == 0:
                print(f"Collected {episode_idx + 1}/{num_episodes} random episodes")
        
        return episodes
    
    def _collect_random_episode(self) -> Episode:
        """Collect a single episode using random actions."""
        experiences = []
        obs = self.env.reset()
        total_reward = 0.0
        step_count = 0
        
        while not self.env.done and step_count < 1000:  # Max episode length
            # Random action (placeholder - would use proper action space)
            action = random.randint(0, 255)  # Assuming 256 possible actions
            
            next_obs, reward, done, info = self.env.step(action)
            
            experience = Experience(
                state=obs.copy(),
                action=action,
                reward=reward,
                next_state=next_obs.copy(),
                done=done
            )
            
            experiences.append(experience)
            total_reward += reward
            obs = next_obs
            step_count += 1
        
        self.episode_count += 1
        
        return Episode(
            experiences=experiences,
            total_reward=total_reward,
            episode_length=len(experiences),
            outcome="random"
        )
    
    def collect_episode_with_policy(self, policy_network, temperature: float = 1.0, decision_logger=None) -> Episode:
        """Collect an episode using a neural network policy."""
        experiences = []
        obs = self.env.reset()
        total_reward = 0.0
        step_count = 0
        
        with torch.no_grad():
            while not self.env.done and step_count < 1000:
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # Get action probabilities from policy
                if hasattr(policy_network, 'forward'):
                    # Handle actor-critic network that returns (action_probs, value)
                    output = policy_network(obs_tensor)
                    if isinstance(output, tuple):
                        action_probs, _ = output
                    else:
                        action_probs = output
                else:
                    action_probs = policy_network(obs_tensor)
                
                # Apply temperature for exploration
                if temperature != 1.0:
                    action_probs = torch.pow(action_probs, 1.0 / temperature)
                    action_probs = action_probs / action_probs.sum()
                
                # Sample action
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
                log_prob = action_dist.log_prob(torch.tensor(action).to(self.device)).item()
                
                # Log decision if logger is provided
                if decision_logger and decision_logger.enabled:
                    # Extract game state information
                    game_state = self._extract_game_state(obs)
                    available_choices = self._extract_available_choices(obs)
                    
                    # Get state value if available
                    state_value = None
                    if hasattr(policy_network, 'forward') and isinstance(policy_network(obs_tensor), tuple):
                        _, value_tensor = policy_network(obs_tensor)
                        state_value = value_tensor.item()
                    
                    decision_logger.log_decision(
                        episode_num=self.episode_count,
                        step_num=step_count,
                        game_state=game_state,
                        available_choices=available_choices,
                        action_probabilities=action_probs.cpu().numpy().flatten(),
                        state_value=state_value,
                        chosen_action=action,
                        additional_info={
                            "temperature": temperature,
                            "log_prob": log_prob
                        }
                    )
                
                next_obs, reward, done, info = self.env.step(action)
                
                experience = Experience(
                    state=obs.copy(),
                    action=action,
                    reward=reward,
                    next_state=next_obs.copy(),
                    done=done,
                    log_prob=log_prob
                )
                
                experiences.append(experience)
                total_reward += reward
                obs = next_obs
                step_count += 1
        
        self.episode_count += 1
        
        return Episode(
            experiences=experiences,
            total_reward=total_reward,
            episode_length=len(experiences),
            outcome="policy"
        )
    
    def _extract_game_state(self, observation: np.ndarray) -> Dict:
        """Extract interpretable game state information from observation."""
        try:
            # Extract basic game state from observation
            # Based on our observation space structure (550 dimensions)
            hp_current = int(observation[0])
            hp_max = int(observation[1]) 
            gold = int(observation[2])
            floor = int(observation[3])
            
            # Boss encoding (features 4-13, one-hot)
            boss_idx = np.argmax(observation[4:14])
            boss_names = ["SLIME_BOSS", "HEXAGHOST", "THE_GUARDIAN", "CHAMP", "AUTOMATON", 
                         "COLLECTOR", "TIME_EATER", "DONU_AND_DECA", "AWAKENED_ONE", "THE_HEART"]
            boss = boss_names[boss_idx] if boss_idx < len(boss_names) else "UNKNOWN"
            
            # Event encoding (features 412-469, one-hot)
            event_start = 412
            event_features = observation[event_start:event_start + 58]
            current_event_idx = np.argmax(event_features) if np.any(event_features) else -1
            
            return {
                "hp_current": hp_current,
                "hp_max": hp_max,
                "gold": gold,
                "floor": floor,
                "boss": boss,
                "current_event_idx": int(current_event_idx),
                "has_active_event": bool(np.any(event_features))
            }
        except Exception as e:
            return {"error": f"Failed to extract game state: {str(e)}"}
    
    def _extract_available_choices(self, observation: np.ndarray) -> List[Dict]:
        """Extract available choice options from observation."""
        try:
            choices = []
            choice_start = 470  # Start of generic choice features
            
            for choice_idx in range(4):
                start_idx = choice_start + (choice_idx * 20)
                choice_features = observation[start_idx:start_idx + 20]
                
                # Parse choice features
                choice_types = choice_features[0:8]
                binary_flags = choice_features[8:10]
                card_id = int(choice_features[10])
                relic_id = int(choice_features[11])
                gold_amount = int(choice_features[12])
                hp_change = int(choice_features[13])
                max_hp_change = int(choice_features[14])
                gold_cost = int(choice_features[15])
                hp_cost = int(choice_features[16])
                
                # Only include non-empty choices
                if np.any(choice_features):
                    choice_info = {
                        "choice_index": choice_idx,
                        "choice_types": {
                            "is_card_reward": bool(choice_types[0]),
                            "is_relic_reward": bool(choice_types[1]),
                            "is_gold_reward": bool(choice_types[2]),
                            "is_hp_change": bool(choice_types[3]),
                            "is_card_upgrade": bool(choice_types[4]),
                            "is_card_removal": bool(choice_types[5]),
                            "is_card_transform": bool(choice_types[6]),
                            "is_shop_action": bool(choice_types[7])
                        },
                        "flags": {
                            "is_curse": bool(binary_flags[0]),
                            "is_upgrade": bool(binary_flags[1])
                        },
                        "card_id": card_id,
                        "relic_id": relic_id,
                        "rewards": {
                            "gold_amount": gold_amount,
                            "hp_change": hp_change,
                            "max_hp_change": max_hp_change
                        },
                        "costs": {
                            "gold_cost": gold_cost,
                            "hp_cost": hp_cost
                        }
                    }
                    choices.append(choice_info)
            
            return choices
        except Exception as e:
            return [{"error": f"Failed to extract choices: {str(e)}"}]

def test_data_collection():
    """Test the data collection system."""
    print("=== Testing STS Data Collection System ===\n")
    
    # Create environment and buffer
    env = STSEnvironmentWrapper()
    buffer = ExperienceBuffer(max_size=10000)
    collector = STSDataCollector(env, buffer)
    
    print("Environment created successfully")
    print(f"Initial observation shape: {env.reset().shape}")
    
    # Test random episode collection
    print("\n--- Collecting Random Episodes ---")
    episodes = collector.collect_random_episodes(5)
    
    print(f"Buffer size after collection: {buffer.size()}")
    print(f"Number of episodes: {len(buffer.episodes)}")
    
    # Print episode statistics
    for i, episode in enumerate(episodes):
        print(f"Episode {i+1}: Length={episode.episode_length}, Reward={episode.total_reward:.3f}")
    
    # Test batch sampling
    print("\n--- Testing Batch Sampling ---")
    batch = buffer.sample_batch(32)
    print(f"Sampled batch size: {len(batch)}")
    print(f"Sample experience - State shape: {batch[0].state.shape}, Action: {batch[0].action}, Reward: {batch[0].reward:.3f}")

if __name__ == "__main__":
    test_data_collection()