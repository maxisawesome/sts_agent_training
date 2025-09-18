#!/usr/bin/env python3
"""
Slay the Spire Neural Network Training System

This module implements the reinforcement learning training loop for the STS agent
using Proximal Policy Optimization (PPO) algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass
import json

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

import slaythespire

from sts_neural_network import STSActorCritic
from sts_data_collection import STSEnvironmentWrapper, ExperienceBuffer, STSDataCollector, Experience

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # PPO hyperparameters
    learning_rate: float = 3e-4
    epsilon: float = 0.2  # PPO clipping parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training schedule
    num_episodes: int = 1000
    batch_size: int = 64
    update_epochs: int = 4  # Number of epochs per PPO update
    collect_episodes_per_update: int = 10
    
    # Network architecture
    hidden_size: int = 512
    num_layers: int = 3
    action_size: int = 256
    
    # Logging and saving
    log_interval: int = 10
    save_interval: int = 100
    model_save_path: str = 'sts_models'
    
    # Environment
    max_episode_length: int = 1000
    ascension: int = 0
    reward_function: str = 'simple'  # Options: 'simple', 'comprehensive', 'sparse', 'shaped'
    
    # Weights & Biases tracking
    use_wandb: bool = True
    wandb_project: str = 'sts-neural-agent'
    wandb_entity: Optional[str] = None  # Your wandb username/team
    wandb_run_name: Optional[str] = None  # Auto-generated if None
    wandb_tags: List[str] = None  # Tags for organizing runs
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for Slay the Spire.
    
    Implements the PPO algorithm for training the actor-critic network.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize network FIRST (needed for wandb.watch)
        self.actor_critic = STSActorCritic(
            obs_size=550,
            hidden_size=config.hidden_size,
            action_size=config.action_size,
            num_layers=config.num_layers
        ).to(self.device)
        
        # Initialize Weights & Biases AFTER network creation
        self.wandb_run = None
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        elif config.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb tracking requested but wandb not available")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate)
        
        # Initialize environment and data collection
        reward_function = getattr(config, 'reward_function', 'simple')
        self.env = STSEnvironmentWrapper(ascension=config.ascension, reward_function=reward_function)
        self.buffer = ExperienceBuffer(max_size=50000)
        self.collector = STSDataCollector(self.env, self.buffer)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        # Create model save directory
        os.makedirs(config.model_save_path, exist_ok=True)
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            # Generate run name if not provided
            run_name = self.config.wandb_run_name
            if run_name is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"sts-{self.config.reward_function}-{timestamp}"
            
            # Initialize wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=run_name,
                tags=self.config.wandb_tags or [],
                config=self.config.__dict__,
                reinit=True
            )
            
            # Watch the model for gradient tracking
            try:
                wandb.watch(self.actor_critic, log_freq=100, log="all")
            except Exception as e:
                print(f"⚠️  Failed to set up wandb model watching: {e}")
                # Continue without model watching
            
            print(f"✅ Initialized wandb tracking: {self.wandb_run.url}")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize wandb: {e}")
            self.wandb_run = None
    
    def _log_to_wandb(self, **kwargs):
        """Log metrics to Weights & Biases."""
        if not self.wandb_run:
            return
        
        try:
            # Basic training metrics
            log_dict = {
                "training/update": kwargs['update_count'],
                "training/episodes": kwargs['total_episodes'],
                "training/avg_reward": kwargs['avg_reward'],
                "training/avg_episode_length": kwargs['avg_length'],
                "training/policy_loss": kwargs['avg_policy_loss'],
                "training/value_loss": kwargs['avg_value_loss'],
                "training/entropy": kwargs['avg_entropy'],
                "training/collection_time": kwargs['collection_time'],
                "training/update_time": kwargs['update_time'],
                "training/fps": kwargs['total_episodes'] / (kwargs['collection_time'] + kwargs['update_time']),
            }
            
            # Episode-specific metrics
            recent_episodes = kwargs.get('recent_episodes', [])
            if recent_episodes:
                episode_rewards = [ep.total_reward for ep in recent_episodes]
                episode_lengths = [ep.episode_length for ep in recent_episodes]
                
                log_dict.update({
                    "episodes/reward_mean": np.mean(episode_rewards),
                    "episodes/reward_std": np.std(episode_rewards),
                    "episodes/reward_min": np.min(episode_rewards),
                    "episodes/reward_max": np.max(episode_rewards),
                    "episodes/length_mean": np.mean(episode_lengths),
                    "episodes/length_std": np.std(episode_lengths),
                })
                
                # Reward distribution histogram
                wandb.log({"episodes/reward_distribution": wandb.Histogram(episode_rewards)})
            
            # Model statistics
            if hasattr(self, 'actor_critic'):
                total_params = sum(p.numel() for p in self.actor_critic.parameters())
                trainable_params = sum(p.numel() for p in self.actor_critic.parameters() if p.requires_grad)
                log_dict.update({
                    "model/total_parameters": total_params,
                    "model/trainable_parameters": trainable_params,
                })
            
            # Performance metrics over time
            if len(self.episode_rewards) > 1:
                # Reward trend (simple moving average)
                window = min(20, len(self.episode_rewards))
                recent_rewards = self.episode_rewards[-window:]
                log_dict["training/reward_trend"] = np.mean(recent_rewards)
                
                # Learning progress (improvement over time)
                if len(self.episode_rewards) >= 100:
                    early_rewards = np.mean(self.episode_rewards[:50])
                    recent_rewards = np.mean(self.episode_rewards[-50:])
                    improvement = recent_rewards - early_rewards
                    log_dict["training/learning_progress"] = improvement
            
            wandb.log(log_dict)
            
        except Exception as e:
            print(f"⚠️  Failed to log to wandb: {e}")
    
    def compute_advantages(self, experiences: List[Experience], gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        states = torch.FloatTensor(np.array([exp.state for exp in experiences])).to(self.device)
        rewards = np.array([exp.reward for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        with torch.no_grad():
            values = self.actor_critic.critic_head(self.actor_critic.backbone(states)).cpu().numpy().flatten()
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        last_advantage = 0
        last_return = 0
        
        for t in reversed(range(len(experiences))):
            if t == len(experiences) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            # Calculate TD error
            delta = rewards[t] + gamma * next_value - values[t]
            
            # Calculate advantage using GAE
            advantages[t] = last_advantage = delta + gamma * lam * last_advantage
            
            # Calculate return
            returns[t] = last_return = rewards[t] + gamma * last_return
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def ppo_update(self, experiences: List[Experience]) -> Dict[str, float]:
        """Perform PPO update on a batch of experiences."""
        if len(experiences) < self.config.batch_size:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(experiences)
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([exp.log_prob for exp in experiences]).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # PPO update epochs
        for epoch in range(self.config.update_epochs):
            # Shuffle data
            indices = torch.randperm(len(experiences))
            
            for start_idx in range(0, len(experiences), self.config.batch_size):
                end_idx = start_idx + self.config.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                if len(batch_indices) < self.config.batch_size:
                    continue
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_probs, state_values = self.actor_critic(batch_states)
                
                # Calculate new log probabilities
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                # Calculate ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                
                # Total loss
                total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Store losses for logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses)
        }
    
    def train(self):
        """Main training loop."""
        print("Starting PPO training for Slay the Spire")
        print(f"Configuration: {self.config.__dict__}")
        
        total_episodes = 0
        update_count = 0
        
        while total_episodes < self.config.num_episodes:
            # Collect episodes
            print(f"\nCollecting {self.config.collect_episodes_per_update} episodes...")
            start_time = time.time()
            
            episodes = []
            for _ in range(self.config.collect_episodes_per_update):
                episode = self.collector.collect_episode_with_policy(self.actor_critic)
                episodes.append(episode)
                
                self.episode_rewards.append(episode.total_reward)
                self.episode_lengths.append(episode.episode_length)
                total_episodes += 1
            
            collection_time = time.time() - start_time
            
            # Prepare experiences for training
            all_experiences = []
            for episode in episodes:
                all_experiences.extend(episode.experiences)
            
            # Perform PPO update
            print(f"Performing PPO update on {len(all_experiences)} experiences...")
            update_start_time = time.time()
            
            losses = self.ppo_update(all_experiences)
            
            if losses:
                self.policy_losses.append(losses['policy_loss'])
                self.value_losses.append(losses['value_loss'])
                self.entropy_losses.append(losses['entropy'])
            
            update_time = time.time() - update_start_time
            update_count += 1
            
            # Logging
            if update_count % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-50:]) if self.episode_lengths else 0
                avg_policy_loss = np.mean(self.policy_losses[-10:]) if self.policy_losses else 0
                avg_value_loss = np.mean(self.value_losses[-10:]) if self.value_losses else 0
                avg_entropy = np.mean(self.entropy_losses[-10:]) if self.entropy_losses else 0
                
                # Console logging
                print(f"\n=== Update {update_count} | Episodes {total_episodes} ===")
                print(f"Avg Reward (last 50): {avg_reward:.3f}")
                print(f"Avg Length (last 50): {avg_length:.1f}")
                print(f"Policy Loss: {avg_policy_loss:.6f}")
                print(f"Value Loss: {avg_value_loss:.6f}")
                print(f"Entropy: {avg_entropy:.6f}")
                print(f"Collection Time: {collection_time:.2f}s")
                print(f"Update Time: {update_time:.2f}s")
                
                # Wandb logging
                if self.wandb_run:
                    self._log_to_wandb(
                        update_count=update_count,
                        total_episodes=total_episodes,
                        avg_reward=avg_reward,
                        avg_length=avg_length,
                        avg_policy_loss=avg_policy_loss,
                        avg_value_loss=avg_value_loss,
                        avg_entropy=avg_entropy,
                        collection_time=collection_time,
                        update_time=update_time,
                        recent_episodes=episodes
                    )
            
            # Save model
            if update_count % self.config.save_interval == 0:
                self.save_model(f"model_update_{update_count}.pt")
                print(f"Model saved at update {update_count}")
        
        print(f"\nTraining completed! Total episodes: {total_episodes}")
        self.save_model("final_model.pt")
        
        # Final wandb logging
        if self.wandb_run:
            wandb.log({
                "training/final_episodes": total_episodes,
                "training/final_avg_reward": np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0,
                "training/total_updates": update_count
            })
            # Save final model to wandb
            wandb.save(os.path.join(self.config.model_save_path, "final_model.pt"))
            wandb.finish()
    
    def save_model(self, filename: str):
        """Save the trained model."""
        filepath = os.path.join(self.config.model_save_path, filename)
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.entropy_losses = checkpoint.get('entropy_losses', [])

def test_training():
    """Test the training system with a quick run."""
    print("=== Testing STS Training System ===\n")
    
    # Create a minimal configuration for testing
    config = TrainingConfig(
        num_episodes=20,
        collect_episodes_per_update=2,
        batch_size=32,
        update_epochs=2,
        log_interval=1,
        save_interval=5
    )
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    print("Trainer initialized successfully")
    print(f"Model parameters: {sum(p.numel() for p in trainer.actor_critic.parameters()):,}")
    
    # Run a short training session
    trainer.train()

if __name__ == "__main__":
    test_training()