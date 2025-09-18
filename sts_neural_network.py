#!/usr/bin/env python3
"""
Slay the Spire Neural Network Architecture for Reinforcement Learning

This module defines the neural network models for training an AI agent
to play Slay the Spire using the sts_lightspeed simulation engine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

import slaythespire

class STSPolicyNetwork(nn.Module):
    """
    Policy network for Slay the Spire agent.
    
    Takes the 550-dimensional game state observation and outputs action probabilities.
    Uses separate heads for different action types (combat vs meta-game decisions).
    """
    
    def __init__(self, obs_size=550, hidden_size=512, num_layers=3):
        super(STSPolicyNetwork, self).__init__()
        
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        
        # Shared backbone network
        layers = []
        layers.append(nn.Linear(obs_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        self.backbone = nn.Sequential(*layers)
        
        # Action heads - will need to be refined based on actual action space
        # For now, using generic action outputs
        self.action_head = nn.Linear(hidden_size, 256)  # Placeholder action space size
        
    def forward(self, obs):
        """Forward pass through the policy network."""
        features = self.backbone(obs)
        action_logits = self.action_head(features)
        return F.softmax(action_logits, dim=-1)

class STSValueNetwork(nn.Module):
    """
    Value network for Slay the Spire agent.
    
    Takes the 550-dimensional game state observation and outputs a state value estimate.
    """
    
    def __init__(self, obs_size=550, hidden_size=512, num_layers=3):
        super(STSValueNetwork, self).__init__()
        
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        
        # Value network layers
        layers = []
        layers.append(nn.Linear(obs_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(hidden_size, 1))  # Single value output
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs):
        """Forward pass through the value network."""
        return self.network(obs)

class STSActorCritic(nn.Module):
    """
    Combined Actor-Critic network for Slay the Spire.
    
    Shares a common backbone between policy and value networks for efficiency.
    """
    
    def __init__(self, obs_size=550, hidden_size=512, action_size=256, num_layers=3):
        super(STSActorCritic, self).__init__()
        
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        
        # Shared backbone
        backbone_layers = []
        backbone_layers.append(nn.Linear(obs_size, hidden_size))
        backbone_layers.append(nn.ReLU())
        backbone_layers.append(nn.Dropout(0.2))
        
        for _ in range(num_layers - 2):
            backbone_layers.append(nn.Linear(hidden_size, hidden_size))
            backbone_layers.append(nn.ReLU())
            backbone_layers.append(nn.Dropout(0.2))
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, obs):
        """Forward pass returning both policy and value."""
        features = self.backbone(obs)
        
        action_logits = self.actor_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        state_value = self.critic_head(features)
        
        return action_probs, state_value
    
    def get_action_and_value(self, obs, action=None):
        """Get action probabilities, value, and optionally evaluate a specific action."""
        action_probs, value = self.forward(obs)
        
        if action is None:
            # Sample action from policy
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
        else:
            # Evaluate given action
            action_dist = torch.distributions.Categorical(action_probs)
            action_log_prob = action_dist.log_prob(action)
        
        return action, action_log_prob, value, action_probs

def test_networks():
    """Test the neural network architectures with sample data."""
    print("=== Testing Neural Network Architectures ===\n")
    
    # Create sample observation (550-dimensional state from sts_lightspeed)
    batch_size = 4
    obs_size = 550
    sample_obs = torch.randn(batch_size, obs_size)
    
    print(f"Sample observation shape: {sample_obs.shape}")
    
    # Test individual networks
    print("\n--- Testing Policy Network ---")
    policy_net = STSPolicyNetwork()
    policy_output = policy_net(sample_obs)
    print(f"Policy output shape: {policy_output.shape}")
    print(f"Policy output sum (should be ~1): {policy_output.sum(dim=1)}")
    
    print("\n--- Testing Value Network ---")
    value_net = STSValueNetwork()
    value_output = value_net(sample_obs)
    print(f"Value output shape: {value_output.shape}")
    print(f"Sample values: {value_output.squeeze()}")
    
    print("\n--- Testing Actor-Critic Network ---")
    actor_critic = STSActorCritic()
    action_probs, state_values = actor_critic(sample_obs)
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"State values shape: {state_values.shape}")
    
    # Test action sampling
    actions, log_probs, values, probs = actor_critic.get_action_and_value(sample_obs)
    print(f"Sampled actions: {actions}")
    print(f"Action log probabilities: {log_probs}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in actor_critic.parameters())
    trainable_params = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def test_with_real_observation():
    """Test networks with real observation from sts_lightspeed."""
    print("\n=== Testing with Real STS Observation ===")
    
    try:
        # Create a real game context
        game_context = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 1234567890, 0)
        nn_interface = slaythespire.getNNInterface()
        
        # Get real observation
        observation = nn_interface.getObservation(game_context)
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)  # Add batch dimension
        
        print(f"Real observation shape: {obs_tensor.shape}")
        print(f"First 10 features: {observation[:10]}")
        
        # Test with actor-critic network
        actor_critic = STSActorCritic()
        with torch.no_grad():
            action_probs, state_value = actor_critic(obs_tensor)
            
        print(f"Action probabilities shape: {action_probs.shape}")
        print(f"State value: {state_value.item():.4f}")
        print(f"Top 5 action probabilities: {torch.topk(action_probs.squeeze(), 5).values}")
        
    except Exception as e:
        print(f"Error testing with real observation: {e}")

if __name__ == "__main__":
    test_networks()
    test_with_real_observation()