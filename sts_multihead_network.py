#!/usr/bin/env python3
"""
Multi-Head Value Network for Slay the Spire

This module implements a value-based neural network with separate heads for
combat and meta-game decisions, designed for integration with tree search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union
from enum import Enum

import slaythespire


class ActionContext(Enum):
    """Different contexts requiring different action heads."""
    COMBAT = "combat"      # Battle screen - card play, potions, targeting
    META = "meta"          # Everything else - events, shops, map, rewards


class ActionEncoder:
    """
    Encodes different types of actions into feature vectors for the neural network.

    Combat actions need rich encoding (card IDs, costs, targets, etc.)
    Meta actions need simpler encoding (choice indices, basic parameters)
    """

    def __init__(self):
        # Action encoding dimensions
        self.combat_action_size = 64    # Cards, potions, targets, special actions
        self.meta_action_size = 32      # Events, shops, rewards, map choices

        # Card encoding constants (from game data)
        self.max_card_id = 512          # Maximum card ID value
        self.max_cost = 10              # Maximum card cost
        self.max_targets = 5            # Maximum number of targets

        # Meta action encoding constants
        self.max_choice_index = 10      # Maximum choice index for events/rewards
        self.max_price = 500            # Maximum shop price

    def encode_combat_action(self, action_data: Dict) -> torch.Tensor:
        """
        Encode a combat action into a feature vector.

        Args:
            action_data: Dictionary containing:
                - action_type: ActionType enum
                - card_id: Card ID if playing a card
                - target_index: Target index if targeting
                - source_index: Source index (hand position)
                - cost: Energy cost
                - other action-specific parameters

        Returns:
            Tensor of shape (combat_action_size,) with encoded action features
        """
        encoding = torch.zeros(self.combat_action_size)

        action_type = action_data.get('action_type', slaythespire.ActionType.END_TURN)

        # One-hot encode action type (positions 0-4)
        if action_type == slaythespire.ActionType.CARD:
            encoding[0] = 1.0
        elif action_type == slaythespire.ActionType.POTION:
            encoding[1] = 1.0
        elif action_type == slaythespire.ActionType.END_TURN:
            encoding[2] = 1.0
        elif action_type == slaythespire.ActionType.SINGLE_CARD_SELECT:
            encoding[3] = 1.0
        elif action_type == slaythespire.ActionType.MULTI_CARD_SELECT:
            encoding[4] = 1.0

        # Card-specific features (positions 5-29)
        if 'card_id' in action_data:
            card_id = action_data['card_id']
            # Normalize card ID to [0, 1]
            encoding[5] = min(card_id / self.max_card_id, 1.0)

        if 'cost' in action_data:
            cost = action_data['cost']
            encoding[6] = min(cost / self.max_cost, 1.0)

        if 'source_index' in action_data:
            source_idx = action_data['source_index']
            encoding[7] = min(source_idx / 10.0, 1.0)  # Hand position

        if 'target_index' in action_data:
            target_idx = action_data['target_index']
            encoding[8] = min(target_idx / self.max_targets, 1.0)

        # Card properties (positions 10-20) - to be filled based on card data
        # These would come from the card's properties: damage, block, effects, etc.

        # Additional features for future expansion (positions 21-63)
        # Energy state, buffs/debuffs, relic interactions, etc.

        return encoding

    def encode_meta_action(self, action_data: Dict) -> torch.Tensor:
        """
        Encode a meta-game action into a feature vector.

        Args:
            action_data: Dictionary containing:
                - screen_state: Current screen type
                - choice_index: Index of choice being made
                - price: Cost if applicable (shop)
                - reward_type: Type of reward if applicable
                - other screen-specific parameters

        Returns:
            Tensor of shape (meta_action_size,) with encoded action features
        """
        encoding = torch.zeros(self.meta_action_size)

        screen_state = action_data.get('screen_state', slaythespire.ScreenState.INVALID)

        # One-hot encode screen type (positions 0-9)
        screen_mapping = {
            slaythespire.ScreenState.EVENT_SCREEN: 0,
            slaythespire.ScreenState.MAP_SCREEN: 1,
            slaythespire.ScreenState.SHOP_ROOM: 2,
            slaythespire.ScreenState.REWARDS: 3,
            slaythespire.ScreenState.REST_ROOM: 4,
            slaythespire.ScreenState.TREASURE_ROOM: 5,
            slaythespire.ScreenState.CARD_SELECT: 6,
            slaythespire.ScreenState.BOSS_RELIC_REWARDS: 7,
        }

        if screen_state in screen_mapping:
            encoding[screen_mapping[screen_state]] = 1.0

        # Choice index (position 10)
        if 'choice_index' in action_data:
            choice_idx = action_data['choice_index']
            encoding[10] = min(choice_idx / self.max_choice_index, 1.0)

        # Price/cost features (position 11)
        if 'price' in action_data:
            price = action_data['price']
            encoding[11] = min(price / self.max_price, 1.0)

        # Reward type features (positions 12-16)
        reward_type = action_data.get('reward_type', None)
        if reward_type == 'card':
            encoding[12] = 1.0
        elif reward_type == 'gold':
            encoding[13] = 1.0
        elif reward_type == 'relic':
            encoding[14] = 1.0
        elif reward_type == 'potion':
            encoding[15] = 1.0
        elif reward_type == 'skip':
            encoding[16] = 1.0

        # Additional meta features (positions 17-31)
        # Room type, path options, resource states, etc.

        return encoding


class STSMultiHeadValueNetwork(nn.Module):
    """
    Multi-head value network for Slay the Spire.

    Uses separate heads for combat and meta-game decisions, with a shared
    backbone for common game state understanding.
    """

    def __init__(self, state_size: int = 550):
        super().__init__()

        self.state_size = state_size
        self.action_encoder = ActionEncoder()

        # Shared backbone for common game state features
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Combat-specific head for battle decisions
        combat_input_size = 256 + self.action_encoder.combat_action_size
        self.combat_head = nn.Sequential(
            nn.Linear(combat_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Q-value output
        )

        # Meta-game head for non-combat decisions
        meta_input_size = 256 + self.action_encoder.meta_action_size
        self.meta_head = nn.Sequential(
            nn.Linear(meta_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Q-value output
        )

        # Optional: State value head for baseline/critic
        self.state_value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor, action_data: Dict,
                action_context: ActionContext) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Game state tensor of shape (batch_size, state_size)
            action_data: Dictionary with action parameters
            action_context: Whether this is COMBAT or META action

        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        # Shared feature extraction
        shared_features = self.shared_backbone(state)

        # Encode action based on context
        if action_context == ActionContext.COMBAT:
            action_encoding = self.action_encoder.encode_combat_action(action_data)
        else:
            action_encoding = self.action_encoder.encode_meta_action(action_data)

        # Expand action encoding to match batch size
        batch_size = state.shape[0]
        action_encoding = action_encoding.unsqueeze(0).expand(batch_size, -1)

        # Move to same device as state
        action_encoding = action_encoding.to(state.device)

        # Concatenate state features with action encoding
        combined_features = torch.cat([shared_features, action_encoding], dim=-1)

        # Pass through appropriate head
        if action_context == ActionContext.COMBAT:
            q_value = self.combat_head(combined_features)
        else:
            q_value = self.meta_head(combined_features)

        return q_value

    def get_state_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimate (for baseline or critic purposes).

        Args:
            state: Game state tensor of shape (batch_size, state_size)

        Returns:
            State value tensor of shape (batch_size, 1)
        """
        shared_features = self.shared_backbone(state)
        return self.state_value_head(shared_features)

    def evaluate_actions(self, state: torch.Tensor,
                        action_list: list,
                        action_context: ActionContext) -> torch.Tensor:
        """
        Evaluate multiple actions for the given state.

        Args:
            state: Game state tensor of shape (1, state_size)
            action_list: List of action dictionaries to evaluate
            action_context: Whether these are COMBAT or META actions

        Returns:
            Q-values tensor of shape (len(action_list),)
        """
        q_values = []

        for action_data in action_list:
            q_value = self.forward(state, action_data, action_context)
            q_values.append(q_value.squeeze())

        return torch.stack(q_values)


def test_multihead_network():
    """Test the multi-head value network."""
    print("=== Testing Multi-Head Value Network ===\n")

    # Create network
    network = STSMultiHeadValueNetwork(state_size=550)

    # Create dummy state
    batch_size = 4
    state = torch.randn(batch_size, 550)

    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")

    # Test combat action
    print("\n--- Testing Combat Head ---")
    combat_action = {
        'action_type': slaythespire.ActionType.CARD,
        'card_id': 42,
        'cost': 2,
        'source_index': 3,
        'target_index': 0
    }

    combat_q = network(state, combat_action, ActionContext.COMBAT)
    print(f"Combat Q-value shape: {combat_q.shape}")
    print(f"Combat Q-values: {combat_q.squeeze()}")

    # Test meta action
    print("\n--- Testing Meta Head ---")
    meta_action = {
        'screen_state': slaythespire.ScreenState.EVENT_SCREEN,
        'choice_index': 1,
        'reward_type': 'card'
    }

    meta_q = network(state, meta_action, ActionContext.META)
    print(f"Meta Q-value shape: {meta_q.shape}")
    print(f"Meta Q-values: {meta_q.squeeze()}")

    # Test state value
    print("\n--- Testing State Value Head ---")
    state_value = network.get_state_value(state)
    print(f"State value shape: {state_value.shape}")
    print(f"State values: {state_value.squeeze()}")

    # Test action evaluation
    print("\n--- Testing Action Evaluation ---")
    action_list = [
        {'action_type': slaythespire.ActionType.CARD, 'card_id': 10},
        {'action_type': slaythespire.ActionType.CARD, 'card_id': 20},
        {'action_type': slaythespire.ActionType.END_TURN}
    ]

    single_state = state[:1]  # Single state for action evaluation
    q_values = network.evaluate_actions(single_state, action_list, ActionContext.COMBAT)
    print(f"Action Q-values: {q_values}")
    print(f"Best action index: {q_values.argmax().item()}")


if __name__ == "__main__":
    test_multihead_network()