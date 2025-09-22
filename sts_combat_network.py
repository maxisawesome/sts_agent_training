#!/usr/bin/env python3
"""
Combat Network for Two-Network STS Architecture

This network handles all combat decisions using policy-based action selection.
Uses softmax over action space with dynamic masking for valid actions.

Selection strategy: Softmax over all possibilities with action masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import slaythespire

from sts_shared_embeddings import SharedEmbeddingSystem


class CombatVectorProcessor(nn.Module):
    """
    Processes the combat-specific input vector for the Combat Network.

    Handles: enemy info, intent, debuffs, health, buffs, energy, hand,
    piles, block, powers, turn number.
    """

    def __init__(self, output_dim: int = 256):
        super().__init__()

        self.output_dim = output_dim

        # Enemy information
        self.enemy_type_embedding = nn.Embedding(100, 32)  # ~100 different enemies
        self.enemy_intent_embedding = nn.Embedding(20, 16)  # Various intent types

        # Enemy status effects (debuffs)
        self.enemy_debuff_processor = nn.Linear(20, 32)  # Up to 20 different debuffs

        # Player status effects (buffs)
        self.player_buff_processor = nn.Linear(20, 32)  # Up to 20 different buffs

        # Powers in play (categorical with counts)
        self.powers_processor = nn.Linear(40, 32)  # Up to 20 powers × 2 (type + count)

        # Hand cards processing
        self.hand_processor = nn.Sequential(
            nn.Linear(20, 64),  # Up to 10 cards × 2 (ID + cost)
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Numerical features
        self.numerical_processor = nn.Linear(8, 32)  # energy, block, turn, enemy_hp, enemy_max_hp, intent_damage, etc.

        # Combine all features
        total_features = 32 + 16 + 32 + 32 + 32 + 32 + 32  # enemy + intent + debuffs + buffs + powers + hand + numerical
        self.combiner = nn.Sequential(
            nn.Linear(total_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, combat_data: Dict) -> torch.Tensor:
        """
        Process combat vector into unified representation.

        Args:
            combat_data: Dictionary with combat state information

        Returns:
            Combat vector embedding of shape (output_dim,)
        """
        device = next(self.parameters()).device

        # Enemy information
        enemy_emb = self.enemy_type_embedding(combat_data['enemy_type'].to(device))
        intent_emb = self.enemy_intent_embedding(combat_data['enemy_intent'].to(device))

        # Status effects
        enemy_debuffs_emb = self.enemy_debuff_processor(combat_data['enemy_debuffs'].to(device).float())
        player_buffs_emb = self.player_buff_processor(combat_data['player_buffs'].to(device).float())

        # Powers
        powers_emb = self.powers_processor(combat_data['powers_in_play'].to(device).float())

        # Hand cards
        hand_emb = self.hand_processor(combat_data['hand_cards'].to(device).float())

        # Numerical features
        numerical = torch.tensor([
            combat_data['energy'] / 10.0,  # Normalize energy
            combat_data['block'] / 100.0,  # Normalize block
            combat_data['turn_number'] / 50.0,  # Normalize turn
            combat_data['enemy_hp'] / 500.0,  # Normalize enemy HP
            combat_data['enemy_max_hp'] / 500.0,  # Normalize enemy max HP
            combat_data['intent_damage'] / 100.0,  # Normalize intent damage
            combat_data['player_hp'] / 100.0,  # Normalize player HP
            combat_data['player_max_hp'] / 100.0  # Normalize player max HP
        ], device=device, dtype=torch.float32)
        numerical_emb = self.numerical_processor(numerical)

        # Combine all features
        features = torch.cat([
            enemy_emb, intent_emb, enemy_debuffs_emb, player_buffs_emb,
            powers_emb, hand_emb, numerical_emb
        ])

        return self.combiner(features)


class CombatNetwork(nn.Module):
    """
    Policy-based network for Combat decisions.

    Input: Game state + Full deck + Combat vector
    Output: Action probabilities over unified action space with masking
    """

    def __init__(self,
                 shared_embeddings: SharedEmbeddingSystem,
                 action_space_size: int = 512,  # Unified combat action space
                 game_state_dim: int = 128,
                 deck_dim: int = 64,
                 combat_dim: int = 256,
                 hidden_dim: int = 512):
        super().__init__()

        self.shared_embeddings = shared_embeddings
        self.combat_processor = CombatVectorProcessor(combat_dim)
        self.action_space_size = action_space_size

        # Calculate total input dimension
        # Game state + deck + relics + combat vector
        total_input_dim = game_state_dim + deck_dim + 32 + combat_dim  # 32 is relic dim

        # Main network layers
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_space_size)  # Action logits
        )

        # Value head (for actor-critic training if needed)
        self.value_head = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self,
                game_state_emb: torch.Tensor,
                deck_emb: torch.Tensor,
                relic_emb: torch.Tensor,
                combat_data: Dict,
                action_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for combat action selection.

        Args:
            game_state_emb: Game state embedding from shared system
            deck_emb: Deck embedding from shared system
            relic_emb: Relic embedding from shared system
            combat_data: Combat vector data
            action_mask: Boolean mask for valid actions (action_space_size,)

        Returns:
            Tuple of (action_probabilities, state_value)
        """
        # Process combat vector
        combat_emb = self.combat_processor(combat_data)

        # Ensure all tensors are 1D for concatenation
        if game_state_emb.dim() > 1:
            game_state_emb = game_state_emb.squeeze()
        if deck_emb.dim() > 1:
            deck_emb = deck_emb.squeeze()
        if relic_emb.dim() > 1:
            relic_emb = relic_emb.squeeze()
        if combat_emb.dim() > 1:
            combat_emb = combat_emb.squeeze()

        # Combine all inputs
        combined_input = torch.cat([game_state_emb, deck_emb, relic_emb, combat_emb])

        # Get action logits and state value
        action_logits = self.network(combined_input)
        state_value = self.value_head(combined_input)

        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid actions to very negative values
            masked_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        else:
            masked_logits = action_logits

        # Convert to probabilities
        action_probs = F.softmax(masked_logits, dim=-1)

        return action_probs, state_value

    def sample_action(self,
                     shared_data: Dict[str, torch.Tensor],
                     combat_data: Dict,
                     action_mask: torch.Tensor = None,
                     temperature: float = 1.0) -> Tuple[int, float, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            shared_data: Output from shared embedding system
            combat_data: Combat-specific data
            action_mask: Boolean mask for valid actions
            temperature: Temperature for action sampling

        Returns:
            Tuple of (action_index, log_probability, state_value)
        """
        with torch.no_grad():
            action_probs, state_value = self.forward(
                shared_data['game_state'],
                shared_data['deck']['full_deck'],
                shared_data['relics'],
                combat_data,
                action_mask
            )

            # Apply temperature
            if temperature != 1.0:
                action_probs = torch.pow(action_probs, 1.0 / temperature)
                action_probs = action_probs / action_probs.sum()

            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
            log_prob = action_dist.log_prob(torch.tensor(action)).item()

            return action, log_prob, state_value

    def get_action_logits(self,
                         shared_data: Dict[str, torch.Tensor],
                         combat_data: Dict,
                         action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Get raw action logits (useful for training).

        Args:
            shared_data: Output from shared embedding system
            combat_data: Combat-specific data
            action_mask: Boolean mask for valid actions

        Returns:
            Action logits tensor
        """
        # Process combat vector
        combat_emb = self.combat_processor(combat_data)

        # Ensure all tensors are 1D for concatenation
        game_state_emb = shared_data['game_state']
        deck_emb = shared_data['deck']['full_deck']
        relic_emb = shared_data['relics']

        if game_state_emb.dim() > 1:
            game_state_emb = game_state_emb.squeeze()
        if deck_emb.dim() > 1:
            deck_emb = deck_emb.squeeze()
        if relic_emb.dim() > 1:
            relic_emb = relic_emb.squeeze()
        if combat_emb.dim() > 1:
            combat_emb = combat_emb.squeeze()

        # Combine all inputs
        combined_input = torch.cat([game_state_emb, deck_emb, relic_emb, combat_emb])

        # Get action logits
        action_logits = self.network(combined_input)

        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))

        return action_logits


class CombatActionSpace:
    """
    Defines the unified combat action space and handles action masking.

    Covers: card play, targeting, potions, end turn, card selection, etc.
    """

    def __init__(self, action_space_size: int = 512):
        self.action_space_size = action_space_size

        # Action type ranges
        self.card_play_start = 0
        self.card_play_count = 200  # 10 hand positions × 10 targets × 2 for upgrades

        self.potion_start = 200
        self.potion_count = 30  # 3 potion slots × 10 targets

        self.end_turn_action = 230

        self.card_select_start = 231  # For cards like Seek, Exhume
        self.card_select_count = 200  # Various card selection actions

        self.special_actions_start = 431
        self.special_actions_count = 81  # Remaining actions for special cases

    def create_action_mask(self, combat_state: Dict) -> torch.Tensor:
        """
        Create boolean mask for valid actions based on current combat state.

        Args:
            combat_state: Dictionary with current combat information

        Returns:
            Boolean tensor of shape (action_space_size,) indicating valid actions
        """
        mask = torch.zeros(self.action_space_size, dtype=torch.bool)

        # Always allow end turn
        mask[self.end_turn_action] = True

        # Card play actions (based on energy and hand)
        energy = combat_state.get('energy', 0)
        hand_cards = combat_state.get('hand_cards', [])
        num_targets = combat_state.get('num_targets', 1)

        for card_idx, card_cost in enumerate(hand_cards):
            if card_cost <= energy and card_idx < 10:  # Max 10 cards in hand
                # Allow playing this card against all valid targets
                for target_idx in range(num_targets):
                    action_idx = self.card_play_start + (card_idx * 20) + target_idx
                    if action_idx < self.card_play_start + self.card_play_count:
                        mask[action_idx] = True

        # Potion actions (based on available potions)
        potions = combat_state.get('potions', [])
        for potion_idx in range(len(potions)):
            if potion_idx < 3:  # Max 3 potions
                for target_idx in range(num_targets):
                    action_idx = self.potion_start + (potion_idx * 10) + target_idx
                    if action_idx < self.potion_start + self.potion_count:
                        mask[action_idx] = True

        # Card selection actions (for cards like Seek)
        if combat_state.get('card_selection_active', False):
            # Enable card selection actions based on available cards
            available_cards = combat_state.get('selectable_cards', [])
            for card_idx in available_cards:
                if card_idx < self.card_select_count:
                    action_idx = self.card_select_start + card_idx
                    mask[action_idx] = True

        return mask

    def decode_action(self, action_index: int) -> Dict:
        """
        Decode action index into action type and parameters.

        Args:
            action_index: Action index in the unified space

        Returns:
            Dictionary with action type and parameters
        """
        if action_index == self.end_turn_action:
            return {'type': 'end_turn'}

        elif self.card_play_start <= action_index < self.card_play_start + self.card_play_count:
            rel_idx = action_index - self.card_play_start
            card_idx = rel_idx // 20
            target_idx = rel_idx % 20
            return {'type': 'play_card', 'card_index': card_idx, 'target_index': target_idx}

        elif self.potion_start <= action_index < self.potion_start + self.potion_count:
            rel_idx = action_index - self.potion_start
            potion_idx = rel_idx // 10
            target_idx = rel_idx % 10
            return {'type': 'use_potion', 'potion_index': potion_idx, 'target_index': target_idx}

        elif self.card_select_start <= action_index < self.card_select_start + self.card_select_count:
            card_idx = action_index - self.card_select_start
            return {'type': 'select_card', 'card_index': card_idx}

        else:
            return {'type': 'special', 'action_index': action_index}


def extract_combat_data_from_observation(observation: np.ndarray,
                                        game_context: slaythespire.GameContext) -> Dict:
    """
    Extract combat-specific data from the observation and game context.

    This would need BattleContext integration for full implementation.
    """
    # Placeholder implementation
    return {
        'enemy_type': torch.tensor(0),
        'enemy_intent': torch.tensor(0),
        'enemy_debuffs': torch.zeros(20),
        'player_buffs': torch.zeros(20),
        'powers_in_play': torch.zeros(40),
        'hand_cards': torch.zeros(20),
        'energy': 3,
        'block': 0,
        'turn_number': 1,
        'enemy_hp': 100,
        'enemy_max_hp': 100,
        'intent_damage': 10,
        'player_hp': game_context.cur_hp,
        'player_max_hp': game_context.max_hp
    }


class CombatNetworkTrainer:
    """
    Trainer for the Combat Network using policy-based learning.
    """

    def __init__(self,
                 network: CombatNetwork,
                 learning_rate: float = 1e-4):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

    def train_step(self,
                   shared_data: Dict[str, torch.Tensor],
                   combat_data: Dict,
                   action_taken: int,
                   target_value: float,
                   action_mask: torch.Tensor = None) -> float:
        """
        Single training step for a combat experience.

        Args:
            shared_data: Shared embedding data
            combat_data: Combat-specific data
            action_taken: Index of the action that was taken
            target_value: Target value for this state
            action_mask: Valid action mask

        Returns:
            Training loss
        """
        self.optimizer.zero_grad()

        # Forward pass
        action_probs, predicted_value = self.network.forward(
            shared_data['game_state'],
            shared_data['deck']['full_deck'],
            shared_data['relics'],
            combat_data,
            action_mask
        )

        # Policy loss (cross entropy with action taken)
        action_target = torch.tensor(action_taken, device=action_probs.device)
        policy_loss = self.policy_loss_fn(action_probs.unsqueeze(0), action_target.unsqueeze(0))

        # Value loss
        target = torch.tensor(target_value, device=predicted_value.device)
        value_loss = self.value_loss_fn(predicted_value.squeeze(), target)

        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item()


def test_combat_network():
    """Test the Combat Network."""
    print("=== Testing Combat Network ===\n")

    # Create shared embedding system
    shared_emb = SharedEmbeddingSystem()

    # Create combat network
    combat_network = CombatNetwork(shared_emb, action_space_size=512)
    action_space = CombatActionSpace(action_space_size=512)

    print(f"Combat network parameters: {sum(p.numel() for p in combat_network.parameters()):,}")

    # Create dummy input data
    game_state_emb = torch.randn(128)
    deck_emb = torch.randn(64)
    relic_emb = torch.randn(32)

    combat_data = {
        'enemy_type': torch.tensor(5),
        'enemy_intent': torch.tensor(2),
        'enemy_debuffs': torch.randn(20),
        'player_buffs': torch.randn(20),
        'powers_in_play': torch.randn(40),
        'hand_cards': torch.randn(20),
        'energy': 3,
        'block': 12,
        'turn_number': 5,
        'enemy_hp': 80,
        'enemy_max_hp': 120,
        'intent_damage': 15,
        'player_hp': 65,
        'player_max_hp': 80
    }

    # Test forward pass
    print("--- Forward Pass ---")
    action_probs, state_value = combat_network(game_state_emb, deck_emb, relic_emb, combat_data)
    print(f"Action probs shape: {action_probs.shape}")
    print(f"State value: {state_value.item():.4f}")

    # Test action masking
    print("\n--- Action Masking ---")
    combat_state = {
        'energy': 3,
        'hand_cards': [1, 2, 0, 3],  # 4 cards with costs
        'num_targets': 3,
        'potions': [0, 1],  # 2 potions
        'card_selection_active': False
    }

    action_mask = action_space.create_action_mask(combat_state)
    print(f"Valid actions: {action_mask.sum().item()} / {len(action_mask)}")

    # Test masked forward pass
    masked_probs, _ = combat_network(game_state_emb, deck_emb, relic_emb, combat_data, action_mask)
    print(f"Masked action probabilities sum: {masked_probs.sum().item():.4f}")

    # Test action sampling
    print("\n--- Action Sampling ---")
    shared_data = {
        'game_state': game_state_emb,
        'deck': {'full_deck': deck_emb},
        'relics': relic_emb
    }

    action, log_prob, value = combat_network.sample_action(shared_data, combat_data, action_mask)
    print(f"Sampled action: {action}")
    print(f"Log probability: {log_prob:.4f}")
    print(f"State value: {value.item():.4f}")

    # Test action decoding
    print("\n--- Action Decoding ---")
    decoded = action_space.decode_action(action)
    print(f"Decoded action: {decoded}")

    print("\n✓ Combat Network working!")


if __name__ == "__main__":
    test_combat_network()