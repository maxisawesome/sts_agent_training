#!/usr/bin/env python3
"""
Events/Planning Network for Two-Network STS Architecture

This network handles all non-combat decisions using value-based evaluation.
Covers: event choices, map actions, card rewards, relic rewards, shops, rest sites.

Selection strategy: Loop over all possible actions, choose highest value option.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import slaythespire

from sts_shared_embeddings import SharedEmbeddingSystem


class EventVectorProcessor(nn.Module):
    """
    Processes the event-specific input vector for the Events/Planning Network.

    Handles current event type and generic choice vectors (4 choices × 20 features).
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.output_dim = output_dim

        # Current event embedding (~60 different events)
        self.event_type_embedding = nn.Embedding(70, 32)  # Extra space for unknown events

        # Generic choice processing (4 choices × 20 features each)
        self.choice_processor = nn.Sequential(
            nn.Linear(20, 32),  # Process each choice's 20 features
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Combine all choices (4 × 32 = 128)
        self.choices_combiner = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Final combination of event type + choices
        self.final_combiner = nn.Sequential(
            nn.Linear(32 + 64, output_dim),  # event_type + choices
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, event_data: Dict) -> torch.Tensor:
        """
        Process event vector into unified representation.

        Args:
            event_data: Dictionary with:
                - event_type: Current event ID
                - choice_vectors: Tensor of shape (4, 20) with choice features

        Returns:
            Event vector embedding of shape (output_dim,)
        """
        device = next(self.parameters()).device

        # Process event type
        event_type_emb = self.event_type_embedding(event_data['event_type'].to(device))

        # Process each choice vector
        choice_vectors = event_data['choice_vectors'].to(device)  # Shape: (4, 20)
        choice_embeddings = []

        for i in range(4):  # Process each of the 4 choices
            choice_emb = self.choice_processor(choice_vectors[i])
            choice_embeddings.append(choice_emb)

        # Combine all choice embeddings
        choices_combined = torch.cat(choice_embeddings, dim=0)  # Shape: (128,)
        choices_processed = self.choices_combiner(choices_combined)

        # Final combination
        features = torch.cat([event_type_emb, choices_processed])
        return self.final_combiner(features)


class EventsPlanningNetwork(nn.Module):
    """
    Value-based network for Events/Planning decisions.

    Input: Game state + Full deck + Event vector
    Output: Single value per choice option (for discrete choice evaluation)
    """

    def __init__(self,
                 shared_embeddings: SharedEmbeddingSystem,
                 game_state_dim: int = 128,
                 deck_dim: int = 64,
                 event_dim: int = 128,
                 hidden_dim: int = 512):
        super().__init__()

        self.shared_embeddings = shared_embeddings
        self.event_processor = EventVectorProcessor(event_dim)

        # Calculate total input dimension
        # Game state + deck + relics + event vector
        total_input_dim = game_state_dim + deck_dim + 32 + event_dim  # 32 is relic dim

        # Main network layers
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)  # Single value output
        )

    def forward(self,
                game_state_emb: torch.Tensor,
                deck_emb: torch.Tensor,
                relic_emb: torch.Tensor,
                event_data: Dict,
                choice_index: int = None) -> torch.Tensor:
        """
        Forward pass for a specific choice evaluation.

        Args:
            game_state_emb: Game state embedding from shared system
            deck_emb: Deck embedding from shared system
            relic_emb: Relic embedding from shared system
            event_data: Event vector data
            choice_index: Which choice to evaluate (if None, process all)

        Returns:
            Value estimate for the choice(s)
        """
        # Process event vector
        event_emb = self.event_processor(event_data)

        # Ensure all tensors are 1D for concatenation
        if game_state_emb.dim() > 1:
            game_state_emb = game_state_emb.squeeze()
        if deck_emb.dim() > 1:
            deck_emb = deck_emb.squeeze()
        if relic_emb.dim() > 1:
            relic_emb = relic_emb.squeeze()
        if event_emb.dim() > 1:
            event_emb = event_emb.squeeze()

        # Combine all inputs
        combined_input = torch.cat([game_state_emb, deck_emb, relic_emb, event_emb])

        # Get value estimate
        value = self.network(combined_input)
        return value

    def evaluate_all_choices(self,
                           shared_data: Dict[str, torch.Tensor],
                           event_data: Dict,
                           valid_choice_mask: torch.Tensor = None) -> Tuple[torch.Tensor, int]:
        """
        Evaluate all available choices and return the best one.

        Args:
            shared_data: Output from shared embedding system
            event_data: Event-specific data
            valid_choice_mask: Boolean mask for valid choices (4,)

        Returns:
            Tuple of (choice_values, best_choice_index)
        """
        choice_values = []

        # Evaluate each choice
        for choice_idx in range(4):
            # Create choice-specific event data
            choice_event_data = event_data.copy()

            # For now, use the same event data for all choices
            # In full implementation, would modify based on choice_idx
            value = self.forward(
                shared_data['game_state'],
                shared_data['deck']['full_deck'],
                shared_data['relics'],
                choice_event_data,
                choice_idx
            )
            choice_values.append(value)

        choice_values = torch.stack(choice_values).squeeze()

        # Apply valid choice mask if provided
        if valid_choice_mask is not None:
            # Set invalid choices to very negative values
            choice_values = choice_values.masked_fill(~valid_choice_mask, float('-inf'))

        # Get best choice
        best_choice = torch.argmax(choice_values).item()

        return choice_values, best_choice


class EventsNetworkTrainer:
    """
    Trainer for the Events/Planning Network using value-based learning.
    """

    def __init__(self,
                 network: EventsPlanningNetwork,
                 learning_rate: float = 1e-4):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def train_step(self,
                   shared_data: Dict[str, torch.Tensor],
                   event_data: Dict,
                   choice_index: int,
                   target_value: float) -> float:
        """
        Single training step for a choice-value pair.

        Args:
            shared_data: Shared embedding data
            event_data: Event-specific data
            choice_index: Index of the choice that was taken
            target_value: Target value for this choice

        Returns:
            Training loss
        """
        self.optimizer.zero_grad()

        # Forward pass
        predicted_value = self.network.forward(
            shared_data['game_state'],
            shared_data['deck']['full_deck'],
            shared_data['relics'],
            event_data,
            choice_index
        )

        # Calculate loss
        target = torch.tensor(target_value, device=predicted_value.device, dtype=torch.float32)
        loss = self.loss_fn(predicted_value.squeeze(), target)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


def extract_event_data_from_observation(observation: np.ndarray,
                                      game_context: slaythespire.GameContext) -> Dict:
    """
    Extract event-specific data from the observation and game context.

    This analyzes the generic choice vectors from the observation space.
    """
    # Based on the observation space analysis, generic choices start at index 470
    # Format: 4 choices × 20 features each
    choice_start_idx = 470
    choice_vectors = []

    for choice_idx in range(4):
        start_idx = choice_start_idx + (choice_idx * 20)
        end_idx = start_idx + 20
        choice_vector = observation[start_idx:end_idx]
        choice_vectors.append(choice_vector)

    choice_tensor = torch.FloatTensor(np.array(choice_vectors))  # Shape: (4, 20)

    # Extract current event type (simplified - would need proper mapping)
    event_type = 0  # Placeholder - would extract from screen state and observation

    return {
        'event_type': torch.tensor(event_type, dtype=torch.long),  # Categorical
        'choice_vectors': choice_tensor
    }


def test_events_network():
    """Test the Events/Planning Network."""
    print("=== Testing Events/Planning Network ===\n")

    # Create shared embedding system
    shared_emb = SharedEmbeddingSystem()

    # Create events network
    events_network = EventsPlanningNetwork(shared_emb)

    print(f"Events network parameters: {sum(p.numel() for p in events_network.parameters()):,}")

    # Create dummy input data
    game_state_emb = torch.randn(128)
    deck_emb = torch.randn(64)
    relic_emb = torch.randn(32)

    event_data = {
        'event_type': torch.tensor(5),
        'choice_vectors': torch.randn(4, 20)
    }

    # Test single choice evaluation
    print("\n--- Single Choice Evaluation ---")
    value = events_network(game_state_emb, deck_emb, relic_emb, event_data, choice_index=0)
    print(f"Choice 0 value: {value.item():.4f}")

    # Test all choices evaluation
    print("\n--- All Choices Evaluation ---")
    shared_data = {
        'game_state': game_state_emb,
        'deck': {'full_deck': deck_emb},
        'relics': relic_emb
    }

    choice_values, best_choice = events_network.evaluate_all_choices(shared_data, event_data)
    print(f"Choice values: {choice_values}")
    print(f"Best choice: {best_choice}")

    # Test with valid choice mask
    print("\n--- Masked Choice Evaluation ---")
    valid_mask = torch.tensor([True, True, False, False])  # Only first 2 choices valid
    choice_values, best_choice = events_network.evaluate_all_choices(shared_data, event_data, valid_mask)
    print(f"Masked choice values: {choice_values}")
    print(f"Best valid choice: {best_choice}")

    # Test trainer
    print("\n--- Training Test ---")
    trainer = EventsNetworkTrainer(events_network)
    loss = trainer.train_step(shared_data, event_data, choice_index=1, target_value=0.8)
    print(f"Training loss: {loss:.4f}")

    print("\n✓ Events/Planning Network working!")


if __name__ == "__main__":
    test_events_network()