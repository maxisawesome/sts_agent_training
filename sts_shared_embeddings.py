#!/usr/bin/env python3
"""
Shared Embedding Infrastructure for Two-Network STS Architecture

This module provides unified card and relic embeddings that are shared between
the Events/Planning Network and Combat Network. It also handles game state
vector processing and pile embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import slaythespire


class CardEmbedding(nn.Module):
    """
    Unified card embedding system shared between both networks.

    Handles card IDs, upgrades, costs, types, rarities with support for
    pile embeddings (draw, discard, exhaust) with order-independence.
    """

    def __init__(self, embedding_dim: int = 64, max_cards: int = 512):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_cards = max_cards

        # Main card ID embedding
        self.card_id_embedding = nn.Embedding(max_cards + 1, embedding_dim // 2)

        # Card property embeddings
        self.cost_embedding = nn.Embedding(11, 8)  # 0-10 cost + X cost
        self.rarity_embedding = nn.Embedding(5, 8)  # Basic, Common, Uncommon, Rare, Special
        self.type_embedding = nn.Embedding(4, 8)   # Attack, Skill, Power, Status/Curse
        self.upgrade_embedding = nn.Embedding(21, 8)  # 0-20 upgrades (Searing Blow)

        # Combine all features
        feature_size = (embedding_dim // 2) + 8 + 8 + 8 + 8  # id + cost + rarity + type + upgrade
        self.feature_combiner = nn.Linear(feature_size, embedding_dim)

        # For pile embeddings (order-independent)
        self.pile_aggregator = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, card_data: Dict) -> torch.Tensor:
        """
        Embed a single card or batch of cards.

        Args:
            card_data: Dictionary with keys:
                - card_ids: Tensor of card IDs
                - costs: Tensor of card costs
                - rarities: Tensor of card rarities
                - types: Tensor of card types
                - upgrades: Tensor of upgrade counts

        Returns:
            Card embeddings of shape (batch_size, embedding_dim)
        """
        device = next(self.parameters()).device

        # Get embeddings for each feature
        id_emb = self.card_id_embedding(card_data['card_ids'].to(device))
        cost_emb = self.cost_embedding(card_data['costs'].to(device))
        rarity_emb = self.rarity_embedding(card_data['rarities'].to(device))
        type_emb = self.type_embedding(card_data['types'].to(device))
        upgrade_emb = self.upgrade_embedding(card_data['upgrades'].to(device))

        # Concatenate all features
        features = torch.cat([id_emb, cost_emb, rarity_emb, type_emb, upgrade_emb], dim=-1)

        # Combine into final embedding
        return self.feature_combiner(features)

    def embed_pile(self, pile_cards: List[Dict]) -> torch.Tensor:
        """
        Embed a pile of cards (draw, discard, exhaust) with order-independence.

        Args:
            pile_cards: List of card dictionaries (max 75 cards)

        Returns:
            Pile embedding of shape (embedding_dim,)
        """
        if not pile_cards:
            return torch.zeros(self.embedding_dim, device=next(self.parameters()).device)

        # Embed each card in the pile
        card_embeddings = []
        for card in pile_cards:
            card_emb = self.forward(card)
            card_embeddings.append(card_emb)

        # Stack and average for order-independence
        pile_tensor = torch.stack(card_embeddings, dim=0)
        pile_avg = torch.mean(pile_tensor, dim=0)

        # Apply aggregation transformation
        return self.pile_aggregator(pile_avg)


class RelicEmbedding(nn.Module):
    """
    Unified relic embedding system shared between both networks.

    Handles relic IDs, counters, stack counts with support for relic
    interactions and synergies.
    """

    def __init__(self, embedding_dim: int = 32, max_relics: int = 200):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_relics = max_relics

        # Main relic ID embedding
        self.relic_id_embedding = nn.Embedding(max_relics + 1, embedding_dim - 8)

        # Counter/stack embedding (for relics with counters)
        self.counter_embedding = nn.Linear(1, 8)  # Numerical counter value

        # Combine ID and counter
        self.feature_combiner = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, relic_data: Dict) -> torch.Tensor:
        """
        Embed relics with their counter information.

        Args:
            relic_data: Dictionary with keys:
                - relic_ids: Tensor of relic IDs
                - counters: Tensor of counter values

        Returns:
            Relic embeddings of shape (batch_size, embedding_dim)
        """
        device = next(self.parameters()).device

        # Get relic ID embeddings
        id_emb = self.relic_id_embedding(relic_data['relic_ids'].to(device))

        # Handle counter values
        counters = relic_data['counters'].to(device).unsqueeze(-1).float()
        counter_emb = self.counter_embedding(counters)

        # Combine features
        features = torch.cat([id_emb, counter_emb], dim=-1)
        return self.feature_combiner(features)

    def embed_relic_collection(self, relics: List[Dict]) -> torch.Tensor:
        """
        Embed a collection of relics into a fixed-size representation.

        Args:
            relics: List of relic dictionaries

        Returns:
            Collection embedding of shape (embedding_dim,)
        """
        if not relics:
            return torch.zeros(self.embedding_dim, device=next(self.parameters()).device)

        # Embed each relic
        relic_embeddings = []
        for relic in relics:
            relic_emb = self.forward(relic)
            relic_embeddings.append(relic_emb)

        # Sum embeddings (relics are additive effects)
        relic_tensor = torch.stack(relic_embeddings, dim=0)
        return torch.sum(relic_tensor, dim=0)


class GameStateProcessor(nn.Module):
    """
    Processes the core game state vector that goes into both networks.

    Handles: HP, gold, floor, ascension, boss type, potions, map state,
    events seen, previous elite, act information.
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.output_dim = output_dim

        # Categorical embeddings
        self.boss_embedding = nn.Embedding(11, 16)  # 10 bosses + unknown
        self.potion_embedding = nn.Embedding(50, 16)  # ~40 potions + combinations
        self.elite_embedding = nn.Embedding(20, 16)  # Various elites
        self.act_embedding = nn.Embedding(4, 8)  # Acts 1-3 + special

        # Numerical feature processing
        self.numerical_processor = nn.Linear(6, 32)  # HP, max_HP, gold, floor, ascension, events_seen_count

        # Events seen (binary vector - assume ~60 events)
        self.events_processor = nn.Linear(60, 32)

        # Map state (placeholder - complex representation)
        self.map_processor = nn.Linear(20, 32)  # Simplified map representation

        # Combine all features
        total_features = 16 + 16 + 16 + 8 + 32 + 32 + 32  # boss + potion + elite + act + numerical + events + map
        self.combiner = nn.Sequential(
            nn.Linear(total_features, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, game_state: Dict) -> torch.Tensor:
        """
        Process game state into a unified representation.

        Args:
            game_state: Dictionary with game state information

        Returns:
            Game state embedding of shape (output_dim,)
        """
        device = next(self.parameters()).device

        # Categorical features
        boss_emb = self.boss_embedding(game_state['boss_type'].to(device))
        potion_emb = self.potion_embedding(game_state['potion_types'].to(device)).sum(dim=0)  # Sum multiple potions
        elite_emb = self.elite_embedding(game_state['previous_elite'].to(device))
        act_emb = self.act_embedding(game_state['act'].to(device))

        # Numerical features
        numerical = torch.tensor([
            game_state['hp'] / 100.0,  # Normalize HP
            game_state['max_hp'] / 100.0,
            game_state['gold'] / 1000.0,  # Normalize gold
            game_state['floor'] / 55.0,  # Normalize floor (max ~55)
            game_state['ascension'] / 20.0,  # Normalize ascension
            game_state['events_seen_count'] / 60.0  # Normalize event count
        ], device=device, dtype=torch.float32)
        numerical_emb = self.numerical_processor(numerical)

        # Events seen (binary vector)
        events_emb = self.events_processor(game_state['events_seen'].to(device).float())

        # Map state (simplified)
        map_emb = self.map_processor(game_state['map_state'].to(device).float())

        # Combine all features
        features = torch.cat([boss_emb, potion_emb, elite_emb, act_emb, numerical_emb, events_emb, map_emb])
        return self.combiner(features)


class SharedEmbeddingSystem:
    """
    Coordinator for all shared embeddings between the two networks.
    """

    def __init__(self, card_dim: int = 64, relic_dim: int = 32, game_state_dim: int = 128):
        self.card_embedding = CardEmbedding(card_dim)
        self.relic_embedding = RelicEmbedding(relic_dim)
        self.game_state_processor = GameStateProcessor(game_state_dim)

    def get_modules(self) -> Dict[str, nn.Module]:
        """Get all embedding modules for training."""
        return {
            'card_embedding': self.card_embedding,
            'relic_embedding': self.relic_embedding,
            'game_state_processor': self.game_state_processor
        }

    def process_full_game_state(self, observation: np.ndarray, game_context: slaythespire.GameContext) -> Dict[str, torch.Tensor]:
        """
        Process the full game state into embeddings for both networks.

        Args:
            observation: Raw observation array from sts_lightspeed
            game_context: Current game context

        Returns:
            Dictionary with processed embeddings
        """
        # This would extract information from the observation and game context
        # For now, return placeholder structure

        # Extract game state info
        game_state = self._extract_game_state(observation, game_context)
        game_state_emb = self.game_state_processor(game_state)

        # Extract deck info
        deck_cards = self._extract_deck_info(observation, game_context)
        deck_emb = self._embed_deck(deck_cards)

        # Extract relic info
        relics = self._extract_relic_info(observation, game_context)
        relic_emb = self.relic_embedding.embed_relic_collection(relics)

        return {
            'game_state': game_state_emb,
            'deck': deck_emb,
            'relics': relic_emb,
            'raw_observation': torch.FloatTensor(observation)
        }

    def _extract_game_state(self, observation: np.ndarray, game_context: slaythespire.GameContext) -> Dict:
        """Extract game state information from observation and context."""
        # Placeholder implementation - would extract from actual game state
        return {
            'hp': torch.tensor(game_context.cur_hp),
            'max_hp': torch.tensor(game_context.max_hp),
            'gold': torch.tensor(game_context.gold),
            'floor': torch.tensor(game_context.floor_num),
            'ascension': torch.tensor(0),  # Would get from game context
            'boss_type': torch.tensor(0),  # Would extract from observation
            'potion_types': torch.tensor([0]),  # Would extract from observation
            'previous_elite': torch.tensor(0),  # Would track from game history
            'act': torch.tensor(game_context.act),
            'events_seen_count': torch.tensor(0),  # Would track from game history
            'events_seen': torch.zeros(60),  # Would track from game history
            'map_state': torch.zeros(20)  # Would extract from game state
        }

    def _extract_deck_info(self, observation: np.ndarray, game_context: slaythespire.GameContext) -> Dict:
        """Extract deck information for embedding."""
        # Placeholder - would extract from actual deck
        deck_cards = game_context.deck
        return {
            'draw_pile': [],
            'discard_pile': [],
            'exhaust_pile': [],
            'full_deck': deck_cards
        }

    def _extract_relic_info(self, observation: np.ndarray, game_context: slaythespire.GameContext) -> List[Dict]:
        """Extract relic information for embedding."""
        # Placeholder - would extract from actual relics
        relics = game_context.relics
        return [{'relic_ids': torch.tensor([0]), 'counters': torch.tensor([0])}]

    def _embed_deck(self, deck_info: Dict) -> Dict[str, torch.Tensor]:
        """Embed all deck components."""
        # Placeholder implementation
        return {
            'draw_pile': torch.zeros(64),
            'discard_pile': torch.zeros(64),
            'exhaust_pile': torch.zeros(64),
            'full_deck': torch.zeros(64)
        }


def test_shared_embeddings():
    """Test the shared embedding systems."""
    print("=== Testing Shared Embedding Systems ===\n")

    # Test card embedding
    print("--- Card Embedding ---")
    card_emb = CardEmbedding(embedding_dim=64)

    # Test single card
    card_data = {
        'card_ids': torch.tensor([42]),
        'costs': torch.tensor([2]),
        'rarities': torch.tensor([1]),  # Common
        'types': torch.tensor([0]),     # Attack
        'upgrades': torch.tensor([1])   # Upgraded once
    }

    card_embedding = card_emb(card_data)
    print(f"Card embedding shape: {card_embedding.shape}")

    # Test pile embedding
    pile_cards = [card_data, card_data, card_data]  # 3 cards in pile
    pile_embedding = card_emb.embed_pile(pile_cards)
    print(f"Pile embedding shape: {pile_embedding.shape}")

    # Test relic embedding
    print("\n--- Relic Embedding ---")
    relic_emb = RelicEmbedding(embedding_dim=32)

    relic_data = {
        'relic_ids': torch.tensor([15]),
        'counters': torch.tensor([3])
    }

    relic_embedding = relic_emb(relic_data)
    print(f"Relic embedding shape: {relic_embedding.shape}")

    # Test game state processing
    print("\n--- Game State Processing ---")
    game_state_proc = GameStateProcessor(output_dim=128)

    game_state = {
        'hp': torch.tensor(80),
        'max_hp': torch.tensor(80),
        'gold': torch.tensor(99),
        'floor': torch.tensor(5),
        'ascension': torch.tensor(0),
        'boss_type': torch.tensor(1),
        'potion_types': torch.tensor([0, 1]),
        'previous_elite': torch.tensor(2),
        'act': torch.tensor(1),
        'events_seen_count': torch.tensor(3),
        'events_seen': torch.zeros(60),
        'map_state': torch.zeros(20)
    }

    game_state_embedding = game_state_proc(game_state)
    print(f"Game state embedding shape: {game_state_embedding.shape}")

    print("\n✓ Shared embedding systems working!")


if __name__ == "__main__":
    test_shared_embeddings()