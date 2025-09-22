#!/usr/bin/env python3
"""
Input Vector Processing and Network Coordination for Two-Network STS Architecture

This module handles:
- Screen state detection (COMBAT vs EVENTS/PLANNING)
- Input vector extraction from observations and game context
- Coordination between Events/Planning and Combat networks
- Unified interface for both networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import slaythespire

from sts_shared_embeddings import SharedEmbeddingSystem
from sts_events_network import EventsPlanningNetwork, extract_event_data_from_observation
from sts_combat_network import CombatNetwork, extract_combat_data_from_observation


class ScreenState(Enum):
    """Possible screen states in Slay the Spire."""
    COMBAT = "combat"
    EVENT = "event"
    MAP = "map"
    SHOP = "shop"
    REST = "rest"
    CARD_REWARD = "card_reward"
    RELIC_REWARD = "relic_reward"
    UNKNOWN = "unknown"


class STSInputProcessor:
    """
    Main coordinator for input processing and network selection.

    Determines which network to use based on screen state and processes
    input vectors accordingly.
    """

    def __init__(self,
                 shared_embeddings: SharedEmbeddingSystem,
                 events_network: EventsPlanningNetwork,
                 combat_network: CombatNetwork):
        self.shared_embeddings = shared_embeddings
        self.events_network = events_network
        self.combat_network = combat_network

        # Screen state detector
        self.screen_detector = ScreenStateDetector()

    def process_and_select_action(self,
                                observation: np.ndarray,
                                game_context: slaythespire.GameContext) -> Tuple[int, float, Dict]:
        """
        Process observation and select action using appropriate network.

        Args:
            observation: Raw observation array
            game_context: Current game context

        Returns:
            Tuple of (action_index, confidence, debug_info)
        """
        # Detect current screen state
        screen_state = self.screen_detector.detect_screen_state(observation, game_context)

        # Process shared data
        shared_data = self.shared_embeddings.process_full_game_state(observation, game_context)

        # Route to appropriate network
        if screen_state == ScreenState.COMBAT:
            return self._process_combat_action(shared_data, observation, game_context)
        else:
            return self._process_events_action(shared_data, observation, game_context, screen_state)

    def _process_combat_action(self,
                             shared_data: Dict[str, torch.Tensor],
                             observation: np.ndarray,
                             game_context: slaythespire.GameContext) -> Tuple[int, float, Dict]:
        """Process combat action using CombatNetwork."""

        # Extract combat-specific data
        combat_data = extract_combat_data_from_observation(observation, game_context)

        # Get action probabilities from combat network
        action_probs, state_value, action_mask = self.combat_network.forward_with_masking(
            shared_data['game_state'],
            shared_data['deck']['full_deck'],
            shared_data['relics'],
            combat_data
        )

        # Select best valid action
        valid_probs = action_probs.masked_fill(~action_mask, 0.0)
        action_index = torch.argmax(valid_probs).item()
        confidence = valid_probs[action_index].item()

        debug_info = {
            'network_used': 'combat',
            'state_value': state_value.item(),
            'valid_actions': action_mask.sum().item(),
            'top_3_actions': torch.topk(valid_probs, k=min(3, action_mask.sum().item())).indices.tolist()
        }

        return action_index, confidence, debug_info

    def _process_events_action(self,
                             shared_data: Dict[str, torch.Tensor],
                             observation: np.ndarray,
                             game_context: slaythespire.GameContext,
                             screen_state: ScreenState) -> Tuple[int, float, Dict]:
        """Process events/planning action using EventsPlanningNetwork."""

        # Extract event-specific data
        event_data = extract_event_data_from_observation(observation, game_context)

        # Create valid choice mask based on screen state
        valid_choice_mask = self._get_valid_choice_mask(observation, game_context, screen_state)

        # Get choice values from events network
        choice_values, best_choice = self.events_network.evaluate_all_choices(
            shared_data,
            event_data,
            valid_choice_mask
        )

        confidence = torch.softmax(choice_values, dim=0)[best_choice].item()

        debug_info = {
            'network_used': 'events',
            'screen_state': screen_state.value,
            'choice_values': choice_values.tolist(),
            'valid_choices': valid_choice_mask.sum().item() if valid_choice_mask is not None else 4
        }

        return best_choice, confidence, debug_info

    def _get_valid_choice_mask(self,
                             observation: np.ndarray,
                             game_context: slaythespire.GameContext,
                             screen_state: ScreenState) -> Optional[torch.Tensor]:
        """
        Determine which choices are valid based on screen state and game context.

        This is a simplified implementation - in practice would analyze the
        generic choice vectors to determine validity.
        """
        # For now, assume all 4 choices are potentially valid
        # In full implementation, would check:
        # - Gold requirements for shop purchases
        # - Card upgrade availability for rest sites
        # - Valid map paths
        # - Event choice prerequisites

        # Simplified: assume first 2-4 choices are valid based on screen
        if screen_state == ScreenState.MAP:
            # Usually 1-3 path choices
            return torch.tensor([True, True, True, False])
        elif screen_state == ScreenState.SHOP:
            # Usually 4 purchase options (cards, relics, remove, leave)
            return torch.tensor([True, True, True, True])
        elif screen_state == ScreenState.REST:
            # Rest, upgrade, recall (if available)
            return torch.tensor([True, True, False, False])
        else:
            # Events usually have 2-4 choices
            return torch.tensor([True, True, True, False])


class ScreenStateDetector:
    """
    Detects current screen state from observation and game context.

    Uses heuristics based on observation values and game context state
    to determine which screen the player is currently viewing.
    """

    def detect_screen_state(self,
                           observation: np.ndarray,
                           game_context: slaythespire.GameContext) -> ScreenState:
        """
        Detect current screen state.

        Args:
            observation: Raw observation array (550 dimensions)
            game_context: Current game context

        Returns:
            Detected screen state
        """
        # This is a simplified heuristic-based implementation
        # In practice, would analyze specific observation indices or
        # use the game context's internal state representation

        # Check if we're in combat by looking for monster HP
        if self._is_in_combat(observation, game_context):
            return ScreenState.COMBAT

        # Check for specific screen indicators in observation
        # These indices would need to be determined from observation space analysis

        # Placeholder logic - would need proper observation space mapping
        if self._has_shop_indicators(observation):
            return ScreenState.SHOP
        elif self._has_event_indicators(observation):
            return ScreenState.EVENT
        elif self._has_map_indicators(observation):
            return ScreenState.MAP
        elif self._has_rest_indicators(observation):
            return ScreenState.REST
        elif self._has_reward_indicators(observation):
            # Would distinguish between card and relic rewards
            return ScreenState.CARD_REWARD

        return ScreenState.UNKNOWN

    def _is_in_combat(self,
                     observation: np.ndarray,
                     game_context: slaythespire.GameContext) -> bool:
        """Check if currently in combat."""
        # Simplified: check if there are monsters with HP > 0
        # In full implementation, would check specific observation indices
        # or use game context battle state

        # Placeholder: assume combat if we have energy/hand size indicators
        # Would need to map these to actual observation indices
        return False  # Simplified for now

    def _has_shop_indicators(self, observation: np.ndarray) -> bool:
        """Check for shop screen indicators."""
        # Would check for shop-specific observation features
        return False

    def _has_event_indicators(self, observation: np.ndarray) -> bool:
        """Check for event screen indicators."""
        # Would check generic choice vectors and event-specific features
        return False

    def _has_map_indicators(self, observation: np.ndarray) -> bool:
        """Check for map screen indicators."""
        # Would check for map paths and room type indicators
        return False

    def _has_rest_indicators(self, observation: np.ndarray) -> bool:
        """Check for rest site indicators."""
        return False

    def _has_reward_indicators(self, observation: np.ndarray) -> bool:
        """Check for reward screen indicators."""
        return False


class InputVectorExtractor:
    """
    Utility class for extracting specific vector types from observations.

    Handles the detailed parsing of the 550-dimensional observation space
    to extract game state, event, and combat vectors.
    """

    @staticmethod
    def extract_game_state_vector(observation: np.ndarray,
                                game_context: slaythespire.GameContext) -> Dict:
        """
        Extract general game state information.

        Covers: HP, gold, floor, ascension, boss type, potions, map state,
        events seen, previous elite, act information.
        """
        # Based on observation space analysis from previous work
        # HP and gold are likely in early indices

        game_state = {
            'hp': game_context.cur_hp,
            'max_hp': game_context.max_hp,
            'gold': game_context.gold,
            'floor': game_context.floor_num,
            'ascension': 0,  # Would extract from observation or context
            'boss_type': 0,  # Would extract from observation (boss_type_idx)
            'potion_types': [0],  # Would extract from observation
            'previous_elite': 0,  # Would track from game history
            'act': game_context.act,
            'events_seen_count': 0,  # Would track from game history
            'events_seen': np.zeros(60),  # Would track from game history
            'map_state': np.zeros(20)  # Would extract from observation
        }

        return game_state

    @staticmethod
    def extract_event_vector(observation: np.ndarray,
                           game_context: slaythespire.GameContext) -> Dict:
        """
        Extract event-specific information.

        Covers: current event type, generic choice vectors (4×20 features).
        """
        # Based on previous analysis, generic choices start at index 470
        choice_start_idx = 470
        choice_vectors = []

        for choice_idx in range(4):
            start_idx = choice_start_idx + (choice_idx * 20)
            end_idx = start_idx + 20
            choice_vector = observation[start_idx:end_idx]
            choice_vectors.append(choice_vector)

        return {
            'event_type': 0,  # Would extract from screen state analysis
            'choice_vectors': np.array(choice_vectors)  # Shape: (4, 20)
        }

    @staticmethod
    def extract_combat_vector(observation: np.ndarray,
                            game_context: slaythespire.GameContext) -> Dict:
        """
        Extract combat-specific information.

        Covers: enemy info, intent, debuffs, health, buffs, energy,
        hand cards, draw/discard/exhaust piles, block, powers, turn number.
        """
        # This would require detailed observation space mapping
        # Placeholder implementation

        combat_data = {
            'enemy_types': [0],  # Would extract from observation
            'enemy_intents': [0],  # Would extract from observation
            'enemy_intent_values': [0],  # Would extract from observation
            'enemy_healths': [50],  # Would extract from observation
            'enemy_max_healths': [50],  # Would extract from observation
            'enemy_debuffs': {},  # Would extract from observation
            'player_buffs': {},  # Would extract from observation
            'energy': 3,  # Would extract from observation
            'hand_cards': [],  # Would extract from observation
            'hand_costs': [],  # Would extract from observation
            'block': 0,  # Would extract from observation
            'powers': {},  # Would extract from observation
            'turn_number': 1  # Would extract from observation or context
        }

        return combat_data


def test_input_processor():
    """Test the input processing system."""
    print("=== Testing Input Processing System ===\n")

    # Create components
    shared_emb = SharedEmbeddingSystem()
    events_network = EventsPlanningNetwork(shared_emb)
    combat_network = CombatNetwork(shared_emb)

    # Create input processor
    processor = STSInputProcessor(shared_emb, events_network, combat_network)

    print(f"Input processor created successfully")
    print(f"Components: SharedEmbeddings, EventsNetwork, CombatNetwork")

    # Test screen state detection
    print("\n--- Screen State Detection ---")
    detector = ScreenStateDetector()

    # Create dummy observation and context
    observation = np.random.randn(550)

    # Mock game context (simplified)
    class MockGameContext:
        def __init__(self):
            self.cur_hp = 80
            self.max_hp = 80
            self.gold = 99
            self.floor_num = 5
            self.act = 1

    game_context = MockGameContext()

    screen_state = detector.detect_screen_state(observation, game_context)
    print(f"Detected screen state: {screen_state}")

    # Test vector extraction
    print("\n--- Vector Extraction ---")
    extractor = InputVectorExtractor()

    game_state = extractor.extract_game_state_vector(observation, game_context)
    print(f"Game state keys: {list(game_state.keys())}")

    event_data = extractor.extract_event_vector(observation, game_context)
    print(f"Event data shape: {event_data['choice_vectors'].shape}")

    combat_data = extractor.extract_combat_vector(observation, game_context)
    print(f"Combat data keys: {list(combat_data.keys())}")

    print("\n✓ Input processing system working!")


if __name__ == "__main__":
    test_input_processor()