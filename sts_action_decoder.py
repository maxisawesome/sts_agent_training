#!/usr/bin/env python3
"""
Slay the Spire Action Decoder

This module provides encoding and decoding between the game's specific action system
and a generic 256-dimensional action space suitable for neural network training.

The generic action space is organized as follows:
- Actions 0-63: Map navigation (4 possible directions × 16 possible nodes)
- Actions 64-127: Card rewards (64 possible card selections)
- Actions 128-191: Event choices (64 possible event options)
- Actions 192-223: Shop actions (32 possible shop items/actions)
- Actions 224-255: Miscellaneous actions (potions, relics, skip, etc.)
"""

import slaythespire
from typing import List, Tuple, Optional, Dict, Any
from enum import IntEnum


class GenericActionType(IntEnum):
    """Generic action types for consistent neural network training."""
    MAP_NAVIGATION = 0      # Actions 0-63
    CARD_REWARD = 64        # Actions 64-127
    EVENT_CHOICE = 128      # Actions 128-191
    SHOP_ACTION = 192       # Actions 192-223
    MISCELLANEOUS = 224     # Actions 224-239
    COMBAT_CARD = 240       # Actions 240-249 (playing cards)
    COMBAT_POTION = 250     # Actions 250-254 (using potions)
    COMBAT_END_TURN = 255   # Action 255 (end turn)


class STSActionDecoder:
    """
    Encoder/Decoder for translating between game-specific actions and generic action space.

    This allows the neural network to work with a consistent 256-dimensional action space
    regardless of the current game state context.
    """

    def __init__(self):
        """Initialize the action decoder."""
        self.action_space_size = 256

        # Action space boundaries
        self.map_action_start = 0
        self.map_action_count = 64

        self.card_reward_start = 64
        self.card_reward_count = 64

        self.event_choice_start = 128
        self.event_choice_count = 64

        self.shop_action_start = 192
        self.shop_action_count = 32

        self.misc_action_start = 224
        self.misc_action_count = 16

        self.combat_card_start = 240
        self.combat_card_count = 10

        self.combat_potion_start = 250
        self.combat_potion_count = 5

        self.combat_end_turn = 255

    def encode_actions(self, game_context: slaythespire.GameContext) -> Tuple[List[int], Dict[int, int]]:
        """
        Encode all valid game actions into generic action space.

        Args:
            game_context: Current game state

        Returns:
            Tuple of:
            - List of generic action indices (0-255) that are valid
            - Mapping from generic action index to game action bits
        """
        valid_actions = game_context.get_valid_actions()
        generic_actions = []
        action_mapping = {}

        screen_state = game_context.screen_state

        for i, action_bits in enumerate(valid_actions):
            try:
                action = slaythespire.GameAction(action_bits)
                generic_idx = self._encode_single_action(action, screen_state, i)

                if generic_idx is not None:
                    generic_actions.append(generic_idx)
                    action_mapping[generic_idx] = action_bits

            except Exception as e:
                # If we can't decode an action, map it to misc space
                generic_idx = self.misc_action_start + (i % self.misc_action_count)
                generic_actions.append(generic_idx)
                action_mapping[generic_idx] = action_bits

        return generic_actions, action_mapping

    def _encode_single_action(self, action: slaythespire.GameAction, screen_state, action_index: int) -> Optional[int]:
        """
        Encode a single game action into generic space.

        Args:
            action: GameAction object
            screen_state: Current screen state
            action_index: Index of this action in the valid actions list

        Returns:
            Generic action index (0-255) or None if encoding fails
        """
        try:
            # Map navigation actions
            if screen_state == slaythespire.ScreenState.MAP_SCREEN:
                return self.map_action_start + (action_index % self.map_action_count)

            # Card reward actions
            elif screen_state == slaythespire.ScreenState.CARD_REWARD:
                return self.card_reward_start + (action_index % self.card_reward_count)

            # Event screen actions
            elif screen_state == slaythespire.ScreenState.EVENT_SCREEN:
                return self.event_choice_start + (action_index % self.event_choice_count)

            # Shop actions
            elif screen_state == slaythespire.ScreenState.SHOP_ROOM:
                return self.shop_action_start + (action_index % self.shop_action_count)

            # Reward screen actions (cards, gold, relics, etc.)
            elif screen_state == slaythespire.ScreenState.REWARDS:
                # Try to categorize by action type
                if action.bits == 0:  # Skip
                    return self.misc_action_start
                else:
                    # Likely card/relic/gold selection
                    return self.card_reward_start + (action_index % self.card_reward_count)

            # Other screens - map to miscellaneous
            else:
                return self.misc_action_start + (action_index % self.misc_action_count)

        except Exception:
            # Fallback to miscellaneous space
            return self.misc_action_start + (action_index % self.misc_action_count)

    def decode_action(self, generic_action: int, action_mapping: Dict[int, int]) -> Optional[int]:
        """
        Decode a generic action back to game action bits.

        Args:
            generic_action: Generic action index (0-255)
            action_mapping: Mapping from generic actions to game action bits

        Returns:
            Game action bits, or None if action is invalid
        """
        return action_mapping.get(generic_action)

    def encode_combat_actions(self, game_context: slaythespire.GameContext) -> Tuple[List[int], Dict[int, Any]]:
        """
        Encode combat actions (cards, potions, end turn) into generic action space.

        Note: This is a placeholder implementation. Full combat action support
        requires BattleContext integration to get actual hand, energy, targets, etc.

        Args:
            game_context: Current game state

        Returns:
            Tuple of:
            - List of generic action indices for combat actions
            - Mapping from generic action to combat action data
        """
        if game_context.screen_state != slaythespire.ScreenState.BATTLE:
            return [], {}

        generic_actions = []
        action_mapping = {}

        # Always include end turn action
        generic_actions.append(self.combat_end_turn)
        action_mapping[self.combat_end_turn] = {
            'action_type': slaythespire.ActionType.END_TURN,
            'action_bits': 0,  # Placeholder - would come from BattleContext
            'description': 'End Turn'
        }

        # Placeholder card actions (would come from actual hand analysis)
        # In real implementation, this would iterate through hand cards
        for i in range(min(self.combat_card_count, 5)):  # Assume max 5 playable cards
            generic_action = self.combat_card_start + i
            generic_actions.append(generic_action)
            action_mapping[generic_action] = {
                'action_type': slaythespire.ActionType.CARD,
                'card_id': 10 + i,  # Placeholder card ID
                'source_index': i,  # Hand position
                'cost': 1,  # Placeholder cost
                'action_bits': i + 1,  # Placeholder action bits
                'description': f'Play Card {i}'
            }

        # Placeholder potion actions
        for i in range(min(self.combat_potion_count, 3)):  # Assume max 3 potions
            generic_action = self.combat_potion_start + i
            generic_actions.append(generic_action)
            action_mapping[generic_action] = {
                'action_type': slaythespire.ActionType.POTION,
                'potion_id': i,
                'source_index': i,
                'action_bits': 100 + i,  # Placeholder action bits
                'description': f'Use Potion {i}'
            }

        return generic_actions, action_mapping

    def is_combat_action(self, generic_action: int) -> bool:
        """Check if a generic action is a combat action."""
        return (self.combat_card_start <= generic_action <= self.combat_card_start + self.combat_card_count - 1 or
                self.combat_potion_start <= generic_action <= self.combat_potion_start + self.combat_potion_count - 1 or
                generic_action == self.combat_end_turn)

    def get_action_type_info(self, generic_action: int) -> Dict[str, Any]:
        """
        Get information about what type of action this generic action represents.

        Args:
            generic_action: Generic action index (0-255)

        Returns:
            Dictionary with action type information
        """
        if self.map_action_start <= generic_action < self.map_action_start + self.map_action_count:
            return {
                'type': 'map_navigation',
                'subtype': 'movement',
                'relative_index': generic_action - self.map_action_start
            }
        elif self.card_reward_start <= generic_action < self.card_reward_start + self.card_reward_count:
            return {
                'type': 'card_reward',
                'subtype': 'selection',
                'relative_index': generic_action - self.card_reward_start
            }
        elif self.event_choice_start <= generic_action < self.event_choice_start + self.event_choice_count:
            return {
                'type': 'event_choice',
                'subtype': 'option',
                'relative_index': generic_action - self.event_choice_start
            }
        elif self.shop_action_start <= generic_action < self.shop_action_start + self.shop_action_count:
            return {
                'type': 'shop_action',
                'subtype': 'purchase',
                'relative_index': generic_action - self.shop_action_start
            }
        elif self.combat_card_start <= generic_action < self.combat_card_start + self.combat_card_count:
            return {
                'type': 'combat_card',
                'subtype': 'play_card',
                'relative_index': generic_action - self.combat_card_start
            }
        elif self.combat_potion_start <= generic_action < self.combat_potion_start + self.combat_potion_count:
            return {
                'type': 'combat_potion',
                'subtype': 'use_potion',
                'relative_index': generic_action - self.combat_potion_start
            }
        elif generic_action == self.combat_end_turn:
            return {
                'type': 'combat_end_turn',
                'subtype': 'end_turn',
                'relative_index': 0
            }
        else:
            return {
                'type': 'miscellaneous',
                'subtype': 'other',
                'relative_index': generic_action - self.misc_action_start
            }

    def get_valid_action_mask(self, game_context: slaythespire.GameContext) -> List[bool]:
        """
        Get a boolean mask indicating which generic actions are valid.

        Args:
            game_context: Current game state

        Returns:
            List of 256 booleans indicating valid actions
        """
        mask = [False] * self.action_space_size
        valid_actions, _ = self.encode_actions(game_context)

        for action in valid_actions:
            mask[action] = True

        return mask


def test_action_decoder():
    """Test the action decoder with various game states."""
    print("=== Testing STS Action Decoder ===\n")

    decoder = STSActionDecoder()

    # Test with different seeds to get variety
    seeds = [1234, 5678, 9999, 4444]

    for seed in seeds:
        print(f"--- Seed {seed} ---")
        gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, seed, 0)

        print(f"Screen: {gc.screen_state}")
        print(f"HP: {gc.cur_hp}/{gc.max_hp}, Gold: {gc.gold}, Floor: {gc.floor_num}")

        # Get raw actions
        raw_actions = gc.get_valid_actions()
        print(f"Raw actions: {raw_actions}")

        # Encode to generic space
        generic_actions, action_mapping = decoder.encode_actions(gc)
        print(f"Generic actions: {generic_actions}")

        # Show action type breakdown
        type_counts = {}
        for action in generic_actions:
            action_info = decoder.get_action_type_info(action)
            action_type = action_info['type']
            type_counts[action_type] = type_counts.get(action_type, 0) + 1

        print(f"Action types: {type_counts}")

        # Test decoding
        for generic_action in generic_actions[:3]:  # Test first 3
            decoded_bits = decoder.decode_action(generic_action, action_mapping)
            action_info = decoder.get_action_type_info(generic_action)
            print(f"  Generic {generic_action} -> bits {decoded_bits} ({action_info['type']})")

        # Test a few steps of gameplay
        step = 0
        while step < 3 and gc.get_valid_actions():
            print(f"\n  Step {step + 1}:")
            generic_actions, action_mapping = decoder.encode_actions(gc)

            if generic_actions:
                # Pick first valid action
                chosen_generic = generic_actions[0]
                chosen_bits = decoder.decode_action(chosen_generic, action_mapping)

                print(f"    Executing generic action {chosen_generic} (bits {chosen_bits})")
                success = gc.execute_action(chosen_bits)
                print(f"    Success: {success}")
                print(f"    New screen: {gc.screen_state}")
            else:
                break

            step += 1

        print()


if __name__ == "__main__":
    test_action_decoder()