#!/usr/bin/env python3
"""
Action Context Classification for Slay the Spire

This module determines whether a given game state requires combat or meta-game
decision making, and helps route actions to the appropriate neural network head.
"""

import slaythespire
from typing import Dict, Any, List, Tuple
from sts_multihead_network import ActionContext
from sts_action_decoder import STSActionDecoder


class STSActionContextClassifier:
    """
    Classifies game states and actions to determine which neural network head to use.

    Combat context: Battle screen with card play, targeting, potion usage
    Meta context: Everything else - events, shops, map navigation, rewards
    """

    def __init__(self):
        self.action_decoder = STSActionDecoder()

    def get_action_context(self, game_context: slaythespire.GameContext) -> ActionContext:
        """
        Determine whether current game state requires combat or meta decisions.

        Args:
            game_context: Current game state

        Returns:
            ActionContext.COMBAT or ActionContext.META
        """
        screen_state = game_context.screen_state

        # Combat contexts
        if screen_state == slaythespire.ScreenState.BATTLE:
            return ActionContext.COMBAT

        # Meta contexts (everything else)
        return ActionContext.META

    def get_available_actions_with_context(self, game_context: slaythespire.GameContext) -> Tuple[List[Dict], ActionContext]:
        """
        Get available actions and their context for the current game state.

        Args:
            game_context: Current game state

        Returns:
            Tuple of (action_list, action_context)
            - action_list: List of action dictionaries with encoded parameters
            - action_context: ActionContext enum indicating which head to use
        """
        action_context = self.get_action_context(game_context)

        if action_context == ActionContext.COMBAT:
            return self._get_combat_actions(game_context), action_context
        else:
            return self._get_meta_actions(game_context), action_context

    def _get_combat_actions(self, game_context: slaythespire.GameContext) -> List[Dict]:
        """
        Get available combat actions (cards, potions, end turn, etc.).

        Args:
            game_context: Current game state

        Returns:
            List of combat action dictionaries
        """
        # Use the action decoder's combat action support
        generic_actions, action_mapping = self.action_decoder.encode_combat_actions(game_context)

        actions = []
        for generic_action in generic_actions:
            combat_data = action_mapping[generic_action]

            action = {
                'action_type': combat_data['action_type'],
                'generic_action_id': generic_action,
                'description': combat_data['description'],
                'action_bits': combat_data.get('action_bits'),
            }

            # Add type-specific data
            if combat_data['action_type'] == slaythespire.ActionType.CARD:
                action.update({
                    'card_id': combat_data.get('card_id'),
                    'source_index': combat_data.get('source_index'),
                    'cost': combat_data.get('cost')
                })
            elif combat_data['action_type'] == slaythespire.ActionType.POTION:
                action.update({
                    'potion_id': combat_data.get('potion_id'),
                    'source_index': combat_data.get('source_index')
                })

            actions.append(action)

        return actions

    def _get_meta_actions(self, game_context: slaythespire.GameContext) -> List[Dict]:
        """
        Get available meta-game actions (events, shops, map, rewards).

        Args:
            game_context: Current game state

        Returns:
            List of meta action dictionaries
        """
        actions = []
        screen_state = game_context.screen_state

        # Use existing action decoder to get valid actions
        generic_actions, action_mapping = self.action_decoder.encode_actions(game_context)

        for generic_action in generic_actions:
            action_info = self.action_decoder.get_action_type_info(generic_action)
            action_bits = self.action_decoder.decode_action(generic_action, action_mapping)

            # Convert to meta action format
            meta_action = {
                'screen_state': screen_state,
                'choice_index': action_info['relative_index'],
                'generic_action_id': generic_action,
                'action_bits': action_bits,
                'description': f"{action_info['type']} choice {action_info['relative_index']}"
            }

            # Add screen-specific parameters
            if screen_state == slaythespire.ScreenState.SHOP_ROOM:
                meta_action['reward_type'] = 'purchase'
                # Price would come from shop state analysis
                meta_action['price'] = 50  # Placeholder

            elif screen_state == slaythespire.ScreenState.REWARDS:
                # Determine reward type from action index
                if action_info['relative_index'] == 0:
                    meta_action['reward_type'] = 'skip'
                else:
                    meta_action['reward_type'] = 'card'

            elif screen_state == slaythespire.ScreenState.EVENT_SCREEN:
                meta_action['reward_type'] = 'event_choice'

            actions.append(meta_action)

        return actions

    def action_to_network_input(self, action: Dict, action_context: ActionContext) -> Dict:
        """
        Convert an action dictionary to the format expected by the neural network.

        Args:
            action: Action dictionary from get_available_actions_with_context
            action_context: ActionContext.COMBAT or ActionContext.META

        Returns:
            Dictionary suitable for neural network action encoding
        """
        if action_context == ActionContext.COMBAT:
            return {
                'action_type': action.get('action_type', slaythespire.ActionType.END_TURN),
                'card_id': action.get('card_id', 0),
                'cost': action.get('cost', 0),
                'source_index': action.get('source_index', 0),
                'target_index': action.get('target_index', 0)
            }
        else:  # META
            return {
                'screen_state': action.get('screen_state', slaythespire.ScreenState.INVALID),
                'choice_index': action.get('choice_index', 0),
                'price': action.get('price', 0),
                'reward_type': action.get('reward_type', 'unknown')
            }

    def execute_action(self, game_context: slaythespire.GameContext, action: Dict) -> bool:
        """
        Execute the given action in the game context.

        Args:
            game_context: Current game state
            action: Action dictionary to execute

        Returns:
            True if action executed successfully, False otherwise
        """
        # For meta actions, use the existing action decoder
        if 'action_bits' in action:
            return game_context.execute_action(action['action_bits'])

        # For combat actions, we would need BattleContext integration
        # This is a placeholder for now
        print(f"Combat action execution not yet implemented: {action}")
        return False


def test_action_classifier():
    """Test the action context classifier."""
    print("=== Testing Action Context Classifier ===\n")

    classifier = STSActionContextClassifier()

    # Test with different game states
    seeds = [1234, 5678, 9999]

    for seed in seeds:
        print(f"--- Seed {seed} ---")
        gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, seed, 0)

        print(f"Screen: {gc.screen_state}")
        print(f"HP: {gc.cur_hp}/{gc.max_hp}, Floor: {gc.floor_num}")

        # Get action context
        context = classifier.get_action_context(gc)
        print(f"Action context: {context}")

        # Get available actions
        actions, action_context = classifier.get_available_actions_with_context(gc)
        print(f"Available actions: {len(actions)}")

        # Show first few actions
        for i, action in enumerate(actions[:3]):
            print(f"  Action {i}: {action['description']}")

            # Convert to network input format
            network_input = classifier.action_to_network_input(action, action_context)
            print(f"    Network input: {network_input}")

        # Test action execution
        if actions:
            first_action = actions[0]
            print(f"Executing first action: {first_action['description']}")
            success = classifier.execute_action(gc, first_action)
            print(f"Execution success: {success}")

        print()


if __name__ == "__main__":
    test_action_classifier()