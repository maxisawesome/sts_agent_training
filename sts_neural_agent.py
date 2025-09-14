#!/usr/bin/env python3
"""
Slay the Spire Neural Network Agent

This module provides a neural network-based agent that can play Slay the Spire
by integrating with the existing Agent system in sts_lightspeed.
"""

import torch
import numpy as np
import sys
import os
from typing import Optional, List, Tuple, Any
import random

import slaythespire

from sts_neural_network import STSActorCritic
from sts_model_manager import STSModelManager

class STSNeuralAgent:
    """
    Neural network-based agent for Slay the Spire.
    
    This agent uses a trained neural network to make decisions in the game,
    bridging the gap between the ML model and the game's action system.
    """
    
    def __init__(self, model_path: Optional[str] = None, temperature: float = 1.0):
        """
        Initialize the neural agent.
        
        Args:
            model_path: Path to trained model file. If None, uses random actions.
            temperature: Temperature for action selection (1.0 = no change, <1.0 = more deterministic)
        """
        self.temperature = temperature
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nn_interface = slaythespire.getNNInterface()
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.model_manager = STSModelManager()
            self.model = self.model_manager.load_model_for_inference(model_path, self.device)
            self.model.eval()
            self.use_neural_network = True
            print(f"Loaded neural network model from: {model_path}")
        else:
            self.model = None
            self.use_neural_network = False
            print("No model provided - using random actions")
        
        # Statistics
        self.action_count = 0
        self.episode_count = 0
    
    def get_action(self, game_context: slaythespire.GameContext) -> int:
        """
        Get the next action to take given the current game state.
        
        This is the main interface method that the game engine can call.
        """
        self.action_count += 1
        
        if not self.use_neural_network:
            return self._get_random_action()
        
        try:
            # Get observation from game state
            observation = self.nn_interface.getObservation(game_context)
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get action probabilities from model
                action_probs, state_value = self.model(obs_tensor)
                
                # Apply temperature for exploration
                if self.temperature != 1.0:
                    action_probs = torch.pow(action_probs, 1.0 / self.temperature)
                    action_probs = action_probs / action_probs.sum()
                
                # Sample action
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
                
                return action
                
        except Exception as e:
            print(f"Error in neural network inference: {e}")
            return self._get_random_action()
    
    def get_action_with_confidence(self, game_context: slaythespire.GameContext) -> Tuple[int, float, float]:
        """
        Get action along with confidence score and state value estimate.
        
        Returns:
            (action, confidence, state_value)
        """
        if not self.use_neural_network:
            return self._get_random_action(), 0.0, 0.0
        
        try:
            observation = self.nn_interface.getObservation(game_context)
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_probs, state_value = self.model(obs_tensor)
                
                # Apply temperature
                if self.temperature != 1.0:
                    action_probs = torch.pow(action_probs, 1.0 / self.temperature)
                    action_probs = action_probs / action_probs.sum()
                
                # Get action and confidence
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
                
                # Confidence is the probability of the chosen action
                confidence = action_probs[0, action].item()
                value = state_value.item()
                
                return action, confidence, value
                
        except Exception as e:
            print(f"Error in neural network inference: {e}")
            return self._get_random_action(), 0.0, 0.0
    
    def get_top_actions(self, game_context: slaythespire.GameContext, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Get the top-k most likely actions with their probabilities.
        
        Returns:
            List of (action, probability) tuples, sorted by probability (descending)
        """
        if not self.use_neural_network:
            # Return random actions
            actions = random.sample(range(256), min(top_k, 256))
            return [(action, 1.0/top_k) for action in actions]
        
        try:
            observation = self.nn_interface.getObservation(game_context)
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_probs, _ = self.model(obs_tensor)
                
                # Apply temperature
                if self.temperature != 1.0:
                    action_probs = torch.pow(action_probs, 1.0 / self.temperature)
                    action_probs = action_probs / action_probs.sum()
                
                # Get top-k actions
                top_probs, top_actions = torch.topk(action_probs.squeeze(), top_k)
                
                return [(action.item(), prob.item()) for action, prob in zip(top_actions, top_probs)]
                
        except Exception as e:
            print(f"Error getting top actions: {e}")
            actions = random.sample(range(256), min(top_k, 256))
            return [(action, 1.0/top_k) for action in actions]
    
    def _get_random_action(self) -> int:
        """Get a random action (fallback when no model is available)."""
        return random.randint(0, 255)  # Assuming 256 possible actions
    
    def reset_episode(self):
        """Reset episode-specific statistics."""
        self.action_count = 0
        self.episode_count += 1
    
    def get_statistics(self) -> dict:
        """Get agent statistics."""
        return {
            'use_neural_network': self.use_neural_network,
            'temperature': self.temperature,
            'action_count': self.action_count,
            'episode_count': self.episode_count,
            'device': str(self.device)
        }
    
    def set_temperature(self, temperature: float):
        """Update the temperature for action selection."""
        self.temperature = max(0.1, temperature)  # Minimum temperature to avoid division by zero

class STSNeuralGameRunner:
    """
    Utility class to run complete games using the neural agent.
    """
    
    def __init__(self, agent: STSNeuralAgent):
        self.agent = agent
        self.game_results = []
    
    def play_game(self, character_class=None, seed: Optional[int] = None, verbose: bool = False) -> dict:
        """
        Play a complete game using the neural agent.
        
        Note: This is a placeholder implementation. The actual game loop would need
        to integrate with the sts_lightspeed game execution system.
        """
        if character_class is None:
            character_class = slaythespire.CharacterClass.IRONCLAD
        
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        # Create game context
        game_context = slaythespire.GameContext(character_class, seed, seed + 1)
        self.agent.reset_episode()
        
        game_result = {
            'character_class': character_class,
            'seed': seed,
            'actions_taken': [],
            'state_values': [],
            'confidences': [],
            'final_hp': game_context.cur_hp,
            'final_gold': game_context.gold,
            'floor_reached': game_context.floor_num,
            'outcome': 'unknown'  # Would be determined by actual game execution
        }
        
        # Simulate some game steps (in real implementation, this would be the actual game loop)
        for step in range(10):  # Placeholder - real game would continue until completion
            action, confidence, state_value = self.agent.get_action_with_confidence(game_context)
            
            game_result['actions_taken'].append(action)
            game_result['confidences'].append(confidence)
            game_result['state_values'].append(state_value)
            
            if verbose:
                print(f"Step {step}: Action={action}, Confidence={confidence:.3f}, Value={state_value:.3f}")
        
        self.game_results.append(game_result)
        return game_result
    
    def play_multiple_games(self, num_games: int, character_class=None, verbose: bool = False) -> List[dict]:
        """Play multiple games and return results."""
        results = []
        
        for game_num in range(num_games):
            if verbose:
                print(f"\n=== Game {game_num + 1}/{num_games} ===")
            
            result = self.play_game(character_class, verbose=verbose)
            results.append(result)
            
            if verbose:
                print(f"Game completed - Final HP: {result['final_hp']}, Gold: {result['final_gold']}")
        
        return results
    
    def get_performance_summary(self) -> dict:
        """Get summary statistics of game performance."""
        if not self.game_results:
            return {}
        
        avg_confidence = np.mean([np.mean(game['confidences']) for game in self.game_results])
        avg_state_value = np.mean([np.mean(game['state_values']) for game in self.game_results])
        avg_actions_per_game = np.mean([len(game['actions_taken']) for game in self.game_results])
        
        return {
            'total_games': len(self.game_results),
            'avg_confidence': avg_confidence,
            'avg_state_value': avg_state_value,
            'avg_actions_per_game': avg_actions_per_game,
            'agent_stats': self.agent.get_statistics()
        }

def test_neural_agent():
    """Test the neural agent system."""
    print("=== Testing STS Neural Agent ===\n")
    
    # Test with random agent (no model)
    print("--- Testing Random Agent ---")
    random_agent = STSNeuralAgent()
    game_context = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 12345, 67890)
    
    print(f"Agent statistics: {random_agent.get_statistics()}")
    
    # Test action selection
    action = random_agent.get_action(game_context)
    print(f"Random action: {action}")
    
    action, confidence, value = random_agent.get_action_with_confidence(game_context)
    print(f"Random action with confidence: action={action}, confidence={confidence}, value={value}")
    
    # Test with trained model (if available)
    print("\n--- Testing Neural Agent ---")
    model_manager = STSModelManager()
    models = model_manager.list_models()
    
    if models:
        latest_model = models[0]['filepath']
        neural_agent = STSNeuralAgent(latest_model, temperature=1.0)
        
        print(f"Neural agent statistics: {neural_agent.get_statistics()}")
        
        # Test action selection
        action = neural_agent.get_action(game_context)
        print(f"Neural action: {action}")
        
        action, confidence, value = neural_agent.get_action_with_confidence(game_context)
        print(f"Neural action with confidence: action={action}, confidence={confidence:.4f}, value={value:.4f}")
        
        # Test top actions
        top_actions = neural_agent.get_top_actions(game_context, top_k=5)
        print(f"Top 5 actions: {[(a, f'{p:.4f}') for a, p in top_actions]}")
        
        # Test game runner
        print("\n--- Testing Game Runner ---")
        game_runner = STSNeuralGameRunner(neural_agent)
        game_result = game_runner.play_game(verbose=True)
        
        print(f"\nGame summary:")
        print(f"Actions taken: {len(game_result['actions_taken'])}")
        print(f"Average confidence: {np.mean(game_result['confidences']):.4f}")
        print(f"Average state value: {np.mean(game_result['state_values']):.4f}")
        
        # Test multiple games
        print(f"\n--- Testing Multiple Games ---")
        results = game_runner.play_multiple_games(3, verbose=False)
        summary = game_runner.get_performance_summary()
        
        print(f"Performance summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    else:
        print("No trained models found. Run training first to test neural agent.")

if __name__ == "__main__":
    test_neural_agent()