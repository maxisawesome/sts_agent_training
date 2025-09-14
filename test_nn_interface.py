#!/usr/bin/env python3
import sys
import os
import numpy as np

import slaythespire

def analyze_nn_interface():
    print("=== Neural Network Interface Analysis ===\n")
    
    # Get the NNInterface instance
    nn_interface = slaythespire.getNNInterface()
    
    # Create a sample game context to understand the observation space
    print("Creating sample game...")
    # Note: We'll need to understand how to create a GameContext
    # For now, let's explore what we can from the interface
    
    # The observation_space_size is a static constant, not an instance variable
    print("Observation space size: 412 (from code analysis)")
    
    # Get observation maximums to understand the feature ranges
    obs_maxs = nn_interface.getObservationMaximums()
    print(f"Observation maximums shape: {len(obs_maxs)}")
    print(f"First 20 maximum values: {list(obs_maxs[:20])}")
    
    print("\n=== Observation Space Structure ===")
    print("Based on the code analysis:")
    print("- Features 0-3: Player HP (current), Player HP (max), Gold, Floor number")
    print("- Features 4-13: Boss type (one-hot encoded, 10 possible bosses)")
    print("- Features 14-233: Deck composition (220 features for card counts)")
    print("- Features 234-411: Relics (178 binary features)")
    print(f"Total: {4 + 10 + 220 + 178} = 412 features")
    
    return nn_interface

def test_game_creation():
    print("\n=== Testing Game Creation ===")
    
    # Test creating a GameContext - needs CharacterClass, seed1, seed2
    try:
        # Use Ironclad character class, and some seed values
        game_context = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 12345, 67890)
        print("Successfully created GameContext with Ironclad character")
        
        nn_interface = slaythespire.getNNInterface()
        observation = nn_interface.getObservation(game_context)
        
        print(f"Observation shape: {len(observation)}")
        print(f"First 20 features: {list(observation[:20])}")
        print(f"Features 4-13 (boss encoding): {list(observation[4:14])}")
        
        return game_context, observation
        
    except Exception as e:
        print(f"Error creating GameContext: {e}")
        return None, None

if __name__ == "__main__":
    nn_interface = analyze_nn_interface()
    game_context, observation = test_game_creation()