#!/usr/bin/env python3
"""
Test script to examine the generic choice features in detail.
"""

import sys
import os
sys.path.insert(0, 'sts_lightspeed')

import slaythespire

def analyze_generic_choices():
    print("=== Generic Choice Features Analysis ===\n")
    
    # Create a game context
    game_context = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 1234567890, 0)
    nn_interface = slaythespire.getNNInterface()
    
    # Get the full observation
    observation = nn_interface.getObservation(game_context)
    
    print(f"Total observation size: {len(observation)}")
    print(f"Expected size: 550")
    print()
    
    # Extract generic choice features (last 80 features)
    choice_features_start = 470  # 412 + 58 (events)
    choice_features = observation[choice_features_start:]
    
    print(f"Generic choice features ({len(choice_features)} features):")
    print(f"Expected: 80 features (4 choices × 20 features per choice)")
    print()
    
    # Analyze each choice option
    for choice_idx in range(4):
        start_idx = choice_idx * 20
        end_idx = start_idx + 20
        choice_data = choice_features[start_idx:end_idx]
        
        print(f"Choice {choice_idx + 1}:")
        print(f"  Choice type flags (8): {choice_data[0:8]}")
        print(f"  Binary flags (2): {choice_data[8:10]}")
        print(f"  Card ID: {choice_data[10]}")
        print(f"  Relic ID: {choice_data[11]}")
        print(f"  Gold amount: {choice_data[12]}")
        print(f"  HP change: {choice_data[13]}")
        print(f"  Max HP change: {choice_data[14]}")
        print(f"  Gold cost: {choice_data[15]}")
        print(f"  HP cost: {choice_data[16]}")
        print(f"  Padding: {choice_data[17:20]}")
        print()
    
    # Check observation maximums
    obs_maxs = nn_interface.getObservationMaximums()
    choice_maxs = obs_maxs[choice_features_start:]
    
    print("Expected maximum values for choice features:")
    for choice_idx in range(4):
        start_idx = choice_idx * 20
        end_idx = start_idx + 20
        max_data = choice_maxs[start_idx:end_idx]
        
        print(f"Choice {choice_idx + 1} maximums:")
        print(f"  Choice type flags (8): {max_data[0:8]}")
        print(f"  Binary flags (2): {max_data[8:10]}")
        print(f"  Card ID max: {max_data[10]}")
        print(f"  Relic ID max: {max_data[11]}")
        print(f"  Gold amount max: {max_data[12]}")
        print(f"  HP change max: {max_data[13]}")
        print(f"  Max HP change max: {max_data[14]}")
        print(f"  Gold cost max: {max_data[15]}")
        print(f"  HP cost max: {max_data[16]}")
        print(f"  Padding max: {max_data[17:20]}")
        print()

if __name__ == "__main__":
    analyze_generic_choices()