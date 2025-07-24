#!/usr/bin/env python3
import sys
import os

# Add the sts_lightspeed directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sts_lightspeed'))

try:
    import slaythespire
    print("Successfully imported sts_lightspeed module!")
    print("Available functions and classes:")
    for item in sorted(dir(slaythespire)):
        if not item.startswith('_'):
            print(f"  - {item}")
    
    # Example usage - start a basic game
    print("\nStarting a simple game example...")
    outcome = slaythespire.play()
    print(f"Game outcome: {outcome}")
    
except ImportError as e:
    print(f"Error importing slaythespire module: {e}")
    print("Make sure the module is compiled and available in the sts_lightspeed directory")
except Exception as e:
    print(f"Error running sts_lightspeed: {e}")