#!/usr/bin/env python3
"""
Quick test of wandb integration with STS training system.
"""

import sys
import os

def test_wandb_import():
    """Test if wandb can be imported."""
    try:
        import wandb
        print(f"✅ wandb imported successfully (version {wandb.__version__})")
        return True
    except ImportError:
        print("❌ wandb not available. Install with: pip install wandb")
        return False

def test_config_update():
    """Test that TrainingConfig properly handles wandb parameters."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sts_lightspeed'))
    
    try:
        from sts_training import TrainingConfig
        
        # Test creating config with wandb parameters
        config = TrainingConfig(
            use_wandb=True,
            wandb_project='test-project',
            wandb_entity='test-entity',
            wandb_run_name='test-run',
            wandb_tags=['test', 'integration']
        )
        
        print("✅ TrainingConfig accepts wandb parameters")
        print(f"   - use_wandb: {config.use_wandb}")
        print(f"   - wandb_project: {config.wandb_project}")
        print(f"   - wandb_entity: {config.wandb_entity}")
        print(f"   - wandb_run_name: {config.wandb_run_name}")
        print(f"   - wandb_tags: {config.wandb_tags}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test TrainingConfig: {e}")
        return False

def test_dry_run():
    """Test trainer creation without actually starting wandb."""
    try:
        from sts_training import TrainingConfig
        
        # Create config with wandb disabled for dry run
        config = TrainingConfig(
            num_episodes=5,
            use_wandb=False,  # Disabled for dry run
            wandb_project='dry-run-test'
        )
        
        print("✅ Can create TrainingConfig with wandb disabled")
        return True
        
    except Exception as e:
        print(f"❌ Dry run test failed: {e}")
        return False

def main():
    print("🧪 Testing wandb integration...")
    print("-" * 40)
    
    # Test wandb import
    if not test_wandb_import():
        return
    
    # Test config updates
    if not test_config_update():
        return
    
    # Test dry run
    if not test_dry_run():
        return
    
    print("\n✅ All tests passed!")
    print("\n🚀 Ready to use wandb with STS training!")
    print("\nNext steps:")
    print("1. Run: python3 setup_wandb.py  (to set up authentication)")
    print("2. Run: python3 train_sts_agent.py train --episodes 50 --wandb")

if __name__ == "__main__":
    main()