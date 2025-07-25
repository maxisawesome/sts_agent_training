#!/usr/bin/env python3
"""
Weights & Biases Setup and Test for STS Neural Agent

This script helps you set up wandb tracking and run a test to ensure everything works.
"""

import sys
import os

def check_wandb_installation():
    """Check if wandb is installed and available."""
    try:
        import wandb
        print(f"✅ wandb is installed (version {wandb.__version__})")
        return True
    except ImportError:
        print("❌ wandb is not installed")
        print("Install with: pip install wandb")
        return False

def setup_wandb():
    """Set up wandb authentication."""
    try:
        import wandb
        
        # Check if already logged in
        try:
            api = wandb.Api()
            user = api.viewer
            print(f"✅ Already logged in as: {user}")
            return True
        except Exception:
            pass
        
        print("Setting up wandb authentication...")
        print("Visit https://wandb.ai/authorize to get your API key")
        
        # Try to login
        wandb.login()
        
        # Verify login worked
        api = wandb.Api()
        user = api.viewer
        print(f"✅ Successfully logged in as: {user}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to set up wandb: {e}")
        return False

def test_wandb_integration():
    """Test wandb integration with a minimal training run."""
    print("\n=== Testing wandb integration ===")
    
    try:
        # Set up path for sts_lightspeed
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sts_lightspeed'))
        
        from sts_training import PPOTrainer, TrainingConfig
        
        # Create minimal test configuration
        config = TrainingConfig(
            num_episodes=5,  # Very short test
            collect_episodes_per_update=2,
            batch_size=32,
            update_epochs=1,
            log_interval=1,
            use_wandb=True,
            wandb_project='sts-neural-agent-test',
            wandb_run_name='setup-test',
            wandb_tags=['test', 'setup']
        )
        
        print("Creating trainer with wandb enabled...")
        trainer = PPOTrainer(config)
        
        if trainer.wandb_run:
            print(f"✅ wandb run created successfully!")
            print(f"🔗 View at: {trainer.wandb_run.url}")
            
            # Log a test metric
            import wandb
            wandb.log({"test/setup_successful": 1})
            
            # Finish the test run
            wandb.finish()
            print("✅ Test completed successfully!")
            return True
        else:
            print("❌ Failed to create wandb run")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_usage_examples():
    """Print examples of how to use wandb with the training system."""
    print("\n" + "="*60)
    print("🚀 WANDB SETUP COMPLETE!")
    print("="*60)
    
    print("\n📊 Usage Examples:")
    print("-" * 40)
    
    print("\n1. Basic training with wandb:")
    print("   python3 train_sts_agent.py train --episodes 1000 --wandb")
    
    print("\n2. Training with custom project and tags:")
    print("   python3 train_sts_agent.py train \\")
    print("     --episodes 2000 \\")
    print("     --reward-function comprehensive \\")
    print("     --wandb-project my-sts-experiments \\")
    print("     --wandb-name experiment-1 \\")
    print("     --wandb-tags baseline comprehensive-reward")
    
    print("\n3. Training without wandb:")
    print("   python3 train_sts_agent.py train --episodes 1000 --no-wandb")
    
    print("\n📈 What gets tracked:")
    print("   • Training progress (rewards, losses, episode lengths)")
    print("   • Model parameters and gradients")
    print("   • Hyperparameters and configuration")
    print("   • Performance metrics (FPS, training time)")
    print("   • Reward distributions and trends")
    print("   • Model checkpoints")
    
    print("\n🌐 View your experiments at: https://wandb.ai/")
    print("\n💡 Pro tip: Use different projects for different experiment types!")

def main():
    print("🎯 STS Neural Agent - Weights & Biases Setup")
    print("=" * 50)
    
    # Check installation
    if not check_wandb_installation():
        return False
    
    # Set up authentication
    if not setup_wandb():
        return False
    
    # Test integration
    if not test_wandb_integration():
        return False
    
    # Show examples
    print_usage_examples()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)