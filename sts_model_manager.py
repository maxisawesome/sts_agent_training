#!/usr/bin/env python3
"""
Slay the Spire Model Management System

This module provides utilities for saving, loading, and managing trained models
for the STS neural network agent.
"""

import torch
import os
import json
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from sts_neural_network import STSActorCritic
from sts_training import TrainingConfig

class STSModelManager:
    """
    Manager class for STS model operations including saving, loading, and evaluation.
    """
    
    def __init__(self, model_dir: str = "sts_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def save_model(self, 
                   model: STSActorCritic, 
                   optimizer: torch.optim.Optimizer,
                   config: TrainingConfig,
                   training_stats: Dict[str, List],
                   model_name: str,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a complete model checkpoint with training statistics and metadata.
        
        Returns the filepath of the saved model.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pt"
        filepath = os.path.join(self.model_dir, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
            'training_stats': training_stats,
            'timestamp': timestamp,
            'model_name': model_name,
            'metadata': metadata or {}
        }
        
        # Add model architecture info
        checkpoint['model_info'] = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'obs_size': model.obs_size,
            'hidden_size': model.hidden_size,
            'action_size': model.action_size
        }
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        # Save human-readable summary
        self._save_model_summary(filepath.replace('.pt', '_summary.json'), checkpoint)
        
        print(f"Model saved: {filepath}")
        return filepath
    
    def load_model(self, filepath: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load a complete model checkpoint.
        
        Returns a dictionary containing the model, optimizer, config, and other data.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model from config
        config_dict = checkpoint['config']
        model = STSActorCritic(
            obs_size=412,
            hidden_size=config_dict.get('hidden_size', 512),
            action_size=config_dict.get('action_size', 256),
            num_layers=config_dict.get('num_layers', 3)
        ).to(device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create optimizer (optional, for continued training)
        optimizer = torch.optim.Adam(model.parameters(), lr=config_dict.get('learning_rate', 3e-4))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Recreate config object
        config = TrainingConfig(**config_dict)
        
        return {
            'model': model,
            'optimizer': optimizer,
            'config': config,
            'training_stats': checkpoint.get('training_stats', {}),
            'metadata': checkpoint.get('metadata', {}),
            'model_info': checkpoint.get('model_info', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
    
    def load_model_for_inference(self, filepath: str, device: Optional[torch.device] = None) -> STSActorCritic:
        """
        Load model for inference only (no optimizer needed).
        """
        checkpoint_data = self.load_model(filepath, device)
        model = checkpoint_data['model']
        model.eval()  # Set to evaluation mode
        return model
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their metadata."""
        models = []
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.pt'):
                filepath = os.path.join(self.model_dir, filename)
                try:
                    # Load minimal info without full model
                    checkpoint = torch.load(filepath, map_location='cpu')
                    
                    model_info = {
                        'filename': filename,
                        'filepath': filepath,
                        'model_name': checkpoint.get('model_name', 'unknown'),
                        'timestamp': checkpoint.get('timestamp', 'unknown'),
                        'total_parameters': checkpoint.get('model_info', {}).get('total_parameters', 'unknown'),
                        'config': checkpoint.get('config', {}),
                        'metadata': checkpoint.get('metadata', {})
                    }
                    
                    # Add training stats summary if available
                    stats = checkpoint.get('training_stats', {})
                    if 'episode_rewards' in stats and stats['episode_rewards']:
                        model_info['final_avg_reward'] = np.mean(stats['episode_rewards'][-10:])
                        model_info['total_episodes'] = len(stats['episode_rewards'])
                    
                    models.append(model_info)
                    
                except Exception as e:
                    print(f"Error reading model {filename}: {e}")
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models
    
    def compare_models(self, model_paths: List[str]) -> Dict[str, Any]:
        """Compare multiple models and their performance."""
        comparison = {
            'models': [],
            'performance_comparison': {},
            'parameter_comparison': {}
        }
        
        for path in model_paths:
            try:
                checkpoint = torch.load(path, map_location='cpu')
                
                model_data = {
                    'path': path,
                    'name': checkpoint.get('model_name', 'unknown'),
                    'timestamp': checkpoint.get('timestamp', 'unknown'),
                    'parameters': checkpoint.get('model_info', {}).get('total_parameters', 'unknown')
                }
                
                # Add performance metrics
                stats = checkpoint.get('training_stats', {})
                if 'episode_rewards' in stats and stats['episode_rewards']:
                    rewards = stats['episode_rewards']
                    model_data['performance'] = {
                        'final_avg_reward': np.mean(rewards[-10:]),
                        'best_reward': max(rewards),
                        'total_episodes': len(rewards),
                        'reward_std': np.std(rewards[-50:]) if len(rewards) >= 50 else np.std(rewards)
                    }
                
                comparison['models'].append(model_data)
                
            except Exception as e:
                print(f"Error comparing model {path}: {e}")
        
        return comparison
    
    def _save_model_summary(self, filepath: str, checkpoint: Dict[str, Any]):
        """Save a human-readable summary of the model."""
        summary = {
            'model_name': checkpoint.get('model_name', 'unknown'),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'model_info': checkpoint.get('model_info', {}),
            'config_summary': {
                'learning_rate': checkpoint.get('config', {}).get('learning_rate'),
                'hidden_size': checkpoint.get('config', {}).get('hidden_size'),
                'num_episodes': checkpoint.get('config', {}).get('num_episodes'),
                'batch_size': checkpoint.get('config', {}).get('batch_size')
            },
            'metadata': checkpoint.get('metadata', {})
        }
        
        # Add training performance summary
        stats = checkpoint.get('training_stats', {})
        if stats:
            performance = {}
            if 'episode_rewards' in stats and stats['episode_rewards']:
                rewards = stats['episode_rewards']
                performance['total_episodes'] = len(rewards)
                performance['final_avg_reward'] = float(np.mean(rewards[-10:]))
                performance['best_reward'] = float(max(rewards))
                performance['worst_reward'] = float(min(rewards))
            
            if 'policy_losses' in stats and stats['policy_losses']:
                performance['final_policy_loss'] = float(np.mean(stats['policy_losses'][-5:]))
            
            summary['performance'] = performance
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

def test_model_manager():
    """Test the model management system."""
    print("=== Testing STS Model Manager ===\n")
    
    manager = STSModelManager()
    
    # List existing models
    print("--- Existing Models ---")
    models = manager.list_models()
    for i, model in enumerate(models):
        print(f"{i+1}. {model['filename']}")
        print(f"   Name: {model['model_name']}")
        print(f"   Timestamp: {model['timestamp']}")
        print(f"   Parameters: {model['total_parameters']}")
        if 'final_avg_reward' in model:
            print(f"   Final Avg Reward: {model['final_avg_reward']:.3f}")
            print(f"   Total Episodes: {model['total_episodes']}")
        print()
    
    if not models:
        print("No models found. Run training first to create models.")
        return
    
    # Test loading the most recent model
    print("--- Testing Model Loading ---")
    latest_model_path = models[0]['filepath']
    print(f"Loading model: {latest_model_path}")
    
    try:
        # Test inference loading
        model = manager.load_model_for_inference(latest_model_path)
        print(f"Successfully loaded model for inference")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test full loading
        checkpoint_data = manager.load_model(latest_model_path)
        print(f"Successfully loaded full checkpoint")
        print(f"Config: {checkpoint_data['config'].__dict__}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    test_model_manager()