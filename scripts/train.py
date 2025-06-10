#!/usr/bin/env python3
"""
Main training script for field delineation models.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import project modules
from utils.config import ConfigManager, create_arg_parser, merge_args_with_config
from training.trainer import TrainingManager


def main():
    """Main training function."""
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = merge_args_with_config(config_manager, args)
    
    # Set environment variables
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Update wandb setting based on args
    config['logging.use_wandb'] = not args.no_wandb
    
    # Create training manager
    print("Initializing training manager...")
    trainer_manager = TrainingManager(config)
    
    # Run training
    results = trainer_manager.train()
    
    # Print results summary
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Final epoch: {results['current_epoch']}")
    print(f"Total steps: {results['global_step']}")
    
    if results['best_model_path']:
        print(f"Best model saved to: {results['best_model_path']}")
        print(f"Best model score: {results['best_model_score']:.4f}")
    
    if results['test_results']:
        print("\nTest Results:")
        for i, test_result in enumerate(results['test_results']):
            print(f"  Test run {i+1}:")
            for metric, value in test_result.items():
                print(f"    {metric}: {value:.4f}")


if __name__ == "__main__":
    main()