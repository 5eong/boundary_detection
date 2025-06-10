#!/usr/bin/env python3
"""
Hyperparameter sweep script for field delineation models.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import project modules
from utils.config import ConfigManager
from training.trainer import HyperparameterSweepManager, create_default_sweep_config


def create_boundary_loss_sweep_config():
    """Create sweep configuration for boundary loss experiments."""
    return {
        'method': 'random',
        'metric': {
            'name': 'val loss',
            'goal': 'minimize'
        },
        'parameters': {
            'max_epochs': {'values': [12]},
            'arch': {'values': ['unetplusplus']},
            'encoder': {'values': ['resnet50']},
            'encoder_weights': {'values': ['imagenet']},
            'in_channels': {'values': [9]},
            'out_channels': {'values': [3]},
            'batch_size': {'values': [60]},
            'learning_rate': {'values': [0.01]},
            'batch_per_epoch': {'values': [120]},
            'accumulate_grad_batches': {'values': [12]},
            'loss_1': {'values': ['BCELoss', 'DiceLoss', 'FocalLoss', 'TanimotoLoss', 'TverskyLoss', 'JaccardLoss']},
            'loss_2': {'values': ['BoundaryLoss']},
            'loss_3': {'values': ['BCELoss', 'DiceLoss', 'FocalLoss', 'TanimotoLoss', 'TverskyLoss', 'JaccardLoss']},
            'weight_1': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
            'weight_2': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
            'weight_3': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
            'weight_4': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
            'dataset_type': {'values': ['Euro_0512']},
            'subset_groups': {'values': list(range(30, 692, 30))},
        }
    }


def create_loss_comparison_sweep_config():
    """Create sweep configuration for loss function comparison."""
    return {
        'method': 'grid',
        'metric': {
            'name': 'val loss',
            'goal': 'minimize'
        },
        'parameters': {
            'max_epochs': {'values': [20]},
            'arch': {'values': ['unetplusplus']},
            'encoder': {'values': ['resnet152']},
            'batch_size': {'values': [48]},
            'learning_rate': {'values': [0.05]},
            'loss_1': {'values': ['DiceLoss', 'BCELoss', 'FocalLoss']},
            'loss_2': {'values': ['ComboLoss', 'TanimotoLoss']},
            'loss_3': {'values': ['BCELoss', 'FocalLoss']},
            'weight_1': {'values': [0.5, 0.8, 1.0]},
            'weight_2': {'values': [0.0, 0.5]},
            'weight_3': {'values': [0.5, 0.8, 1.0]},
        }
    }


def create_architecture_sweep_config():
    """Create sweep configuration for architecture comparison."""
    return {
        'method': 'random',
        'metric': {
            'name': 'val f1_interior',
            'goal': 'maximize'
        },
        'parameters': {
            'max_epochs': {'values': [15]},
            'arch': {'values': ['unet', 'unetplusplus', 'deeplabv3plus', 'manet', 'linknet']},
            'encoder': {'values': ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d']},
            'batch_size': {'values': [32, 48, 64]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
            'weight_1': {'distribution': 'uniform', 'min': 0.5, 'max': 1.0},
            'weight_3': {'distribution': 'uniform', 'min': 0.5, 'max': 1.0},
        }
    }


def main():
    """Main sweep function."""
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for field delineation')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to base configuration file')
    parser.add_argument('--project', type=str, default='field-delineation-sweep',
                       help='Wandb project name')
    parser.add_argument('--entity', type=str, default=None,
                       help='Wandb entity name')
    parser.add_argument('--count', type=int, default=100,
                       help='Number of sweep runs')
    parser.add_argument('--sweep-type', type=str, 
                       choices=['default', 'boundary_loss', 'loss_comparison', 'architecture'],
                       default='default',
                       help='Type of sweep to run')
    parser.add_argument('--sweep-config-file', type=str, default=None,
                       help='Custom sweep configuration file')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Load base configuration
    if args.config:
        config_manager = ConfigManager(args.config)
        base_config = config_manager.to_flat_dict()
    else:
        # Create minimal base config
        base_config = {
            'data.dataset_type': 'MyanmarSatellite',
            'data.train_dir': '../Datasets/MyanmarAnnotations/Resolution_0_5/Masks/Cropped/Train/',
            'data.val_dir': '../Datasets/MyanmarAnnotations/Resolution_0_5/Masks/Cropped/Val/',
            'data.channels': 9,
            'data.augment': True,
            'model.in_channels': 9,
            'model.out_channels': 3,
            'logging.project': args.project,
            'logging.entity': args.entity,
            'training.precision': '16-mixed',
            'training.num_sanity_val_steps': 0,
        }
    
    # Create sweep configuration
    if args.sweep_config_file:
        import yaml
        with open(args.sweep_config_file, 'r') as f:
            sweep_config = yaml.safe_load(f)
    elif args.sweep_type == 'boundary_loss':
        sweep_config = create_boundary_loss_sweep_config()
    elif args.sweep_type == 'loss_comparison':
        sweep_config = create_loss_comparison_sweep_config()
    elif args.sweep_type == 'architecture':
        sweep_config = create_architecture_sweep_config()
    else:
        sweep_config = create_default_sweep_config()
    
    print(f"Running {args.sweep_type} sweep with {args.count} runs")
    print(f"Project: {args.project}")
    if args.entity:
        print(f"Entity: {args.entity}")
    
    # Create and run sweep
    sweep_manager = HyperparameterSweepManager(base_config, sweep_config)
    sweep_manager.run_sweep(count=args.count)
    
    print("Sweep completed!")


if __name__ == "__main__":
    main()