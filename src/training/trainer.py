"""
Training utilities and trainer classes for field delineation models.

This module provides reusable training components that can be used across
different training scenarios (main training, hyperparameter sweeps, etc.).
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, 
    DeviceStatsMonitor, RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from data.datasets import AI4Boundaries, Euro_0512, MyanmarSatellite, SingleImage
from models.segmentator import PLSegmentator
from models.boundary_segmentator import PLSegmentator as BoundaryPLSegmentator
from utils.config import ConfigManager


class TrainingManager:
    """
    Main training manager that orchestrates the entire training process.
    
    This class handles dataset creation, model instantiation, trainer setup,
    and the complete training workflow.
    """
    
    def __init__(self, config: Union[Dict[str, Any], ConfigManager]):
        """
        Initialize training manager.
        
        Args:
            config: Configuration dictionary or ConfigManager instance
        """
        if isinstance(config, ConfigManager):
            self.config = config.to_flat_dict()
            self.config_manager = config
        else:
            self.config = config
            self.config_manager = None
        
        self.model = None
        self.trainer = None
        self.datasets = {}
        
        # Setup random seeds for reproducibility
        self._setup_seeds()
        
    def _setup_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config.get('training.seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Set deterministic algorithms for even better reproducibility
        torch.backends.cudnn.deterministic = self.config.get('training.deterministic', False)
        torch.backends.cudnn.benchmark = not self.config.get('training.deterministic', False)
    
    def create_datasets(self) -> Tuple[torch.utils.data.Dataset, ...]:
        """
        Create train, validation, and test datasets based on configuration.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        dataset_type = self.config.get('data.dataset_type', 'MyanmarSatellite')
        
        if dataset_type == 'AI4Boundaries':
            return self._create_ai4boundaries_datasets()
        elif dataset_type == 'Euro_0512':
            return self._create_euro0512_datasets()
        elif dataset_type == 'MyanmarSatellite':
            return self._create_myanmar_datasets()
        elif dataset_type == 'SingleImage':
            return self._create_single_image_datasets()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _create_ai4boundaries_datasets(self) -> Tuple[torch.utils.data.Dataset, ...]:
        """Create AI4Boundaries datasets."""
        train_dataset = AI4Boundaries(
            self.config.get('data.train_dir'),
            source=self.config.get('data.source'),
            augment=self.config.get('data.augment', True),
            cache_data=self.config.get('data.cache_data', False)
        )
        val_dataset = AI4Boundaries(
            self.config.get('data.val_dir'),
            source=self.config.get('data.source'),
            augment=False,
            cache_data=self.config.get('data.cache_data', False)
        )
        test_dataset = AI4Boundaries(
            self.config.get('data.test_dir', self.config.get('data.val_dir')),
            source=self.config.get('data.source'),
            augment=False,
            cache_data=self.config.get('data.cache_data', False)
        )
        return train_dataset, val_dataset, test_dataset
    
    def _create_euro0512_datasets(self) -> Tuple[torch.utils.data.Dataset, ...]:
        """Create Euro_0512 datasets with train/val/test splits."""
        dataset = Euro_0512(
            self.config.get('data.train_dir'),
            channels=self.config.get('data.channels', 9),
            cache_data=self.config.get('data.cache_data', False)
        )
        
        # Create splits
        dataset_size = len(dataset)
        val_size = int(dataset_size * self.config.get('data.val_split', 0.1))
        test_size = int(dataset_size * self.config.get('data.test_split', 0.1))
        train_size = dataset_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Set augmentation for training
        train_dataset.dataset.channels = self.config.get('data.channels', 9)
        train_dataset.dataset.augment = self.config.get('data.augment', True)
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_myanmar_datasets(self) -> Tuple[torch.utils.data.Dataset, ...]:
        """Create Myanmar satellite datasets."""
        train_dataset = MyanmarSatellite(
            self.config.get('data.train_dir'),
            channels=self.config.get('data.channels', 9),
            augment=self.config.get('data.augment', True),
            cache_data=self.config.get('data.cache_data', False)
        )
        val_dataset = MyanmarSatellite(
            self.config.get('data.val_dir'),
            channels=self.config.get('data.channels', 9),
            augment=False,
            cache_data=self.config.get('data.cache_data', False)
        )
        test_dataset = MyanmarSatellite(
            self.config.get('data.test_dir', self.config.get('data.val_dir')),
            channels=self.config.get('data.channels', 9),
            augment=False,
            cache_data=self.config.get('data.cache_data', False)
        )
        return train_dataset, val_dataset, test_dataset
    
    def _create_single_image_datasets(self) -> Tuple[torch.utils.data.Dataset, ...]:
        """Create single image datasets."""
        train_dataset = SingleImage(
            self.config.get('data.train_dir'),
            augment=self.config.get('data.augment', True),
            cache_data=self.config.get('data.cache_data', False)
        )
        val_dataset = SingleImage(
            self.config.get('data.val_dir'),
            augment=False,
            cache_data=self.config.get('data.cache_data', False)
        )
        test_dataset = SingleImage(
            self.config.get('data.test_dir', self.config.get('data.val_dir')),
            augment=False,
            cache_data=self.config.get('data.cache_data', False)
        )
        return train_dataset, val_dataset, test_dataset
    
    def create_model(self, train_dataset, val_dataset, test_dataset) -> pl.LightningModule:
        """
        Create model based on configuration.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset  
            test_dataset: Test dataset
            
        Returns:
            PyTorch Lightning module
        """
        # Prepare model context
        model_context = {
            'arch': self.config.get('model.arch', 'unetplusplus'),
            'encoder': self.config.get('model.encoder', 'resnet152'),
            'encoder_weights': self.config.get('model.encoder_weights', 'imagenet'),
            'in_channels': self.config.get('model.in_channels', 9),
            'out_channels': self.config.get('model.out_channels', 3),
            'batch_size': self.config.get('training.batch_size', 48),
            'learning_rate': self.config.get('training.learning_rate', 0.05),
            
            # Loss configuration
            'loss_1': self.config.get('losses.loss_1', 'DiceLoss'),
            'loss_2': self.config.get('losses.loss_2', 'ComboLoss'),
            'loss_3': self.config.get('losses.loss_3', 'BCELoss'),
            'params_1': self.config.get('losses.params_1', {}),
            'params_2': self.config.get('losses.params_2', {}),
            'params_3': self.config.get('losses.params_3', {}),
            'weight_1': self.config.get('losses.weight_1', 1.0),
            'weight_2': self.config.get('losses.weight_2', 0.0),
            'weight_3': self.config.get('losses.weight_3', 1.0),
        }
        
        # Determine model class based on configuration
        model_arch = self.config.get('model.arch', 'unetplusplus')
        use_boundary_loss = self.config.get('losses.weight_4', 0.0) > 0 or 'BoundaryLoss' in [
            self.config.get('losses.loss_1', ''),
            self.config.get('losses.loss_2', ''), 
            self.config.get('losses.loss_3', '')
        ]
        
        if model_arch == 'segformer':
            # Add SegFormer specific parameters
            model_context.update({
                'alpha': self.config.get('losses.alpha', 1.0),
                'beta': self.config.get('losses.beta', 1.0),
                'gamma': self.config.get('losses.gamma', 0.4),
            })
            model_class = SegFormerPLSegmentator
        elif use_boundary_loss:
            model_context['params_4'] = self.config.get('losses.params_4', {})
            model_context['weight_4'] = self.config.get('losses.weight_4', 0.0)
            model_class = BoundaryPLSegmentator
        else:
            model_class = PLSegmentator
        
        # Load from checkpoint if specified
        checkpoint_path = self.config.get('checkpoint.checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            model = model_class.load_from_checkpoint(
                checkpoint_path,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
            )
            # Update context if needed
            if hasattr(model, 'update_context'):
                model.update_context(model_context)
        else:
            model = model_class(
                model_context,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
            )
        
        return model
    
    def create_callbacks(self) -> List[pl.Callback]:
        """
        Create training callbacks.
        
        Returns:
            List of PyTorch Lightning callbacks
        """
        callbacks = []
        
        # Learning rate monitor
        if self.config.get('training.log_lr', True):
            callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.get('checkpoint.dirpath', './checkpoints'),
            filename=self.config.get('checkpoint.filename', '{epoch}-{val_loss:.2f}'),
            save_top_k=self.config.get('checkpoint.save_top_k', 3),
            monitor=self.config.get('checkpoint.monitor', 'val loss'),
            mode=self.config.get('checkpoint.mode', 'min'),
            save_last=self.config.get('checkpoint.save_last', True),
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.get('training.early_stopping', False):
            early_stop_callback = EarlyStopping(
                monitor=self.config.get('training.early_stopping_monitor', 'val loss'),
                patience=self.config.get('training.early_stopping_patience', 10),
                mode=self.config.get('training.early_stopping_mode', 'min'),
                min_delta=self.config.get('training.early_stopping_min_delta', 0.001),
            )
            callbacks.append(early_stop_callback)
        
        # Progress bar
        if self.config.get('training.rich_progress', True):
            callbacks.append(RichProgressBar())
        
        # Device stats monitoring
        if self.config.get('training.monitor_device_stats', False):
            callbacks.append(DeviceStatsMonitor())
        
        return callbacks
    
    def create_logger(self) -> Optional[pl.loggers.Logger]:
        """
        Create training logger.
        
        Returns:
            PyTorch Lightning logger or None
        """
        logger_type = self.config.get('logging.logger_type', 'wandb')
        
        if logger_type == 'wandb' and self.config.get('logging.use_wandb', True):
            try:
                import wandb
                if not wandb.api.api_key:
                    wandb.login()
                
                return WandbLogger(
                    project=self.config.get('logging.project', 'field-delineation'),
                    entity=self.config.get('logging.entity'),
                    name=self.config.get('logging.experiment_name'),
                    tags=self.config.get('logging.tags', []),
                    log_model=self.config.get('logging.log_model', False),
                    save_dir=self.config.get('logging.save_dir', './logs')
                )
            except ImportError:
                print("Warning: wandb not available, falling back to TensorBoard")
                logger_type = 'tensorboard'
        
        if logger_type == 'tensorboard':
            return TensorBoardLogger(
                save_dir=self.config.get('logging.save_dir', './logs'),
                name=self.config.get('logging.experiment_name', 'field_delineation'),
                version=self.config.get('logging.version')
            )
        
        return None
    
    def create_trainer(self) -> pl.Trainer:
        """
        Create PyTorch Lightning trainer.
        
        Returns:
            Configured trainer instance
        """
        # Determine accelerator
        accelerator = self.config.get('training.accelerator', 'auto')
        if accelerator == 'auto':
            accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        
        # Determine devices
        devices = self.config.get('training.devices', 'auto')
        if devices == 'auto':
            devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        trainer_kwargs = {
            'max_epochs': self.config.get('training.max_epochs', 30),
            'accelerator': accelerator,
            'devices': devices,
            'logger': self.create_logger(),
            'callbacks': self.create_callbacks(),
            'accumulate_grad_batches': self.config.get('training.accumulate_grad_batches', 1),
            'limit_train_batches': self.config.get('training.limit_train_batches', 1.0),
            'limit_val_batches': self.config.get('training.limit_val_batches', 1.0),
            'num_sanity_val_steps': self.config.get('training.num_sanity_val_steps', 2),
            'precision': self.config.get('training.precision', 32),
            'gradient_clip_val': self.config.get('training.gradient_clip_val', 0.0),
            'enable_checkpointing': self.config.get('training.enable_checkpointing', True),
            'enable_progress_bar': self.config.get('training.enable_progress_bar', True),
            'enable_model_summary': self.config.get('training.enable_model_summary', True),
        }
        
        # Remove None values
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}
        
        return pl.Trainer(**trainer_kwargs)
    
    def setup_training(self) -> Tuple[pl.LightningModule, pl.Trainer]:
        """
        Setup complete training pipeline.
        
        Returns:
            Tuple of (model, trainer)
        """
        print("Setting up training pipeline...")
        
        # Create datasets
        print("Creating datasets...")
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        self.datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        print(f"Dataset sizes - Train: {len(train_dataset)}, "
              f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create model
        print("Creating model...")
        self.model = self.create_model(train_dataset, val_dataset, test_dataset)
        
        # Create trainer
        print("Creating trainer...")
        self.trainer = self.create_trainer()
        
        return self.model, self.trainer
    
    def train(self) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Returns:
            Dictionary with training results
        """
        start_time = time.time()
        
        # Setup training if not already done
        if self.model is None or self.trainer is None:
            self.setup_training()
        
        print("Starting training...")
        
        # Run training
        self.trainer.fit(self.model)
        
        # Run testing
        if self.config.get('training.run_test', True):
            print("Running final evaluation...")
            test_results = self.trainer.test(self.model)
        else:
            test_results = []
        
        training_time = time.time() - start_time
        
        # Collect results
        results = {
            'training_time': training_time,
            'best_model_path': self.trainer.checkpoint_callback.best_model_path,
            'best_model_score': self.trainer.checkpoint_callback.best_model_score,
            'test_results': test_results,
            'current_epoch': self.trainer.current_epoch,
            'global_step': self.trainer.global_step,
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        if self.trainer.checkpoint_callback.best_model_path:
            print(f"Best model saved to: {self.trainer.checkpoint_callback.best_model_path}")
        
        return results
    
    def validate(self) -> Dict[str, Any]:
        """
        Run validation on the model.
        
        Returns:
            Validation results
        """
        if self.trainer is None or self.model is None:
            raise RuntimeError("Model and trainer must be setup before validation")
        
        print("Running validation...")
        val_results = self.trainer.validate(self.model)
        return val_results
    
    def test(self) -> Dict[str, Any]:
        """
        Run testing on the model.
        
        Returns:
            Test results
        """
        if self.trainer is None or self.model is None:
            raise RuntimeError("Model and trainer must be setup before testing")
        
        print("Running testing...")
        test_results = self.trainer.test(self.model)
        return test_results


class HyperparameterSweepManager:
    """
    Manager for hyperparameter sweeps using wandb.
    
    This class handles the setup and execution of hyperparameter sweeps
    with automatic experiment tracking.
    """
    
    def __init__(self, base_config: Dict[str, Any], sweep_config: Dict[str, Any]):
        """
        Initialize sweep manager.
        
        Args:
            base_config: Base configuration
            sweep_config: Sweep configuration for wandb
        """
        self.base_config = base_config
        self.sweep_config = sweep_config
        self.sweep_id = None
    
    def create_sweep(self) -> str:
        """
        Create wandb sweep.
        
        Returns:
            Sweep ID
        """
        try:
            import wandb
            
            self.sweep_id = wandb.sweep(
                self.sweep_config,
                project=self.base_config.get('logging.project', 'field-delineation-sweep'),
                entity=self.base_config.get('logging.entity')
            )
            
            print(f"Created sweep with ID: {self.sweep_id}")
            return self.sweep_id
            
        except ImportError:
            raise ImportError("wandb is required for hyperparameter sweeps")
    
    def sweep_train_function(self):
        """Training function for wandb agent."""
        try:
            import wandb
            wandb.init()
            
            # Merge wandb config with base config
            sweep_config = dict(wandb.config)
            merged_config = self._merge_configs(self.base_config, sweep_config)
            
            # Create training manager
            trainer_manager = TrainingManager(merged_config)
            
            # Run training
            results = trainer_manager.train()
            
            # Log final metrics
            if results['test_results']:
                for result in results['test_results']:
                    for key, value in result.items():
                        wandb.log({f"final_{key}": value})
            
            wandb.finish()
            
        except Exception as e:
            print(f"Error in sweep training: {e}")
            raise
    
    def _merge_configs(self, base_config: Dict[str, Any], sweep_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base config with sweep parameters.
        
        Args:
            base_config: Base configuration
            sweep_config: Sweep parameters from wandb
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        # Map sweep parameters to config structure
        param_mapping = {
            'learning_rate': 'training.learning_rate',
            'batch_size': 'training.batch_size',
            'max_epochs': 'training.max_epochs',
            'arch': 'model.arch',
            'encoder': 'model.encoder',
            'in_channels': 'model.in_channels',
            'loss_1': 'losses.loss_1',
            'loss_2': 'losses.loss_2',
            'loss_3': 'losses.loss_3',
            'weight_1': 'losses.weight_1',
            'weight_2': 'losses.weight_2',
            'weight_3': 'losses.weight_3',
        }
        
        for sweep_key, value in sweep_config.items():
            if sweep_key in param_mapping:
                config_key = param_mapping[sweep_key]
                self._set_nested_config(merged, config_key, value)
            else:
                # Direct mapping
                merged[sweep_key] = value
        
        return merged
    
    def _set_nested_config(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def run_sweep(self, count: int = 10) -> None:
        """
        Run hyperparameter sweep.
        
        Args:
            count: Number of sweep runs
        """
        if self.sweep_id is None:
            self.create_sweep()
        
        try:
            import wandb
            print(f"Starting {count} sweep runs...")
            wandb.agent(self.sweep_id, function=self.sweep_train_function, count=count)
            print("Sweep completed!")
            
        except ImportError:
            raise ImportError("wandb is required for hyperparameter sweeps")


def create_default_sweep_config() -> Dict[str, Any]:
    """
    Create default sweep configuration for hyperparameter optimization.
    
    Returns:
        Default sweep configuration
    """
    return {
        'method': 'random',
        'metric': {
            'name': 'val loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'batch_size': {
                'values': [16, 32, 48, 64]
            },
            'weight_1': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'weight_2': {
                'distribution': 'uniform', 
                'min': 0.0,
                'max': 1.0
            },
            'weight_3': {
                'distribution': 'uniform',
                'min': 0.1, 
                'max': 1.0
            },
            'loss_1': {
                'values': ['DiceLoss', 'BCELoss', 'FocalLoss', 'TverskyLoss']
            },
            'loss_3': {
                'values': ['BCELoss', 'FocalLoss', 'TverskyLoss']
            }
        }
    }


# Convenience functions
def quick_train(config_path: str, **overrides) -> Dict[str, Any]:
    """
    Convenience function for quick training with minimal setup.
    
    Args:
        config_path: Path to configuration file
        **overrides: Configuration overrides
        
    Returns:
        Training results
    """
    config_manager = ConfigManager(config_path)
    config_manager.update(overrides)
    
    trainer = TrainingManager(config_manager)
    return trainer.train()


def create_trainer_from_args(args) -> TrainingManager:
    """
    Create trainer from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured TrainingManager
    """
    # Convert args to config dictionary
    config_dict = {}
    
    # Map common arguments
    arg_mappings = {
        'epochs': 'training.max_epochs',
        'batch_size': 'training.batch_size', 
        'lr': 'training.learning_rate',
        'arch': 'model.arch',
        'encoder': 'model.encoder',
        'train_dir': 'data.train_dir',
        'val_dir': 'data.val_dir',
        'test_dir': 'data.test_dir',
        'dataset_type': 'data.dataset_type',
        'project': 'logging.project',
        'entity': 'logging.entity',
    }
    
    for arg_name, config_key in arg_mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            # Set nested config value
            keys = config_key.split('.')
            current = config_dict
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
    
    return TrainingManager(config_dict)