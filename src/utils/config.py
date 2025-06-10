import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Configuration management for field delineation project."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default config path relative to project root."""
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        return str(project_root / "config" / "default_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'model.arch')."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert nested config to flat dictionary for easy access."""
        def _flatten(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        return _flatten(self.config)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(description='Field Delineation Training')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for resuming training')
    
    # Model arguments
    parser.add_argument('--arch', type=str, default=None,
                       help='Model architecture')
    parser.add_argument('--encoder', type=str, default=None,
                       help='Encoder backbone')
    parser.add_argument('--in-channels', type=int, default=None,
                       help='Number of input channels')
    parser.add_argument('--out-channels', type=int, default=None,
                       help='Number of output channels')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                       help='Learning rate')
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, default=None,
                       help='Training data directory')
    parser.add_argument('--val-dir', type=str, default=None,
                       help='Validation data directory')
    parser.add_argument('--test-dir', type=str, default=None,
                       help='Test data directory')
    parser.add_argument('--dataset-type', type=str, default=None,
                       help='Dataset type')
    
    # Logging
    parser.add_argument('--project', type=str, default=None,
                       help='Wandb project name')
    parser.add_argument('--entity', type=str, default=None,
                       help='Wandb entity name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    
    return parser


def merge_args_with_config(config: ConfigManager, args: argparse.Namespace) -> Dict[str, Any]:
    """Merge command line arguments with configuration."""
    # Map command line args to config keys
    arg_mapping = {
        'arch': 'model.arch',
        'encoder': 'model.encoder',
        'in_channels': 'model.in_channels',
        'out_channels': 'model.out_channels',
        'epochs': 'training.max_epochs',
        'batch_size': 'training.batch_size',
        'lr': 'training.learning_rate',
        'train_dir': 'data.train_dir',
        'val_dir': 'data.val_dir',
        'test_dir': 'data.test_dir',
        'dataset_type': 'data.dataset_type',
        'project': 'logging.project',
        'entity': 'logging.entity',
        'checkpoint': 'checkpoint.checkpoint_path',
    }
    
    # Create updates dict from non-None args
    updates = {}
    for arg_name, config_key in arg_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            keys = config_key.split('.')
            current = updates
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = arg_value
    
    # Update config
    config.update(updates)
    
    return config.to_flat_dict()