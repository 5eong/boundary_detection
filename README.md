# Field Delineation Using Satellite Imagery

A production-ready deep learning framework for field delineation using satellite imagery, built with PyTorch Lightning.

## Project Structure

```
field_delineation/
├── config/
│   └── default_config.yaml          # Default configuration
├── src/
│   ├── data/
│   │   ├── datasets.py               # Dataset classes
│   │   └── preprocessing.py          # Data preprocessing utilities
│   ├── models/
│   │   ├── segmentator.py           # Standard segmentation model
│   │   ├── boundary_segmentator.py  # Boundary loss segmentation model
│   │   └── segformer.py             # SegFormer model
│   ├── losses/
│   │   ├── loss_functions.py        # Loss function implementations
│   │   └── boundary_loss.py         # Boundary loss utilities
│   ├── metrics/
│   │   └── metrics.py               # Evaluation metrics
│   ├── optimizers/
│   │   └── lookahead.py             # Lookahead optimizer
│   └── utils/
│       ├── config.py                # Configuration management
│       └── visualization.py         # Visualization utilities
├── scripts/
│   ├── train.py                     # Training script
│   ├── inference.py                 # Inference script
│   └── sweep.py                     # Hyperparameter sweep script
├── notebooks/
│   └── UNet.ipynb                   # Original development notebook
├── setup.py                         # Package setup
└── requirements.txt                 # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd field-delineation
```

2. Install dependencies:
```bash
pip install -e .
```

Or install with development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Training a Model

1. **Using default configuration:**
```bash
python scripts/train.py
```

2. **With custom configuration:**
```bash
python scripts/train.py --config config/custom_config.yaml
```

3. **With command line arguments:**
```bash
python scripts/train.py \
    --arch unetplusplus \
    --encoder resnet152 \
    --epochs 30 \
    --batch-size 48 \
    --lr 0.05 \
    --train-dir /path/to/train \
    --val-dir /path/to/val
```

### Running Inference

```bash
python scripts/inference.py \
    --checkpoint path/to/model.ckpt \
    --data-dir /path/to/test/data \
    --output-dir ./outputs \
    --visualize \
    --save-predictions
```

### Hyperparameter Sweeps

```bash
python scripts/sweep.py \
    --project my-sweep-project \
    --entity my-wandb-entity \
    --count 100
```

## Configuration

The project uses YAML configuration files for easy parameter management. See `config/default_config.yaml` for all available options.

### Key Configuration Sections:

- **model**: Architecture, encoder, channels
- **training**: Epochs, batch size, learning rate
- **losses**: Loss functions and weights
- **data**: Dataset paths and preprocessing options
- **logging**: Wandb configuration

### Example Configuration:

```yaml
model:
  arch: 'unetplusplus'
  encoder: 'resnet152'
  in_channels: 9
  out_channels: 3

training:
  max_epochs: 30
  batch_size: 48
  learning_rate: 0.05

losses:
  loss_1: 'DiceLoss'
  loss_2: 'ComboLoss'
  loss_3: 'BCELoss'
  weight_1: 0.78
  weight_2: 0.0
  weight_3: 0.81
```

## Supported Datasets

- **AI4Boundaries**: Sentinel-2 imagery for agricultural field boundaries
- **Euro_0512**: European satellite imagery dataset
- **MyanmarSatellite**: Myanmar satellite imagery with multi-source data

## Supported Models

- **UNet variants**: UNet, UNet++, UNet3+
- **DeepLab**: DeepLabV3+
- **Other architectures**: MANet, LinkNet, FPN, PSPNet
- **SegFormer**: Vision Transformer-based segmentation

## Loss Functions

- **Standard losses**: BCE, Dice, Focal, Tversky
- **Advanced losses**: Tanimoto, Lovász, Combo
- **Boundary loss**: Specialized for field boundary detection

## Metrics

- **F1 Score**: Per-class F1 scores
- **IoU**: Intersection over Union
- **MCC**: Matthews Correlation Coefficient

## Development

### Adding New Datasets

1. Create a new dataset class in `src/data/datasets.py`
2. Inherit from `torch.utils.data.Dataset`
3. Implement `__len__` and `__getitem__` methods
4. Add to the dataset factory in `scripts/train.py`

### Adding New Loss Functions

1. Implement the loss in `src/losses/loss_functions.py`
2. Add to the loss registry in the model files
3. Update configuration documentation

### Adding New Models

1. Create a new model file in `src/models/`
2. Inherit from `pytorch_lightning.LightningModule`
3. Implement required methods
4. Add to the model factory in training scripts

## Command Line Interface

After installation, you can use the console scripts:

```bash
# Training
field-delineation-train --config config/my_config.yaml

# Inference
field-delineation-inference --checkpoint model.ckpt --data-dir ./data
```

## Logging and Monitoring

The framework integrates with Weights & Biases for experiment tracking:

- Automatic metric logging
- Model checkpointing
- Hyperparameter sweeps
- Visualization dashboards

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended
- **Memory**: 16GB+ RAM for large datasets
- **Storage**: Sufficient space for datasets and model checkpoints

## License

[Add your license information here]

## Citation

[Add citation information if applicable]

## Contributing

[Add contribution guidelines]
