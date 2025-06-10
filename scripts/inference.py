#!/usr/bin/env python3
"""
Inference script for field delineation models.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import project modules
from data.datasets import AI4Boundaries, Euro_0512, MyanmarSatellite
from models.segmentator import PLSegmentator
from models.boundary_segmentator import PLSegmentator as BoundaryPLSegmentator
from utils.visualization import visualize_geotiff_samples


def create_inference_parser():
    """Create argument parser for inference script."""
    parser = argparse.ArgumentParser(description='Field Delineation Inference')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing data for inference')
    
    # Model arguments
    parser.add_argument('--dataset-type', type=str, 
                       choices=['AI4Boundaries', 'Euro_0512', 'MyanmarSatellite'],
                       default='MyanmarSatellite',
                       help='Dataset type')
    parser.add_argument('--channels', type=int, default=9,
                       help='Number of input channels')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save prediction masks')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to visualize')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    
    return parser


def create_inference_dataset(data_dir: str, dataset_type: str, channels: int):
    """Create dataset for inference."""
    if dataset_type == 'AI4Boundaries':
        dataset = AI4Boundaries(data_dir, augment=False)
    elif dataset_type == 'Euro_0512':
        dataset = Euro_0512(data_dir, channels=channels, augment=False)
    elif dataset_type == 'MyanmarSatellite':
        dataset = MyanmarSatellite(data_dir, channels=channels, augment=False)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return dataset


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to load as boundary segmentator first, then fallback to regular
    try:
        model = BoundaryPLSegmentator.load_from_checkpoint(checkpoint_path)
    except:
        try:
            model = PLSegmentator.load_from_checkpoint(checkpoint_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from checkpoint: {e}")
    
    model.eval()
    model.to(device)
    
    return model


def run_inference(model, dataloader, device: str, save_predictions: bool = False, output_dir: str = None):
    """Run inference on dataset."""
    predictions = []
    
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            images = batch['image'].to(device)
            
            # Run inference
            outputs = model(images)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs)
            
            # Store predictions
            predictions.append({
                'batch_idx': batch_idx,
                'predictions': probabilities.cpu(),
                'file_paths': batch.get('file_path', []),
            })
            
            # Save predictions if requested
            if save_predictions and output_dir:
                for i, pred in enumerate(probabilities):
                    save_path = Path(output_dir) / f"prediction_batch_{batch_idx}_sample_{i}.pt"
                    torch.save(pred.cpu(), save_path)
            
            print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    return predictions


def create_visualizations(model, dataset, device: str, num_samples: int, output_dir: str):
    """Create and save visualizations."""
    # Create a small dataloader for visualization
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    viz_dataloader = DataLoader(subset, batch_size=1, shuffle=False)
    
    # Create visualizations
    try:
        visualize_geotiff_samples(model, viz_dataloader, plot_mask=True)
        
        if output_dir:
            plt.savefig(Path(output_dir) / "visualizations.png", dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {output_dir}/visualizations.png")
        
        plt.show()
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")


def calculate_metrics(model, dataset, device: str):
    """Calculate metrics on dataset with ground truth."""
    from metrics.metrics import pixelwise_f1, mean_iou, mcc
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    f1_scores = []
    iou_scores = []
    mcc_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            if 'mask' not in batch:
                print("Warning: No ground truth masks found, skipping metrics calculation")
                return None
            
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            
            # Calculate metrics
            f1_interior, f1_border = pixelwise_f1(outputs, masks)
            iou_interior, iou_border = mean_iou(outputs, masks)
            mcc_interior, mcc_border = mcc(outputs, masks)
            
            f1_scores.append((f1_interior, f1_border))
            iou_scores.append((iou_interior, iou_border))
            mcc_scores.append((mcc_interior, mcc_border))
    
    # Calculate averages
    avg_f1 = np.mean(f1_scores, axis=0)
    avg_iou = np.mean(iou_scores, axis=0)
    avg_mcc = np.mean(mcc_scores, axis=0)
    
    metrics = {
        'f1_interior': avg_f1[0],
        'f1_border': avg_f1[1],
        'iou_interior': avg_iou[0],
        'iou_border': avg_iou[1],
        'mcc_interior': avg_mcc[0],
        'mcc_border': avg_mcc[1],
    }
    
    return metrics


def main():
    """Main inference function."""
    parser = create_inference_parser()
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_inference_dataset(args.data_dir, args.dataset_type, args.channels)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")
    
    # Run inference
    print("Running inference...")
    predictions = run_inference(
        model, dataloader, device, 
        save_predictions=args.save_predictions, 
        output_dir=args.output_dir
    )
    
    # Calculate metrics if ground truth is available
    print("Calculating metrics...")
    metrics = calculate_metrics(model, dataset, device)
    if metrics:
        print("\nMetrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Create visualizations
    if args.visualize:
        print("Creating visualizations...")
        create_visualizations(model, dataset, device, args.num_samples, args.output_dir)
    
    print(f"Inference completed! Processed {len(predictions)} batches.")
    if args.output_dir:
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()