"""
Visualization utilities for field delineation models.

This module contains functions for visualizing model predictions,
input data, and evaluation results.
"""

from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import torch
import numpy as np
import rasterio
from torch.utils.data import DataLoader


def normalize_for_display(img: np.ndarray, vmin: float = 0, vmax: float = 3000) -> np.ndarray:
    """
    Normalize image for display purposes.
    
    Args:
        img: Input image array
        vmin: Minimum value for clipping
        vmax: Maximum value for clipping
        
    Returns:
        Normalized image array
    """
    img_norm = (img - img.min()) / (img.max() - img.min())
    img_8bit = (img_norm * 255).astype(np.uint8)
    return img_8bit


def create_rgb_composite(
    image_tensor: torch.Tensor, 
    bands: Tuple[int, int, int] = (0, 1, 2)
) -> np.ndarray:
    """
    Create RGB composite from multi-band tensor.
    
    Args:
        image_tensor: Input image tensor [C, H, W]
        bands: Tuple of band indices for RGB channels
        
    Returns:
        RGB composite as numpy array
    """
    r, g, b = bands
    rgb = torch.stack([image_tensor[r], image_tensor[g], image_tensor[b]], dim=0)
    rgb = rgb.permute(1, 2, 0).cpu().numpy()
    return normalize_for_display(rgb)


def plot_true_color_geotiff(
    geotiff_path: str, 
    ax: plt.Axes, 
    vmin: float = 0, 
    vmax: float = 3000
) -> None:
    """
    Plot true color image from GeoTIFF file.
    
    Args:
        geotiff_path: Path to GeoTIFF file
        ax: Matplotlib axes to plot on
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
    """
    try:
        with rasterio.open(geotiff_path) as src:
            red = src.read(3)
            green = src.read(2)
            blue = src.read(1)
            
            true_color = np.stack((red, green, blue), axis=-1)
            true_color = np.clip(true_color, vmin, vmax)
            true_color = (true_color - vmin) / (vmax - vmin)
            
        ax.imshow(true_color)
        ax.axis('off')
    except Exception as e:
        print(f"Warning: Could not load GeoTIFF {geotiff_path}: {e}")
        ax.text(0.5, 0.5, 'Image not available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.axis('off')


def visualize_samples(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    num_samples: int = 10,
    plot_mask: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize model predictions on sample data.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing test data
        num_samples: Number of samples to visualize
        plot_mask: Whether to plot ground truth masks
        save_path: Optional path to save the visualization
    """
    model.eval()
    
    all_batches = list(dataloader)
    sample_indices = np.random.choice(len(all_batches), min(num_samples, len(all_batches)), replace=False)
    sampled_batches = [all_batches[i] for i in sample_indices]

    sample_batch = sampled_batches[0]
    channels = sample_batch['image'].shape[1]
    
    if channels not in [3, 9]:
        print(f"Warning: Unexpected number of channels ({channels}). Visualization may not work correctly.")

    cols = 5 if plot_mask else 4
    fig, axes = plt.subplots(len(sampled_batches), cols, figsize=(15, 3 * len(sampled_batches)))
    
    if len(sampled_batches) == 1:
        axes = axes.reshape(1, -1)

    for idx, batch in enumerate(sampled_batches):
        inputs = batch["image"].to(model.device)

        with torch.no_grad():
            predictions = model(inputs)
            if not isinstance(predictions, torch.Tensor):
                predictions = predictions.logits if hasattr(predictions, 'logits') else predictions

        input_img = inputs[0].cpu()
        pred_img = torch.sigmoid(predictions[0]).cpu()

        if channels == 9:
            stitched_input = torch.cat([
                input_img[i*3:(i+1)*3] for i in range(3)
            ], dim=2).permute(1, 2, 0).numpy()
            stitched_input = normalize_for_display(stitched_input)
            axes[idx, 0].imshow(stitched_input)
        else:
            rgb_input = input_img[:3].permute(1, 2, 0).numpy()
            rgb_input = normalize_for_display(rgb_input)
            axes[idx, 0].imshow(rgb_input)

        col_offset = 1
        
        if plot_mask and 'mask' in batch:
            masks = batch["mask"].to(model.device)
            
            if 'binary' in batch:
                binary = batch["binary"].to(model.device)
                predictions_masked = predictions * binary
            else:
                predictions_masked = predictions
            
            f1_interior, f1_border = pixelwise_f1(predictions_masked, masks)
            iou_interior, iou_border = mean_iou(predictions_masked, masks)
            
            axes[idx, 0].set_title(f"F1: {f1_interior:.2f}, IoU: {iou_interior:.2f}")

            mask_img = masks[0].cpu()
            if mask_img.shape[0] > 1:
                mask_display = mask_img[0].numpy()
            else:
                mask_display = mask_img.squeeze().numpy()
            
            axes[idx, 1].imshow(mask_display, cmap="gray")
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 1].axis('off')
            col_offset = 2

        num_pred_channels = min(pred_img.shape[0], cols - col_offset)
        channel_names = ['Interior', 'Border', 'Distance']
        
        for i in range(num_pred_channels):
            col_idx = col_offset + i
            if col_idx < cols:
                pred_channel = pred_img[i].numpy()
                axes[idx, col_idx].imshow(pred_channel, cmap="gray", vmin=0, vmax=1)
                title = channel_names[i] if i < len(channel_names) else f"Channel {i}"
                axes[idx, col_idx].set_title(title)
                axes[idx, col_idx].axis('off')

        axes[idx, 0].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def visualize_geotiff_samples(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    num_samples: int = 10,
    plot_mask: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize model predictions with GeoTIFF true color images.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing test data
        num_samples: Number of samples to visualize
        plot_mask: Whether to plot ground truth masks
        save_path: Optional path to save the visualization
    """
    model.eval()
    
    all_batches = list(dataloader)
    sample_indices = np.random.choice(len(all_batches), min(num_samples, len(all_batches)), replace=False)
    sampled_batches = [all_batches[i] for i in sample_indices]

    channels = sampled_batches[0]['image'].shape[1]
    if channels not in [3, 9]:
        print(f"Warning: Unexpected number of channels ({channels}). Visualization may not work correctly.")

    cols = 5 if plot_mask else 4
    fig, axes = plt.subplots(len(sampled_batches), cols, figsize=(15, 3 * len(sampled_batches)))
    
    if len(sampled_batches) == 1:
        axes = axes.reshape(1, -1)

    for idx, batch in enumerate(sampled_batches):
        inputs = batch["image"].to(model.device)

        with torch.no_grad():
            predictions = model(inputs)
            if not isinstance(predictions, torch.Tensor):
                predictions = predictions.logits if hasattr(predictions, 'logits') else predictions

        if 'file_path' in batch and len(batch['file_path']) > 0:
            geotiff_path = batch["file_path"][0]
            plot_true_color_geotiff(geotiff_path, axes[idx, 0])
        else:
            input_img = inputs[0].cpu()
            if channels >= 3:
                rgb_input = input_img[:3].permute(1, 2, 0).numpy()
                rgb_input = normalize_for_display(rgb_input)
                axes[idx, 0].imshow(rgb_input)
            axes[idx, 0].axis('off')

        pred_img = torch.sigmoid(predictions[0]).cpu()
        col_offset = 1

        if plot_mask and 'mask' in batch:
            masks = batch["mask"].to(model.device)
            
            if 'binary' in batch:
                binary = batch["binary"].to(model.device)
                predictions_masked = predictions * binary
            else:
                predictions_masked = predictions

            f1_interior, f1_border = pixelwise_f1(predictions_masked, masks)
            iou_interior, iou_border = mean_iou(predictions_masked, masks)

            axes[idx, 0].set_title(f"F1: {f1_interior:.2f}, IoU: {iou_interior:.2f}")

            mask_img = masks[0].cpu()
            if mask_img.shape[0] > 1:
                mask_display = mask_img[0].numpy()
            else:
                mask_display = mask_img.squeeze().numpy()
            
            axes[idx, 1].imshow(mask_display, cmap="gray")
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 1].axis('off')
            col_offset = 2

        num_pred_channels = min(pred_img.shape[0], cols - col_offset)
        channel_names = ['Interior', 'Border', 'Distance']
        
        for i in range(num_pred_channels):
            col_idx = col_offset + i
            if col_idx < cols:
                pred_channel = pred_img[i].numpy()
                axes[idx, col_idx].imshow(pred_channel, cmap="gray", vmin=0, vmax=1)
                title = channel_names[i] if i < len(channel_names) else f"Channel {i}"
                axes[idx, col_idx].set_title(title)
                axes[idx, col_idx].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def create_prediction_overlay(
    image: np.ndarray, 
    prediction: np.ndarray, 
    alpha: float = 0.5,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Create overlay of prediction on input image.
    
    Args:
        image: Input RGB image [H, W, 3]
        prediction: Prediction mask [H, W] or [H, W, C]
        alpha: Overlay transparency
        colors: List of RGB colors for each class
        
    Returns:
        Overlaid image
    """
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    overlay = image.copy()
    
    if prediction.ndim == 2:
        mask = prediction > 0.5
        color = colors[0]
        overlay[mask] = overlay[mask] * (1 - alpha) + np.array(color) * alpha
    else:
        for c in range(prediction.shape[2]):
            mask = prediction[:, :, c] > 0.5
            if c < len(colors):
                color = colors[c]
                overlay[mask] = overlay[mask] * (1 - alpha) + np.array(color) * alpha
    
    return overlay.astype(np.uint8)


def compare_predictions(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    dataloader: DataLoader,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    num_samples: int = 5,
    save_path: Optional[str] = None
) -> None:
    """
    Compare predictions from two different models.
    
    Args:
        model1: First model to compare
        model2: Second model to compare
        dataloader: DataLoader containing test data
        model1_name: Name for first model
        model2_name: Name for second model
        num_samples: Number of samples to compare
        save_path: Optional path to save the comparison
    """
    model1.eval()
    model2.eval()
    
    all_batches = list(dataloader)
    sample_indices = np.random.choice(len(all_batches), min(num_samples, len(all_batches)), replace=False)
    sampled_batches = [all_batches[i] for i in sample_indices]

    cols = 6
    fig, axes = plt.subplots(len(sampled_batches), cols, figsize=(18, 3 * len(sampled_batches)))
    
    if len(sampled_batches) == 1:
        axes = axes.reshape(1, -1)

    for idx, batch in enumerate(sampled_batches):
        inputs = batch["image"].to(model1.device)
        
        with torch.no_grad():
            pred1 = torch.sigmoid(model1(inputs)[0]).cpu()
            pred2 = torch.sigmoid(model2(inputs)[0]).cpu()

        input_img = inputs[0].cpu()
        if input_img.shape[0] >= 3:
            rgb_input = input_img[:3].permute(1, 2, 0).numpy()
            rgb_input = normalize_for_display(rgb_input)
            axes[idx, 0].imshow(rgb_input)
        axes[idx, 0].set_title("Input")
        axes[idx, 0].axis('off')

        if 'mask' in batch:
            mask = batch['mask'][0].cpu()
            mask_display = mask[0].numpy() if mask.shape[0] > 1 else mask.squeeze().numpy()
            axes[idx, 1].imshow(mask_display, cmap="gray")
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 1].axis('off')

        pred1_display = pred1[0].numpy()
        pred2_display = pred2[0].numpy()
        
        axes[idx, 2].imshow(pred1_display, cmap="gray", vmin=0, vmax=1)
        axes[idx, 2].set_title(model1_name)
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(pred2_display, cmap="gray", vmin=0, vmax=1)
        axes[idx, 3].set_title(model2_name)
        axes[idx, 3].axis('off')

        if 'mask' in batch:
            mask_np = mask[0].numpy() if mask.shape[0] > 1 else mask.squeeze().numpy()
            diff1 = np.abs(pred1_display - mask_np)
            diff2 = np.abs(pred2_display - mask_np)
            
            axes[idx, 4].imshow(diff1, cmap="hot", vmin=0, vmax=1)
            axes[idx, 4].set_title(f"{model1_name} Error")
            axes[idx, 4].axis('off')
            
            axes[idx, 5].imshow(diff2, cmap="hot", vmin=0, vmax=1)
            axes[idx, 5].set_title(f"{model2_name} Error")
            axes[idx, 5].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.show()