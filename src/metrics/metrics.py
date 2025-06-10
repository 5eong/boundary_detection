"""
Evaluation metrics for segmentation tasks.

This module contains various metrics commonly used for evaluating
segmentation model performance, including F1 score, IoU, and MCC.
"""

from typing import Tuple, Optional
import torch
import numpy as np


def pixelwise_f1(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    channels: int = 2, 
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute pixelwise F1 score for each channel.
    
    Args:
        pred: Predicted logits or probabilities [B, C, H, W]
        target: Ground truth targets [B, C, H, W]
        channels: Number of channels to evaluate
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Tuple of F1 scores for each channel
    """
    pred = (pred > threshold).float()

    if channels is None:
        channels = pred.size(1)
        
    f1_scores = []
    
    for channel in range(channels):
        tp = torch.sum(pred[:, channel] * target[:, channel])
        fp = torch.sum(pred[:, channel] * (1 - target[:, channel]))
        fn = torch.sum((1 - pred[:, channel]) * target[:, channel])

        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        f1_scores.append(f1.item())
        
    return tuple(f1_scores) if len(f1_scores) > 1 else (f1_scores[0], 0.0)


def mean_iou(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    channels: int = 2, 
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute mean Intersection over Union (IoU) for each channel.
    
    Args:
        pred: Predicted logits or probabilities [B, C, H, W]
        target: Ground truth targets [B, C, H, W]
        channels: Number of channels to evaluate
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Tuple of IoU scores for each channel
    """
    pred = (pred > threshold).float()
    
    if channels is None:
        channels = pred.size(1)

    iou_scores = []

    for channel in range(channels):
        intersection = torch.sum(pred[:, channel] * target[:, channel])
        union = (
            torch.sum(pred[:, channel]) + 
            torch.sum(target[:, channel]) - 
            intersection
        )
        
        iou = intersection / (union + 1e-8)
        iou_scores.append(iou.item())

    return tuple(iou_scores) if len(iou_scores) > 1 else (iou_scores[0], 0.0)


def mcc(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    channels: int = 2, 
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute Matthews Correlation Coefficient (MCC) for each channel.
    
    MCC is a balanced metric that works well even with imbalanced classes.
    It returns a value between -1 and 1, where 1 indicates perfect prediction,
    0 indicates random prediction, and -1 indicates inverse prediction.
    
    Args:
        pred: Predicted logits or probabilities [B, C, H, W]
        target: Ground truth targets [B, C, H, W]
        channels: Number of channels to evaluate
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Tuple of MCC scores for each channel
    """
    pred = (pred > threshold).float()
    
    mcc_scores = []
    
    for channel in range(channels):
        target_channel = target[:, channel].cpu().numpy()
        pred_channel = pred[:, channel].cpu().numpy()
        
        tp = np.sum((target_channel == 1) & (pred_channel == 1))
        tn = np.sum((target_channel == 0) & (pred_channel == 0))
        fp = np.sum((target_channel == 0) & (pred_channel == 1))
        fn = np.sum((target_channel == 1) & (pred_channel == 0))

        numerator = (tp * tn) - (fp * fn)
        
        term1 = np.sqrt(tp + fp)
        term2 = np.sqrt(tp + fn)
        term3 = np.sqrt(tn + fp)
        term4 = np.sqrt(tn + fn)

        if term1 == 0 or term2 == 0 or term3 == 0 or term4 == 0:
            mcc_scores.append(0.0)
        else:
            denominator = term1 * term2 * term3 * term4
            mcc_value = numerator / denominator
            mcc_scores.append(float(mcc_value))

    return tuple(mcc_scores) if len(mcc_scores) > 1 else (mcc_scores[0], 0.0)


def dice_coefficient(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    smooth: float = 1e-8,
    threshold: float = 0.5
) -> float:
    """
    Compute Dice coefficient (F1 score for binary segmentation).
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth targets
        smooth: Smoothing factor to avoid division by zero
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Dice coefficient
    """
    pred = (pred > threshold).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def jaccard_index(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    smooth: float = 1e-8,
    threshold: float = 0.5
) -> float:
    """
    Compute Jaccard index (IoU for binary segmentation).
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth targets
        smooth: Smoothing factor to avoid division by zero
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Jaccard index
    """
    pred = (pred > threshold).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    
    return jaccard.item()


def precision_recall(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute precision and recall.
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth targets
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Tuple of (precision, recall)
    """
    pred = (pred > threshold).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return precision.item(), recall.item()


def specificity(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> float:
    """
    Compute specificity (true negative rate).
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth targets
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Specificity score
    """
    pred = (pred > threshold).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    
    spec = tn / (tn + fp + 1e-8)
    return spec.item()


def compute_all_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    channels: Optional[int] = None,
    threshold: float = 0.5
) -> dict:
    """
    Compute all available metrics for segmentation evaluation.
    
    Args:
        pred: Predicted logits or probabilities [B, C, H, W]
        target: Ground truth targets [B, C, H, W]
        channels: Number of channels to evaluate (None for all)
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Dictionary containing all computed metrics
    """
    if channels is None:
        channels = pred.size(1)
    
    metrics = {}
    
    # Per-channel metrics
    f1_scores = pixelwise_f1(pred, target, channels, threshold)
    iou_scores = mean_iou(pred, target, channels, threshold)
    mcc_scores = mcc(pred, target, channels, threshold)
    
    for i in range(min(len(f1_scores), channels)):
        metrics[f'f1_channel_{i}'] = f1_scores[i]
        metrics[f'iou_channel_{i}'] = iou_scores[i]
        metrics[f'mcc_channel_{i}'] = mcc_scores[i]
    
    if pred.size(1) == 1:
        pred_binary = pred.squeeze(1) if pred.dim() > 3 else pred
        target_binary = target.squeeze(1) if target.dim() > 3 else target
        
        metrics['dice'] = dice_coefficient(pred_binary, target_binary, threshold=threshold)
        metrics['jaccard'] = jaccard_index(pred_binary, target_binary, threshold=threshold)
        
        precision, recall = precision_recall(pred_binary, target_binary, threshold)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['specificity'] = specificity(pred_binary, target_binary, threshold)
    
    if len(f1_scores) > 1:
        metrics['mean_f1'] = np.mean(f1_scores)
        metrics['mean_iou'] = np.mean(iou_scores)
        metrics['mean_mcc'] = np.mean(mcc_scores)
    
    return metrics


class MetricsCalculator:
    """
    A class for computing segmentation metrics with running averages.
    
    This class maintains running averages of metrics across multiple batches,
    which is useful for validation and testing loops.
    """
    
    def __init__(self, channels: int = 2, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            channels: Number of channels to evaluate
            threshold: Threshold for converting probabilities to binary
        """
        self.channels = channels
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_samples = 0
        self.sum_f1 = [0.0] * self.channels
        self.sum_iou = [0.0] * self.channels
        self.sum_mcc = [0.0] * self.channels
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a new batch.
        
        Args:
            pred: Predicted logits or probabilities
            target: Ground truth targets
        """
        f1_scores = pixelwise_f1(pred, target, self.channels, self.threshold)
        iou_scores = mean_iou(pred, target, self.channels, self.threshold)
        mcc_scores = mcc(pred, target, self.channels, self.threshold)
        
        batch_size = pred.size(0)
        self.total_samples += batch_size
        
        for i in range(self.channels):
            self.sum_f1[i] += f1_scores[i] * batch_size
            self.sum_iou[i] += iou_scores[i] * batch_size
            self.sum_mcc[i] += mcc_scores[i] * batch_size
    
    def compute(self) -> dict:
        """
        Compute average metrics across all accumulated samples.
        
        Returns:
            Dictionary of average metrics
        """
        if self.total_samples == 0:
            return {}
        
        metrics = {}
        
        for i in range(self.channels):
            metrics[f'f1_channel_{i}'] = self.sum_f1[i] / self.total_samples
            metrics[f'iou_channel_{i}'] = self.sum_iou[i] / self.total_samples
            metrics[f'mcc_channel_{i}'] = self.sum_mcc[i] / self.total_samples
        
        if self.channels > 1:
            metrics['mean_f1'] = np.mean([metrics[f'f1_channel_{i}'] for i in range(self.channels)])
            metrics['mean_iou'] = np.mean([metrics[f'iou_channel_{i}'] for i in range(self.channels)])
            metrics['mean_mcc'] = np.mean([metrics[f'mcc_channel_{i}'] for i in range(self.channels)])
        
        return metrics