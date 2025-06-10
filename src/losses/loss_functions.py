"""
Loss functions for segmentation tasks.

This module contains various loss functions commonly used in segmentation tasks,
including binary and multi-class variants.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(label: torch.Tensor, n_classes: int, requires_grad: bool = True) -> torch.Tensor:
    """
    Convert label tensor to one-hot encoding.
    
    Args:
        label: Input label tensor
        n_classes: Number of classes
        requires_grad: Whether to require gradients
        
    Returns:
        One-hot encoded tensor
    """
    device = label.device
    one_hot_label = torch.eye(n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)
    return one_hot_label


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss function for segmentation.
    
    This loss function weights pixels based on their proximity to boundaries,
    helping improve boundary detection accuracy.
    """
    
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth targets
            
        Returns:
            Computed loss value
        """
        if inputs.dim() == 3:  # Add channel dimension if necessary
            inputs = inputs.unsqueeze(1)

        log_p = inputs.permute(0, 2, 3, 1).contiguous().view(-1)
        target_t = targets.view(-1).float()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)

        # Compute weights for balanced loss
        pos_num = pos_index.sum().float()
        neg_num = neg_index.sum().float()
        sum_num = pos_num + neg_num
        
        weight = torch.zeros_like(log_p)
        weight[pos_index] = neg_num / sum_num
        weight[neg_index] = pos_num / sum_num

        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight=weight, reduction='mean'
        )
        return loss


class BCELoss(nn.Module):
    """Binary Cross Entropy Loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss."""
        return F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    The Dice loss is based on the Dice coefficient (also known as F1 score),
    which measures the overlap between predicted and ground truth masks.
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth targets
            
        Returns:
            Computed Dice loss
        """
        inputs = F.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss.
    
    This loss combines the benefits of both Dice loss (good for class imbalance)
    and BCE loss (good for pixel-wise accuracy).
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + BCE loss."""
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = 1 - (2. * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        bce = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='mean')
        
        return bce + dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss down-weights easy examples and focuses learning on hard examples.
    Reference: "Focal Loss for Dense Object Detection" by Lin et al.
    """
    
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, smooth: float = 1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal loss."""
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce
        
        return focal_loss


class TanimotoLoss(nn.Module):
    """
    Tanimoto Loss (also known as Jaccard Loss).
    
    The Tanimoto coefficient is equivalent to the Jaccard index and measures
    the similarity between predicted and ground truth sets.
    """
    
    def __init__(self, smooth: float = 1e-5):
        super(TanimotoLoss, self).__init__()
        self.smooth = smooth

    def forward(self, labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """Compute Tanimoto complement loss."""
        return self.tanimoto_complement_loss(labels, predictions).median()

    def tanimoto_loss(self, labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """Compute basic Tanimoto loss."""
        # Volume-weighted approach
        vli = torch.mean(torch.sum(labels, dim=[1, 2]), dim=0)
        wli = 1.0 / (vli ** 2)

        # Handle infinite weights
        if torch.isinf(wli).all():
            max_wli = torch.tensor(0.0, device=wli.device)
        else:
            max_wli = torch.max(wli[~torch.isinf(wli)])
        
        wli[wli == float('inf')] = max_wli

        # Compute Tanimoto coefficient
        square_pred = predictions ** 2
        square_label = labels ** 2
        add_squared = square_pred + square_label
        sum_square = torch.sum(add_squared, dim=[1, 2])

        product = predictions * labels
        sum_product = torch.sum(product, dim=[1, 2])

        sum_product_labels = torch.sum(wli * sum_product, dim=-1)
        denominator = sum_square - sum_product
        denominator_sum_labels = torch.sum(wli * denominator, dim=-1)

        tanimoto = (sum_product_labels + self.smooth) / (denominator_sum_labels + self.smooth)
        return tanimoto

    def tanimoto_complement_loss(self, labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """Compute Tanimoto complement loss."""
        loss_1 = self.tanimoto_loss(predictions, labels)
        pred_complement = 1.0 - predictions
        label_complement = 1.0 - labels
        loss_2 = self.tanimoto_loss(label_complement, pred_complement)
        tanimoto = (loss_1 + loss_2) * 0.5
        return 1.0 - tanimoto


class TverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation.
    
    The Tversky loss is a generalization of Dice loss that allows for weighting
    of false positives and false negatives differently.
    """
    
    def __init__(self, smooth: float = 1.0, alpha: float = 0.5, beta: float = 0.5):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Tversky loss."""
        inputs = F.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
       
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss.
    
    Combines the benefits of Focal loss and Tversky loss for handling
    class imbalance and hard examples.
    """
    
    def __init__(
        self, 
        smooth: float = 1.0, 
        alpha: float = 0.5, 
        beta: float = 0.5, 
        gamma: float = 1.0
    ):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Tversky loss."""
        inputs = F.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class ComboLoss(nn.Module):
    """
    Combo Loss combining Dice and Cross Entropy.
    
    This loss combines weighted cross entropy and Dice loss for better
    handling of both pixel-wise accuracy and shape preservation.
    """
    
    def __init__(
        self, 
        smoothing: float = 0.1, 
        alpha: float = 0.5, 
        ce_ratio: float = 0.5, 
        eps: float = 1e-6
    ):
        super(ComboLoss, self).__init__()
        self.smoothing = smoothing
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Combo loss."""
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # Dice component
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smoothing) / (
            inputs.sum() + targets.sum() + self.smoothing
        )
        
        # Weighted cross entropy component
        inputs = torch.clamp(inputs, self.eps, 1.0 - self.eps)
        out = -(
            self.alpha * (targets * torch.log(inputs)) + 
            (1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs)
        )
        weighted_ce = out.mean(-1)

        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)
        return combo


# Lovász Loss implementation
def flatten_binary_scores(scores: torch.Tensor, labels: torch.Tensor, ignore: Optional[int] = None):
    """Flatten predictions for binary case."""
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    return scores[valid], labels[valid]


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # Cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovász hinge loss (flat version)."""
    if len(labels) == 0:
        return logits.sum() * 0.
    
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def lovasz_hinge(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    per_image: bool = True, 
    ignore: Optional[int] = None
) -> torch.Tensor:
    """Binary Lovász hinge loss."""
    if per_image:
        def mean(losses):
            return sum(losses) / len(losses)
        
        loss = mean(
            lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovász Hinge Loss for binary segmentation.
    
    The Lovász loss is based on the submodular Lovász extension of submodular losses,
    providing a surrogate for optimizing intersection-over-union.
    """
    
    def __init__(self, per_image: bool = False):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Lovász hinge loss."""
        inputs = F.sigmoid(inputs)
        return lovasz_hinge(inputs, targets, per_image=self.per_image)


# Multi-class Lovász loss functions
def flatten_probas(probas: torch.Tensor, labels: torch.Tensor, ignore: Optional[int] = None):
    """Flatten predictions in the multiclass case."""
    if probas.dim() == 3:
        # Assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
        
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes: str = 'present'):
    """Multi-class Lovász-Softmax loss (flat version)."""
    if probas.numel() == 0:
        return probas * 0.
    
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    
    for c in class_to_sum:
        fg = (labels == c).float()  # Foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    
    return sum(losses) / len(losses) if losses else torch.tensor(0., device=probas.device)


def lovasz_softmax(
    probas: torch.Tensor, 
    labels: torch.Tensor, 
    classes: str = 'present', 
    per_image: bool = False, 
    ignore: Optional[int] = None
):
    """Multi-class Lovász-Softmax loss."""
    if per_image:
        def mean(losses):
            return sum(losses) / len(losses)
        
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), 
                classes=classes
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss