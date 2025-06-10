import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from optimizers.lookahead import Lookahead
from metrics.metrics import pixelwise_f1, mean_iou
from losses import loss_functions as loss


loss_registry = {
    "BCELoss": loss.BCELoss,
    "ComboLoss": loss.ComboLoss,
    "DiceBCELoss": loss.DiceBCELoss,
    "FocalLoss": loss.FocalLoss,
    "TanimotoLoss": loss.TanimotoLoss,
    "FocalTverskyLoss": loss.FocalTverskyLoss,
    "DiceLoss": smp.losses.DiceLoss,
    "TverskyLoss": smp.losses.TverskyLoss,
    "LovaszHingeLoss": smp.losses.LovaszLoss,
    "MCCLoss": smp.losses.MCCLoss,
    "JaccardLoss": smp.losses.JaccardLoss,
    "SoftBCEWithLogitsLoss": smp.losses.SoftBCEWithLogitsLoss,
    "SoftCrossEntropyLoss": smp.losses.SoftCrossEntropyLoss,
}


def loss_function(name, **kwargs):
    """Factory function to instantiate loss functions."""
    loss_class = loss_registry.get(name)
    if not loss_class:
        raise ValueError(f"No such loss function: {name}")
    
    # Add default parameters for SMP losses
    if name == "DiceLoss" and not kwargs:
        kwargs = {"mode": "multilabel", "from_logits": True}
    
    return loss_class(**kwargs)


class PLSegmentator(pl.LightningModule):
    """PyTorch Lightning segmentation model."""

    def __init__(self, context, train_dataset=None, val_dataset=None, test_dataset=None):
        super(PLSegmentator, self).__init__()

        self.in_channels = context['in_channels']

        self.loss_1 = loss_function(context["loss_1"], **context.get("params_1", {}))
        self.loss_2 = loss_function(context["loss_2"], **context.get("params_2", {}))
        self.loss_3 = loss_function(context["loss_3"], **context.get("params_3", {}))
        self.weight_1 = context['weight_1']
        self.weight_2 = context['weight_2']
        self.weight_3 = context['weight_3']

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.model = smp.create_model(
            context['arch'], 
            encoder_name=context['encoder'], 
            encoder_weights=context['encoder_weights'], 
            in_channels=context['out_channels'],  # context['in_channels'], 
            classes=context['out_channels']
        )
                    
        params = smp.encoders.get_preprocessing_params(
            context['encoder'], 
            pretrained=context['encoder_weights']
        )
        self.std = nn.Parameter(
            torch.tensor(params["std"]).view(1, 3, 1, 1), 
            requires_grad=False
        )
        self.mean = nn.Parameter(
            torch.tensor(params["mean"]).view(1, 3, 1, 1), 
            requires_grad=False
        )

        self.temporal_compress = nn.Conv2d(self.in_channels, 3, kernel_size=1)
                
        self.learning_rate = context['learning_rate']
        self.batch_size = context['batch_size']

        self.save_hyperparameters()

    def update_context(self, context):
        """Update model configuration."""
        for key, value in context.items():
            setattr(self, key, value)

    def normalise(self, image):
        """Normalize image using pretrained model statistics."""
        channels = self.in_channels // 3
        
        for index in range(channels):
            stacked_indices = range((index*3), ((index*3) + 3))
            image[:, stacked_indices, :, :] = (
                image[:, stacked_indices, :, :] - self.mean
            ) / self.std
        return image
    
    def forward(self, image):
        """Forward pass through the model."""
        outputs = self.normalise(image)
        outputs = self.temporal_compress(outputs)
        outputs = self.model(outputs)
        return outputs

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, _, _, _, _ = self.compute_loss(batch, train=True)
        self.log('train loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, f1_interior, f1_border, iou_interior, iou_border = self.compute_loss(batch)
        self.log('val loss', loss)
        self.log('val f1_interior', f1_interior)
        self.log('val f1_border', f1_border)
        self.log('val iou_interior', iou_interior)
        self.log('val iou_border', iou_border)

    def test_step(self, batch, batch_idx):
        """Test step."""
        _, f1_interior, f1_border, iou_interior, iou_border = self.compute_loss(batch)
        self.log('test f1_interior', f1_interior)
        self.log('test f1_border', f1_border)
        self.log('test iou_interior', iou_interior)
        self.log('test iou_border', iou_border)

    def predict(self, image):
        """Make prediction on image."""
        self.eval()
        with torch.no_grad():
            mask = self.forward(image)
            mask = torch.sigmoid(mask)
        return mask

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4, 
            persistent_workers=True, 
            pin_memory=True
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=4, 
            persistent_workers=True, 
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=4, 
            persistent_workers=True, 
            pin_memory=True
        )

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = optim.RAdam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=3e-5
        )
        optimizer = Lookahead(optimizer=optimizer, alpha=0.6, k=10)
        optimizer.defaults = optimizer.optimizer.defaults
        
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=4
            ),
            'monitor': 'val loss',
            'interval': 'epoch',
            'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def compute_loss(self, batch, train=False):
        """Compute loss and metrics."""
        image = batch["image"]
        mask = batch["mask"]
        binary = batch["binary"]

        logits = self.forward(image)
        logits = logits * binary

        loss = (
            (self.weight_1 * self.loss_1(logits, mask)) + 
            (self.weight_2 * self.loss_2(logits, mask)) + 
            (self.weight_3 * self.loss_3(logits, mask))
        )

        if train:
            return loss, [], [], [], []
        else:
            f1_interior, f1_border = pixelwise_f1(logits, mask)
            iou_interior, iou_border = mean_iou(logits, mask)
            return loss, f1_interior, f1_border, iou_interior, iou_border