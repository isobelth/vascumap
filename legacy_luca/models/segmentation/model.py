import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .model_utils import build_model
import torchmetrics

class SegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Segmentation.

    Wraps a segmentation model (U-Net, etc.) with loss functions, metrics, 
    and optimization logic.

    Args:
        model_name (str): Architecture name (e.g. 'Unet').
        encoder_name (str): Encoder name (e.g. 'mit_b5').
        in_channels (int): Number of input channels.
        encoder_weights (str, optional): Pretrained weights name. Defaults to "imagenet".
        learning_rate (float, optional): Learning rate. Defaults to 1e-3.
        max_epochs (int, optional): Max epochs (used for scheduler). Defaults to 200.
    """
    def __init__(self, model_name, encoder_name, in_channels, encoder_weights="imagenet", learning_rate=1e-3, max_epochs=200):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = build_model(model_name, encoder_name, in_channels, encoder_weights)
        
        # Losses
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_dice = torchmetrics.Dice(task="binary")
        self.train_iou = torchmetrics.JaccardIndex(task="binary")
        self.val_dice = torchmetrics.Dice(task="binary")
        self.val_iou = torchmetrics.JaccardIndex(task="binary")

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        
        # Ensure masks are correct shape (B, 1, H, W)
        if masks.ndim == 3:
             masks = masks.unsqueeze(1)
        
        logits = self(images)
        
        # Calculate losses
        loss_dice = self.dice_loss(logits, masks)
        loss_bce = self.bce_loss(logits, masks.float())
        loss = loss_dice + loss_bce
        
        # Metrics
        probs = torch.sigmoid(logits)
        self.train_dice(probs, masks.int())
        self.train_iou(probs, masks.int())
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        
        if masks.ndim == 3:
             masks = masks.unsqueeze(1)

        logits = self(images)
        
        loss_dice = self.dice_loss(logits, masks)
        loss_bce = self.bce_loss(logits, masks.float())
        loss = loss_dice + loss_bce
        
        probs = torch.sigmoid(logits)
        self.val_dice(probs, masks.int())
        self.val_iou(probs, masks.int())
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        
        # Replicating original StepLR logic: step_size=int(n_epochs/2), gamma=0.5
        step_size = int(self.hparams.max_epochs / 2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, verbose=False)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }
