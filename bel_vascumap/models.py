from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
from monai.networks.nets import UNet
from monai.inferers import SliceInferer, SlidingWindowInferer
import torchmetrics
import numpy as np
from typing import Literal
import segmentation_models_pytorch as smp
import gc
from cupyx.scipy import ndimage
import math
import tqdm
import cupy as cp
from skimage.filters import apply_hysteresis_threshold
from utils import cupy_chunk_processing


class Generator(nn.Module):
    """
    3D U-Net Generator for Pix2Pix.

    Based on MONAI UNet implementation.
    """

    def __init__(
        self, 
        dropout_p: float = 0.4
    ):
        """
        """
        super(Generator, self).__init__()
        self.dropout_p = dropout_p
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(1, 2, 2, 2, 1),
            num_res_units=3,
            dropout=self.dropout_p,
        )

    def forward(
        self, 
        x
    ):
        """
        """
        x = self.unet(x)
        x = F.relu(x)
        return x
    
class Discriminator(nn.Module):
    """
    3D PatchGAN Discriminator.
    """
    def __init__(
        self, 
        dropout_p: float = 0.4
    ):
        """
        """
        super(Discriminator, self).__init__()
        self.dropout_p = dropout_p
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ('conv1', nn.Conv3d(2, 64, kernel_size=4, stride=(1, 2, 2), padding=(1, 1, 1))),  # Output shape: [64, 14, 128, 128]
                    ('bn1', nn.BatchNorm3d(64)),
                    ('lrelu1', nn.LeakyReLU(0.2)),

                    ('conv2', nn.Conv3d(64, 128, kernel_size=4, stride=(1, 2, 2), padding=(1, 1, 1))),  # Output shape: [128, 12, 64, 64]
                    ('bn2', nn.BatchNorm3d(128)),
                    ('lrelu2', nn.LeakyReLU(0.2)),

                    ('conv3', nn.Conv3d(128, 256, kernel_size=4, stride=(2, 2, 2), padding=(1, 1, 1))),  # Output shape: [256, 6, 32, 32]
                    ('bn3', nn.BatchNorm3d(256)),
                    ('lrelu3', nn.LeakyReLU(0.2)),

                    ('conv4', nn.Conv3d(256, 512, kernel_size=4, stride=(2, 2, 2), padding=(1, 1, 1))),  # Output shape: [512, 3, 16, 16]
                    ('bn4', nn.BatchNorm3d(512)),
                    ('lrelu4', nn.LeakyReLU(0.2)),

                    ('conv5', nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1)),  # Output shape: [1, 2, 15, 15]
                    ('sigmoid', nn.Sigmoid())
                ]
            )
        )
        
    def forward(
        self, 
        x
    ):
        """
        """
        x = self.model(x)
        return x

class Pix2Pix(pl.LightningModule):
    """
    PyTorch Lightning Module for Pix2Pix 3D Image Translation.

    Implements the training logic for GANs:
    - Generator attempts to fool Discriminator and minimize pixel-wise error (L1/L2).
    - Discriminator attempts to distinguish real pairs from fake pairs.

    Args:
        generator_dropout_p (float): Dropout for generator.
        discriminator_dropout_p (float): Dropout for discriminator.
        generator_lr (float): Learning rate for generator.
        discriminator_lr (float): Learning rate for discriminator.
        weight_decay (float): L2 regularization.
        lr_scheduler_T_0 (float): Cosine annealing restart interval.
        lr_scheduler_T_mult (float): Cosine annealing multiplier.
    """
    
    def __init__(
        self, 
        generator_dropout_p=0.4, 
        discriminator_dropout_p=0.4, 
        generator_lr=1e-3, 
        discriminator_lr=1e-6, 
        weight_decay=1e-5, 
        lr_scheduler_T_0=5e3, 
        lr_scheduler_T_mult=2,
        model_path: str = None
    ):

        super(Pix2Pix, self).__init__()

        self.save_hyperparameters()
        # Important to disable automatic optimization as it 
        # will be done manually as there are two optimizers
        self.automatic_optimization = False
        self.generator_lr = generator_lr               # Generator learning rate
        self.discriminator_lr = discriminator_lr       # Discriminator learning rate
        self.weight_decay = weight_decay               # Weight decay e.g. L2 regularization
        self.lr_scheduler_T_0 = lr_scheduler_T_0       # Optimizer initial restart step number
        self.lr_scheduler_T_mult = lr_scheduler_T_mult # Optimizer restart step number factor
        
        # Models
        self.generator = Generator(dropout_p=generator_dropout_p)
        self.discriminator = Discriminator(dropout_p=discriminator_dropout_p)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            
        
    def forward(self, x):
        return self.generator(x)
    
    def generator_loss(self, prediction_image, target_image, prediction_label, target_label):
        """
        Generator loss (a combination of): 
            1 - Binary Cross-Entropy
                Between predicted labels (generated by the discriminator) and target labels which is all 1s
            2 - L1 / Mean Absolute Error (weighted by lambda)
                Between generated image and target image
            3 - L2 / Mean Squared Error (weighted by lambda)
                Between generated image and target image
        """
        bce_loss = F.binary_cross_entropy(prediction_label, target_label)
        l1_loss = F.l1_loss(prediction_image, target_image)
        mse_loss = F.mse_loss(prediction_image, target_image)
        return bce_loss, l1_loss, mse_loss
    
    def discriminator_loss(self, prediction_label, target_label):
        """
        Discriminator loss: 
            1 - Binary Cross-Entropy
                Between predicted labels (generated by the discriminator) and target labels
                The target would be all 0s if the input of the discriminator is the generated image (generator)
                The target would be all 1s if the input of the discriminator is the target image (dataloader)
        """
        bce_loss = F.binary_cross_entropy(prediction_label, target_label)
        return bce_loss
    
    def configure_optimizers(self):
        """
        Using Adam optimizer for both generator and discriminator including L2 regularization
        Both would have different initial learning rates
        Stochastic Gradient Descent with Warm Restarts is also added as learning scheduler (https://arxiv.org/abs/1608.03983)
        """
        # Optimizers
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr, weight_decay=self.weight_decay)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, weight_decay=self.weight_decay)
        # Learning Scheduler
        genertator_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(generator_optimizer, T_0=self.lr_scheduler_T_0, T_mult=self.lr_scheduler_T_mult)
        discriminator_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(discriminator_optimizer, T_0=self.lr_scheduler_T_0, T_mult=self.lr_scheduler_T_mult)
        return [generator_optimizer, discriminator_optimizer], [genertator_lr_scheduler, discriminator_lr_scheduler]

    def training_step(self, batch, batch_idx, TRAIN_BATCH_SIZE=6, LAMBDA=100):
        # Optimizers
        generator_optimizer, discriminator_optimizer = self.optimizers()
        generator_lr_scheduler, discriminator_lr_scheduler = self.lr_schedulers()
        
        image, target = batch

        image_i, image_j = torch.split(image, TRAIN_BATCH_SIZE // 2)
        target_i, target_j = torch.split(target, TRAIN_BATCH_SIZE // 2)
        
        ######################################
        #  Discriminator Loss and Optimizer  #
        ######################################
        # Generator Feed-Forward
        generator_prediction = self.forward(image_i)
        generator_prediction = torch.clip(generator_prediction, 0, 1)
        # Discriminator Feed-Forward
        discriminator_prediction_real = self.discriminator(torch.cat((image_i, target_i), dim=1))
        discriminator_prediction_fake = self.discriminator(torch.cat((image_i, generator_prediction), dim=1))
        # Discriminator Loss
        discriminator_label_real = self.discriminator_loss(discriminator_prediction_real, 
                                                           torch.ones_like(discriminator_prediction_real))
        discriminator_label_fake = self.discriminator_loss(discriminator_prediction_fake,
                                                           torch.zeros_like(discriminator_prediction_fake))
        discriminator_loss = discriminator_label_real + discriminator_label_fake
        # Discriminator Optimizer
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()
        discriminator_lr_scheduler.step()
        
        ##################################
        #  Generator Loss and Optimizer  #
        ##################################
        #  Generator Feed-Forward
        generator_prediction = self.forward(image_j)
        generator_prediction = torch.clip(generator_prediction, 0, 1)
        # Discriminator Feed-Forward
        discriminator_prediction_fake = self.discriminator(torch.cat((image_j, generator_prediction), dim=1))
        # Generator loss
        generator_bce_loss, generator_l1_loss, generator_mse_loss = self.generator_loss(generator_prediction, target_j,
                                                                                        discriminator_prediction_fake,
                                                                                        torch.ones_like(discriminator_prediction_fake))
        generator_loss = generator_bce_loss + (generator_l1_loss * LAMBDA) + (generator_mse_loss * LAMBDA)
        # Generator Optimizer
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()
        generator_lr_scheduler.step()
        
        # Progressbar and Logging
        loss = OrderedDict({'train_g_bce_loss': generator_bce_loss, 'train_g_l1_loss': generator_l1_loss, 'train_g_mse_loss': generator_mse_loss,
                            'train_g_loss': generator_loss, 'train_d_loss': discriminator_loss,
                            'train_g_lr': generator_lr_scheduler.get_last_lr()[0], 'train_d_lr': discriminator_lr_scheduler.get_last_lr()[0]})
        self.log_dict(loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        image, target = batch

        psnr = torchmetrics.PeakSignalNoiseRatio().to(device)
        ssim = torchmetrics.StructuralSimilarityIndexMeasure().to(device)
        # C4: use separate Accuracy instances so each call doesn't reset the others' state
        gen_acc_metric       = torchmetrics.Accuracy(task="binary").to(device)
        disc_acc_real_metric = torchmetrics.Accuracy(task="binary").to(device)
        disc_acc_fake_metric = torchmetrics.Accuracy(task="binary").to(device)
        
        # Generator Feed-Forward
        generator_prediction = self.forward(image)
        generator_prediction = torch.clip(generator_prediction, 0, 1)
        # Generator Metrics
        generator_psnr = psnr(generator_prediction, target)
        generator_ssim = ssim(generator_prediction, target)
        discriminator_prediction_fake = self.discriminator(torch.cat((image, generator_prediction), dim=1))
        generator_accuracy = gen_acc_metric(discriminator_prediction_fake, torch.ones_like(discriminator_prediction_fake, dtype=torch.int32))
        
        # Discriminator Feed-Forward
        discriminator_prediction_real = self.discriminator(torch.cat((image, target), dim=1))
        discriminator_prediction_fake = self.discriminator(torch.cat((image, generator_prediction), dim=1))
        # Discriminator Metrics
        discriminator_accuracy = (
            disc_acc_real_metric(discriminator_prediction_real, torch.ones_like(discriminator_prediction_real, dtype=torch.int32)) * 0.5
            + disc_acc_fake_metric(discriminator_prediction_fake, torch.zeros_like(discriminator_prediction_fake, dtype=torch.int32)) * 0.5
        )
            
        # Progressbar and Logging
        metrics = OrderedDict({'val_g_psnr': generator_psnr, 'val_g_ssim': generator_ssim,
                               'val_g_accuracy': generator_accuracy, 'val_d_accuracy': discriminator_accuracy})
        
        self.log_dict(metrics, prog_bar=True)
        return metrics
    
    def predict(
        self,
        stack_bf: np.ndarray,
        device: Literal["cuda", "cpu"],
        n_iter: int = 1
    ) -> np.ndarray:
        """
            Perform inference using the Pix2Pix 3D model.

            Translates a brightfield stack to a fluorescence-like volume.
            Uses sliding window inference.

            Args:
                model_p2p (torch.nn.Module): The loaded Pix2Pix model.
                stack_bf (np.ndarray): Input brightfield stack.
                device (str): Device for inference.
                n_iter (int, optional): Number of iterations for averaging (if stochastic). Defaults to 1.

            Returns:
                np.ndarray: Predicted volume.
        """
        
        self.eval()
        input_p2p_tensor = torch.tensor(stack_bf[None, None, ...], device='cpu').float()

        inferer = SlidingWindowInferer(
            roi_size=(32, 512, 512), 
            overlap=(0.75, 0.25, 0.25), 
            sw_batch_size=1, 
            sw_device=device, 
            device='cpu', 
            progress=True
        )
        
        self.to(device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
            reconstruction_mean = torch.stack([inferer(inputs=input_p2p_tensor, network=self).to('cpu') for _ in range(n_iter)])
            reconstruction_mean = torch.clip(reconstruction_mean, 0, 1)
            reconstruction_mean = torch.mean(reconstruction_mean, dim=0)

        vessel_pred = reconstruction_mean.squeeze().cpu().numpy()

        self.to('cpu')
        del input_p2p_tensor, reconstruction_mean
        if 'cuda' in device:
            torch.cuda.empty_cache()

        return vessel_pred

def adapt_input_conv(in_chans, conv_weight):
    """
    Adapt input channels of a convolutional layer's weights.

    Handles resizing of weights for 1-channel (grayscale) or N-channel inputs, 
    initializing from 3-channel (RGB) pretrained weights.

    Args:
        in_chans (int): Number of target input channels.
        conv_weight (torch.Tensor): Original weights (O, I, K, K).

    Returns:
        torch.Tensor: Adapted weights.
    """
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)

    return conv_weight

def adapt_input_model(model):
    """
    Adapt the first layer of a segmentation model to accept 1-channel input.

    Specifically targets models with 'encoder.patch_embed1.proj' structure (e.g. MiT/SegFormer).

    Args:
        model (nn.Module): The segmentation model.

    Returns:
        nn.Module: The model with modified first layer.
    """
    # Adapt first layer to take 1 channel as input - timm approach = sum weights
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'patch_embed1'):
         # Specific for certain encoders like MiT (SegFormer)
        new_weights = adapt_input_conv(in_chans=1, conv_weight=model.encoder.patch_embed1.proj.weight)
        model.encoder.patch_embed1.proj = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))

        with torch.no_grad():
            model.encoder.patch_embed1.proj.weight = torch.nn.parameter.Parameter(new_weights)
    
    return model

def load_segmentation_model(model_path, device='cuda'):
    """
    Load a segmentation model (U-Net with MiT-B5 encoder) from a checkpoint.

    Supports both raw PyTorch state dicts and PyTorch Lightning checkpoints.

    Args:
        model_path (str): Path to the checkpoint file.
        device (str, optional): Device to map model to. Defaults to 'cuda'.

    Returns:
        torch.nn.Module: The loaded model in eval mode.
    """
    # Original code used smp.Unet with adapt_input_model. 
    # Since we refactored model_utils, we should use that or just recreate logic here if we want to support old checkpoints.
    # Assuming we want to support the checkpoints trained with the new code or old code? 
    # If old code used smp.Unet directly + adapt, we should replicate.
    
    # Replicating the load_smp logic from original inference.py
    
    model = smp.Unet(encoder_name='mit_b5', classes=1, in_channels=3, encoder_weights=None)
    model = adapt_input_model(model)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint: # Lightning checkpoint
         # Lightning checkpoints have "model." prefix usually if using LightningModule
         state_dict = checkpoint['state_dict']
         # Remove "model." prefix if present
         new_state_dict = {}
         for k, v in state_dict.items():
             if k.startswith('model.'):
                 new_state_dict[k[6:]] = v
             else:
                 new_state_dict[k] = v
         model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
        
    return model.eval()
    
def predict_mask_ortho(model_smp, vessel_pred_iso, device):
    """
    Perform segmentation inference using 2.5D approach (axial, coronal, sagittal averaging).

    Predictions are made along all three axes and averaged for better 3D consistency.

    Args:
        model_smp (torch.nn.Module): The segmentation model.
        vessel_pred_iso (np.ndarray): Isotropic input volume.
        device (str): Device for inference.

    Returns:
        np.ndarray: Averaged probability map.
    """
    
    vessel_pred_tensor = torch.tensor(vessel_pred_iso[None, None, ...], device='cpu').float()

    tensor_shape = vessel_pred_tensor.shape
    
    model_smp.to(device)

    axial_inferer = SliceInferer(
        roi_size=(1024, 1024), 
        sw_batch_size=4, 
        progress=True, 
        mode="gaussian",
        overlap=0.5,
        device='cpu', 
        sw_device=device)

    with torch.no_grad():
        output3D_axial = torch.sigmoid(axial_inferer(vessel_pred_tensor, model_smp))

    vessel_proba = output3D_axial.squeeze().to('cpu').numpy().astype(np.float32, copy=False)
    del output3D_axial
    if 'cuda' in device:
        torch.cuda.empty_cache()

    # A3: compute padding params once, reuse for both coronal and sagittal unpadding
    z_depth = tensor_shape[2]
    pad_pre = 0
    if z_depth < 256:
        pad_d = 256 - z_depth
        pad_pre = pad_d // 2
        pad_post = pad_d - pad_pre
        vessel_pred_tensor = F.pad(vessel_pred_tensor, (0, 0, 0, 0, pad_pre, pad_post), mode='constant', value=-1)

    coronal_inferer = SliceInferer(
        roi_size=(256, 256),
        sw_batch_size=32,
        spatial_dim=1, 
        progress=True, 
        mode="gaussian",
        overlap=0.5,
        device='cpu', 
        sw_device=device
        )

    sagital_inferer = SliceInferer(
        roi_size=(256, 256),
        sw_batch_size=32,
        spatial_dim=2, 
        progress=True, 
        mode="gaussian",
        overlap=0.5,
        device='cpu', 
        sw_device=device
        )

    with torch.no_grad():
        output3D_coronal = torch.sigmoid(coronal_inferer(vessel_pred_tensor, model_smp))

    if pad_pre > 0:
        vessel_coronal = output3D_coronal.squeeze().to('cpu').numpy()[pad_pre:pad_pre + z_depth, :, :]
    else:
        vessel_coronal = output3D_coronal.squeeze().to('cpu').numpy()

    vessel_proba += vessel_coronal
    del output3D_coronal, vessel_coronal
    if 'cuda' in device:
        torch.cuda.empty_cache()

    with torch.no_grad():
        output3D_sagital = torch.sigmoid(sagital_inferer(vessel_pred_tensor, model_smp))

    if pad_pre > 0:
        vessel_sagital = output3D_sagital.squeeze().to('cpu').numpy()[pad_pre:pad_pre + z_depth, :, :]
    else:
        vessel_sagital = output3D_sagital.squeeze().to('cpu').numpy()

    vessel_proba += vessel_sagital
    vessel_proba /= 3.0
    del output3D_sagital, vessel_sagital

    model_smp.to('cpu')
    del vessel_pred_tensor
    if 'cuda' in device:
        torch.cuda.empty_cache()
    gc.collect()

    return vessel_proba

def process_vessel_mask(vessel_proba, ortho=False):
    """
    Process vessel probability map into a binary mask.

    Applies median filtering (if ortho is True) and hysteresis thresholding.

    Args:
        vessel_proba (np.ndarray): Vessel probability map (values 0-1).
        ortho (bool, optional): Whether orthogonal plane predictions were used. 
            If True, applies extra filtering. Defaults to False.

    Returns:
        np.ndarray: Binary vessel mask.
    """
    if ortho:
        vessel_filtered = median_filter_3d_gpu(vessel_proba, size=7, chunk_size=(32, 1024, 1024))
        vessel_out = apply_hysteresis_threshold(vessel_filtered, 0.2, 0.5)
    else:
        vessel_out = apply_hysteresis_threshold(vessel_proba, 0.1, 0.5)
    return vessel_out


def median_filter_3d_gpu(volume, size=3, chunk_size=(64, 64, 64)):
    """
    Apply 3D median filter using GPU with chunked processing.

    Args:
        volume (np.ndarray): 3D input volume (Z, Y, X).
        size (int or tuple, optional): Median filter kernel size. Defaults to 3.
        chunk_size (tuple, optional): Chunk size. Defaults to (64, 64, 64).

    Returns:
        np.ndarray: Filtered volume with same shape as input.
    """
    return cupy_chunk_processing(
        volume, 
        ndimage.median_filter, 
        chunk_size=chunk_size, 
        overlap=(size//2, size//2, size//2) if isinstance(size, int) else (size[0]//2, size[1]//2, size[2]//2),
        size=size
    )
