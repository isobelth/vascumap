
import glob
import tifffile as tif
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

from torchvision import transforms
import torchio as tio

import pytorch_lightning as pl # pip install pytorch-lightning

from torchsummary import summary # pip install torchsummary
from monai.networks.nets import UNet
# import segmentation_models_pytorch as smp # pip install segmentation-models-pytorch
import torchmetrics
from monai.networks import normal_init

class ImageDataset3D(Dataset):
    """
    PyTorch Dataset for 3D image volumes (NIfTI).

    Args:
        input_paths (list): List of paths to input volumes.
        target_paths (list, optional): List of paths to target volumes. Defaults to None.
        split (str, optional): 'train', 'val', or 'test'. Defaults to 'train'.
        transform (callable, optional): TorchIO transforms. Defaults to None.
    """

    # def __init__(self, filenames, split='train', transform=None, TRAIN_IMAGE_SIZE=256, VAL_IMAGE_SIZE=256):
    def __init__(self, input_paths, target_paths=None, split='train', transform=None):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.split = split
        self.transform = transform
        self.split = split

        # Data transform
        if not transform:
            if self.split == 'train':
                self.transform = tio.Compose([
                    transforms.Lambda(lambda t: torch.tensor(t).float()),
                    # tio.transforms.RandomAffine(
                    #     scales=, 
                    #     degrees=0, 
                    #     # translation=50, 
                    #     default_pad_value=0, 
                    #     # image_interpolation='bspline'
                    #     ),
                    tio.transforms.RandomFlip(axes=('LR','AP')),
                ])
            elif self.split == 'val':
                self.transform = tio.Compose([
                    transforms.Lambda(lambda t: torch.tensor(t).float()),
                ])
            else:
                self.transform = tio.Compose([
                    transforms.Lambda(lambda t: torch.tensor(t).float()),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):

        input_image = nib.load(self.input_paths[idx]).get_fdata(dtype=np.float32) 
        input_image = np.expand_dims(input_image, axis=0)  # Shape becomes [C, H, W, D]

        if self.target_paths is not None: 
            target_image = nib.load(self.target_paths[idx]).get_fdata(dtype=np.float32)
            target_image = np.expand_dims(target_image, axis=0)  # Shape becomes [C, H, W, D]

            combined = np.concatenate((input_image, target_image), axis=0)  # Shape becomes [2C, D, H, W]
        
        else: 
            combined = input_image

        if self.transform:
            combined = self.transform(combined)

        if self.target_paths is not None: 
            input_image = combined[:1, :, :, :]
            target_image = combined[1:, :, :, :]
            return input_image, target_image
        
        else: 
            input_image = combined
            return input_image

class Generator(nn.Module):
    """
    3D U-Net Generator for Pix2Pix.

    Based on MONAI UNet implementation.

    Args:
        dropout_p (float, optional): Dropout probability. Defaults to 0.4.
    """

    def __init__(self, dropout_p=0.4):
        super(Generator, self).__init__()

        self.dropout_p = dropout_p
        # Load Unet with Resnet34 Embedding from SMP library. Pre-trained on Imagenet
        # self.unet = smp.Unet(encoder_name="efficientnet-b7", encoder_weights=None, 
        #                      in_channels=2, classes=1, activation=None)
        
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(1, 2, 2, 2, 1),
            num_res_units=3,
            dropout=self.dropout_p,
        )

        # self.unet.apply(normal_init)
        # Adding two layers of Dropout as the original Unet doesn't have any
        # This will be used to feed noise into the networking during both training and evaluation
        # These extra layers will be added on decoder part where 2D transposed convolution is occured
        # for idx in range(1, 3):
        #     self.unet.decoder.blocks[idx].conv1.add_module('3', nn.Dropout2d(p=self.dropout_p))

        # Disabling in-place ReLU as to avoid in-place operations as it will
        # cause issues for double backpropagation on the same graph
        # for module in self.unet.modules():
        #     if isinstance(module, nn.ReLU):
        #         module.inplace = False

    def forward(self, x):
        x = self.unet(x)
        x = F.relu(x)
        return x

class Discriminator(nn.Module):
    """
    3D PatchGAN Discriminator.

    Args:
        dropout_p (float, optional): Dropout probability. Defaults to 0.4.
    """
    def __init__(self, dropout_p=0.4):
        super(Discriminator, self).__init__()

        self.dropout_p = dropout_p

        self.model = nn.Sequential(OrderedDict([
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
        ]))

    def forward(self, x):
        x = self.model(x)
        # x = F.sigmoid(x)
        return  x

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
    def __init__(self, generator_dropout_p=0.4, discriminator_dropout_p=0.4, generator_lr=1e-3, discriminator_lr=1e-6, 
                 weight_decay=1e-5, lr_scheduler_T_0=5e3, lr_scheduler_T_mult=2):

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
        accuracy = torchmetrics.Accuracy(task="binary").to(device)
        
        # Generator Feed-Forward
        generator_prediction = self.forward(image)
        generator_prediction = torch.clip(generator_prediction, 0, 1)
        # Generator Metrics
        generator_psnr = psnr(generator_prediction, target)
        generator_ssim = ssim(generator_prediction, target)
        discriminator_prediction_fake = self.discriminator(torch.cat((image, generator_prediction), dim=1))
        generator_accuracy = accuracy(discriminator_prediction_fake, torch.ones_like(discriminator_prediction_fake, dtype=torch.int32))
        
        # Discriminator Feed-Forward
        discriminator_prediction_real = self.discriminator(torch.cat((image, target), dim=1))
        discriminator_prediction_fake = self.discriminator(torch.cat((image, generator_prediction), dim=1))
        # Discriminator Metrics
        discriminator_accuracy = accuracy(discriminator_prediction_real, torch.ones_like(discriminator_prediction_real, dtype=torch.int32)) * 0.5 + \
                                accuracy(discriminator_prediction_fake, torch.zeros_like(discriminator_prediction_fake, dtype=torch.int32)) * 0.5
            
        # Progressbar and Logging
        metrics = OrderedDict({'val_g_psnr': generator_psnr, 'val_g_ssim': generator_ssim,
                               'val_g_accuracy': generator_accuracy, 'val_d_accuracy': discriminator_accuracy})
        
        self.log_dict(metrics, prog_bar=True)
        return metrics