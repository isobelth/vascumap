"""
VascuMap3D - 3D Image Translation Training Module

This module provides training functionality for the Pix2Pix 3D GAN model
that translates brightfield Z-stacks to fluorescence-like volumes.

Usage:
    python -m vascumap.models.image_translation.train \\
        --input_path /path/to/input \\
        --target_path /path/to/target \\
        --model_path /path/to/save/models
"""

import argparse
import glob
import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torchsummary import summary

from .utils import ImageDataset3D, Generator, Discriminator, Pix2Pix


def train(
    input_path: str,
    target_path: str,
    model_path: str,
    batch_size: int = 4,
    epochs: int = 200,
    generator_lr: float = 1e-3,
    discriminator_lr: float = 1e-5,
    generator_dropout: float = 0.1,
    discriminator_dropout: float = 0.25,
    val_split: float = 0.1,
    num_workers: int = 0,
    device: str = 'cuda'
):
    """
    Train the Pix2Pix 3D model for brightfield to fluorescence translation.

    Args:
        input_path (str): Path to directory containing input NIfTI volumes (.nii.gz).
        target_path (str): Path to directory containing target NIfTI volumes (.nii.gz).
        model_path (str): Path to save model checkpoints.
        batch_size (int, optional): Training batch size. Defaults to 4.
        epochs (int, optional): Number of training epochs. Defaults to 200.
        generator_lr (float, optional): Generator learning rate. Defaults to 1e-3.
        discriminator_lr (float, optional): Discriminator learning rate. Defaults to 1e-5.
        generator_dropout (float, optional): Generator dropout rate. Defaults to 0.1.
        discriminator_dropout (float, optional): Discriminator dropout rate. Defaults to 0.25.
        val_split (float, optional): Validation split fraction. Defaults to 0.1.
        num_workers (int, optional): DataLoader workers. Defaults to 0.
        device (str, optional): Device for training. Defaults to 'cuda'.
    """
    
    # Load file paths
    input_filenames = sorted(glob.glob(os.path.join(input_path, '*.nii.gz')))
    target_filenames = sorted(glob.glob(os.path.join(target_path, '*.nii.gz')))
    
    if len(input_filenames) == 0:
        raise ValueError(f"No .nii.gz files found in {input_path}")
    if len(target_filenames) == 0:
        raise ValueError(f"No .nii.gz files found in {target_path}")
    if len(input_filenames) != len(target_filenames):
        raise ValueError(f"Mismatch: {len(input_filenames)} inputs vs {len(target_filenames)} targets")
    
    # Split data
    train_inputs, val_inputs = train_test_split(input_filenames, test_size=val_split, random_state=42)
    train_targets, val_targets = train_test_split(target_filenames, test_size=val_split, random_state=42)
    
    print(f'Training samples: {len(train_inputs)} | Validation samples: {len(val_inputs)}')
    
    # Create datasets and dataloaders
    train_dataset = ImageDataset3D(train_inputs, train_targets, split='train')
    val_dataset = ImageDataset3D(val_inputs, val_targets, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True, 
        drop_last=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        pin_memory=True
    )
    
    # Print model summaries
    print("\n=== Generator Architecture ===")
    generator = Generator().to(device)
    summary(generator, input_size=(1, 16, 256, 256), device=device)
    
    print("\n=== Discriminator Architecture ===")
    discriminator = Discriminator().to(device)
    summary(discriminator, input_size=(2, 16, 256, 256), device=device)
    
    # Initialize model
    model = Pix2Pix(
        generator_dropout_p=generator_dropout,
        discriminator_dropout_p=discriminator_dropout,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        lr_scheduler_T_0=len(train_loader) * 2
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename='{epoch}-{val_g_psnr:.2f}-{val_g_ssim:.2f}',
        monitor='val_g_ssim',
        save_top_k=5,
        mode='max'
    )
    
    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=epochs,
        val_check_interval=0.5,
        default_root_dir=model_path,
        callbacks=[checkpoint_callback],
        logger=True,
        num_sanity_val_steps=1
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_path = os.path.join(model_path, 'last_epoch_full.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to: {final_path}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Train Pix2Pix 3D model for brightfield to fluorescence translation."
    )
    
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to directory containing input NIfTI volumes.")
    parser.add_argument("--target_path", type=str, required=True,
                        help="Path to directory containing target NIfTI volumes.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to save model checkpoints.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size. Default: 4")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs. Default: 200")
    parser.add_argument("--generator_lr", type=float, default=1e-3,
                        help="Generator learning rate. Default: 1e-3")
    parser.add_argument("--discriminator_lr", type=float, default=1e-5,
                        help="Discriminator learning rate. Default: 1e-5")
    parser.add_argument("--generator_dropout", type=float, default=0.1,
                        help="Generator dropout rate. Default: 0.1")
    parser.add_argument("--discriminator_dropout", type=float, default=0.25,
                        help="Discriminator dropout rate. Default: 0.25")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split fraction. Default: 0.1")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers. Default: 0")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device for training. Default: cuda")
    
    args = parser.parse_args()
    
    train(
        input_path=args.input_path,
        target_path=args.target_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        generator_dropout=args.generator_dropout,
        discriminator_dropout=args.discriminator_dropout,
        val_split=args.val_split,
        num_workers=args.num_workers,
        device=args.device
    )
