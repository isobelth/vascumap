import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from .dataset import load_data
from .model import SegmentationModule

def train(images_path, masks_path, model_path, model_str, encoder_str, 
          weights, in_channels, batch_size, epochs, learning_rate, fp16):
    """
    Train a segmentation model using PyTorch Lightning.

    Args:
        images_path (str): Path to training images directory.
        masks_path (str): Path to training masks directory.
        model_path (str): Path to save model checkpoints.
        model_str (str): Architecture name (e.g., 'Unet').
        encoder_str (str): Encoder name (e.g., 'mit_b5').
        weights (str): Pretrained weights (e.g., 'imagenet').
        in_channels (int): Number of input channels.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        fp16 (bool): Whether to use mixed precision (16-bit).
    """
          
    # Load data and prepare loaders
    loaders = load_data(images_path, masks_path, batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["valid"]

    # Initialize model
    model = SegmentationModule(
        model_name=model_str,
        encoder_name=encoder_str,
        in_channels=in_channels,
        encoder_weights=weights,
        learning_rate=learning_rate,
        max_epochs=epochs # Pass epochs for scheduler calculation
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename='{epoch}-{val_loss:.2f}-{val_dice:.2f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=model_path,
        precision="16-mixed" if fp16 else 32
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a segmentation model on provided images and masks.")
    
    parser.add_argument("--images_dir_path", type=str, required=True,
                        help="Path to the directory containing training images.")
    parser.add_argument("--masks_dir_path", type=str, required=True,
                        help="Path to the directory containing corresponding mask images.")
    parser.add_argument("--model_dir_path", type=str, required=True,
                        help="Path to the directory where model checkpoints and logs will be saved.")
    parser.add_argument("--model_architecture", type=str, default="Unet",
                        help="Model architecture to use for segmentation. Default is 'Unet'.")
    parser.add_argument("--encoder_architecture", type=str, default="mit_b5",
                        help="Encoder architecture to use within the segmentation model. Default is 'mit_b5'.")
    parser.add_argument("--input_channels", type=int, default=1,
                        help="Number of input channels for the model. Default is 1.")
    parser.add_argument("--weights", type=str, default="imagenet",
                        help="Pretrained weights to initialize the encoder. Default is 'imagenet'.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training. Default is 16.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs to train the model. Default is 200.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer during training. Default is 0.001 (1e-3).")
    parser.add_argument("--fp16", action='store_true',
                        help="Enable mixed precision training using FP16. Reduces memory usage and can improve performance.")

    args = parser.parse_args()

    train(
      images_path=args.images_dir_path,
      masks_path=args.masks_dir_path,
      model_path=args.model_dir_path,
      model_str=args.model_architecture,
      encoder_str=args.encoder_architecture,
      in_channels=args.input_channels,
      weights=args.weights,
      batch_size=args.batch_size,
      epochs=args.epochs,
      learning_rate=args.learning_rate, 
      fp16=args.fp16
   )
