import torch
import torch.nn.functional as F
import numpy as np
import gc
from monai.inferers import SliceInferer, SlidingWindowInferer
import segmentation_models_pytorch as smp
from luca_version_vascumap.models.image_translation.utils import Pix2Pix
# from vascumap.models.segmentation.model import SegmentationModule # Use if loading lightning checkpoint, but original code loaded a .pth or .ckpt with specific logic

def load_pix2pix(model_path, device='cuda'):
    """
    Load a Pix2Pix 3D model from a checkpoint.

    Args:
        model_path (str): Path to the checkpoint file.
        device (str, optional): Device to map model to ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        Pix2Pix: The loaded model in eval mode.
    """
    checkpoint = torch.load(model_path, map_location='cpu') # Load to cpu first
    model = Pix2Pix()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval()

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
    from luca_version_vascumap.models.segmentation.model_utils import adapt_input_model
    
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

def predict_pix2pix(model_p2p, stack_bf, device, n_iter=1):
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

    input_p2p_tensor = torch.tensor(stack_bf[None, None, ...], device='cpu').float()

    inferer = SlidingWindowInferer(
        roi_size=(32, 512, 512), 
        overlap=(0.75, 0.25, 0.25), 
        sw_batch_size=1, 
        sw_device=device, 
        device='cpu', 
        progress=True
        )
    
    model_p2p.to(device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
        reconstruction_mean = torch.stack([inferer(inputs=input_p2p_tensor, network=model_p2p).to('cpu') for _ in range(n_iter)])
        reconstruction_mean = torch.clip(reconstruction_mean, 0, 1)
        reconstruction_mean = torch.mean(reconstruction_mean, dim=0)

    vessel_pred = reconstruction_mean.squeeze().cpu().numpy()

    model_p2p.to('cpu')
    del input_p2p_tensor, reconstruction_mean
    if 'cuda' in device:
        torch.cuda.empty_cache()

    return vessel_pred

def predict_mask(model_smp, vessel_pred, device):
    """
    Perform segmentation inference using a 2D model with axial slicing.

    Args:
        model_smp (torch.nn.Module): The segmentation model.
        vessel_pred (np.ndarray): Input volume (e.g., predicted fluorescence).
        device (str): Device for inference.

    Returns:
        np.ndarray: Probability map (values 0-1).
    """
    
    vessel_pred_tensor = torch.tensor(vessel_pred[None, None, ...]).float().to(device)

    axial_inferer = SliceInferer(
        roi_size=(1024, 1024), 
        sw_batch_size=3, 
        progress=True, 
        mode="gaussian",
        overlap=0.5,
        device='cpu', 
        sw_device=device)

    model_smp.to(device)

    with torch.no_grad():
        output3D_axial = axial_inferer(vessel_pred_tensor, model_smp)
        output3D_axial = torch.sigmoid(output3D_axial)

    vessel_proba = output3D_axial.squeeze().to('cpu').numpy()

    model_smp.to('cpu')
    del vessel_pred_tensor, output3D_axial
    if 'cuda' in device:
        torch.cuda.empty_cache()
    gc.collect()

    return vessel_proba

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
        output3D_axial =  torch.sigmoid(axial_inferer(vessel_pred_tensor, model_smp))

    vessel_axial = output3D_axial.squeeze().to('cpu').numpy()

    if tensor_shape[2] < 256:
        pad_d = 256 - tensor_shape[2]
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
        output3D_sagital = torch.sigmoid(sagital_inferer(vessel_pred_tensor, model_smp))

    if tensor_shape[2] < 256:
        pad_d = 256 - tensor_shape[2]
        pad_pre = pad_d // 2
        vessel_coronal = output3D_coronal.squeeze().to('cpu').numpy()[pad_pre:pad_pre + tensor_shape[2], :, :]
        vessel_sagital = output3D_sagital.squeeze().to('cpu').numpy()[pad_pre:pad_pre + tensor_shape[2], :, :]

    else:
        vessel_coronal = output3D_coronal.squeeze().to('cpu').numpy()
        vessel_sagital = output3D_sagital.squeeze().to('cpu').numpy()


    vessel_proba = np.mean(
        np.array([
        vessel_axial, 
        vessel_coronal,
        vessel_sagital]), axis=0
        )

    model_smp.to('cpu')
    del vessel_pred_tensor, output3D_axial, output3D_coronal, output3D_sagital
    if 'cuda' in device:
        torch.cuda.empty_cache()
    gc.collect()

    return vessel_proba
