import math
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

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

def build_model(model_str, encoder_str, in_channels=1, encoder_weights=None):
    """
    Factory function to build a segmentation model using segmentation_models_pytorch.

    Args:
        model_str (str): Architecture name (e.g. 'Unet').
        encoder_str (str): Encoder name (e.g. 'mit_b5').
        in_channels (int, optional): Number of input channels. Defaults to 1.
        encoder_weights (str, optional): Pretrained weights ('imagenet'). Defaults to None.

    Returns:
        nn.Module: The constructed model.
    """

    # Create a dictionary mapping from string to actual SMP function
    ARCHITECTURES = {
       'Unet': smp.Unet,
       'Unet++': smp.UnetPlusPlus,
       'MAnet': smp.MAnet,
       'Linknet': smp.Linknet,
       'FPN': smp.FPN,
       'PSPNet': smp.PSPNet,
       'PAN': smp.PAN,
       'DeepLabV3': smp.DeepLabV3,
       'DeepLabV3+': smp.DeepLabV3Plus
    }

    try:
        # Get constructor method for desired architecture
        constructor = ARCHITECTURES[model_str]
       
        if encoder_str.startswith('mit') and in_channels == 1:
            model = constructor(encoder_name=encoder_str, classes=1, in_channels=3, encoder_weights=encoder_weights)
            model = adapt_input_model(model)
        else: 
            model = constructor(encoder_name=encoder_str, classes=1, in_channels=in_channels, encoder_weights=encoder_weights)

        return model
   
    except KeyError:
       raise ValueError(f'Model type "{model_str}" not understood. Valid options are: {list(ARCHITECTURES.keys())}')
