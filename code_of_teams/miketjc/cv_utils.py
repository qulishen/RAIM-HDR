# complex-valued related utils

import numpy as np
import torch


def normalize_img_2d_complex(img):
    # Assuming img shape is [channels, height, width] or [height, width]
    if img.ndim == 3: # Multi-channel image
        norm_img = np.empty_like(img, dtype=np.complex64)
        minner = np.empty(img.shape[0])
        maxer = np.empty(img.shape[0])
        for i in range(img.shape[0]):
            norm_img[i], minner[i], maxer[i] = normalize_channel(img[i])
        return norm_img, minner, maxer
    else: # Single channel image
        return normalize_channel(img)

def normalize_channel(channel):
    mag = np.abs(channel)
    minner = np.amin(mag)
    maxer = np.amax(mag)

    if maxer - minner > 1e-8: # Avoid division by zero
        mag = (mag - minner) / (maxer - minner) # normalize to [0, 1]
    else:
        mag = np.zeros_like(mag)

    phase = np.angle(channel)
    norm_channel = mag * np.exp(1j * phase)
    return norm_channel, minner, maxer

def denormalize_img_2d_complex(norm_img, minner, maxer):
    if norm_img.ndim == 3: # Multi-channel image
        denorm_img = np.empty_like(norm_img, dtype=np.complex64)
        for i in range(norm_img.shape[0]):
            denorm_img[i] = denormalize_channel(norm_img[i], minner[i], maxer[i])
        return denorm_img
    else: # Single channel image
        return denormalize_channel(norm_img, minner, maxer)

def denormalize_channel(norm_channel, minner, maxer):
    mag = np.abs(norm_channel)
    phase = np.angle(norm_channel)

    if maxer - minner > 1e-8: # Avoid multiplication by zero
        mag = (mag * (maxer - minner)) + minner
    else:
        mag = np.full_like(mag, minner)

    denorm_channel = mag * np.exp(1j * phase)
    return denorm_channel


def convert_to_complex_numpy(image_multi_coil):
    """
    Convert a multi-channel PyTorch tensor to a complex NumPy array.

    Parameters:
    image_multi_coil (torch.Tensor): A PyTorch tensor of shape [n, height, width, 2],
                                     where the last dimension contains the real and imaginary parts.

    Returns:
    numpy.ndarray: A complex NumPy array of shape [n, height, width].
    """
    # Split the tensor into real and imaginary parts
    real_part = image_multi_coil[..., 0]
    imaginary_part = image_multi_coil[..., 1]

    # Combine the real and imaginary parts to form a complex number
    complex_image = torch.complex(real_part, imaginary_part)

    # Convert the PyTorch tensor to a NumPy array
    complex_numpy_array = complex_image.numpy()

    return complex_numpy_array


# import random

# def get_params(img, output_size, n):
#     w, h = img.shape
#     th, tw = output_size
#     if w == tw and h == th:
#         return 0, 0, h, w

#     i_list = [random.randint(0, h - th) for _ in range(n)]
#     j_list = [random.randint(0, w - tw) for _ in range(n)]
#     return i_list, j_list, th, tw


# def n_random_crops(img, x, y, h, w):
#     crops = []
#     for i in range(len(x)):
#         new_crop = img[y[i]:y[i] + h, x[i]:x[i] + w]
#         crops.append(new_crop)
#     return tuple(crops)


def overlapping_grid_indices(inp, output_size, r=None):
    """
    Generate overlapping grid indices for patch-based processing.
    """
    _, c, h, w = inp.shape
    r = 16 if r is None else r
    h_list = [i for i in range(0, h - output_size + 1, r)]
    w_list = [i for i in range(0, w - output_size + 1, r)]
    
    # Make sure we cover the entire image by adding the final index if needed
    if h - output_size not in h_list:
        h_list.append(h - output_size)
    if w - output_size not in w_list:
        w_list.append(w - output_size)
        
    return h_list, w_list

def unpatchify_restore_overlapping(x, model=None, corners=None, p_size=64, time=None, model_time_conditioning=False):
    """
    Restore image by processing overlapping patches with time conditioning support.
    
    Args:
        x: Input tensor
        model: Neural network model
        corners: List of (h, w) coordinates for patches
        p_size: Patch size
        time: Time parameter for conditioning (noise level)
        model_time_conditioning: Whether model uses time conditioning
    """
    with torch.no_grad():
        x_grid_mask = torch.zeros_like(x, device=x.device)
        e_output = torch.zeros_like(x, device=x.device)
        
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
            
        for (hi, wi) in corners:
            x_patch = x[:, :, hi:hi+p_size, wi:wi+p_size]
            if model is None:
                e_output[:, :, hi:hi + p_size, wi:wi + p_size] += x_patch
            else:
                if model_time_conditioning and time is not None:
                    num_channels = x_patch.shape[1]
                    is_color = (num_channels == 3)
                    sigma_values = torch.zeros(x_patch.shape[0]).to(x_patch.device)
                    # Iterate over the batch to estimate sigma for each image
                    for i in range(x_patch.shape[0]):
                        if is_color:
                            # For color images, estimate sigma for each channel separately and average
                            channel_sigmas = []
                            for c in range(num_channels):
                                img_channel = x_patch[i, c, :, :].detach().cpu().numpy()
                                channel_sigmas.append(estimate_sigma(img_channel))
                            # Average sigma across channels
                            sigma_values[i] = torch.tensor(np.mean(channel_sigmas), device=x_patch.device)
                        else:
                            # For grayscale, estimate sigma on the single channel
                            img = x_patch[i, 0, :, :].detach().cpu().numpy()
                            sigma_values[i] = estimate_sigma(img)
                    noise_amount = sigma_values                            

                    # Use time conditioning if enabled
                    output_patch = model(x1=x_patch, time=noise_amount)
                else:
                    # Regular model forward pass
                    output_patch = model(x_patch)
                e_output[:, :, hi:hi + p_size, wi:wi + p_size] += output_patch
                
        # Average the overlapping regions
        e = torch.div(e_output, x_grid_mask)
    return e

def patch_based_restoration(x, model=None, r=None, patch_size=64, time=None, model_time_conditioning=False):
    """
    Perform patch-based restoration with time conditioning support.
    
    Args:
        x: Input tensor
        model: Neural network model
        r: Stride for overlapping patches
        patch_size: Size of each patch
        time: Time parameter for conditioning (noise level)
        model_time_conditioning: Whether model uses time conditioning
    """
    p_size = patch_size
    h_list, w_list = overlapping_grid_indices(x, output_size=p_size, r=r)
    corners = [(i, j) for i in h_list for j in w_list]
    x_output = unpatchify_restore_overlapping(
        x, 
        model=model, 
        corners=corners, 
        p_size=p_size, 
        time=time,
        model_time_conditioning=model_time_conditioning
    )
    return x_output

