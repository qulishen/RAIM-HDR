from typing import List, Tuple
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
import random


def corrupt(x, amount, diffusion2self=True):
    """
    Apply a corruption process to an image using Gaussian noise.
    
    Parameters:
    x (numpy.ndarray): The input image array.
    amount (float): The strength of the corruption, should be a single float.
    diffusion2self (bool, optional): If True, uses a corruption method involving square roots.
                                     If False, uses a linear combination of the image and noise.
                                     If None, the method is chosen randomly.
    
    Returns:
    numpy.ndarray: The corrupted image.
    """
    noise = np.random.randn(*x.shape)
    if diffusion2self is None:
        diffusion2self = random.choice([True, False])
    if diffusion2self:
        return np.sqrt(1 - amount) * x + np.sqrt(amount) * noise
    else:
        return x * (1 - amount) + noise * amount


def get_stratified_coords(
    box_size: int,
    shape: Tuple[int, ...],
    resample: bool = False,
) -> Tuple[List[int], ...]:
    """
    Create stratified blind spot coordinates
    :param box_size: int, size of stratification box
    :param shape: tuple, image shape
    :param resample: bool, resample if out o box
    :return:
    """
    box_count = [int(np.ceil(s / box_size)) for s in shape]
    coords = []

    for ic in product(*[np.arange(bc) for bc in box_count]):
        sampled = False
        while not sampled:
            coord = tuple(np.random.rand() * box_size for _ in shape)
            coord = [int(i * box_size + c) for i, c in zip(ic, coord)]
            if all(c < s for c, s in zip(coord, shape)):
                coords.append(coord)
                sampled = True
            if not resample:
                break

    coords = tuple(zip(*coords))  # transpose (N, 3) -> (3, N)
    return coords


def mask_like_image(
    image: np.ndarray, mask_percentage: float = 0.25, channels_last: bool = True
) -> np.ndarray:
    """
    Generates a stratified mask of image.shape
    :param image: ndarray, reference image to mask
    :param mask_percentage: float, percentage of pixels to mask, default 0.5%
    :param channels_last: bool, true to process image as channel-last (256, 256, 3)
    :return: ndarray, mask
    """
    # todo understand generator_val
    # https://github.com/divelab/Noise2Same/blob/8cdbfef5c475b9f999dcb1a942649af7026c887b/models.py#L130
    mask = np.zeros_like(image)
    n_channels = image.shape[-1 if channels_last else 0]
    channel_shape = image.shape[:-1] if channels_last else image.shape[1:]
    n_dim = len(channel_shape)
    # I think, here comes a mistake in original implementation (np.sqrt used both for 2D and 3D images)
    # If we use square root for 3D images, we do not reach the required masking percentage
    # See test_dataset.py for checks
    box_size = np.round(np.power(100 / mask_percentage, 1 / n_dim)).astype(int)
    for c in range(n_channels):
        mask_coords = get_stratified_coords(box_size=box_size, shape=channel_shape)
        mask_coords = (mask_coords + (c,)) if channels_last else ((c,) + mask_coords)
        mask[mask_coords] = 1.0
    return mask


class DonutMask(nn.Module):
    def __init__(self, n_dim: int = 2, in_channels: int = 1):
        """
        Local average excluding the center pixel
        :param n_dim:
        :param in_channels:
        """
        super(DonutMask, self).__init__()
        assert n_dim in (2, 3)
        self.n_dim = n_dim

        kernel = (
            np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
            if n_dim == 2
            else np.array(
                [
                    [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                    [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                    [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                ]
            )
        )
        kernel = kernel / kernel.sum()
        kernel = torch.from_numpy(kernel)[None, None]
        shape = (
            in_channels,
            in_channels,
        ) + (-1,) * n_dim
        kernel = kernel.expand(shape)
        self.register_buffer("kernel", kernel)

    def forward(self, x: T) -> T:
        conv = F.conv2d if self.n_dim == 2 else F.conv3d
        # todo understand stride
        return conv(x, self.kernel, padding=1, stride=1)

def apply_complex_mask(
    image: np.ndarray, mask_percentage: float = 0.25, channels_last: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies a stratified mask to a complex-valued image using the same mask for both real and imaginary parts.
    :param image: ndarray, complex-valued image to mask
    :param mask_percentage: float, percentage of pixels to mask, default 0.25
    :param channels_last: bool, true to process image as channel-last (256, 256, 3)
    :return: tuple, (masked_image, mask, mask, label_image)
    """
    # Create a single mask for both real and imaginary parts
    mask = mask_like_image(image.real, mask_percentage=mask_percentage, channels_last=channels_last)

    # Generate random noise
    noise = np.random.rand(*image.shape)

    # Apply mask and add noise to both real and imaginary parts
    masked_image_real = image.real * (1 - mask) + noise.real * mask
    masked_image_imag = image.imag * (1 - mask) + noise.imag * mask

    # Combine real and imaginary parts
    masked_image = masked_image_real + 1j * masked_image_imag

    # Label image (masked original image)
    label_image = image.real * mask + 1j * image.imag * mask

    return masked_image, mask, mask, label_image


# def apply_real_mask(
#     image: np.ndarray, mask_percentage: float = 0.25, channels_last: bool = False, 
#     interpolate: bool = False
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Applies a stratified mask to a real-valued image and fills the masked regions 
#     either using convolutional interpolation or random values.

#     :param image: ndarray, real-valued image to mask
#     :param mask_percentage: float, percentage of pixels to mask, default 0.25
#     :param channels_last: bool, true if image format is channel-last (e.g., 256, 256, 3)
#     :param interpolate: bool, true to fill masked regions using conv interpolation, 
#                         false to fill with random values
#     :return: tuple, (masked_image, mask)
#     """
#     mask = mask_like_image(image, mask_percentage=mask_percentage, channels_last=channels_last)
#     mask_kernel = DonutMask(n_dim=2, in_channels=1).float() 

#     def conv_interpolate(image, mask):
#         image_tensor = torch.from_numpy(image).float()
#         interpolated_image_T = mask_kernel(image_tensor[None])
#         interpolated_image = interpolated_image_T.numpy()[0, :, :]
#         return image * (1 - mask) + interpolated_image * mask

#     if interpolate:
#         masked_image = conv_interpolate(image, mask)
#     else:
#         noise = np.random.rand(*image.shape)
#         masked_image = image * (1 - mask) + noise * mask

#     return masked_image, mask

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def identity(t, *args, **kwargs):
    return t



def apply_real_mask(image, mask_percentage=0.25, channels_last=False, 
                    interpolate=False, C2S=False, autonorm=True):
    """
    Applies a stratified mask to a real-valued image and fills the masked regions 
    either using convolutional interpolation, a corrupted image, or random values.

    Parameters:
    image (np.ndarray): The input image to mask.
    mask_percentage (float): The percentage of pixels to mask.
    channels_last (bool): True if image format is channel-last (e.g., 256, 256, 3).
    interpolate (bool): True to fill masked regions using convolutional interpolation.
    C2S (bool): True to fill masked regions with a corrupted image using `corrupt` function.
    autonorm (bool): True to normalize image before corruption and unnormalize afterwards.

    Returns:
    tuple: A tuple containing the masked image and the mask used.
    """
    if mask_percentage == 0:
        return image, np.zeros_like(image, dtype=np.float32)

    mask = mask_like_image(image, mask_percentage=mask_percentage, channels_last=channels_last)
    mask_kernel = DonutMask(n_dim=2, in_channels=1).float()

    def conv_interpolate(image, mask):
        image_tensor = torch.from_numpy(image).float()
        interpolated_image_T = mask_kernel(image_tensor[None])
        interpolated_image = interpolated_image_T.numpy()[0, :, :]
        return image * (1 - mask) + interpolated_image * mask

    if interpolate:
        masked_image = conv_interpolate(image, mask)
    else:
        if C2S:
            if autonorm:
                image = normalize_to_neg_one_to_one(image)
            amount = np.random.rand()  # Random corruption strength
            noisy_image = corrupt(image, amount)
            if autonorm:
                noisy_image = unnormalize_to_zero_to_one(noisy_image)
            masked_image = image * (1 - mask) + noisy_image * mask
        else:
            noise = np.random.rand(*image.shape)
            masked_image = image * (1 - mask) + noise * mask

    return masked_image, mask


def apply_dual_contrast_mask(
    image_T1: np.ndarray, image_T2: np.ndarray, mask_percentage: float = 0.25, 
    channels_last: bool = False, apply_same_mask: bool = True, 
    interpolate: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies a stratified mask to two MRI images (dual-contrast) separately, 
    with an option to use the same mask for both and an option to interpolate masked points.
    """
    mask_kernel = DonutMask(n_dim=2, in_channels=1).float() 

    def conv_interpolate(image, mask):
        # Assuming DonutMask and mask_kernel are defined and properly initialized
        image_tensor = torch.from_numpy(image).float() 
        interpolated_image_T = mask_kernel(image_tensor[None])
        interpolated_image = interpolated_image_T.numpy()[0, :, :]
        return image * (1 - mask) + interpolated_image * mask

    if apply_same_mask:
        mask = mask_like_image(image_T1, mask_percentage=mask_percentage, channels_last=channels_last)
        mask_T1 = mask
        mask_T2 = mask
    else:
        mask_T1 = mask_like_image(image_T1, mask_percentage=mask_percentage, channels_last=channels_last)
        mask_T2 = mask_like_image(image_T2, mask_percentage=mask_percentage, channels_last=channels_last)

    if interpolate:
        masked_image_T1 = conv_interpolate(image_T1, mask_T1)
        masked_image_T2 = conv_interpolate(image_T2, mask_T2)
    else:
        noise_T1 = np.random.rand(*image_T1.shape)
        noise_T2 = np.random.rand(*image_T2.shape)
        masked_image_T1 = image_T1 * (1 - mask_T1) + noise_T1 * mask_T1
        masked_image_T2 = image_T2 * (1 - mask_T2) + noise_T2 * mask_T2

    label_image_T1 = image_T1 * mask_T1
    label_image_T2 = image_T2 * mask_T2
    label_image = np.concatenate((label_image_T1, label_image_T2), axis=0)

    return masked_image_T1, masked_image_T2, mask_T1, mask_T2, label_image